import os
import yaml
import pprint
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('high')
from easydict import EasyDict
from tqdm import tqdm

from dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs
from cg_detr import build_model
from basic_utils import mkdirp, save_jsonl, save_json
from postprocessing import PostProcessorDETR
from span_utils import span_cxw_to_xx
from standalone_eval.eval import eval_submission

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt):
    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores", 
                      bar_format='{percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
        query_meta = batch[0]
        model_inputs, targets = cg_detr_prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs, targets=targets)

        _saliency_scores = outputs["saliency_scores"].half()
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())

        pred_spans = outputs["pred_spans"].cpu()
        prob = torch.nn.functional.softmax(outputs["pred_logits"], -1)
        scores = prob[..., 0].cpu()

        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans, scores)):
            spans = span_cxw_to_xx(spans) * meta["duration"]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            mr_res.append(dict(
                qid=meta["qid"], query=meta["query"], vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
            ))

    post_processor = PostProcessorDETR(
        clip_length=opt.clip_length, min_ts_val=0, max_ts_val=300,
        min_w_l=1, max_w_l=300, move_window_method="left",
        process_func_names=("clip_ts", "round_multiple")
    )
    return post_processor(mr_res)


def start_inference(opt):
    logger.info("Setup config, data and model...")
    cudnn.benchmark = True
    load_labels = opt.eval_split_name in ['val', 'test']

    data_path = opt.val_path if opt.eval_split_name == 'val' else opt.test_path
    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=data_path,
        ctx_mode=opt.ctx_mode,
        v_feat_dirs=None,
        a_feat_dirs=[opt.a_feat_dir],
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        v_feat_types=None,
        a_feat_types=opt.a_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_a_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=load_labels,
    )
    eval_dataset = CGDETR_StartEndDataset(**dataset_config)

    model, criterion = build_model(opt)
    if opt.device == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
    checkpoint = torch.load(opt.model_path, weights_only=False)
    loaded_state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    adapted_state_dict = {}
    for k, v in loaded_state_dict.items():
        clean_k = k.replace('_orig_mod.', '')
        compiled_k = clean_k.replace('transformer.', 'transformer._orig_mod.')
        if compiled_k in model_state_dict:
            adapted_state_dict[compiled_k] = v
        else:
            adapted_state_dict[clean_k] = v
    model.load_state_dict(adapted_state_dict)
    logger.info(f"Model checkpoint: {opt.model_path}")
    model.eval()

    eval_loader = DataLoader(
        eval_dataset, collate_fn=cg_detr_start_end_collate,
        batch_size=opt.eval_bsz, num_workers=opt.num_workers, shuffle=False,
    )

    logger.info("Starting inference...")
    submission = compute_mr_results(model, eval_loader, opt)

    submission_path = os.path.join(opt.results_dir, f"submission_{opt.eval_split_name}.jsonl")
    save_jsonl(submission, submission_path)

    if load_labels:
        metrics = eval_submission(submission, eval_dataset.data)
        metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, metrics_path, save_pretty=True, sort_keys=False)
        logger.info("metrics:\n{}".format(pprint.pformat(dict(metrics["brief"]), indent=2)))
    else:
        logger.info(f"Submission saved to {submission_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--split', '-s', type=str, default='val', choices=['val', 'test'])
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    model_path = os.path.abspath(args.model_path)
    os.chdir(os.path.dirname(config_path))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    opt = EasyDict(cfg)
    opt.model_path = model_path
    opt.eval_split_name = args.split
    opt.max_v_l = opt.max_a_l
    mkdirp(opt.results_dir)

    start_inference(opt)
