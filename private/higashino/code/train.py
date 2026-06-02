import os
import time
import yaml
import pprint
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydict import EasyDict

from dataset import CGDETR_StartEndDataset, cg_detr_start_end_collate, cg_detr_prepare_batch_inputs
from cg_detr import build_model
from basic_utils import AverageMeter, write_log, save_checkpoint, rename_latest_to_best, mkdirp, save_jsonl, save_json
from model_utils import count_parameters, ModelEMA
from postprocessing import PostProcessorDETR
from span_utils import span_cxw_to_xx
from standalone_eval.eval import eval_submission

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_model(opt):
    model, criterion = build_model(opt)
    if opt.device == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    scheduler_type = getattr(opt, 'lr_scheduler', 'step')
    if scheduler_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epoch, eta_min=getattr(opt, 'lr_min', 0.0))
        logger.info(f"Using CosineAnnealingLR (T_max={opt.n_epoch}, eta_min={getattr(opt, 'lr_min', 0.0)})")
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
        logger.info(f"Using StepLR (step_size={opt.lr_drop})")
    return model, criterion, optimizer, lr_scheduler


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    logger.info("Saving/Evaluating results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)
    if opt.eval_split_name in ["val", "test"]:
        metrics = eval_submission(submission, gt_data)
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path]
    return metrics, latest_file_paths


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt):
    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
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


def eval_epoch(model, val_dataset, opt, save_submission_filename, criterion=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None:
        criterion.eval()
    eval_loader = DataLoader(
        val_dataset, collate_fn=cg_detr_start_end_collate,
        batch_size=opt.eval_bsz, num_workers=opt.num_workers, shuffle=False,
    )
    submission = compute_mr_results(model, eval_loader, opt)
    return eval_epoch_post_processing(submission, opt, val_dataset.data, save_submission_filename)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()
    loss_meters = defaultdict(AverageMeter)
    for batch in tqdm(train_loader, desc="Training"):
        model_inputs, targets = cg_detr_prepare_batch_inputs(batch[1], opt.device)
        outputs = model(**model_inputs, targets=targets)
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))
    write_log(opt, epoch_i, loss_meters)


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = f"latest_{opt.dset_name}_val_preds.jsonl"
    train_loader = DataLoader(
        train_dataset, collate_fn=cg_detr_start_end_collate,
        batch_size=opt.bsz, num_workers=opt.num_workers, shuffle=True,
    )
    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    prev_best_score = 0.
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i)
        lr_scheduler.step()
        if opt.model_ema:
            model_ema.update(model)
        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            with torch.no_grad():
                eval_model = model_ema.module if opt.model_ema else model
                metrics, latest_file_paths = eval_epoch(eval_model, val_dataset, opt, save_submission_filename, criterion)
            write_log(opt, epoch_i, defaultdict(AverageMeter), metrics=metrics, mode='val')
            logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
            stop_score = metrics["brief"]["MR-full-mAP"]
            if stop_score > prev_best_score:
                prev_best_score = stop_score
                save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
                logger.info("The checkpoint file has been updated.")
                rename_latest_to_best(latest_file_paths)


def main(opt, resume=None):
    logger.info("Setup config, data and model...")
    cudnn.benchmark = True
    set_seed(opt.seed)
    mkdirp(opt.results_dir)

    dataset_config = EasyDict(
        dset_name=opt.dset_name,
        domain=None,
        data_path=opt.train_path,
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
        load_labels=True,
    )
    train_dataset = CGDETR_StartEndDataset(**dataset_config)
    eval_config = copy.deepcopy(dataset_config)
    eval_config.data_path = opt.val_path
    val_dataset = CGDETR_StartEndDataset(**eval_config)

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model: {opt.model_name}")
    count_parameters(model)

    if resume is not None:
        checkpoint = torch.load(resume, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info(f"Loaded checkpoint: {resume}")

    logger.info("Start Training...")
    train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--resume', '-r', type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    os.chdir(os.path.dirname(config_path))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    opt = EasyDict(cfg)
    opt.ckpt_filepath = os.path.join(opt.results_dir, opt.ckpt_filename)
    opt.train_log_filepath = os.path.join(opt.results_dir, opt.train_log_filename)
    opt.eval_log_filepath = os.path.join(opt.results_dir, opt.eval_log_filename)

    resume = os.path.abspath(args.resume) if args.resume else None
    opt.max_v_l = opt.max_a_l
    main(opt, resume=resume)
