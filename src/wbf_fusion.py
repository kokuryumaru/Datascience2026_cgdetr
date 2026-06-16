"""
Weighted Box Fusion (WBF) for moment retrieval predictions.

Fuses multiple per-seed submission jsonl files into one merged prediction.
For each (qid), groups overlapping windows across seeds, then averages
their (start, end) weighted by confidence score.

Reference (image domain):
  Solovyev et al., "Weighted boxes fusion: Ensembling boxes for object detection",
  Image and Vision Computing, 2021.

Usage:
    python wbf_fusion.py \
        -i ../results_v2_btfix/submission_test.jsonl \
           ../results_v2_btfix_seed1/submission_test.jsonl \
           ../results_v2_btfix_seed11/submission_test.jsonl \
           ../results_v2_btfix_seed101/submission_test.jsonl \
           ../results_v2_btfix_seed2024/submission_test.jsonl \
        -o ../results_v2_btfix/submission_test_wbf.jsonl \
        --iou_thr 0.5
"""

import json
import argparse
from collections import defaultdict


def temporal_iou(a, b):
    """1D IoU for two [start, end] windows."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def wbf_one_query(preds_per_seed, iou_thr=0.5):
    """
    preds_per_seed: list of list of [start, end, score]
                    (one list per seed; each seed has K candidates)

    Returns merged list of [start, end, score], sorted by score desc.
    """
    # flatten all candidates with seed-id (just for traceability, not used)
    all_cands = []
    for seed_idx, preds in enumerate(preds_per_seed):
        for p in preds:
            if len(p) >= 3:
                all_cands.append([float(p[0]), float(p[1]), float(p[2])])
    if not all_cands:
        return []

    # sort by score desc
    all_cands.sort(key=lambda x: x[2], reverse=True)

    clusters = []  # list of lists of candidates
    for cand in all_cands:
        # find existing cluster whose representative IoU >= thr
        merged = False
        for cl in clusters:
            rep = cl[0]
            if temporal_iou(cand[:2], rep[:2]) >= iou_thr:
                cl.append(cand)
                merged = True
                break
        if not merged:
            clusters.append([cand])

    # for each cluster, compute weighted-average (start, end) and mean score
    fused = []
    for cl in clusters:
        weights = [c[2] for c in cl]
        wsum = sum(weights)
        if wsum <= 0:
            # fallback to simple mean
            s = sum(c[0] for c in cl) / len(cl)
            e = sum(c[1] for c in cl) / len(cl)
            sc = sum(c[2] for c in cl) / len(cl)
        else:
            s = sum(c[0] * c[2] for c in cl) / wsum
            e = sum(c[1] * c[2] for c in cl) / wsum
            # mean score boosted by number of supporters (capped)
            sc = (sum(c[2] for c in cl) / len(cl))
        fused.append([s, e, sc])

    fused.sort(key=lambda x: x[2], reverse=True)
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '-i', nargs='+', required=True,
                        help='per-seed submission jsonl files')
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--iou_thr', type=float, default=0.5,
                        help='IoU threshold to merge windows')
    parser.add_argument('--top_k', type=int, default=10,
                        help='keep this many top windows per query in output')
    args = parser.parse_args()

    # load all jsonl files into qid -> list of seed preds
    qid_seed_preds = defaultdict(list)
    qid_meta = {}

    for f_idx, fname in enumerate(args.inputs):
        with open(fname) as f:
            for line in f:
                d = json.loads(line)
                qid = d['qid']
                preds = d.get('pred_relevant_windows', [])
                qid_seed_preds[qid].append(preds)
                if qid not in qid_meta:
                    qid_meta[qid] = {
                        'qid': qid,
                        'query': d.get('query', ''),
                        'vid': d.get('vid', ''),
                    }

    print(f"Loaded {len(qid_seed_preds)} queries from {len(args.inputs)} files")

    # fuse per query
    with open(args.output, 'w') as out:
        for qid, preds_per_seed in qid_seed_preds.items():
            fused = wbf_one_query(preds_per_seed, iou_thr=args.iou_thr)
            fused = fused[:args.top_k]
            data = dict(qid_meta[qid])
            data['pred_relevant_windows'] = [
                [float(f"{e[0]:.4f}"), float(f"{e[1]:.4f}"), float(f"{e[2]:.4f}")]
                for e in fused
            ]
            out.write(json.dumps(data) + '\n')

    print(f"Saved fused predictions to {args.output}  (top_k={args.top_k}, iou_thr={args.iou_thr})")


if __name__ == '__main__':
    main()
