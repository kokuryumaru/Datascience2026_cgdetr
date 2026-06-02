"""
Weighted Box Fusion (WBF) for moment retrieval predictions.

Fuses multiple per-seed submission jsonl files into one merged prediction.
For each (qid), groups overlapping windows across seeds, then averages
their (start, end) weighted by confidence score.

Reference (image domain):
  Solovyev et al., "Weighted boxes fusion: Ensembling boxes for object detection",
  Image and Vision Computing, 2021.

Usage:
    # Recommended: hybrid Top-1 (preserves coordinates, boosts agreed windows)
    python wbf_fusion.py \
        -i results_seed2023/submission_test.jsonl \
           results_seed1/submission_test.jsonl \
           results_seed11/submission_test.jsonl \
           results_seed101/submission_test.jsonl \
           results_seed2024/submission_test.jsonl \
        -o results/submission_test_hybrid.jsonl \
        --strategy hybrid --top_k_in 1 --iou_thr 0.5

    # Cross-architecture ensemble (src + higashino predictions)
    python wbf_fusion.py \
        -i results/submission_test.jsonl \
           private/higashino/predictions/higashino_5seed_hybrid_top1.jsonl \
        -o results/submission_test_team_ensemble.jsonl \
        --strategy hybrid --top_k_in 1 --iou_thr 0.5

    # Standard WBF (NOT recommended for moment retrieval: ~-7pt on R1@0.7)
    python wbf_fusion.py \
        -i results_seed2023/submission_test.jsonl \
           results_seed1/submission_test.jsonl \
        -o results/submission_test_wbf.jsonl \
        --strategy wbf --iou_thr 0.5
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


def _cluster(all_cands, iou_thr):
    """Greedily cluster candidates by IoU. Returns list of clusters (each a list of cands)."""
    clusters = []
    for cand in all_cands:
        merged = False
        for cl in clusters:
            if temporal_iou(cand[:2], cl[0][:2]) >= iou_thr:
                cl.append(cand)
                merged = True
                break
        if not merged:
            clusters.append([cand])
    return clusters


def wbf_one_query(preds_per_seed, iou_thr=0.5, top_k_in=None):
    """Standard WBF: coordinate-weighted average within each cluster.

    WARNING: averaging coordinates hurts R1@0.7 in moment retrieval (~-7pt).
    Use hybrid_one_query instead for better accuracy.
    """
    all_cands = []
    for preds in preds_per_seed:
        for p in (preds[:top_k_in] if top_k_in else preds):
            if len(p) >= 3:
                all_cands.append([float(p[0]), float(p[1]), float(p[2])])
    if not all_cands:
        return []

    all_cands.sort(key=lambda x: x[2], reverse=True)
    clusters = _cluster(all_cands, iou_thr)

    fused = []
    for cl in clusters:
        wsum = sum(c[2] for c in cl)
        if wsum <= 0:
            s = sum(c[0] for c in cl) / len(cl)
            e = sum(c[1] for c in cl) / len(cl)
        else:
            s = sum(c[0] * c[2] for c in cl) / wsum
            e = sum(c[1] * c[2] for c in cl) / wsum
        sc = sum(c[2] for c in cl) / len(cl)
        fused.append([s, e, sc])

    fused.sort(key=lambda x: x[2], reverse=True)
    return fused


def hybrid_one_query(preds_per_seed, iou_thr=0.5, top_k_in=None):
    """Hybrid Top-1: coordinates from highest-score candidate, score by agreement.

    For each cluster:
      - (start, end) = coordinates of the highest-scoring candidate (no averaging)
      - score        = mean_score_in_cluster * (num_agreeing_seeds / total_seeds)

    Preserves precise boundaries (critical for R1@0.7) while boosting windows
    that multiple seeds agree on.
    """
    num_seeds = len(preds_per_seed)

    all_cands = []
    for seed_idx, preds in enumerate(preds_per_seed):
        for p in (preds[:top_k_in] if top_k_in else preds):
            if len(p) >= 3:
                all_cands.append([float(p[0]), float(p[1]), float(p[2]), seed_idx])
    if not all_cands:
        return []

    all_cands.sort(key=lambda x: x[2], reverse=True)
    clusters = _cluster(all_cands, iou_thr)

    fused = []
    for cl in clusters:
        # coordinates: highest-score candidate (list is already sorted desc)
        s, e = cl[0][0], cl[0][1]
        # agreement: how many distinct seeds contributed to this cluster
        n_agree = len(set(c[3] for c in cl))
        mean_score = sum(c[2] for c in cl) / len(cl)
        score = mean_score * (n_agree / num_seeds)
        fused.append([s, e, score])

    fused.sort(key=lambda x: x[2], reverse=True)
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', '-i', nargs='+', required=True,
                        help='per-seed submission jsonl files')
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--strategy', type=str, default='hybrid',
                        choices=['wbf', 'hybrid'],
                        help='wbf: weighted-avg coords (bad for R1@0.7); '
                             'hybrid: keep best-score coords, score by agreement (recommended)')
    parser.add_argument('--iou_thr', type=float, default=0.5,
                        help='IoU threshold to merge windows into one cluster')
    parser.add_argument('--top_k_in', type=int, default=None,
                        help='use only top-N candidates per seed (None = use all)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='keep this many top windows per query in output')
    args = parser.parse_args()

    fuse_fn = hybrid_one_query if args.strategy == 'hybrid' else wbf_one_query

    # load all jsonl files into qid -> list of seed preds
    qid_seed_preds = defaultdict(list)
    qid_meta = {}

    for fname in args.inputs:
        with open(fname) as f:
            for line in f:
                d = json.loads(line)
                qid = d['qid']
                qid_seed_preds[qid].append(d.get('pred_relevant_windows', []))
                if qid not in qid_meta:
                    qid_meta[qid] = {k: d[k] for k in ('qid', 'query', 'vid') if k in d}

    print(f"Loaded {len(qid_seed_preds)} queries from {len(args.inputs)} files "
          f"(strategy={args.strategy}, top_k_in={args.top_k_in}, iou_thr={args.iou_thr})")

    with open(args.output, 'w') as out:
        for qid, preds_per_seed in qid_seed_preds.items():
            fused = fuse_fn(preds_per_seed, iou_thr=args.iou_thr, top_k_in=args.top_k_in)
            fused = fused[:args.top_k]
            data = dict(qid_meta[qid])
            data['pred_relevant_windows'] = [
                [float(f"{e[0]:.4f}"), float(f"{e[1]:.4f}"), float(f"{e[2]:.4f}")]
                for e in fused
            ]
            out.write(json.dumps(data) + '\n')

    print(f"Saved fused predictions to {args.output}")


if __name__ == '__main__':
    main()
