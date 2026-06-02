#!/bin/bash
# Plan B (Cosine LR Annealing + B-team fix) を 4 seed で順次学習し、
# それぞれの test inference まで実行する。
#
# seed=2023 は既存 (results_planB_cosine/) なので、合計 5seed で hybrid Top-1 に統合できる。
#
# Usage:
#   screen -S planB_multi
#   cd ~/Datascience2026_cgdetr
#   bash multi_seed_planB.sh 2>&1 | tee multi_seed_planB.log
#   # 起動確認後 Ctrl-A D
#
# 1seed = 約20分(Plan B 実測)なので 4 seed = 約80分 = 1.5時間で完走見込み。

set -e

cd ~/Datascience2026_cgdetr/src
source ../.venv/bin/activate

SEEDS=(1 11 101 2024)
RESUME=$HOME/Datascience2026_cgdetr/results_pretraining/best.ckpt
CONFIG_BASE=$HOME/Datascience2026_cgdetr/config_planB_cosine.yml

START_TIME=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    RESDIR="results_planB_seed${SEED}"
    CONFIG_TMP=$HOME/Datascience2026_cgdetr/config_planB_seed${SEED}.yml

    # config を Plan B base からコピーして seed と results_dir だけ変更
    sed -e "s/^seed:.*/seed: ${SEED}/" \
        -e "s|^results_dir:.*|results_dir: ${RESDIR}|" \
        $CONFIG_BASE > $CONFIG_TMP

    echo ""
    echo "============================================"
    echo "  $(date '+%H:%M:%S') Plan B Training seed=${SEED}"
    echo "  results_dir = ${RESDIR}"
    echo "============================================"

    python train.py --config $CONFIG_TMP --resume $RESUME

    echo ""
    echo "--- $(date '+%H:%M:%S') Inference (test) for seed=${SEED} ---"

    python evaluate.py --config $CONFIG_TMP \
        --model_path $HOME/Datascience2026_cgdetr/${RESDIR}/best.ckpt \
        --split test

    echo "--- seed=${SEED} done ---"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "============================================"
echo "  All 4 Plan B seeds finished in $((ELAPSED / 60)) min"
echo "============================================"
date
