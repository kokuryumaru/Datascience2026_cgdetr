#!/bin/bash
# Multi-seed Stage2 training script.
# Trains 4 additional seeds (the original seed=2023 already lives in results_v2_btfix/),
# then runs test inference for each. Run inside a `screen` session.
#
# Usage:
#   screen -S multi
#   cd ~/Datascience2026_cgdetr
#   bash multi_seed_train.sh 2>&1 | tee multi_seed.log
#   # then Ctrl-A D to detach

set -e

cd ~/Datascience2026_cgdetr/src
source ../.venv/bin/activate

SEEDS=(1 11 101 2024)
RESUME=$HOME/Datascience2026_cgdetr/results_pretraining/best.ckpt
CONFIG_BASE=$HOME/Datascience2026_cgdetr/config.yml

START_TIME=$(date +%s)

for SEED in "${SEEDS[@]}"; do
    RESDIR="results_v2_btfix_seed${SEED}"
    CONFIG_TMP=$HOME/Datascience2026_cgdetr/config_seed${SEED}.yml

    # Generate per-seed config
    sed -e "s/^seed:.*/seed: ${SEED}/" \
        -e "s/^results_dir:.*/results_dir: ${RESDIR}/" \
        $CONFIG_BASE > $CONFIG_TMP

    echo ""
    echo "============================================"
    echo "  $(date '+%H:%M:%S') Training seed=${SEED}"
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
echo "  All seeds finished in $((ELAPSED / 60)) min"
echo "============================================"
date
