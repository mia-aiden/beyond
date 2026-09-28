#!/usr/bin/env bash
set -euo pipefail
GPU="$1"; MODE="$2"; ADAPTER="$3"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
cd /root/autodl-tmp/beyond/training
CE=/root/autodl-fs/sft_experiments/conditioned_eval
python infer_conditioned.py --mode "$MODE" --adapter "$ADAPTER" \
  --gold /root/autodl-fs/train_set_stage1_test_eval.csv \
  --output "$CE/$MODE/stage1/predictions/pred.csv"
python infer_conditioned.py --mode "$MODE" --adapter "$ADAPTER" \
  --gold /root/autodl-fs/manual_test_ekman_eval.csv \
  --output "$CE/$MODE/manual/predictions/pred.csv"
echo "DONE $MODE"
