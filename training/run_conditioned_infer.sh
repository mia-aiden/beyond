#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
GPU="$1"; MODE="$2"; ADAPTER="$3"
source "$CONDA_SH"
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
cd $PROJECT_ROOT/training
CE=$EXPERIMENTS_DIR/conditioned_eval
python infer_conditioned.py --mode "$MODE" --adapter "$ADAPTER" \
  --gold $STAGE1_TEST \
  --output "$CE/$MODE/stage1/predictions/pred.csv"
python infer_conditioned.py --mode "$MODE" --adapter "$ADAPTER" \
  --gold $MANUAL_TEST \
  --output "$CE/$MODE/manual/predictions/pred.csv"
echo "DONE $MODE"
