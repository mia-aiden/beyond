#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

TASK="${1:?Usage: run_direct_sft_eval.sh <label|narrative> <gold.csv> <output-root> [batch-size]}"
GOLD_PATH="${2:?Missing gold CSV path.}"
OUTPUT_ROOT="${3:?Missing output root.}"
BATCH_SIZE="${4:-2}"
FS_ROOT="$EXPERIMENTS_DIR"

case "${TASK}" in
  label)
    ADAPTER_PATH="${FS_ROOT}/models/label_lora_r8"
    RUN_DEFAULTS="${PROJECT_ROOT}/training/configs/run_label_eval.yaml"
    ;;
  narrative)
    ADAPTER_PATH="${FS_ROOT}/models/narrative_lora_r8"
    RUN_DEFAULTS="${PROJECT_ROOT}/training/configs/run_narrative_eval.yaml"
    ;;
  *)
    echo "Task must be label or narrative." >&2
    exit 2
    ;;
esac

source "$CONDA_SH"
conda activate llamafactory
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${OUTPUT_ROOT}/predictions" "${OUTPUT_ROOT}/evaluation"
PREDICTION_PATH="${OUTPUT_ROOT}/predictions/sft_transformers_${TASK}.csv"

python "${PROJECT_ROOT}/training/infer_sft_task_transformers.py" \
  --task "${TASK}" \
  --gold "${GOLD_PATH}" \
  --output "${PREDICTION_PATH}" \
  --adapter "${ADAPTER_PATH}" \
  --batch-size "${BATCH_SIZE}"

conda activate vllm-eval
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python "${PROJECT_ROOT}/evaluation/src/run_eval.py" \
  --gold "${GOLD_PATH}" \
  --pred "${PREDICTION_PATH}" \
  --run-defaults "${RUN_DEFAULTS}" \
  --output-root "${OUTPUT_ROOT}/evaluation"
