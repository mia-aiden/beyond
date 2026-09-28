#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: run_task_eval.sh <label|narrative> <model-name> <gold.csv> <output-root>}"
MODEL_NAME="${2:?Missing served model name.}"
GOLD_PATH="${3:?Missing gold CSV path.}"
OUTPUT_ROOT="${4:?Missing output root.}"
PROJECT_ROOT="/root/autodl-tmp/beyond"

case "${TASK}" in
  label)
    RUN_DEFAULTS="${PROJECT_ROOT}/training/configs/run_label_eval.yaml"
    ;;
  narrative)
    RUN_DEFAULTS="${PROJECT_ROOT}/training/configs/run_narrative_eval.yaml"
    ;;
  *)
    echo "Task must be label or narrative." >&2
    exit 2
    ;;
esac

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-eval
export OMP_NUM_THREADS=8

mkdir -p "${OUTPUT_ROOT}/predictions" "${OUTPUT_ROOT}/evaluation"
PREDICTION_PATH="${OUTPUT_ROOT}/predictions/${MODEL_NAME}_${TASK}.csv"

python "${PROJECT_ROOT}/training/infer_sft_task_vllm.py" \
  --task "${TASK}" \
  --gold "${GOLD_PATH}" \
  --output "${PREDICTION_PATH}" \
  --model-name "${MODEL_NAME}"

python "${PROJECT_ROOT}/evaluation/src/run_eval.py" \
  --gold "${GOLD_PATH}" \
  --pred "${PREDICTION_PATH}" \
  --run-defaults "${RUN_DEFAULTS}" \
  --output-root "${OUTPUT_ROOT}/evaluation"

