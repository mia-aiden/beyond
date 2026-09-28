#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

ROOT="$PROJECT_ROOT"
FS_ROOT="$EXPERIMENTS_DIR"
STAGE1_GOLD="$STAGE1_TEST"
MANUAL_GOLD="$MANUAL_TEST"

source "$CONDA_SH"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

evaluate_prediction() {
  local model="$1"
  local gold="$2"
  local dataset_name="$3"
  local prediction_path="${FS_ROOT}/predictions/narrative/${dataset_name}_${model}.csv"
  local output_root="${FS_ROOT}/evaluation/narrative/${dataset_name}_${model}"
  test -f "${prediction_path}"
  mkdir -p "${output_root}"
  conda run -n vllm-eval python "${ROOT}/evaluation/src/run_eval.py" \
    --gold "${gold}" \
    --pred "${prediction_path}" \
    --run-defaults "${ROOT}/training/configs/run_narrative_eval.yaml" \
    --output-root "${output_root}"
}

evaluate_prediction base "${STAGE1_GOLD}" stage1
evaluate_prediction narrative "${STAGE1_GOLD}" stage1
evaluate_prediction base "${MANUAL_GOLD}" manual
evaluate_prediction narrative "${MANUAL_GOLD}" manual

conda run -n vllm-eval python "${ROOT}/training/summarize_sft_results.py"
touch "${FS_ROOT}/PIPELINE_COMPLETE"
echo "Narrative metrics completed."
