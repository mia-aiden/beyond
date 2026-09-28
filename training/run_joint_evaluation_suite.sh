#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/beyond"
FS_ROOT="/root/autodl-fs/sft_experiments"
BATCH_SIZE="${1:-8}"
OUTPUT_ROOT="${FS_ROOT}/evaluation_joint"
LOG_PATH="${FS_ROOT}/logs/joint_evaluation_suite.log"

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Model C evaluation suite started at $(date --iso-8601=seconds)"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  label \
  /root/autodl-fs/train_set_stage1_test_eval.csv \
  "${OUTPUT_ROOT}/label/stage1" \
  "${BATCH_SIZE}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  label \
  /root/autodl-fs/manual_test_ekman_eval.csv \
  "${OUTPUT_ROOT}/label/manual" \
  "${BATCH_SIZE}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  narrative \
  /root/autodl-fs/train_set_stage1_test_eval.csv \
  "${OUTPUT_ROOT}/narrative/stage1" \
  "${BATCH_SIZE}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  narrative \
  /root/autodl-fs/manual_test_ekman_eval.csv \
  "${OUTPUT_ROOT}/narrative/manual" \
  "${BATCH_SIZE}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-eval
python "${PROJECT_ROOT}/training/compare_model_abc.py"

date --iso-8601=seconds > "${FS_ROOT}/MODEL_C_EVALUATION_COMPLETE"
echo "Model C evaluation suite completed at $(date --iso-8601=seconds)"
