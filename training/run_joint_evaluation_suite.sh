#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

FS_ROOT="$EXPERIMENTS_DIR"
BATCH_SIZE="${1:-8}"
OUTPUT_ROOT="${FS_ROOT}/evaluation_joint"
LOG_PATH="${FS_ROOT}/logs/joint_evaluation_suite.log"

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Model C evaluation suite started at $(date --iso-8601=seconds)"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  label \
  $STAGE1_TEST \
  "${OUTPUT_ROOT}/label/stage1" \
  "${BATCH_SIZE}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  label \
  $MANUAL_TEST \
  "${OUTPUT_ROOT}/label/manual" \
  "${BATCH_SIZE}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  narrative \
  $STAGE1_TEST \
  "${OUTPUT_ROOT}/narrative/stage1" \
  "${BATCH_SIZE}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  narrative \
  $MANUAL_TEST \
  "${OUTPUT_ROOT}/narrative/manual" \
  "${BATCH_SIZE}"

source "$CONDA_SH"
conda activate vllm-eval
python "${PROJECT_ROOT}/analysis/compare_model_abc.py"

date --iso-8601=seconds > "${FS_ROOT}/MODEL_C_EVALUATION_COMPLETE"
echo "Model C evaluation suite completed at $(date --iso-8601=seconds)"
