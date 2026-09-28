#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

EXPERIMENT_NAME="${1:?Usage: finish_joint_ablation.sh <name> <gpu> <config> <model-dir> <train-screen> <experiment-root>}"
GPU_INDEX="${2:?Missing GPU index.}"
CONFIG_PATH="${3:?Missing config path.}"
MODEL_DIR="${4:?Missing model directory.}"
TRAIN_SCREEN="${5:?Missing training screen name.}"
EXPERIMENT_ROOT="${6:?Missing experiment root.}"

LOG_PATH="${EXPERIMENT_ROOT}/logs/finish_pipeline.log"
DECISION_PATH="${MODEL_DIR}/joint_early_stopping_decision.json"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "$CONDA_SH"
conda activate llamafactory

echo "${EXPERIMENT_NAME} finish pipeline started at $(date --iso-8601=seconds)"

if [[ -f "${DECISION_PATH}" ]]; then
  echo "Using existing early-stopping decision."
else
  python "${PROJECT_ROOT}/training/monitor_joint_training.py" \
    --model-dir "${MODEL_DIR}" \
    --screen-name "${TRAIN_SCREEN}" \
    --output "${DECISION_PATH}"
fi

python "${PROJECT_ROOT}/training/finalize_joint_model.py" \
  --model-dir "${MODEL_DIR}"
python "${PROJECT_ROOT}/training/evaluate_joint_checkpoint.py" \
  --config "${CONFIG_PATH}" \
  --adapter "${MODEL_DIR}" \
  --output "${MODEL_DIR}/joint_validation_components.json"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  label \
  $STAGE1_TEST \
  "${EXPERIMENT_ROOT}/evaluation/label/stage1" \
  8 \
  "${MODEL_DIR}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  label \
  $MANUAL_TEST \
  "${EXPERIMENT_ROOT}/evaluation/label/manual" \
  8 \
  "${MODEL_DIR}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  narrative \
  $STAGE1_TEST \
  "${EXPERIMENT_ROOT}/evaluation/narrative/stage1" \
  8 \
  "${MODEL_DIR}"

"${PROJECT_ROOT}/training/run_joint_eval.sh" \
  narrative \
  $MANUAL_TEST \
  "${EXPERIMENT_ROOT}/evaluation/narrative/manual" \
  8 \
  "${MODEL_DIR}"

mkdir -p "${EXPERIMENT_ROOT}/artifacts"
cp "${CONFIG_PATH}" "${EXPERIMENT_ROOT}/artifacts/training_config.yaml"
sha256sum "${MODEL_DIR}/adapter_model.safetensors" \
  > "${EXPERIMENT_ROOT}/artifacts/adapter_sha256.txt"
date --iso-8601=seconds > "${EXPERIMENT_ROOT}/ABLATION_COMPLETE"
echo "${EXPERIMENT_NAME} completed at $(date --iso-8601=seconds)"
