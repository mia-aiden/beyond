#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

FS_ROOT="$EXPERIMENTS_DIR"
MODEL_DIR="${FS_ROOT}/models/joint_lora_r8"
LOG_PATH="${FS_ROOT}/logs/joint_finish_pipeline.log"
ARTIFACT_ROOT="${FS_ROOT}/model_c_artifacts"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Model C finish pipeline started at $(date --iso-8601=seconds)"

source "$CONDA_SH"
conda activate llamafactory
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -f "${MODEL_DIR}/joint_early_stopping_decision.json" ]]; then
  echo "Using existing early-stopping decision."
else
  python "${PROJECT_ROOT}/training/monitor_joint_training.py"
fi
python "${PROJECT_ROOT}/training/finalize_joint_model.py"
python "${PROJECT_ROOT}/training/evaluate_joint_checkpoint.py" \
  --adapter "${MODEL_DIR}" \
  --output "${MODEL_DIR}/joint_validation_components.json"

"${PROJECT_ROOT}/training/run_joint_evaluation_suite.sh" 8

conda activate llamafactory
python "${PROJECT_ROOT}/training/build_model_c_report.py"

mkdir -p "${ARTIFACT_ROOT}/code"
cp "${PROJECT_ROOT}/training/prepare_joint_data.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/train_joint_multitask.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/finalize_joint_model.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/evaluate_joint_checkpoint.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/monitor_joint_training.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/infer_sft_task_transformers.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/run_joint_eval.sh" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/run_joint_evaluation_suite.sh" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/finish_joint_pipeline.sh" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/analysis/compare_model_abc.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/build_model_c_report.py" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/README.md" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/configs/joint_sft.yaml" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/configs/run_label_eval.yaml" "${ARTIFACT_ROOT}/code/"
cp "${PROJECT_ROOT}/training/configs/run_narrative_eval.yaml" "${ARTIFACT_ROOT}/code/"

find "${ARTIFACT_ROOT}/code" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${ARTIFACT_ROOT}/code_sha256.txt"

date --iso-8601=seconds > "${FS_ROOT}/MODEL_C_PIPELINE_COMPLETE"
echo "Model C finish pipeline completed at $(date --iso-8601=seconds)"
