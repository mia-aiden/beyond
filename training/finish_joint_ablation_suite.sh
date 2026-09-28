#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

FS_ROOT="$EXPERIMENTS_DIR"
ABLATION_ROOT="${FS_ROOT}/ablations"
LOG_PATH="${ABLATION_ROOT}/lambda_comparison/suite.log"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Lambda ablation suite monitor started at $(date --iso-8601=seconds)"

experiments=(lambda_0_3 lambda_0_5 lambda_0_8)
finish_screens=(joint_l03_finish joint_l05_finish joint_l08_finish)

for index in 0 1 2; do
  experiment="${experiments[$index]}"
  marker="${ABLATION_ROOT}/${experiment}/ABLATION_COMPLETE"
  screen_name="${finish_screens[$index]}"
  while [[ ! -f "${marker}" ]]; do
    if ! screen -list 2>/dev/null | grep -q "\.${screen_name}"; then
      echo "${screen_name} ended before writing ${marker}." >&2
      exit 1
    fi
    sleep 30
  done
  echo "${experiment} complete at $(cat "${marker}")"
done

source "$CONDA_SH"
conda activate vllm-eval
python "${PROJECT_ROOT}/training/compare_joint_lambdas.py"

mkdir -p "${ABLATION_ROOT}/artifacts"
cp "${PROJECT_ROOT}/training/compare_joint_lambdas.py" "${ABLATION_ROOT}/artifacts/"
cp "${PROJECT_ROOT}/training/finish_joint_ablation.sh" "${ABLATION_ROOT}/artifacts/"
cp "${PROJECT_ROOT}/training/finish_joint_ablation_suite.sh" "${ABLATION_ROOT}/artifacts/"
cp "${PROJECT_ROOT}/training/configs/joint_lambda_03.yaml" "${ABLATION_ROOT}/artifacts/"
cp "${PROJECT_ROOT}/training/configs/joint_lambda_05.yaml" "${ABLATION_ROOT}/artifacts/"
cp "${PROJECT_ROOT}/training/configs/joint_lambda_08.yaml" "${ABLATION_ROOT}/artifacts/"
find "${ABLATION_ROOT}/artifacts" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${ABLATION_ROOT}/artifacts_sha256.txt"

date --iso-8601=seconds > "${ABLATION_ROOT}/LAMBDA_ABLATION_COMPLETE"
echo "Lambda ablation suite completed at $(date --iso-8601=seconds)"
