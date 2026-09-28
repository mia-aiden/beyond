#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/beyond"
EXPERIMENT_ROOT="/root/autodl-fs/sft_experiments/mentor_followup"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
MODEL_ROOT="${EXPERIMENT_ROOT}/models"
LLAMAFACTORY_ROOT="/root/autodl-tmp/LLaMA-Factory"

joint_jobs=(
  "joint_lambda_0_0_seed42:0:joint_lambda_00_seed42.yaml"
  "joint_lambda_0_8_seed1:1:joint_lambda_08_seed1.yaml"
  "joint_lambda_0_8_seed2:2:joint_lambda_08_seed2.yaml"
)

model_dirs=(
  "${MODEL_ROOT}/joint_lambda_0_0_seed_42"
  "${MODEL_ROOT}/joint_lambda_0_8_seed_1"
  "${MODEL_ROOT}/joint_lambda_0_8_seed_2"
  "${MODEL_ROOT}/model_a_seed_1"
  "${MODEL_ROOT}/model_a_seed_2"
)

for model_dir in "${model_dirs[@]}"; do
  if [[ -e "${model_dir}" ]] && [[ -n "$(find "${model_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty model directory: ${model_dir}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_ROOT}" "${MODEL_ROOT}"

for job_spec in "${joint_jobs[@]}"; do
  IFS=: read -r job_name gpu config_name <<< "${job_spec}"
  screen -dmS "${job_name}" bash -lc \
    "source /root/miniconda3/etc/profile.d/conda.sh && conda activate llamafactory && export CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 && python ${PROJECT_ROOT}/training/train_joint_multitask.py --config ${PROJECT_ROOT}/training/configs/${config_name} >> ${LOG_ROOT}/${job_name}.log 2>&1"
done

screen -dmS model_a_seed_queue bash -lc \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate llamafactory && cd ${LLAMAFACTORY_ROOT} && export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 && echo '[model_a_seed1] START' \$(date) >> ${LOG_ROOT}/model_a_seed_queue.log && llamafactory-cli train ${PROJECT_ROOT}/training/configs/label_sft_seed1.yaml >> ${LOG_ROOT}/model_a_seed_queue.log 2>&1 && echo '[model_a_seed1] DONE' \$(date) >> ${LOG_ROOT}/model_a_seed_queue.log && echo '[model_a_seed2] START' \$(date) >> ${LOG_ROOT}/model_a_seed_queue.log && llamafactory-cli train ${PROJECT_ROOT}/training/configs/label_sft_seed2.yaml >> ${LOG_ROOT}/model_a_seed_queue.log 2>&1 && echo '[model_a_seed2] DONE' \$(date) >> ${LOG_ROOT}/model_a_seed_queue.log"

sleep 8
screen -list
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
