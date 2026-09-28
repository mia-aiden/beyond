#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/beyond"
FS_ROOT="/root/autodl-fs/sft_experiments"
ABLATION_ROOT="${FS_ROOT}/ablations"

names=(lambda_0_3 lambda_0_5 lambda_0_8)
short_names=(l03 l05 l08)
gpus=(0 1 2)
configs=(
  "${PROJECT_ROOT}/training/configs/joint_lambda_03.yaml"
  "${PROJECT_ROOT}/training/configs/joint_lambda_05.yaml"
  "${PROJECT_ROOT}/training/configs/joint_lambda_08.yaml"
)

for index in 0 1 2; do
  root="${ABLATION_ROOT}/${names[$index]}"
  model_dir="${root}/models/joint_lora_r8"
  if [[ -e "${model_dir}" ]]; then
    echo "Refusing to overwrite existing model directory: ${model_dir}" >&2
    exit 1
  fi
  mkdir -p "${root}/logs"
done

for index in 0 1 2; do
  name="${names[$index]}"
  short="${short_names[$index]}"
  gpu="${gpus[$index]}"
  config="${configs[$index]}"
  root="${ABLATION_ROOT}/${name}"
  model_dir="${root}/models/joint_lora_r8"
  train_screen="joint_${short}_train"
  finish_screen="joint_${short}_finish"

  screen -dmS "${train_screen}" bash -lc \
    "source /root/miniconda3/etc/profile.d/conda.sh && conda activate llamafactory && export CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 && python ${PROJECT_ROOT}/training/train_joint_multitask.py --config ${config} >> ${root}/logs/train.log 2>&1"

  screen -dmS "${finish_screen}" bash -lc \
    "source /root/miniconda3/etc/profile.d/conda.sh && conda activate llamafactory && ${PROJECT_ROOT}/training/finish_joint_ablation.sh ${name} ${gpu} ${config} ${model_dir} ${train_screen} ${root}"
done

screen -dmS joint_lambda_suite bash -lc \
  "${PROJECT_ROOT}/training/finish_joint_ablation_suite.sh"

sleep 5
screen -list
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
