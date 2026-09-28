#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: serve_sft_adapter.sh <label|narrative>}"
BASE_MODEL="/root/autodl-tmp/Meta-Llama-3-8B-Instruct"

case "${TASK}" in
  label)
    ADAPTER="/root/autodl-fs/sft_experiments/models/label_lora_r8"
    ;;
  narrative)
    ADAPTER="/root/autodl-fs/sft_experiments/models/narrative_lora_r8"
    ;;
  *)
    echo "Task must be label or narrative." >&2
    exit 2
    ;;
esac

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-eval
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve "${BASE_MODEL}" \
  --served-model-name base \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 8 \
  --lora-modules "${TASK}=${ADAPTER}"
