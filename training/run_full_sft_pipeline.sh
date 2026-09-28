#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

LABEL_TRAIN_PID="${1:?Usage: run_full_sft_pipeline.sh <running-label-train-pid>}"
ROOT="$PROJECT_ROOT"
FS_ROOT="$EXPERIMENTS_DIR"
STAGE1_GOLD="$STAGE1_TEST"
MANUAL_GOLD="$MANUAL_TEST"
BASE_URL="http://127.0.0.1:8000/v1"

source "$CONDA_SH"
export OMP_NUM_THREADS=8

wait_for_process() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
}

wait_for_server() {
  local attempts=0
  until curl --fail --silent "${BASE_URL}/models" >/dev/null; do
    attempts=$((attempts + 1))
    if (( attempts >= 180 )); then
      echo "vLLM server did not become ready." >&2
      return 1
    fi
    sleep 2
  done
}

start_server() {
  local task="$1"
  local log_path="${FS_ROOT}/logs/vllm_${task}.log"
  conda deactivate 2>/dev/null || true
  setsid "${ROOT}/training/serve_sft_adapter.sh" "${task}" >"${log_path}" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

stop_server() {
  kill -- "-${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  sleep 10
}

generate_prediction() {
  local task="$1"
  local model="$2"
  local gold="$3"
  local dataset_name="$4"
  local prediction_path="${FS_ROOT}/predictions/${task}/${dataset_name}_${model}.csv"
  mkdir -p "$(dirname "${prediction_path}")"
  conda run -n vllm-eval python "${ROOT}/training/infer_sft_task_vllm.py" \
    --task "${task}" \
    --gold "${gold}" \
    --output "${prediction_path}" \
    --model-name "${model}"
}

evaluate_prediction() {
  local task="$1"
  local model="$2"
  local gold="$3"
  local dataset_name="$4"
  local prediction_path="${FS_ROOT}/predictions/${task}/${dataset_name}_${model}.csv"
  local defaults="${ROOT}/training/configs/run_${task}_eval.yaml"
  local output_root="${FS_ROOT}/evaluation/${task}/${dataset_name}_${model}"
  mkdir -p "${output_root}"
  conda run -n vllm-eval python "${ROOT}/evaluation/src/run_eval.py" \
    --gold "${gold}" \
    --pred "${prediction_path}" \
    --run-defaults "${defaults}" \
    --output-root "${output_root}"
}

mkdir -p "${FS_ROOT}/logs" "${FS_ROOT}/predictions" "${FS_ROOT}/evaluation"

echo "Waiting for label training process ${LABEL_TRAIN_PID}."
wait_for_process "${LABEL_TRAIN_PID}"
test -f "${FS_ROOT}/models/label_lora_r8/adapter_model.safetensors"

echo "Running base and label-adapter inference."
start_server label
generate_prediction label base "${STAGE1_GOLD}" stage1
generate_prediction label label "${STAGE1_GOLD}" stage1
generate_prediction label base "${MANUAL_GOLD}" manual
generate_prediction label label "${MANUAL_GOLD}" manual
stop_server

evaluate_prediction label base "${STAGE1_GOLD}" stage1
evaluate_prediction label label "${STAGE1_GOLD}" stage1
evaluate_prediction label base "${MANUAL_GOLD}" manual
evaluate_prediction label label "${MANUAL_GOLD}" manual

echo "Starting narrative LoRA training."
conda activate llamafactory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$LLAMAFACTORY_DIR"
llamafactory-cli train "${ROOT}/training/configs/narrative_sft.yaml" \
  >"${FS_ROOT}/logs/narrative_sft.log" 2>&1
conda deactivate
test -f "${FS_ROOT}/models/narrative_lora_r8/adapter_model.safetensors"

echo "Running base and narrative-adapter inference."
start_server narrative
generate_prediction narrative base "${STAGE1_GOLD}" stage1
generate_prediction narrative narrative "${STAGE1_GOLD}" stage1
generate_prediction narrative base "${MANUAL_GOLD}" manual
generate_prediction narrative narrative "${MANUAL_GOLD}" manual
stop_server

evaluate_prediction narrative base "${STAGE1_GOLD}" stage1
evaluate_prediction narrative narrative "${STAGE1_GOLD}" stage1
evaluate_prediction narrative base "${MANUAL_GOLD}" manual
evaluate_prediction narrative narrative "${MANUAL_GOLD}" manual

touch "${FS_ROOT}/PIPELINE_COMPLETE"
echo "Full two-task SFT pipeline completed."
