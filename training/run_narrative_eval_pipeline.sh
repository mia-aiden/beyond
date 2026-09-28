#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"

ROOT="$PROJECT_ROOT"
FS_ROOT="$EXPERIMENTS_DIR"
STAGE1_GOLD="$STAGE1_TEST"
MANUAL_GOLD="$MANUAL_TEST"
BASE_URL="http://127.0.0.1:8000/v1"
SERVER_PID=""

source "$CONDA_SH"
export OMP_NUM_THREADS=8

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    sleep 10
  fi
}
trap stop_server EXIT

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

generate_prediction() {
  local model="$1"
  local gold="$2"
  local dataset_name="$3"
  local prediction_path="${FS_ROOT}/predictions/narrative/${dataset_name}_${model}.csv"
  mkdir -p "$(dirname "${prediction_path}")"
  conda run -n vllm-eval python "${ROOT}/training/infer_sft_task_vllm.py" \
    --task narrative \
    --gold "${gold}" \
    --output "${prediction_path}" \
    --model-name "${model}"
}

evaluate_prediction() {
  local model="$1"
  local gold="$2"
  local dataset_name="$3"
  local prediction_path="${FS_ROOT}/predictions/narrative/${dataset_name}_${model}.csv"
  local output_root="${FS_ROOT}/evaluation/narrative/${dataset_name}_${model}"
  mkdir -p "${output_root}"
  conda run -n vllm-eval python "${ROOT}/evaluation/src/run_eval.py" \
    --gold "${gold}" \
    --pred "${prediction_path}" \
    --run-defaults "${ROOT}/training/configs/run_narrative_eval.yaml" \
    --output-root "${output_root}"
}

test -f "${FS_ROOT}/models/narrative_lora_r8/adapter_model.safetensors"
mkdir -p "${FS_ROOT}/logs" "${FS_ROOT}/predictions/narrative"

setsid "${ROOT}/training/serve_sft_adapter.sh" narrative \
  >"${FS_ROOT}/logs/vllm_narrative.log" 2>&1 &
SERVER_PID=$!
wait_for_server

generate_prediction base "${STAGE1_GOLD}" stage1
generate_prediction narrative "${STAGE1_GOLD}" stage1
generate_prediction base "${MANUAL_GOLD}" manual
generate_prediction narrative "${MANUAL_GOLD}" manual

stop_server
SERVER_PID=""

evaluate_prediction base "${STAGE1_GOLD}" stage1
evaluate_prediction narrative "${STAGE1_GOLD}" stage1
evaluate_prediction base "${MANUAL_GOLD}" manual
evaluate_prediction narrative "${MANUAL_GOLD}" manual

conda run -n vllm-eval python "${ROOT}/training/summarize_sft_results.py"
touch "${FS_ROOT}/PIPELINE_COMPLETE"
echo "Narrative inference and evaluation completed."
