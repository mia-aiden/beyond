#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
GPU="$1"; shift
source "$CONDA_SH"
conda activate llamafactory
cd "$LLAMAFACTORY_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
for cfg in "$@"; do
  echo "[queue gpu$GPU] START $cfg $(date)"
  llamafactory-cli train "$cfg"
  echo "[queue gpu$GPU] DONE  $cfg $(date)"
done
echo "[queue gpu$GPU] QUEUE COMPLETE $(date)"
