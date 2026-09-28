# Paths shared by the shell scripts. Source this file; the defaults are the
# AutoDL server layout used for the thesis runs, and any of them can be
# overridden by exporting a variable with the same name first.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/root/autodl-fs}"
STAGE1_TEST="${STAGE1_TEST:-$DATA_DIR/train_set_stage1_test_eval.csv}"
MANUAL_TEST="${MANUAL_TEST:-$DATA_DIR/manual_test_ekman_eval.csv}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/root/autodl-fs/sft_experiments}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/Meta-Llama-3-8B-Instruct}"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-/root/autodl-tmp/LLaMA-Factory}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
VLLM_PYTHON="${VLLM_PYTHON:-/root/miniconda3/envs/vllm-eval/bin/python}"
