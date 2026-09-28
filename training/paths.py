"""Paths shared by the training and analysis scripts.

The defaults are the AutoDL server layout used for the thesis runs. Set an
environment variable with the same name to point a path somewhere else.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_data = Path(os.environ.get("DATA_DIR", "/root/autodl-fs"))
STAGE1_TEST = Path(os.environ.get("STAGE1_TEST", _data / "train_set_stage1_test_eval.csv"))
MANUAL_TEST = Path(os.environ.get("MANUAL_TEST", _data / "manual_test_ekman_eval.csv"))
TRAIN_POOL = Path(os.environ.get("TRAIN_POOL", _data / "train_set_stage1_train_no_test_overlap.csv"))
RAW_PARQUET = Path(os.environ.get("RAW_PARQUET", "/root/autodl-tmp/data.parquet"))

EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", "/root/autodl-fs/sft_experiments"))
BASE_MODEL = os.environ.get("BASE_MODEL", "/root/autodl-tmp/Meta-Llama-3-8B-Instruct")
LLAMAFACTORY_DIR = Path(os.environ.get("LLAMAFACTORY_DIR", "/root/autodl-tmp/LLaMA-Factory"))

LLAMAFACTORY_PYTHON = os.environ.get("LLAMAFACTORY_PYTHON", "/root/miniconda3/envs/llamafactory/bin/python")
LLAMAFACTORY_CLI = os.environ.get("LLAMAFACTORY_CLI", "/root/miniconda3/envs/llamafactory/bin/llamafactory-cli")
VLLM_PYTHON = os.environ.get("VLLM_PYTHON", "/root/miniconda3/envs/vllm-eval/bin/python")
VLLM_BIN = os.environ.get("VLLM_BIN", "/root/miniconda3/envs/vllm-eval/bin/vllm")
