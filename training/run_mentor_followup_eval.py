#!/usr/bin/env python3
"""Evaluate the lambda=0 endpoint and three-seed lambda=0.8/Model A runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from paths import (
    PROJECT_ROOT,
    EXPERIMENTS_DIR,
    STAGE1_TEST,
    MANUAL_TEST,
    BASE_MODEL,
    VLLM_PYTHON,
    VLLM_BIN,
)


EXPERIMENT_ROOT = EXPERIMENTS_DIR
RUN_ROOT = EXPERIMENT_ROOT / "mentor_followup"
OUTPUT_ROOT = RUN_ROOT / "evaluation"
LOG_PATH = RUN_ROOT / "logs" / "evaluation_pipeline.log"
STATE_PATH = RUN_ROOT / "evaluation_state.json"
SUMMARY_PATH = RUN_ROOT / "evaluation_summary.csv"
PYTHON = VLLM_PYTHON
VLLM = VLLM_BIN
INFERENCE_SCRIPT = PROJECT_ROOT / "training" / "infer_sft_task_vllm.py"
EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "src" / "run_eval.py"
RUN_DEFAULTS = {
    "label": PROJECT_ROOT / "training" / "configs" / "run_label_eval.yaml",
    "narrative": PROJECT_ROOT / "training" / "configs" / "run_narrative_eval.yaml",
}
DATASETS = {
    "stage1": STAGE1_TEST,
    "manual": MANUAL_TEST,
}
MODELS = {
    "lambda_0_seed42": {
        "adapter": RUN_ROOT / "models" / "joint_lambda_0_0_seed_42",
        "tasks": ("label", "narrative"),
    },
    "lambda_08_seed42": {
        "adapter": EXPERIMENT_ROOT / "ablations" / "lambda_0_8" / "models" / "joint_lora_r8",
        "tasks": ("label", "narrative"),
    },
    "lambda_08_seed1": {
        "adapter": RUN_ROOT / "models" / "joint_lambda_0_8_seed_1",
        "tasks": ("label", "narrative"),
    },
    "lambda_08_seed2": {
        "adapter": RUN_ROOT / "models" / "joint_lambda_0_8_seed_2",
        "tasks": ("label", "narrative"),
    },
    "model_a_seed42": {
        "adapter": EXPERIMENT_ROOT / "models" / "label_lora_r8",
        "tasks": ("label",),
    },
    "model_a_seed1": {
        "adapter": RUN_ROOT / "models" / "model_a_seed_1",
        "tasks": ("label",),
    },
    "model_a_seed2": {
        "adapter": RUN_ROOT / "models" / "model_a_seed_2",
        "tasks": ("label",),
    },
}
PORT = 8400


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_state(phase: str, current: str = "", error: str = "") -> None:
    atomic_json(
        STATE_PATH,
        {
            "updated_at": now(),
            "phase": phase,
            "current": current,
            "error": error,
        },
    )


def latest_complete_run(root: Path, expected_rows: int) -> Path | None:
    for summary_path in reversed(sorted(root.glob("run_*/metrics/summary.json"))):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            int(summary.get("num_aligned_rows", 0)) == expected_rows
            and int(summary.get("num_missing_prediction_rows", 0)) == 0
        ):
            return summary_path.parent.parent
    return None


def prediction_is_complete(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    if len(frame) != expected_rows or "parse_success" not in frame:
        return False
    return bool(frame["parse_success"].astype(bool).all())


def wait_for_server(process: subprocess.Popen, expected_models: set[str]) -> None:
    endpoint = f"http://127.0.0.1:{PORT}/v1/models"
    deadline = time.time() + 480
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                available = {row["id"] for row in json.load(response).get("data", [])}
            if expected_models <= available:
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not expose all requested LoRA models")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_checked(command: list[str], env: dict[str, str], log) -> None:
    log.write(f"[{now()}] RUN {' '.join(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Command exited with code {completed.returncode}: {' '.join(command)}"
        )


def prediction_path(model_name: str, task: str, dataset: str) -> Path:
    return OUTPUT_ROOT / model_name / task / dataset / "predictions" / "pred.csv"


def evaluation_root(model_name: str, task: str, dataset: str) -> Path:
    return OUTPUT_ROOT / model_name / task / dataset / "evaluation"


def validate_inputs() -> None:
    for dataset, path in DATASETS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {dataset} gold data: {path}")
    for name, model in MODELS.items():
        adapter = model["adapter"] / "adapter_model.safetensors"
        if not adapter.is_file():
            raise FileNotFoundError(f"Missing adapter for {name}: {adapter}")


def run_inference(env: dict[str, str], log) -> None:
    lora_arguments = [
        f"mentor-{name.replace('_', '-')}={model['adapter']}"
        for name, model in MODELS.items()
    ]
    command = [
        VLLM,
        "serve",
        BASE_MODEL,
        "--served-model-name",
        "base",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.90",
        "--enable-lora",
        "--max-lora-rank",
        "8",
        "--max-loras",
        "1",
        "--max-cpu-loras",
        str(len(MODELS)),
        "--lora-modules",
        *lora_arguments,
    ]
    server = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        expected_models = {
            f"mentor-{name.replace('_', '-')}" for name in MODELS
        }
        wait_for_server(server, expected_models)
        for name, model in MODELS.items():
            served_name = f"mentor-{name.replace('_', '-')}"
            for task in model["tasks"]:
                for dataset, gold_path in DATASETS.items():
                    expected_rows = len(pd.read_csv(gold_path))
                    output = prediction_path(name, task, dataset)
                    key = f"{name}/{task}/{dataset}"
                    if prediction_is_complete(output, expected_rows):
                        log.write(f"[{now()}] SKIP complete prediction {key}\n")
                        log.flush()
                        continue
                    write_state("inference", key)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    run_checked(
                        [
                            PYTHON,
                            str(INFERENCE_SCRIPT),
                            "--task",
                            task,
                            "--gold",
                            str(gold_path),
                            "--output",
                            str(output),
                            "--model-name",
                            served_name,
                            "--base-url",
                            f"http://127.0.0.1:{PORT}/v1",
                            "--concurrency",
                            "16",
                        ],
                        env,
                        log,
                    )
                    if not prediction_is_complete(output, expected_rows):
                        raise RuntimeError(f"Incomplete or invalid predictions: {key}")
    finally:
        stop_server(server)


def run_evaluations(env: dict[str, str], log) -> None:
    for name, model in MODELS.items():
        for task in model["tasks"]:
            for dataset, gold_path in DATASETS.items():
                expected_rows = len(pd.read_csv(gold_path))
                root = evaluation_root(name, task, dataset)
                key = f"{name}/{task}/{dataset}"
                if latest_complete_run(root, expected_rows) is not None:
                    log.write(f"[{now()}] SKIP complete evaluation {key}\n")
                    log.flush()
                    continue
                write_state("evaluation", key)
                root.mkdir(parents=True, exist_ok=True)
                run_checked(
                    [
                        PYTHON,
                        str(EVAL_SCRIPT),
                        "--gold",
                        str(gold_path),
                        "--pred",
                        str(prediction_path(name, task, dataset)),
                        "--run-defaults",
                        str(RUN_DEFAULTS[task]),
                        "--output-root",
                        str(root),
                    ],
                    env,
                    log,
                )
                if latest_complete_run(root, expected_rows) is None:
                    raise RuntimeError(f"Evaluation output is incomplete: {key}")


def write_summary() -> None:
    rows = []
    for name, model in MODELS.items():
        for task in model["tasks"]:
            for dataset, gold_path in DATASETS.items():
                run = latest_complete_run(
                    evaluation_root(name, task, dataset), len(pd.read_csv(gold_path))
                )
                if run is None:
                    raise RuntimeError(f"Missing summary for {name}/{task}/{dataset}")
                summary = json.loads((run / "metrics" / "summary.json").read_text())
                row = {
                    "model": name,
                    "task": task,
                    "dataset": dataset,
                    "run_path": str(run),
                }
                metric_names = (
                    "accuracy",
                    "macro_f1",
                    "bleu_corpus",
                    "rouge1_mean",
                    "rouge2_mean",
                    "rougeL_mean",
                    "bertscore_f1_mean",
                    "normalized_dtw_distance_mean",
                )
                row.update({metric: summary.get(metric) for metric in metric_names})
                rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)


def main() -> None:
    validate_inputs()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "OMP_NUM_THREADS": "8",
        }
    )
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log:
        try:
            write_state("starting")
            run_inference(env, log)
            run_evaluations(env, log)
            write_summary()
            write_state("completed")
            log.write(f"[{now()}] Pipeline completed\n")
        except Exception as exc:
            write_state("failed", error=f"{type(exc).__name__}: {exc}")
            log.write(f"[{now()}] FAILED {type(exc).__name__}: {exc}\n")
            raise


if __name__ == "__main__":
    main()
