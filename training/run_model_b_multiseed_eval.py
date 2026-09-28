#!/usr/bin/env python3
"""Evaluate Model B seeds 42, 1, and 2 with the current vLLM stack."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
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


EXPERIMENT_DIR = EXPERIMENTS_DIR / "model_b_multiseed"
OUTPUT_DIR = EXPERIMENT_DIR / "evaluation"
LOG_DIR = EXPERIMENT_DIR / "eval_logs"
STATE_PATH = EXPERIMENT_DIR / "eval_state.json"
PYTHON = VLLM_PYTHON
VLLM = VLLM_BIN
INFERENCE_SCRIPT = PROJECT_ROOT / "training" / "infer_sft_task_vllm.py"
EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "src" / "run_eval.py"
RUN_DEFAULTS = PROJECT_ROOT / "training" / "configs" / "run_narrative_eval.yaml"
DATASETS = {
    "stage1": STAGE1_TEST,
    "manual": MANUAL_TEST,
}
MODELS = {
    42: EXPERIMENTS_DIR / "models" / "narrative_lora_r8",
    1: EXPERIMENT_DIR / "models" / "seed_1",
    2: EXPERIMENT_DIR / "models" / "seed_2",
}
GPU_PORTS = {0: 8200, 1: 8201}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def latest_complete_run(root: Path, expected_rows: int) -> Path | None:
    for summary_path in reversed(sorted(root.glob("run_*/metrics/summary.json"))):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("num_aligned_rows", 0)) == expected_rows:
            return summary_path.parent.parent
    return None


def wait_for_server(process: subprocess.Popen, port: int, model_name: str) -> None:
    endpoint = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + 360
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                models = {row["id"] for row in json.load(response).get("data", [])}
            if model_name in models:
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become ready on port {port}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_checked(command: list[str], env: dict[str, str], log_handle) -> None:
    log_handle.write(f"[{now()}] RUN {' '.join(command)}\n")
    log_handle.flush()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(command)}")


def evaluate_seed(seed: int, gpu: int) -> dict[str, str]:
    model_path = MODELS[seed]
    if not (model_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter for seed {seed}: {model_path}")
    model_name = f"model-b-s{seed}"
    port = GPU_PORTS[gpu]
    root = OUTPUT_DIR / f"seed_{seed}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "OMP_NUM_THREADS": "6",
        }
    )
    runs: dict[str, str] = {}
    with (LOG_DIR / f"seed_{seed}.log").open("a", encoding="utf-8", buffering=1) as log:
        server = subprocess.Popen(
            [
                VLLM,
                "serve",
                BASE_MODEL,
                "--served-model-name",
                "base",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "2048",
                "--gpu-memory-utilization",
                "0.90",
                "--enable-lora",
                "--max-lora-rank",
                "8",
                "--lora-modules",
                f"{model_name}={model_path}",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(server, port, model_name)
            for dataset, gold_path in DATASETS.items():
                prediction_path = root / dataset / "predictions" / "pred.csv"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                run_checked(
                    [
                        PYTHON,
                        str(INFERENCE_SCRIPT),
                        "--task",
                        "narrative",
                        "--gold",
                        str(gold_path),
                        "--output",
                        str(prediction_path),
                        "--model-name",
                        model_name,
                        "--base-url",
                        f"http://127.0.0.1:{port}/v1",
                        "--concurrency",
                        "16",
                    ],
                    env,
                    log,
                )
                predictions = pd.read_csv(prediction_path)
                if len(predictions) != len(pd.read_csv(gold_path)):
                    raise RuntimeError(f"Incomplete predictions for seed {seed}/{dataset}")
                if not predictions["parse_success"].astype(bool).all():
                    raise RuntimeError(f"Generation failures for seed {seed}/{dataset}")
        finally:
            stop_server(server)

        for dataset, gold_path in DATASETS.items():
            prediction_path = root / dataset / "predictions" / "pred.csv"
            evaluation_root = root / dataset / "evaluation"
            evaluation_root.mkdir(parents=True, exist_ok=True)
            run_checked(
                [
                    PYTHON,
                    str(EVAL_SCRIPT),
                    "--gold",
                    str(gold_path),
                    "--pred",
                    str(prediction_path),
                    "--run-defaults",
                    str(RUN_DEFAULTS),
                    "--output-root",
                    str(evaluation_root),
                ],
                env,
                log,
            )
            run = latest_complete_run(evaluation_root, len(pd.read_csv(gold_path)))
            if run is None:
                raise RuntimeError(f"No complete evaluation for seed {seed}/{dataset}")
            runs[dataset] = str(run)
    return runs


def main() -> None:
    jobs = [{"seed": seed, "status": "pending", "attempts": 0} for seed in (42, 1, 2)]
    pending: queue.Queue = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()

    def write_state(phase: str = "evaluating") -> None:
        atomic_json(
            STATE_PATH,
            {
                "updated_at": now(),
                "phase": phase,
                "counts": {
                    status: sum(job["status"] == status for job in jobs)
                    for status in ("pending", "running", "completed", "failed")
                },
                "jobs": jobs,
            },
        )

    def worker(gpu: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                job.update(status="running", gpu=gpu, attempts=job["attempts"] + 1, started_at=now())
                write_state()
            try:
                job["evaluation_runs"] = evaluate_seed(job["seed"], gpu)
                job.update(status="completed", finished_at=now())
            except Exception as exc:
                job.update(status="failed", last_error=f"{type(exc).__name__}: {exc}")
            finally:
                with lock:
                    write_state()
                pending.task_done()

    write_state()
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in GPU_PORTS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [job for job in jobs if job["status"] == "failed"]
    write_state("failed" if failures else "completed")
    if failures:
        raise RuntimeError(f"Failed seeds: {[job['seed'] for job in failures]}")


if __name__ == "__main__":
    main()
