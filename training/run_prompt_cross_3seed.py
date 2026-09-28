#!/usr/bin/env python3
"""Run the two missing cells of the three-seed model-by-prompt experiment."""

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


PROJECT_ROOT = Path("/root/autodl-tmp/beyond")
ROOT = Path("/root/autodl-fs/sft_experiments")
EXPERIMENT_DIR = ROOT / "prompt_cross_3seed"
OUTPUT_DIR = EXPERIMENT_DIR / "evaluation"
LOG_DIR = EXPERIMENT_DIR / "logs"
STATE_PATH = EXPERIMENT_DIR / "state.json"
BASE_MODEL = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
PYTHON = "/root/autodl-tmp/vllm-cu128/bin/python"
VLLM = "/root/autodl-tmp/vllm-cu128/bin/vllm"
INFERENCE_SCRIPT = PROJECT_ROOT / "training" / "infer_sft_task_vllm.py"
EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "src" / "run_eval.py"
RUN_DEFAULTS = PROJECT_ROOT / "training" / "configs" / "run_narrative_eval.yaml"
DATASETS = {
    "stage1": Path("/root/autodl-fs/train_set_stage1_test_eval.csv"),
    "manual": Path("/root/autodl-fs/manual_test_ekman_eval.csv"),
}
MODELS = {
    "model_b": {
        42: ROOT / "models" / "narrative_lora_r8",
        1: ROOT / "model_b_multiseed" / "models" / "seed_1",
        2: ROOT / "model_b_multiseed" / "models" / "seed_2",
    },
    "seqjoint": {
        42: ROOT / "models" / "seqjoint_lora_r8",
        1: ROOT / "models" / "seqjoint_lora_r8_s1",
        2: ROOT / "models" / "seqjoint_lora_r8_s2",
    },
}
CONDITIONS = {
    "model_b_joint": ("model_b", "joint"),
    "seqjoint_narrative": ("seqjoint", "narrative"),
}
GPU_PORTS = {0: 8300, 1: 8301}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def latest_complete_run(root: Path, expected_rows: int) -> Path | None:
    for summary_path in reversed(sorted(root.glob("run_*/metrics/summary.json"))):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            int(summary.get("num_aligned_rows", 0)) == expected_rows
            and int(summary.get("num_missing_prediction_rows", 0)) == 0
        ):
            return summary_path.parent.parent
    return None


def job_is_complete(condition: str, seed: int) -> bool:
    root = OUTPUT_DIR / condition / f"seed_{seed}"
    return all(
        latest_complete_run(root / dataset / "evaluation", len(pd.read_csv(gold)))
        is not None
        for dataset, gold in DATASETS.items()
    )


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


def run_job(condition: str, seed: int, gpu: int) -> dict[str, str]:
    model, task = CONDITIONS[condition]
    model_path = MODELS[model][seed]
    if not (model_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter: {model_path}")

    model_name = f"cross-{condition.replace('_', '-')}-s{seed}"
    port = GPU_PORTS[gpu]
    root = OUTPUT_DIR / condition / f"seed_{seed}"
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
    with (LOG_DIR / f"{condition}_seed_{seed}.log").open(
        "a", encoding="utf-8", buffering=1
    ) as log:
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
                        task,
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
                expected = len(pd.read_csv(gold_path))
                if len(predictions) != expected:
                    raise RuntimeError(
                        f"Expected {expected} predictions for {condition}/{seed}/{dataset}, "
                        f"found {len(predictions)}"
                    )
                if not predictions["parse_success"].astype(bool).all():
                    failures = int((~predictions["parse_success"].astype(bool)).sum())
                    raise RuntimeError(
                        f"{failures} generation failures for {condition}/{seed}/{dataset}"
                    )
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
                raise RuntimeError(f"No complete evaluation for {condition}/{seed}/{dataset}")
            runs[dataset] = str(run)
    return runs


def main() -> None:
    jobs = []
    for condition in CONDITIONS:
        for seed in (42, 1, 2):
            jobs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "status": "completed" if job_is_complete(condition, seed) else "pending",
                    "attempts": 0,
                }
            )
    pending: queue.Queue = queue.Queue()
    for job in jobs:
        if job["status"] == "pending":
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
                job.update(
                    status="running",
                    gpu=gpu,
                    attempts=int(job["attempts"]) + 1,
                    started_at=now(),
                )
                write_state()
            try:
                job["evaluation_runs"] = run_job(job["condition"], job["seed"], gpu)
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
        raise RuntimeError(
            "Failed jobs: "
            + ", ".join(f"{job['condition']}/seed_{job['seed']}" for job in failures)
        )


if __name__ == "__main__":
    main()
