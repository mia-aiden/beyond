#!/usr/bin/env python3
"""Wait for training, then evaluate every condition/seed on two GPUs."""

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
EXPERIMENT_DIR = Path("/root/autodl-fs/sft_experiments/format_semantics_multiseed")
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"
TRAIN_STATE_PATH = EXPERIMENT_DIR / "queue_state.json"
STATE_PATH = EXPERIMENT_DIR / "eval_queue_state.json"
LOCK_PATH = EXPERIMENT_DIR / "eval_queue.pid"
LOG_DIR = EXPERIMENT_DIR / "eval_logs"
OUTPUT_DIR = EXPERIMENT_DIR / "evaluation"
BASE_MODEL = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
PYTHON = "/root/autodl-tmp/vllm-cu128/bin/python"
VLLM = "/root/autodl-tmp/vllm-cu128/bin/vllm"
INFERENCE_SCRIPT = PROJECT_ROOT / "training" / "infer_sft_task_vllm.py"
EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "src" / "run_eval.py"
RUN_DEFAULTS = PROJECT_ROOT / "training" / "configs" / "run_narrative_eval.yaml"
SUMMARY_SCRIPT = PROJECT_ROOT / "training" / "summarize_format_semantics_multiseed.py"
DATASETS = {
    "stage1": Path("/root/autodl-fs/train_set_stage1_test_eval.csv"),
    "manual": Path("/root/autodl-fs/manual_test_ekman_eval.csv"),
}
GPU_PORTS = {0: 8100, 1: 8101}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def acquire_lock() -> None:
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            raise RuntimeError(f"Evaluation queue already running as PID {pid}")
        except (ProcessLookupError, ValueError):
            LOCK_PATH.unlink(missing_ok=True)
    LOCK_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")


def wait_for_training() -> None:
    while True:
        if TRAIN_STATE_PATH.is_file():
            state = json.loads(TRAIN_STATE_PATH.read_text(encoding="utf-8"))
            counts = state.get("counts", {})
            atomic_json(
                STATE_PATH,
                {
                    "updated_at": now(),
                    "phase": "waiting_for_training",
                    "training_counts": counts,
                },
            )
            if int(counts.get("failed", 0)):
                raise RuntimeError("Training queue contains failed jobs; evaluation will not start.")
            if int(counts.get("completed", 0)) == 20:
                return
        time.sleep(60)


def latest_complete_run(dataset_root: Path, dataset_name: str) -> Path | None:
    candidates = sorted(dataset_root.glob("run_*/metrics/summary.json"))
    for summary_path in reversed(candidates):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = len(pd.read_csv(DATASETS[dataset_name]))
            if int(summary.get("num_aligned_rows", 0)) == expected:
                return summary_path.parent.parent
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return None


def job_is_complete(job: dict) -> bool:
    root = OUTPUT_DIR / job["condition"] / f'seed_{job["seed"]}'
    return all(
        latest_complete_run(root / dataset / "evaluation", dataset) is not None
        for dataset in DATASETS
    )


def wait_for_server(process: subprocess.Popen, port: int, model_name: str) -> None:
    endpoint = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + 360
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                payload = json.load(response)
            if model_name in {model["id"] for model in payload.get("data", [])}:
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
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")


def run_job(job: dict, gpu: int) -> dict[str, str]:
    condition = job["condition"]
    seed = int(job["seed"])
    key = f"{condition}_seed_{seed}"
    model_name = f"fs-{condition.replace('_', '-')}-s{seed}"
    port = GPU_PORTS[gpu]
    task = "marker_joint" if condition == "marker_control" else "joint"
    root = OUTPUT_DIR / condition / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
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
    with (LOG_DIR / f"{key}.log").open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n[{now()}] evaluating on physical GPU {gpu}\n")
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
                f'{model_name}={job["model_path"]}',
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
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
                    log_handle,
                )
                predictions = pd.read_csv(prediction_path)
                if len(predictions) != len(pd.read_csv(gold_path)):
                    raise RuntimeError(f"Incomplete predictions for {key}/{dataset}")
                if not predictions["parse_success"].astype(bool).all():
                    failures = int((~predictions["parse_success"].astype(bool)).sum())
                    raise RuntimeError(f"{failures} structured-output failures for {key}/{dataset}")
        finally:
            stop_server(server)

        # Metric models need the GPU, so run them only after vLLM has released memory.
        for dataset, gold_path in DATASETS.items():
            dataset_root = root / dataset / "evaluation"
            prediction_path = root / dataset / "predictions" / "pred.csv"
            dataset_root.mkdir(parents=True, exist_ok=True)
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
                    str(dataset_root),
                ],
                env,
                log_handle,
            )
            run_dir = latest_complete_run(dataset_root, dataset)
            if run_dir is None:
                raise RuntimeError(f"No complete evaluation output for {key}/{dataset}")
            runs[dataset] = str(run_dir)
    return runs


def main() -> None:
    acquire_lock()
    try:
        wait_for_training()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        jobs = []
        for row in manifest:
            job = dict(row)
            job["key"] = f'{job["condition"]}_seed_{job["seed"]}'
            job["status"] = "completed" if job_is_complete(job) else "pending"
            job["attempts"] = 0
            jobs.append(job)

        pending: queue.Queue = queue.Queue()
        for job in jobs:
            if job["status"] == "pending":
                pending.put(job)
        state_lock = threading.Lock()

        def write_state() -> None:
            atomic_json(
                STATE_PATH,
                {
                    "updated_at": now(),
                    "phase": "evaluating",
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
                with state_lock:
                    job.update(
                        {
                            "status": "running",
                            "gpu": gpu,
                            "attempts": int(job["attempts"]) + 1,
                            "started_at": now(),
                        }
                    )
                    write_state()
                try:
                    job["evaluation_runs"] = run_job(job, gpu)
                    job["status"] = "completed"
                    job["finished_at"] = now()
                except Exception as exc:
                    job["last_error"] = f"{type(exc).__name__}: {exc}"
                    if int(job["attempts"]) < 2:
                        job["status"] = "pending"
                        pending.put(job)
                    else:
                        job["status"] = "failed"
                    time.sleep(10)
                finally:
                    with state_lock:
                        write_state()
                    pending.task_done()

        write_state()
        threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in GPU_PORTS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        write_state()
        failures = [job["key"] for job in jobs if job["status"] == "failed"]
        if failures:
            raise RuntimeError(f"Evaluation failed for: {', '.join(failures)}")
        summary_log_path = LOG_DIR / "summary.log"
        with summary_log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            run_checked(
                [PYTHON, str(SUMMARY_SCRIPT)],
                os.environ.copy(),
                log_handle,
            )
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state["phase"] = "completed"
        state["results_dir"] = str(EXPERIMENT_DIR / "results")
        atomic_json(STATE_PATH, state)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
