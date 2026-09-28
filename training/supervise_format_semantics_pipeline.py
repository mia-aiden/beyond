#!/usr/bin/env python3
"""Recover and supervise the format-semantics training/evaluation pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/root/autodl-tmp/beyond")
EXPERIMENT_DIR = Path("/root/autodl-fs/sft_experiments/format_semantics_multiseed")
TRAIN_SCRIPT = PROJECT_ROOT / "training" / "run_format_semantics_queue.py"
EVAL_SUPERVISOR_SCRIPT = PROJECT_ROOT / "training" / "supervise_format_semantics_eval_queue.py"
PYTHON = "/root/miniconda3/bin/python"
EVAL_PYTHON = "/root/autodl-tmp/vllm-cu128/bin/python"
TRAIN_PID_PATH = EXPERIMENT_DIR / "queue.pid"
EVAL_SUPERVISOR_PID_PATH = EXPERIMENT_DIR / "eval_supervisor.pid"
PIPELINE_PID_PATH = EXPERIMENT_DIR / "pipeline_supervisor.pid"
STATE_PATH = EXPERIMENT_DIR / "pipeline_supervisor_state.json"
LOG_PATH = EXPERIMENT_DIR / "pipeline_supervisor_children.log"
POLL_SECONDS = 60
REQUIRED_GPUS = {0, 1}
MAX_TRAIN_RESTARTS = 10
MAX_EVAL_RESTARTS = 10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def process_matches(pid: int | None, script: Path) -> bool:
    if pid is None:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return False
    return str(script) in command


def visible_gpus() -> set[int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    if completed.returncode:
        return set()
    result = set()
    for line in completed.stdout.splitlines():
        try:
            result.add(int(line.strip()))
        except ValueError:
            continue
    return result


def launch(python: str, script: Path, label: str) -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    log_handle.write(f"[{now()}] launching {label}: {script}\n")
    process = subprocess.Popen(
        [python, str(script)],
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return process.pid


def acquire_lock() -> None:
    existing = read_pid(PIPELINE_PID_PATH)
    if process_matches(existing, Path(__file__)):
        raise RuntimeError(f"Pipeline supervisor already running as PID {existing}")
    PIPELINE_PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")


def main() -> None:
    acquire_lock()
    train_restarts = 0
    eval_restarts = 0
    try:
        while True:
            train_state = read_json(EXPERIMENT_DIR / "queue_state.json")
            train_counts = train_state.get("counts", {})
            completed = int(train_counts.get("completed", 0))
            failed = int(train_counts.get("failed", 0))
            eval_state = read_json(EXPERIMENT_DIR / "eval_queue_state.json")
            eval_phase = eval_state.get("phase")
            gpus = visible_gpus()
            phase = "waiting_for_gpus"

            train_pid = read_pid(TRAIN_PID_PATH)
            train_alive = process_matches(train_pid, TRAIN_SCRIPT)
            eval_pid = read_pid(EVAL_SUPERVISOR_PID_PATH)
            eval_alive = process_matches(eval_pid, EVAL_SUPERVISOR_SCRIPT)

            if completed < 20:
                if REQUIRED_GPUS.issubset(gpus):
                    phase = "training"
                    if not train_alive:
                        if train_restarts >= MAX_TRAIN_RESTARTS:
                            raise RuntimeError("Training queue exceeded automatic restart limit")
                        TRAIN_PID_PATH.unlink(missing_ok=True)
                        train_pid = launch(PYTHON, TRAIN_SCRIPT, "training queue")
                        train_alive = True
                        train_restarts += 1
                else:
                    phase = "waiting_for_gpus"
            elif eval_phase == "completed":
                phase = "completed"
            elif REQUIRED_GPUS.issubset(gpus):
                phase = "evaluating"
                if not eval_alive:
                    if eval_restarts >= MAX_EVAL_RESTARTS:
                        raise RuntimeError("Evaluation supervisor exceeded automatic restart limit")
                    EVAL_SUPERVISOR_PID_PATH.unlink(missing_ok=True)
                    eval_pid = launch(EVAL_PYTHON, EVAL_SUPERVISOR_SCRIPT, "evaluation supervisor")
                    eval_alive = True
                    eval_restarts += 1
            else:
                phase = "waiting_for_gpus_for_evaluation"

            atomic_json(
                STATE_PATH,
                {
                    "updated_at": now(),
                    "status": phase,
                    "visible_gpus": sorted(gpus),
                    "training_counts": train_counts,
                    "training_failed_count": failed,
                    "training_pid": train_pid,
                    "training_alive": train_alive,
                    "evaluation_phase": eval_phase,
                    "evaluation_supervisor_pid": eval_pid,
                    "evaluation_supervisor_alive": eval_alive,
                    "training_restart_count": train_restarts,
                    "evaluation_restart_count": eval_restarts,
                },
            )
            if phase == "completed":
                return
            time.sleep(POLL_SECONDS)
    finally:
        PIPELINE_PID_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
