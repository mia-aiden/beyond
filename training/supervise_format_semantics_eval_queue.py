#!/usr/bin/env python3
"""Keep the format-semantics evaluation queue alive until it completes."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from paths import PROJECT_ROOT, EXPERIMENTS_DIR, VLLM_PYTHON


EXPERIMENT_DIR = EXPERIMENTS_DIR / "format_semantics_multiseed"
EVAL_SCRIPT = PROJECT_ROOT / "training" / "run_format_semantics_eval_queue.py"
PYTHON = VLLM_PYTHON
EVAL_PID_PATH = EXPERIMENT_DIR / "eval_queue.pid"
STATE_PATH = EXPERIMENT_DIR / "eval_queue_state.json"
SUPERVISOR_PID_PATH = EXPERIMENT_DIR / "eval_supervisor.pid"
SUPERVISOR_STATE_PATH = EXPERIMENT_DIR / "eval_supervisor_state.json"
LOG_PATH = EXPERIMENT_DIR / "eval_queue.log"
POLL_SECONDS = 60
MAX_RESTARTS = 5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def read_phase() -> str | None:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8")).get("phase")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def acquire_lock() -> None:
    existing = read_pid(SUPERVISOR_PID_PATH)
    if existing is not None and process_is_alive(existing):
        raise RuntimeError(f"Evaluation supervisor already running as PID {existing}")
    SUPERVISOR_PID_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")


def launch_watcher() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = LOG_PATH.open("a", encoding="utf-8", buffering=1)
    log_handle.write(f"\n[{now()}] supervisor launching evaluation queue\n")
    process = subprocess.Popen(
        [PYTHON, str(EVAL_SCRIPT)],
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return process.pid


def main() -> None:
    acquire_lock()
    restart_count = 0
    try:
        while True:
            phase = read_phase()
            if phase == "completed":
                atomic_json(
                    SUPERVISOR_STATE_PATH,
                    {
                        "updated_at": now(),
                        "status": "completed",
                        "restart_count": restart_count,
                    },
                )
                return

            watcher_pid = read_pid(EVAL_PID_PATH)
            watcher_alive = watcher_pid is not None and process_is_alive(watcher_pid)
            if not watcher_alive:
                if restart_count >= MAX_RESTARTS:
                    atomic_json(
                        SUPERVISOR_STATE_PATH,
                        {
                            "updated_at": now(),
                            "status": "failed",
                            "restart_count": restart_count,
                            "last_phase": phase,
                        },
                    )
                    raise RuntimeError("Evaluation queue exceeded automatic restart limit")
                EVAL_PID_PATH.unlink(missing_ok=True)
                watcher_pid = launch_watcher()
                restart_count += 1

            atomic_json(
                SUPERVISOR_STATE_PATH,
                {
                    "updated_at": now(),
                    "status": "monitoring",
                    "watcher_pid": watcher_pid,
                    "watcher_alive": True,
                    "evaluation_phase": phase,
                    "restart_count": restart_count,
                },
            )
            time.sleep(POLL_SECONDS)
    finally:
        SUPERVISOR_PID_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
