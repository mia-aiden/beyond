#!/usr/bin/env python3
"""Keep two GPUs occupied with independent LLaMA-Factory training jobs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from paths import EXPERIMENTS_DIR, LLAMAFACTORY_DIR, LLAMAFACTORY_CLI


EXPERIMENT_DIR = EXPERIMENTS_DIR / "format_semantics_multiseed"
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"
STATE_PATH = EXPERIMENT_DIR / "queue_state.json"
LOCK_PATH = EXPERIMENT_DIR / "queue.pid"
LOG_DIR = EXPERIMENT_DIR / "logs"
CLI = LLAMAFACTORY_CLI
GPUS = (0, 1)
POLL_SECONDS = 30


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def adapter_is_complete(model_path: str) -> bool:
    path = Path(model_path)
    result_path = path / "all_results.json"
    state_path = path / "trainer_state.json"
    if not (path / "adapter_model.safetensors").is_file() or not result_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return float(state.get("epoch", 0)) >= 2.99
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False


def latest_resumable_checkpoint(model_path: str) -> Path | None:
    """Return the latest checkpoint with the state required by Trainer.resume."""
    checkpoints = []
    for path in Path(model_path).glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        required_files = (
            path / "adapter_model.safetensors",
            path / "trainer_state.json",
            path / "optimizer.pt",
            path / "scheduler.pt",
        )
        if all(file_path.is_file() for file_path in required_files):
            checkpoints.append((step, path))
    return max(checkpoints, default=(None, None))[1]


def priority(job: dict) -> tuple[int, int]:
    # Produce a complete seed-42 control comparison first, then fill each seed block.
    condition_order = {"forced_wrong": 0, "constant_neutral": 1, "marker_control": 2, "true": 3}
    seed_order = {42: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    return seed_order[int(job["seed"])], condition_order[job["condition"]]


def load_jobs() -> list[dict]:
    jobs = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for job in jobs:
        job["key"] = f'{job["condition"]}_seed_{job["seed"]}'
        if job["action"] == "reuse" or adapter_is_complete(job["model_path"]):
            job["status"] = "completed"
        else:
            job["status"] = "pending"
        job["attempts"] = 0
    return sorted(jobs, key=priority)


def acquire_lock() -> None:
    if LOCK_PATH.exists():
        try:
            existing_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
            os.kill(existing_pid, 0)
            raise RuntimeError(f"Queue already running as PID {existing_pid}")
        except ProcessLookupError:
            LOCK_PATH.unlink()
        except ValueError:
            LOCK_PATH.unlink()
    LOCK_PATH.write_text(f"{os.getpid()}\n", encoding="utf-8")


def gpu_snapshot() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == 5:
            rows.append(
                {
                    "gpu": int(values[0]),
                    "utilization_percent": int(values[1]),
                    "memory_used_mib": int(values[2]),
                    "memory_total_mib": int(values[3]),
                    "temperature_c": int(values[4]),
                }
            )
    return rows


def write_state(jobs: list[dict], running: dict[int, dict]) -> None:
    serializable_jobs = []
    for job in jobs:
        serializable_jobs.append({key: value for key, value in job.items() if key != "process"})
    atomic_json(
        STATE_PATH,
        {
            "updated_at": now(),
            "scheduler_pid": os.getpid(),
            "gpu_snapshot": gpu_snapshot(),
            "counts": {
                status: sum(job["status"] == status for job in jobs)
                for status in ("pending", "running", "completed", "failed")
            },
            "running": {
                str(gpu): {"key": item["job"]["key"], "pid": item["process"].pid}
                for gpu, item in running.items()
            },
            "jobs": serializable_jobs,
        },
    )


def launch(job: dict, gpu: int) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'{job["key"]}.log'
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    resume_checkpoint = latest_resumable_checkpoint(job["model_path"])
    command = [CLI, "train", job["config_path"]]
    if resume_checkpoint is not None:
        command.extend(
            [
                f"resume_from_checkpoint={resume_checkpoint}",
                "overwrite_output_dir=false",
            ]
        )
    log_handle.write(
        f"\n[{now()}] launching on GPU {gpu}; "
        f"resume_from_checkpoint={resume_checkpoint}\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=LLAMAFACTORY_DIR,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job.update(
        {
            "status": "running",
            "gpu": gpu,
            "pid": process.pid,
            "started_at": now(),
            "attempts": job["attempts"] + 1,
            "log_path": str(log_path),
            "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }
    )
    return {"job": job, "process": process, "log_handle": log_handle}


def terminate_children(running: dict[int, dict]) -> None:
    for item in running.values():
        process = item["process"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)


def main() -> None:
    acquire_lock()
    jobs = load_jobs()
    running: dict[int, dict] = {}
    try:
        visible_gpus = {row["gpu"] for row in gpu_snapshot()}
        missing_gpus = set(GPUS) - visible_gpus
        if missing_gpus:
            raise RuntimeError(
                "Required GPUs are not visible: "
                + ", ".join(str(gpu) for gpu in sorted(missing_gpus))
            )
        while True:
            for gpu, item in list(running.items()):
                return_code = item["process"].poll()
                if return_code is None:
                    continue
                item["log_handle"].close()
                job = item["job"]
                job["finished_at"] = now()
                job["return_code"] = return_code
                if return_code == 0 and adapter_is_complete(job["model_path"]):
                    job["status"] = "completed"
                    job.pop("last_error", None)
                elif job["attempts"] < 2:
                    job["status"] = "pending"
                    job["last_error"] = f"attempt exited with code {return_code}"
                else:
                    job["status"] = "failed"
                    job["last_error"] = f"attempt exited with code {return_code}"
                del running[gpu]

            pending = [job for job in jobs if job["status"] == "pending"]
            free_gpus = [gpu for gpu in GPUS if gpu not in running]
            for gpu, job in zip(free_gpus, pending):
                running[gpu] = launch(job, gpu)

            write_state(jobs, running)
            if not running and not any(job["status"] == "pending" for job in jobs):
                failed = [job["key"] for job in jobs if job["status"] == "failed"]
                if failed:
                    print("Queue finished with failed jobs:", ", ".join(failed), file=sys.stderr)
                    raise SystemExit(1)
                print("All training jobs completed.")
                return
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        terminate_children(running)
        raise
    finally:
        write_state(jobs, running)
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
