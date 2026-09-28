from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/root/autodl-fs/sft_experiments/ablations")
EXPERIMENTS = (
    ("lambda_0_3", "joint_l03_train", "joint_l03_finish", 0),
    ("lambda_0_5", "joint_l05_train", "joint_l05_finish", 1),
    ("lambda_0_8", "joint_l08_train", "joint_l08_finish", 2),
)
PROGRESS_PATTERN = re.compile(r"(\d+)/3915")


def run(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout


def screen_names() -> set[str]:
    output = run(["screen", "-list"])
    return {
        match.group(1)
        for match in re.finditer(r"\d+\.([^\s]+)\s+", output)
    }


def gpu_status() -> dict[int, dict[str, int]]:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    status = {}
    for line in output.splitlines():
        try:
            index, memory, utilization = (int(value.strip()) for value in line.split(","))
        except ValueError:
            continue
        status[index] = {"memory_used_mib": memory, "utilization_percent": utilization}
    return status


def latest_progress(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    matches = PROGRESS_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    return int(matches[-1]) if matches else None


def validation_history(model_dir: Path) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for state_path in model_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in state.get("log_history", []):
            if "eval_loss" in row and row.get("step") is not None:
                step = int(row["step"])
                by_step[step] = {"step": step, "eval_loss": float(row["eval_loss"])}
    return [by_step[step] for step in sorted(by_step)]


def experiment_status(
    root: Path,
    experiment: str,
    train_screen: str,
    finish_screen: str,
    gpu_index: int,
    active_screens: set[str],
    gpus: dict[int, dict[str, int]],
) -> dict[str, Any]:
    experiment_root = root / experiment
    model_dir = experiment_root / "models/joint_lora_r8"
    complete = (experiment_root / "ABLATION_COMPLETE").exists()
    train_active = train_screen in active_screens
    finish_active = finish_screen in active_screens
    if complete:
        phase = "complete"
    elif train_active:
        phase = "training"
    elif finish_active:
        phase = "finalizing_or_evaluating"
    else:
        phase = "failed_or_stopped"

    history = validation_history(model_dir)
    checkpoints = []
    for path in model_dir.glob("checkpoint-*"):
        try:
            checkpoints.append(int(path.name.rsplit("-", 1)[1]))
        except ValueError:
            continue
    return {
        "experiment": experiment,
        "phase": phase,
        "optimizer_step": latest_progress(experiment_root / "logs/train.log"),
        "latest_checkpoint": max(checkpoints, default=None),
        "latest_validation": history[-1] if history else None,
        "best_validation": min(history, key=lambda row: row["eval_loss"]) if history else None,
        "gpu": gpu_index,
        **gpus.get(gpu_index, {}),
    }


def snapshot(root: Path) -> dict[str, Any]:
    active_screens = screen_names()
    gpus = gpu_status()
    experiments = [
        experiment_status(root, *definition, active_screens, gpus)
        for definition in EXPERIMENTS
    ]
    suite_complete = (root / "LAMBDA_ABLATION_COMPLETE").exists()
    failed = any(row["phase"] == "failed_or_stopped" for row in experiments)
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "suite_complete": suite_complete,
        "failed": failed,
        "experiments": experiments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report Model C ablation progress periodically.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval-seconds", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        result = snapshot(args.root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.once or result["suite_complete"]:
            return
        if result["failed"]:
            raise RuntimeError("At least one ablation pipeline stopped before completion.")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
