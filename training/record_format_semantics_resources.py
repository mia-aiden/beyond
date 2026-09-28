#!/usr/bin/env python3
"""Record GPU and queue utilization throughout training and evaluation."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path("/root/autodl-fs/sft_experiments/format_semantics_multiseed")
TRAIN_STATE = EXPERIMENT_DIR / "queue_state.json"
EVAL_STATE = EXPERIMENT_DIR / "eval_queue_state.json"
OUTPUT = EXPERIMENT_DIR / "resource_history.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def gpu_snapshot() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        rows.append(
            {
                "gpu": int(values[0]),
                "utilization_percent": int(values[1]),
                "memory_used_mib": int(values[2]),
                "memory_total_mib": int(values[3]),
                "power_w": float(values[4]),
                "temperature_c": int(values[5]),
            }
        )
    return rows


def main() -> None:
    while True:
        train = load_json(TRAIN_STATE)
        evaluation = load_json(EVAL_STATE)
        row = {
            "timestamp": now(),
            "gpus": gpu_snapshot(),
            "training_counts": train.get("counts", {}),
            "training_running": train.get("running", {}),
            "evaluation_phase": evaluation.get("phase", "not_started"),
            "evaluation_counts": evaluation.get("counts", {}),
        }
        with OUTPUT.open("a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(row) + "\n")
        if evaluation.get("phase") == "completed":
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
