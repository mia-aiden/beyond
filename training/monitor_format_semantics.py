#!/usr/bin/env python3
"""Print a compact status snapshot for the multi-seed training queue."""

from __future__ import annotations

import json
from pathlib import Path


EXPERIMENT_DIR = Path("/root/autodl-fs/sft_experiments/format_semantics_multiseed")
STATE_PATH = EXPERIMENT_DIR / "queue_state.json"


def tail(path: Path, line_count: int = 5) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]


def main() -> None:
    if not STATE_PATH.is_file():
        raise SystemExit(f"No queue state found at {STATE_PATH}")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    print("Updated:", state["updated_at"])
    print("Counts:", state["counts"])
    for gpu in state.get("gpu_snapshot", []):
        print(
            f'GPU {gpu["gpu"]}: util={gpu["utilization_percent"]}% '
            f'memory={gpu["memory_used_mib"]}/{gpu["memory_total_mib"]} MiB '
            f'temp={gpu["temperature_c"]}C'
        )
    for gpu, running in state.get("running", {}).items():
        print(f'\nGPU {gpu}: {running["key"]} (PID {running["pid"]})')
        job = next(item for item in state["jobs"] if item["key"] == running["key"])
        for line in tail(Path(job["log_path"])):
            print("  ", line[-240:])


if __name__ == "__main__":
    main()
