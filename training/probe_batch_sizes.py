from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import yaml


def monitor_gpu(stop_event: threading.Event, samples: list[tuple[int, int]]) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(command, text=True).strip().splitlines()[0]
            memory_mib, utilization = (int(value.strip()) for value in output.split(","))
            samples.append((memory_mib, utilization))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        stop_event.wait(0.5)


def parse_training_speed(log_text: str) -> float | None:
    matches = re.findall(r"'train_samples_per_second': ([0-9.]+)", log_text)
    return float(matches[-1]) if matches else None


def run_probe(
    *,
    base_config: dict,
    batch_size: int,
    effective_batch_size: int,
    max_steps: int,
    output_root: Path,
) -> dict[str, object]:
    probe_dir = output_root / f"batch_{batch_size}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    config = dict(base_config)
    config.update(
        {
            "output_dir": str(probe_dir / "adapter"),
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": max(1, effective_batch_size // batch_size),
            "max_steps": max_steps,
            "max_samples": 512,
            "save_strategy": "no",
            "eval_strategy": "no",
            "load_best_model_at_end": False,
            "overwrite_output_dir": True,
            "plot_loss": False,
        }
    )
    config.pop("eval_dataset", None)
    config.pop("num_train_epochs", None)

    config_path = probe_dir / "probe.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log_path = probe_dir / "train.log"

    gpu_samples: list[tuple[int, int]] = []
    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop_event, gpu_samples), daemon=True)
    monitor.start()
    start = time.monotonic()
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    with log_path.open("w", encoding="utf-8") as logfile:
        process = subprocess.run(
            ["llamafactory-cli", "train", str(config_path)],
            stdout=logfile,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            check=False,
        )
    runtime_seconds = time.monotonic() - start
    stop_event.set()
    monitor.join(timeout=2)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    active_utilization = [util for memory, util in gpu_samples if memory > 1000]
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "return_code": process.returncode,
        "success": process.returncode == 0,
        "out_of_memory": "out of memory" in log_text.lower(),
        "runtime_seconds": round(runtime_seconds, 2),
        "peak_gpu_memory_mib": max((memory for memory, _ in gpu_samples), default=0),
        "mean_active_gpu_utilization": (
            round(sum(active_utilization) / len(active_utilization), 2)
            if active_utilization
            else 0.0
        ),
        "train_samples_per_second": parse_training_speed(log_text),
        "config_path": str(config_path),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe LoRA batch sizes on the current GPU.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("/root/autodl-tmp/beyond/training/configs/label_sft.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-fs/sft_experiments/batch_probe/label"),
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    results = []
    for batch_size in args.batch_sizes:
        print(f"Probing per-device batch size {batch_size}...")
        result = run_probe(
            base_config=base_config,
            batch_size=batch_size,
            effective_batch_size=args.effective_batch_size,
            max_steps=args.max_steps,
            output_root=args.output_root,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    output_path = args.output_root / "summary.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
