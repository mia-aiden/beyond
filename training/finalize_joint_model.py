from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from paths import EXPERIMENTS_DIR, BASE_MODEL


DEFAULT_MODEL_DIR = EXPERIMENTS_DIR / "models" / "joint_lora_r8"
DEFAULT_BASE_MODEL = Path(BASE_MODEL)
FINAL_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "training_args.bin",
)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint directory: {path}") from exc


def read_validation_history(model_dir: Path) -> list[dict[str, Any]]:
    history_by_step: dict[int, dict[str, Any]] = {}
    checkpoints = sorted(model_dir.glob("checkpoint-*"), key=checkpoint_step)
    for checkpoint in checkpoints:
        state_path = checkpoint / "trainer_state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for entry in state.get("log_history", []):
            if "eval_loss" not in entry or entry.get("step") is None:
                continue
            step = int(entry["step"])
            history_by_step[step] = {
                "step": step,
                "eval_loss": float(entry["eval_loss"]),
                "epoch": entry.get("epoch"),
            }
    return [history_by_step[step] for step in sorted(history_by_step)]


def select_checkpoint(
    model_dir: Path,
    history: list[dict[str, Any]],
    explicit_checkpoint: Path | None,
) -> tuple[Path, dict[str, Any]]:
    if explicit_checkpoint is not None:
        checkpoint = explicit_checkpoint
        step = checkpoint_step(checkpoint)
        metric = next((row for row in history if row["step"] == step), None)
        if metric is None:
            raise ValueError(f"No validation metric found for {checkpoint}")
    else:
        available = {
            checkpoint_step(path): path
            for path in model_dir.glob("checkpoint-*")
            if (path / "adapter_model.safetensors").exists()
        }
        candidates = [row for row in history if row["step"] in available]
        if not candidates:
            raise ValueError(f"No complete checkpoint with validation metrics under {model_dir}")
        metric = min(candidates, key=lambda row: row["eval_loss"])
        checkpoint = available[metric["step"]]

    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    missing = [name for name in FINAL_ADAPTER_FILES[:2] if not (checkpoint / name).exists()]
    if missing:
        raise FileNotFoundError(f"{checkpoint} is missing required files: {missing}")
    return checkpoint, metric


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize(
    *,
    model_dir: Path,
    base_model: Path,
    checkpoint: Path | None,
) -> dict[str, Any]:
    history = read_validation_history(model_dir)
    selected, selected_metric = select_checkpoint(model_dir, history, checkpoint)

    for filename in FINAL_ADAPTER_FILES:
        source = selected / filename
        destination = model_dir / filename
        if source.exists():
            shutil.copy2(source, destination)

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, use_fast=True)
    tokenizer.save_pretrained(model_dir)

    adapter_path = model_dir / "adapter_model.safetensors"
    summary = {
        "status": "finalized",
        "selection_rule": "minimum validation loss among available complete checkpoints",
        "selected_checkpoint": str(selected),
        "selected_step": selected_metric["step"],
        "selected_validation_loss": selected_metric["eval_loss"],
        "validation_history": history,
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256(adapter_path),
        "base_model_path": str(base_model),
    }
    (model_dir / "joint_finalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the best Model C checkpoint.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = finalize(
        model_dir=args.model_dir,
        base_model=args.base_model,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
