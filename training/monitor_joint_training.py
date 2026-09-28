from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from finalize_joint_model import read_validation_history


DEFAULT_MODEL_DIR = Path("/root/autodl-fs/sft_experiments/models/joint_lora_r8")


def latest_complete_checkpoint(model_dir: Path) -> Path | None:
    candidates = []
    for path in model_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        required = ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors")
        if all((path / filename).exists() for filename in required):
            candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def screen_exists(name: str) -> bool:
    result = subprocess.run(
        ["screen", "-list"],
        check=False,
        capture_output=True,
        text=True,
    )
    return f".{name}" in result.stdout


def stop_screen(name: str) -> None:
    subprocess.run(["screen", "-S", name, "-X", "quit"], check=False)
    for _ in range(60):
        if not screen_exists(name):
            return
        time.sleep(1)
    raise RuntimeError(f"Training screen {name!r} did not stop within 60 seconds.")


def training_finished(model_dir: Path, planned_steps: int) -> bool:
    summary_path = model_dir / "joint_training_summary.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary.get("global_step", 0)) >= planned_steps


def decision_from_history(
    history: list[dict[str, Any]],
    *,
    patience: int,
    immediate_relative_degradation: float,
) -> dict[str, Any] | None:
    if not history:
        return None
    best_index = min(range(len(history)), key=lambda index: history[index]["eval_loss"])
    best = history[best_index]
    latest = history[-1]
    relative_degradation = latest["eval_loss"] / best["eval_loss"] - 1.0
    validations_without_improvement = len(history) - best_index - 1

    reason = None
    if latest["step"] != best["step"] and relative_degradation >= immediate_relative_degradation:
        reason = "immediate_relative_degradation_threshold"
    elif validations_without_improvement >= patience:
        reason = "validation_patience_exhausted"
    if reason is None:
        return None
    return {
        "decision": "early_stop_and_select_best_checkpoint",
        "reason": reason,
        "selected_step": best["step"],
        "selected_validation_loss": best["eval_loss"],
        "latest_step": latest["step"],
        "latest_validation_loss": latest["eval_loss"],
        "relative_degradation": relative_degradation,
        "validations_without_improvement": validations_without_improvement,
        "patience": patience,
        "immediate_relative_degradation_threshold": immediate_relative_degradation,
        "validation_history": history,
    }


def monitor(
    *,
    model_dir: Path,
    screen_name: str,
    output_path: Path,
    poll_seconds: int,
    patience: int,
    immediate_relative_degradation: float,
    planned_steps: int,
) -> dict[str, Any]:
    last_seen_step = None
    while True:
        checkpoint = latest_complete_checkpoint(model_dir)
        history = read_validation_history(model_dir) if checkpoint else []
        if history and history[-1]["step"] != last_seen_step:
            last_seen_step = history[-1]["step"]
            print(
                f"Validation step {last_seen_step}: {history[-1]['eval_loss']:.6f}",
                flush=True,
            )
            decision = decision_from_history(
                history,
                patience=patience,
                immediate_relative_degradation=immediate_relative_degradation,
            )
            if decision is not None:
                stop_screen(screen_name)
                decision["training_screen"] = screen_name
                decision["last_complete_checkpoint"] = str(checkpoint)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return decision

        if training_finished(model_dir, planned_steps):
            best = min(history, key=lambda row: row["eval_loss"])
            decision = {
                "decision": "training_completed_and_select_best_checkpoint",
                "reason": "planned_steps_completed",
                "selected_step": best["step"],
                "selected_validation_loss": best["eval_loss"],
                "latest_step": history[-1]["step"],
                "patience": patience,
                "immediate_relative_degradation_threshold": immediate_relative_degradation,
                "validation_history": history,
            }
            output_path.write_text(
                json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return decision

        if not screen_exists(screen_name):
            raise RuntimeError(
                f"Training screen {screen_name!r} ended before a valid stopping decision."
            )
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor and early-stop Model C training.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--screen-name", default="model_c")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_DIR / "joint_early_stopping_decision.json",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--immediate-relative-degradation", type=float, default=0.10)
    parser.add_argument("--planned-steps", type=int, default=3915)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = monitor(
        model_dir=args.model_dir,
        screen_name=args.screen_name,
        output_path=args.output,
        poll_seconds=args.poll_seconds,
        patience=args.patience,
        immediate_relative_degradation=args.immediate_relative_degradation,
        planned_steps=args.planned_steps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
