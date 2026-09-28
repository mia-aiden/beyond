from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from paths import EXPERIMENTS_DIR


DEFAULT_ROOT = EXPERIMENTS_DIR
TASK_METRICS = {
    "label": ("accuracy", "macro_f1"),
    "narrative": (
        "bleu_corpus",
        "rouge1_mean",
        "rouge2_mean",
        "rougeL_mean",
        "bertscore_f1_mean",
        "normalized_dtw_distance_mean",
    ),
}
DATASETS = ("stage1", "manual")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_summary(root: Path, task: str, dataset: str) -> Path:
    matches = sorted(root.glob(f"{task}/{dataset}/evaluation/run_*/metrics/summary.json"))
    if not matches:
        raise FileNotFoundError(f"No summary for {task}/{dataset} under {root}")
    return matches[-1]


def specialist_summary(root: Path, task: str, dataset: str) -> Path:
    matches = sorted(
        root.glob(
            f"evaluation_direct/{task}/{dataset}_sft/evaluation/run_*/metrics/summary.json"
        )
    )
    if not matches:
        raise FileNotFoundError(f"No specialist summary for {task}/{dataset}")
    return matches[-1]


def build_comparison(root: Path) -> dict[str, Any]:
    evaluation_roots = {
        "0.3": root / "ablations/lambda_0_3/evaluation",
        "0.5": root / "ablations/lambda_0_5/evaluation",
        "0.8": root / "ablations/lambda_0_8/evaluation",
        "1.0": root / "evaluation_joint",
    }
    model_dirs = {
        "0.3": root / "ablations/lambda_0_3/models/joint_lora_r8",
        "0.5": root / "ablations/lambda_0_5/models/joint_lora_r8",
        "0.8": root / "ablations/lambda_0_8/models/joint_lora_r8",
        "1.0": root / "models/joint_lora_r8",
    }

    metric_rows = []
    sources: dict[str, str] = {}
    for task, metrics in TASK_METRICS.items():
        for dataset in DATASETS:
            specialist_path = specialist_summary(root, task, dataset)
            specialist = load_json(specialist_path)
            values: dict[str, float] = {}
            for weight, evaluation_root in evaluation_roots.items():
                path = latest_summary(evaluation_root, task, dataset)
                sources[f"lambda_{weight}_{task}_{dataset}"] = str(path)
                summary = load_json(path)
                for metric in metrics:
                    values[f"{weight}:{metric}"] = float(summary[metric])

            for metric in metrics:
                higher_is_better = metric != "normalized_dtw_distance_mean"
                lambda_values = {
                    weight: values[f"{weight}:{metric}"] for weight in evaluation_roots
                }
                best_lambda = (
                    max(lambda_values, key=lambda_values.get)
                    if higher_is_better
                    else min(lambda_values, key=lambda_values.get)
                )
                metric_rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "specialist_value": float(specialist[metric]),
                        "lambda_0.3": lambda_values["0.3"],
                        "lambda_0.5": lambda_values["0.5"],
                        "lambda_0.8": lambda_values["0.8"],
                        "lambda_1.0": lambda_values["1.0"],
                        "best_lambda": best_lambda,
                        "best_value": lambda_values[best_lambda],
                    }
                )

    training_rows = []
    for weight, model_dir in model_dirs.items():
        finalization = load_json(model_dir / "joint_finalization_summary.json")
        components = load_json(model_dir / "joint_validation_components.json")
        training_rows.append(
            {
                "lambda": weight,
                "selected_step": finalization["selected_step"],
                "joint_validation_loss": finalization["selected_validation_loss"],
                "emotion_validation_loss": components["emotion_validation_loss"],
                "narrative_validation_loss": components["narrative_validation_loss"],
                "adapter_sha256": finalization["adapter_sha256"],
            }
        )
    return {
        "controlled_variables": {
            "seed": 42,
            "data_split": "identical paired train/validation and held-out tests",
            "changed_variable": "emotion_loss_weight",
        },
        "sources": sources,
        "training": training_rows,
        "metrics": metric_rows,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Model C Lambda Ablation",
        "",
        "All runs use seed 42 and identical data, prompts, LoRA settings, decoding, and metrics. Only the emotion loss weight changes.",
        "",
        "## Training",
        "",
        "| Lambda | Selected step | Joint val loss | Emotion val loss | Narrative val loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["training"]:
        lines.append(
            f"| {row['lambda']} | {row['selected_step']} | "
            f"{row['joint_validation_loss']:.6f} | "
            f"{row['emotion_validation_loss']:.6f} | "
            f"{row['narrative_validation_loss']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Task | Dataset | Metric | Specialist | λ=0.3 | λ=0.5 | λ=0.8 | λ=1.0 | Best λ |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["metrics"]:
        lines.append(
            f"| {row['task']} | {row['dataset']} | {row['metric']} | "
            f"{row['specialist_value']:.6f} | {row['lambda_0.3']:.6f} | "
            f"{row['lambda_0.5']:.6f} | {row['lambda_0.8']:.6f} | "
            f"{row['lambda_1.0']:.6f} | {row['best_lambda']} |"
        )
    lines.extend(
        [
            "",
            "For normalized DTW distance, lower is better. For every other metric, higher is better. No single aggregate score is used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Model C emotion loss weights.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "ablations/lambda_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_comparison(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result["training"], args.output_dir / "training_summary.csv")
    write_csv(result["metrics"], args.output_dir / "test_metrics.csv")
    (args.output_dir / "lambda_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(result, args.output_dir / "LAMBDA_ABLATION_REPORT.md")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
