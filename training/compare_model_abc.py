from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from paths import EXPERIMENTS_DIR


DEFAULT_ROOT = EXPERIMENTS_DIR
DATASETS = ("stage1", "manual")
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
SPECIALIST_NAMES = {"label": "Model A", "narrative": "Model B"}


def find_one(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No summary matched {root / pattern}")
    return matches[-1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_paths(root: Path, task: str, dataset: str) -> tuple[Path, Path]:
    specialist = find_one(
        f"evaluation_direct/{task}/{dataset}_sft/evaluation/run_*/metrics/summary.json",
        root,
    )
    joint = find_one(
        f"evaluation_joint/{task}/{dataset}/evaluation/run_*/metrics/summary.json",
        root,
    )
    return specialist, joint


def compare(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}
    for task, metrics in TASK_METRICS.items():
        for dataset in DATASETS:
            specialist_path, joint_path = summary_paths(root, task, dataset)
            specialist = load_json(specialist_path)
            joint = load_json(joint_path)
            source_key = f"{task}_{dataset}"
            sources[source_key] = {
                "specialist": str(specialist_path),
                "joint": str(joint_path),
            }
            for metric in metrics:
                specialist_value = float(specialist[metric])
                joint_value = float(joint[metric])
                higher_is_better = metric != "normalized_dtw_distance_mean"
                improvement = (
                    joint_value - specialist_value
                    if higher_is_better
                    else specialist_value - joint_value
                )
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "metric": metric,
                        "specialist_model": SPECIALIST_NAMES[task],
                        "specialist_value": specialist_value,
                        "model_c_value": joint_value,
                        "higher_is_better": higher_is_better,
                        "model_c_improvement": improvement,
                    }
                )
    return {
        "comparison_policy": "direct Transformers inference for all adapters",
        "sources": sources,
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Model A/B/C comparison",
        "",
        (
            "All values use direct Transformers inference with identical prompts, "
            "greedy decoding, test sets, and metric implementations."
        ),
        "",
        "| Task | Dataset | Metric | Specialist | Specialist value | Model C | C improvement |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            (
                "| {task} | {dataset} | {metric} | {specialist_model} | "
                "{specialist_value:.6f} | {model_c_value:.6f} | "
                "{model_c_improvement:+.6f} |"
            ).format(**row)
        )
    lines.extend(
        [
            "",
            (
                "Positive `C improvement` always means Model C is better. For "
                "normalized DTW distance, lower is better and the delta direction is reversed."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare specialist Models A/B with joint Model C."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "comparison_model_abc",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result["rows"], args.output_dir / "model_abc_metrics.csv")
    (args.output_dir / "model_abc_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(result, args.output_dir / "model_abc_comparison.md")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
