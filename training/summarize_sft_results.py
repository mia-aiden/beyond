from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paths import EXPERIMENTS_DIR


FS_ROOT = EXPERIMENTS_DIR
EVALUATION_ROOT = FS_ROOT / "evaluation"


def latest_summary(run_root: Path) -> dict:
    summaries = sorted(run_root.glob("run_*/metrics/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No evaluation summary under {run_root}")
    return json.loads(summaries[-1].read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for task in ("label", "narrative"):
        task_root = EVALUATION_ROOT / task
        if not task_root.exists():
            continue
        for experiment_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            summary = latest_summary(experiment_dir)
            dataset, model = experiment_dir.name.rsplit("_", 1)
            row = {
                "task": task,
                "dataset": dataset,
                "model": model,
                "aligned_rows": summary.get("num_aligned_rows"),
            }
            if task == "label":
                row.update(
                    {
                        "accuracy": summary.get("accuracy"),
                        "macro_f1": summary.get("macro_f1"),
                    }
                )
            else:
                row.update(
                    {
                        "bleu_corpus": summary.get("bleu_corpus"),
                        "rouge1_mean": summary.get("rouge1_mean"),
                        "rouge2_mean": summary.get("rouge2_mean"),
                        "rougeL_mean": summary.get("rougeL_mean"),
                        "bertscore_f1_mean": summary.get("bertscore_f1_mean"),
                        "normalized_dtw_distance_mean": summary.get(
                            "normalized_dtw_distance_mean"
                        ),
                    }
                )
            rows.append(row)

    result = pd.DataFrame(rows)
    output_path = FS_ROOT / "evaluation" / "sft_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
