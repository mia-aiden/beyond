from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU


PER_SAMPLE_METRICS = {
    "rouge1_mean": "rouge1_fmeasure",
    "rouge2_mean": "rouge2_fmeasure",
    "rougeL_mean": "rougeL_fmeasure",
    "bertscore_f1_mean": "bertscore_f1",
    "normalized_dtw_distance_mean": "normalized_dtw_distance",
}


def load_run(run_dir: Path, suffix: str) -> pd.DataFrame:
    metrics_path = run_dir / "metrics/per_sample_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    columns = [
        "sample_id",
        "reference_narrative",
        "predicted_narrative",
        *PER_SAMPLE_METRICS.values(),
    ]
    frame = pd.read_csv(metrics_path, usecols=columns)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id in {metrics_path}")
    return frame.rename(columns={column: f"{column}_{suffix}" for column in columns[1:]})


def align_runs(model_a_run: Path, model_b_run: Path) -> pd.DataFrame:
    model_a = load_run(model_a_run, "a")
    model_b = load_run(model_b_run, "b")
    aligned = model_a.merge(model_b, on="sample_id", how="inner", validate="one_to_one")
    if len(aligned) != len(model_a) or len(aligned) != len(model_b):
        raise ValueError("Runs do not contain identical sample_id sets.")
    if not aligned["reference_narrative_a"].equals(aligned["reference_narrative_b"]):
        raise ValueError("Reference narratives differ between the two runs.")
    return aligned.sort_values("sample_id").reset_index(drop=True)


def bleu_statistics(
    metric: BLEU,
    predictions: list[str],
    references: list[str],
) -> np.ndarray:
    return np.asarray(
        metric._extract_corpus_statistics(predictions, [references]),  # noqa: SLF001
        dtype=np.int64,
    )


def bleu_score(metric: BLEU, statistics: np.ndarray) -> float:
    return float(metric._compute_score_from_stats(statistics.sum(axis=0)).score)  # noqa: SLF001


def bootstrap(
    aligned: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    batch_size: int = 250,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    row_count = len(aligned)
    bleu = BLEU()
    references = aligned["reference_narrative_a"].fillna("").astype(str).tolist()
    stats_a = bleu_statistics(
        bleu,
        aligned["predicted_narrative_a"].fillna("").astype(str).tolist(),
        references,
    )
    stats_b = bleu_statistics(
        bleu,
        aligned["predicted_narrative_b"].fillna("").astype(str).tolist(),
        references,
    )
    distributions = {"bleu_corpus": np.empty(samples, dtype=np.float64)}
    point_estimates = {
        "bleu_corpus": (bleu_score(bleu, stats_a), bleu_score(bleu, stats_b))
    }
    per_sample_differences = {}
    for metric_name, column in PER_SAMPLE_METRICS.items():
        values_a = aligned[f"{column}_a"].to_numpy(dtype=np.float64)
        values_b = aligned[f"{column}_b"].to_numpy(dtype=np.float64)
        if np.isnan(values_a).any() or np.isnan(values_b).any():
            raise ValueError(f"Missing values found for {metric_name}")
        per_sample_differences[metric_name] = values_a - values_b
        distributions[metric_name] = np.empty(samples, dtype=np.float64)
        point_estimates[metric_name] = (float(values_a.mean()), float(values_b.mean()))

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, row_count, size=(stop - start, row_count))
        for offset, sampled_indices in enumerate(indices, start=start):
            distributions["bleu_corpus"][offset] = (
                bleu_score(bleu, stats_a[sampled_indices])
                - bleu_score(bleu, stats_b[sampled_indices])
            )
        for metric_name, differences in per_sample_differences.items():
            distributions[metric_name][start:stop] = differences[indices].mean(axis=1)

    rows = []
    for metric_name, distribution in distributions.items():
        value_a, value_b = point_estimates[metric_name]
        difference = value_a - value_b
        ci_low, ci_high = np.quantile(distribution, [0.025, 0.975])
        probability_nonpositive = (np.count_nonzero(distribution <= 0) + 1) / (samples + 1)
        probability_nonnegative = (np.count_nonzero(distribution >= 0) + 1) / (samples + 1)
        p_value = min(1.0, 2 * min(probability_nonpositive, probability_nonnegative))
        lower_is_better = metric_name == "normalized_dtw_distance_mean"
        if ci_low > 0:
            favored_model = "model_b" if lower_is_better else "model_a"
        elif ci_high < 0:
            favored_model = "model_a" if lower_is_better else "model_b"
        else:
            favored_model = "no_significant_difference"
        rows.append(
            {
                "metric": metric_name,
                "model_a": value_a,
                "model_b": value_b,
                "difference_a_minus_b": difference,
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "p_value_two_sided": float(p_value),
                "significant_95": bool(ci_low > 0 or ci_high < 0),
                "favored_model": favored_model,
                "lower_is_better": lower_is_better,
            }
        )
    return rows


def write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_bootstrap_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"metadata": metadata, "results": rows}
    (output_dir / "paired_bootstrap_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Paired Bootstrap: {metadata['model_a_name']} vs {metadata['model_b_name']}",
        "",
        f"Dataset: {metadata['dataset']} ({metadata['aligned_rows']} paired samples)",
        f"Bootstrap samples: {metadata['bootstrap_samples']}; seed: {metadata['seed']}",
        "",
        "| Metric | Model A | Model B | A - B | 95% CI | p | Favored |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['model_a']:.6f} | {row['model_b']:.6f} | "
            f"{row['difference_a_minus_b']:+.6f} | "
            f"[{row['ci_95_low']:+.6f}, {row['ci_95_high']:+.6f}] | "
            f"{row['p_value_two_sided']:.4f} | {row['favored_model']} |"
        )
    (output_dir / "PAIRED_BOOTSTRAP_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap for narrative metrics.")
    parser.add_argument("--model-a-run", type=Path, required=True)
    parser.add_argument("--model-b-run", type=Path, required=True)
    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aligned = align_runs(args.model_a_run, args.model_b_run)
    rows = bootstrap(aligned, samples=args.samples, seed=args.seed)
    metadata = {
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "model_a_run": str(args.model_a_run),
        "model_b_run": str(args.model_b_run),
        "dataset": args.dataset,
        "aligned_rows": len(aligned),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "difference_definition": "model_a - model_b",
    }
    write_outputs(rows, args.output_dir, metadata)
    print(json.dumps({"metadata": metadata, "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
