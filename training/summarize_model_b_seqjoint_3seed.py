#!/usr/bin/env python3
"""Compare Model B and true Seq-joint over fixed seeds 42, 1, and 2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paths import EXPERIMENTS_DIR


ROOT = EXPERIMENTS_DIR
MODEL_B_DIR = ROOT / "model_b_multiseed" / "evaluation"
SEQJOINT_DIR = ROOT / "format_semantics_multiseed" / "evaluation" / "true"
RESULTS_DIR = ROOT / "model_b_multiseed" / "comparison"
SEEDS = (42, 1, 2)
DATASETS = ("stage1", "manual")
METRICS = (
    "bleu_corpus",
    "rouge1_mean",
    "rouge2_mean",
    "rougeL_mean",
    "bertscore_f1_mean",
    "normalized_dtw_distance_mean",
)
PER_SAMPLE = {
    "rouge1_mean": "rouge1_fmeasure",
    "rouge2_mean": "rouge2_fmeasure",
    "rougeL_mean": "rougeL_fmeasure",
    "bertscore_f1_mean": "bertscore_f1",
    "normalized_dtw_distance_mean": "normalized_dtw_distance",
}
LOWER_IS_BETTER = {"normalized_dtw_distance_mean"}


def latest_run(model: str, seed: int, dataset: str) -> Path:
    if model == "model_b":
        root = MODEL_B_DIR / f"seed_{seed}" / dataset / "evaluation"
    else:
        root = SEQJOINT_DIR / f"seed_{seed}" / dataset / "evaluation"
    candidates = sorted(root.glob("run_*/metrics/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation under {root}")
    return candidates[-1].parent.parent


def load_seed_metrics() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in ("model_b", "seqjoint"):
            for seed in SEEDS:
                run = latest_run(model, seed, dataset)
                summary = json.loads((run / "metrics" / "summary.json").read_text())
                if int(summary["num_missing_prediction_rows"]):
                    raise ValueError(f"Missing rows for {model}/{seed}/{dataset}")
                for metric in METRICS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "metric": metric,
                            "value": float(summary[metric]),
                            "run_dir": str(run),
                        }
                    )
    return pd.DataFrame(rows)


def condition_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, metric), group in seed_metrics.groupby(
        ["dataset", "model", "metric"], sort=False
    ):
        values = group["value"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def paired_seed_comparisons(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    signs = np.asarray(
        [[1 if (mask >> bit) & 1 else -1 for bit in range(len(SEEDS))] for mask in range(8)]
    )
    for dataset in DATASETS:
        for metric in METRICS:
            pivot = seed_metrics.query("dataset == @dataset and metric == @metric").pivot(
                index="seed", columns="model", values="value"
            ).reindex(SEEDS)
            raw = pivot["seqjoint"].to_numpy() - pivot["model_b"].to_numpy()
            advantage = -raw if metric in LOWER_IS_BETTER else raw
            mean = float(advantage.mean())
            sd = float(advantage.std(ddof=1))
            margin = float(stats.t.ppf(0.975, 2) * stats.sem(advantage))
            permutation = (signs * advantage).mean(axis=1)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "model_b_mean": float(pivot["model_b"].mean()),
                    "seqjoint_mean": float(pivot["seqjoint"].mean()),
                    "seqjoint_advantage": mean,
                    "advantage_ci_95_low": mean - margin,
                    "advantage_ci_95_high": mean + margin,
                    "paired_t_p": float(stats.ttest_1samp(advantage, 0).pvalue),
                    "exact_sign_flip_p": float(
                        np.mean(np.abs(permutation) >= abs(mean) - 1e-15)
                    ),
                    "cohen_dz": mean / sd if sd else math.nan,
                }
            )
    return pd.DataFrame(rows)


def sample_advantages(seed: int, dataset: str, metric: str) -> np.ndarray:
    column = PER_SAMPLE[metric]
    frames = []
    for model in ("model_b", "seqjoint"):
        path = latest_run(model, seed, dataset) / "metrics" / "per_sample_metrics.csv"
        frame = pd.read_csv(path, usecols=["sample_id", "reference_narrative", column])
        frames.append(frame)
    aligned = frames[1].merge(frames[0], on="sample_id", suffixes=("_seq", "_b"), validate="one_to_one")
    if len(aligned) != len(frames[0]) or len(aligned) != len(frames[1]):
        raise ValueError(f"Alignment failure for {seed}/{dataset}")
    if not aligned["reference_narrative_seq"].equals(aligned["reference_narrative_b"]):
        raise ValueError(f"Reference mismatch for {seed}/{dataset}")
    difference = aligned[f"{column}_seq"].to_numpy() - aligned[f"{column}_b"].to_numpy()
    return -difference if metric in LOWER_IS_BETTER else difference


def hierarchical_bootstrap(samples: int = 10_000, random_seed: int = 20260824) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = []
    for dataset in DATASETS:
        for metric in PER_SAMPLE:
            differences = [sample_advantages(seed, dataset, metric) for seed in SEEDS]
            within = np.empty((len(SEEDS), samples))
            for seed_index, values in enumerate(differences):
                for start in range(0, samples, 250):
                    stop = min(samples, start + 250)
                    indices = rng.integers(0, len(values), size=(stop - start, len(values)))
                    within[seed_index, start:stop] = values[indices].mean(axis=1)
            selected = rng.integers(0, len(SEEDS), size=(samples, len(SEEDS)))
            replicate = np.arange(samples)
            distribution = np.stack(
                [within[selected[:, draw], replicate] for draw in range(len(SEEDS))]
            ).mean(axis=0)
            point = float(np.mean([values.mean() for values in differences]))
            low, high = np.quantile(distribution, [0.025, 0.975])
            p = min(
                1.0,
                2
                * min(
                    (np.count_nonzero(distribution <= 0) + 1) / (samples + 1),
                    (np.count_nonzero(distribution >= 0) + 1) / (samples + 1),
                ),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "seqjoint_advantage": point,
                    "ci_95_low": float(low),
                    "ci_95_high": float(high),
                    "p_two_sided": float(p),
                    "significant_95": bool(low > 0 or high < 0),
                    "favored": "seqjoint" if low > 0 else "model_b" if high < 0 else "no_clear_difference",
                    "bootstrap_samples": samples,
                    "bootstrap_seed": random_seed,
                }
            )
    return pd.DataFrame(rows)


def label(metric: str) -> str:
    return {
        "bleu_corpus": "BLEU",
        "rouge1_mean": "ROUGE-1",
        "rouge2_mean": "ROUGE-2",
        "rougeL_mean": "ROUGE-L",
        "bertscore_f1_mean": "BERTScore F1",
        "normalized_dtw_distance_mean": "Normalized DTW",
    }[metric]


def build_report(summary: pd.DataFrame, paired: pd.DataFrame, hierarchical: pd.DataFrame) -> str:
    lines = [
        "# Model B vs Seq-joint: Three-Seed Narrative Comparison",
        "",
        "Fixed seeds are 42, 1, and 2. Both models use identical train/validation rows, LoRA hyperparameters, current vLLM inference, deterministic decoding, prompts matched to their training formats, and the same evaluation implementation. Values are mean +/- SD across seeds.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend([f"## {dataset.title()}", "", "| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Normalized DTW |", "|---|---:|---:|---:|---:|---:|---:|"])
        for model in ("model_b", "seqjoint"):
            values = []
            for metric in METRICS:
                row = summary.query(
                    "dataset == @dataset and model == @model and metric == @metric"
                ).iloc[0]
                values.append(f'{row["mean"]:.5f} +/- {row["sd"]:.5f}')
            lines.append(f"| {model} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## Hierarchical Bootstrap",
            "",
            "Positive advantage favors Seq-joint; normalized DTW has been direction-corrected. The bootstrap resamples both training seeds and test samples.",
            "",
            "| Dataset | Metric | Advantage | 95% CI | p | Favored |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in hierarchical.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {label(row.metric)} | {row.seqjoint_advantage:+.6f} | "
            f"[{row.ci_95_low:+.6f}, {row.ci_95_high:+.6f}] | {row.p_two_sided:.4f} | {row.favored} |"
        )
    primary = hierarchical[hierarchical["metric"].isin(["rougeL_mean", "bertscore_f1_mean"])]
    seq_wins = int((primary["favored"] == "seqjoint").sum())
    b_wins = int((primary["favored"] == "model_b").sum())
    if seq_wins == len(primary):
        conclusion = "Seq-joint consistently improves both primary metrics on both test sets."
    elif seq_wins and not b_wins:
        conclusion = "Seq-joint shows partial evidence of improvement, but the result is not consistent across both primary metrics and test sets."
    elif b_wins:
        conclusion = "Seq-joint is not consistently superior; at least one primary comparison favors Model B."
    else:
        conclusion = "No robust primary-metric difference is detected between Seq-joint and Model B."
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            conclusion,
            "",
            "This is an exploratory three-seed comparison. With only three training seeds, exact seed-level sign-flip tests have low resolution; effect directions and hierarchical confidence intervals should be emphasized over isolated p-values.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_metrics = load_seed_metrics()
    summary = condition_summary(seed_metrics)
    paired = paired_seed_comparisons(seed_metrics)
    hierarchical = hierarchical_bootstrap()
    seed_metrics.to_csv(RESULTS_DIR / "seed_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "condition_summary.csv", index=False)
    paired.to_csv(RESULTS_DIR / "paired_seed_comparisons.csv", index=False)
    hierarchical.to_csv(RESULTS_DIR / "hierarchical_bootstrap.csv", index=False)
    (RESULTS_DIR / "REPORT.md").write_text(
        build_report(summary, paired, hierarchical) + "\n", encoding="utf-8"
    )
    print(f"Wrote comparison to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
