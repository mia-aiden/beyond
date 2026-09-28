#!/usr/bin/env python3
"""Summarize the three-seed 2x2 model-by-inference-prompt experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import EXPERIMENTS_DIR


ROOT = EXPERIMENTS_DIR
RESULTS_DIR = ROOT / "prompt_cross_3seed" / "comparison"
SEEDS = (42, 1, 2)
DATASETS = ("stage1", "manual")
CONDITIONS = (
    "model_b_narrative",
    "model_b_joint",
    "seqjoint_narrative",
    "seqjoint_joint",
)
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
CONTRASTS = {
    "seqjoint_minus_model_b_under_narrative_prompt": {
        "seqjoint_narrative": 1,
        "model_b_narrative": -1,
    },
    "seqjoint_minus_model_b_under_joint_prompt": {
        "seqjoint_joint": 1,
        "model_b_joint": -1,
    },
    "joint_minus_narrative_prompt_for_model_b": {
        "model_b_joint": 1,
        "model_b_narrative": -1,
    },
    "joint_minus_narrative_prompt_for_seqjoint": {
        "seqjoint_joint": 1,
        "seqjoint_narrative": -1,
    },
    "model_by_prompt_interaction": {
        "seqjoint_joint": 1,
        "seqjoint_narrative": -1,
        "model_b_joint": -1,
        "model_b_narrative": 1,
    },
}


def evaluation_root(condition: str, seed: int, dataset: str) -> Path:
    if condition == "model_b_narrative":
        return (
            ROOT
            / "model_b_multiseed"
            / "evaluation"
            / f"seed_{seed}"
            / dataset
            / "evaluation"
        )
    if condition == "seqjoint_joint":
        return (
            ROOT
            / "format_semantics_multiseed"
            / "evaluation"
            / "true"
            / f"seed_{seed}"
            / dataset
            / "evaluation"
        )
    return (
        ROOT
        / "prompt_cross_3seed"
        / "evaluation"
        / condition
        / f"seed_{seed}"
        / dataset
        / "evaluation"
    )


def latest_run(condition: str, seed: int, dataset: str) -> Path:
    root = evaluation_root(condition, seed, dataset)
    candidates = sorted(root.glob("run_*/metrics/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation under {root}")
    return candidates[-1].parent.parent


def load_seed_metrics() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run = latest_run(condition, seed, dataset)
                summary = json.loads((run / "metrics" / "summary.json").read_text())
                if int(summary.get("num_missing_prediction_rows", 0)):
                    raise ValueError(f"Missing rows for {condition}/{seed}/{dataset}")
                for metric in METRICS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "seed": seed,
                            "metric": metric,
                            "value": float(summary[metric]),
                            "run_dir": str(run),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_conditions(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_metrics.groupby(["dataset", "condition", "metric"], sort=False)["value"]
        .agg(n_seeds="count", mean="mean", sd="std")
        .reset_index()
    )


def load_aligned_samples(seed: int, dataset: str, metric: str) -> pd.DataFrame:
    column = PER_SAMPLE[metric]
    aligned = None
    reference_column = None
    for condition in CONDITIONS:
        path = latest_run(condition, seed, dataset) / "metrics" / "per_sample_metrics.csv"
        frame = pd.read_csv(path, usecols=["sample_id", "reference_narrative", column]).rename(
            columns={
                "reference_narrative": f"reference_{condition}",
                column: condition,
            }
        )
        aligned = frame if aligned is None else aligned.merge(frame, on="sample_id", validate="one_to_one")
        current_reference = f"reference_{condition}"
        if reference_column is None:
            reference_column = current_reference
        elif not aligned[reference_column].equals(aligned[current_reference]):
            raise ValueError(f"Reference mismatch for {seed}/{dataset}/{condition}")
    if aligned is None:
        raise RuntimeError("No conditions loaded")
    return aligned


def hierarchical_contrasts(
    samples: int = 10_000, random_seed: int = 20260824
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows = []
    for dataset in DATASETS:
        for metric in PER_SAMPLE:
            sign = -1 if metric in LOWER_IS_BETTER else 1
            aligned_by_seed = {
                seed: load_aligned_samples(seed, dataset, metric) for seed in SEEDS
            }
            for contrast, coefficients in CONTRASTS.items():
                differences = []
                for seed in SEEDS:
                    frame = aligned_by_seed[seed]
                    values = sum(
                        coefficient * frame[condition].to_numpy(dtype=float)
                        for condition, coefficient in coefficients.items()
                    )
                    differences.append(sign * values)

                within = np.empty((len(SEEDS), samples))
                for seed_index, values in enumerate(differences):
                    for start in range(0, samples, 250):
                        stop = min(samples, start + 250)
                        indices = rng.integers(
                            0, len(values), size=(stop - start, len(values))
                        )
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
                        "contrast": contrast,
                        "direction_corrected_effect": point,
                        "ci_95_low": float(low),
                        "ci_95_high": float(high),
                        "p_two_sided": float(p),
                        "significant_95": bool(low > 0 or high < 0),
                        "bootstrap_samples": samples,
                        "bootstrap_seed": random_seed,
                    }
                )
    return pd.DataFrame(rows)


def metric_label(metric: str) -> str:
    return {
        "bleu_corpus": "BLEU",
        "rouge1_mean": "ROUGE-1",
        "rouge2_mean": "ROUGE-2",
        "rougeL_mean": "ROUGE-L",
        "bertscore_f1_mean": "BERTScore F1",
        "normalized_dtw_distance_mean": "Normalized DTW",
    }[metric]


def build_report(summary: pd.DataFrame, contrasts: pd.DataFrame) -> str:
    lines = [
        "# Three-Seed Model-by-Prompt 2x2 Experiment",
        "",
        "Models: narrative-only Model B and Seq-joint. Inference prompts: narrative-only and joint Emotion -> Narrative. Fixed seeds: 42, 1, and 2. Values are mean +/- SD across seeds.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend(
            [
                f"## {dataset.title()}",
                "",
                "| Condition | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Normalized DTW |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in CONDITIONS:
            values = []
            for metric in METRICS:
                row = summary.query(
                    "dataset == @dataset and condition == @condition and metric == @metric"
                ).iloc[0]
                values.append(f'{row["mean"]:.5f} +/- {row["sd"]:.5f}')
            lines.append(f"| {condition} | " + " | ".join(values) + " |")
        lines.append("")

    lines.extend(
        [
            "## Hierarchical Bootstrap Contrasts",
            "",
            "Effects are direction-corrected so positive values favor the first-named side of each contrast. The interaction is positive when the joint prompt benefits Seq-joint more than it benefits Model B.",
            "",
            "| Dataset | Metric | Contrast | Effect | 95% CI | p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {metric_label(row.metric)} | {row.contrast} | "
            f"{row.direction_corrected_effect:+.6f} | "
            f"[{row.ci_95_low:+.6f}, {row.ci_95_high:+.6f}] | "
            f"{row.p_two_sided:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "The causal question about prompt-format compatibility is supported only when the model-by-prompt interaction is directionally consistent and its hierarchical 95% interval excludes zero. Main effects alone cannot distinguish training semantics from inference-format matching.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_metrics = load_seed_metrics()
    summary = summarize_conditions(seed_metrics)
    contrasts = hierarchical_contrasts()
    seed_metrics.to_csv(RESULTS_DIR / "seed_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "condition_summary.csv", index=False)
    contrasts.to_csv(RESULTS_DIR / "hierarchical_contrasts.csv", index=False)
    (RESULTS_DIR / "REPORT.md").write_text(
        build_report(summary, contrasts) + "\n", encoding="utf-8"
    )
    print(f"Wrote comparison to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
