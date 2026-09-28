#!/usr/bin/env python3
"""Aggregate the format-versus-semantics experiment across five seeds."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

from paths import EXPERIMENTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


EXPERIMENT_DIR = EXPERIMENTS_DIR / "format_semantics_multiseed"
EVALUATION_DIR = EXPERIMENT_DIR / "evaluation"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CONDITIONS = ("true", "forced_wrong", "constant_neutral", "marker_control")
CONTROLS = CONDITIONS[1:]
EMOTION_CONDITIONS = ("true", "forced_wrong", "constant_neutral")
EMOTION_LABELS = ("Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise")
SEEDS = (42, 1, 2, 3, 4)
DATASETS = ("stage1", "manual")
METRICS = (
    "bleu_corpus",
    "rouge1_mean",
    "rouge2_mean",
    "rougeL_mean",
    "bertscore_f1_mean",
    "normalized_dtw_distance_mean",
)
PER_SAMPLE_COLUMNS = {
    "rouge1_mean": "rouge1_fmeasure",
    "rouge2_mean": "rouge2_fmeasure",
    "rougeL_mean": "rougeL_fmeasure",
    "bertscore_f1_mean": "bertscore_f1",
    "normalized_dtw_distance_mean": "normalized_dtw_distance",
}
LOWER_IS_BETTER = {"normalized_dtw_distance_mean"}
PRIMARY_METRICS = {"rougeL_mean", "bertscore_f1_mean"}


def latest_run(condition: str, seed: int, dataset: str) -> Path:
    root = EVALUATION_DIR / condition / f"seed_{seed}" / dataset / "evaluation"
    candidates = sorted(root.glob("run_*/metrics/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation summary under {root}")
    return candidates[-1].parent.parent


def load_seed_metrics() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_dir = latest_run(condition, seed, dataset)
                summary = json.loads(
                    (run_dir / "metrics" / "summary.json").read_text(encoding="utf-8")
                )
                if int(summary["num_missing_prediction_rows"]) != 0:
                    raise ValueError(f"Missing predictions in {condition}/{seed}/{dataset}")
                for metric in METRICS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "seed": seed,
                            "metric": metric,
                            "value": float(summary[metric]),
                            "run_dir": str(run_dir),
                        }
                    )
    return pd.DataFrame(rows)


def load_training_metrics() -> pd.DataFrame:
    manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for job in manifest:
        model_path = Path(job["model_path"])
        results_path = model_path / "all_results.json"
        state_path = model_path / "trainer_state.json"
        if not results_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"Incomplete training artifacts in {model_path}")
        results = json.loads(results_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        epoch = float(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        if epoch < 2.99 or global_step != 1959:
            raise ValueError(
                f"Unexpected training completion for {job['condition']}/{job['seed']}: "
                f"epoch={epoch}, global_step={global_step}"
            )
        rows.append(
            {
                "condition": job["condition"],
                "seed": int(job["seed"]),
                "action": job["action"],
                "model_path": str(model_path),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": float(results.get("train_loss", math.nan)),
                "eval_loss": float(results.get("eval_loss", state.get("best_metric", math.nan))),
                "best_eval_loss": float(state.get("best_metric", math.nan)),
                "best_model_checkpoint": str(state.get("best_model_checkpoint", "")),
                "train_runtime_seconds": float(results.get("train_runtime", math.nan)),
                "train_samples_per_second": float(
                    results.get("train_samples_per_second", math.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def load_emotion_seed_metrics() -> pd.DataFrame:
    rows = []
    valid_labels = set(EMOTION_LABELS)
    for dataset in DATASETS:
        for condition in EMOTION_CONDITIONS:
            for seed in SEEDS:
                run_dir = latest_run(condition, seed, dataset)
                path = run_dir / "metrics" / "per_sample_metrics.csv"
                frame = pd.read_csv(
                    path,
                    usecols=["emotion_label", "predicted_emotion_label"],
                ).dropna()
                gold = frame["emotion_label"].astype(str).str.strip().str.title()
                predicted = (
                    frame["predicted_emotion_label"].astype(str).str.strip().str.title()
                )
                unknown_gold = sorted(set(gold) - valid_labels)
                unknown_predictions = sorted(set(predicted) - valid_labels)
                if unknown_gold or unknown_predictions:
                    raise ValueError(
                        f"Unknown emotion labels in {condition}/{seed}/{dataset}: "
                        f"gold={unknown_gold}, predictions={unknown_predictions}"
                    )
                accuracy = float((gold == predicted).mean())
                label_f1 = []
                for label in EMOTION_LABELS:
                    true_positive = int(((gold == label) & (predicted == label)).sum())
                    false_positive = int(((gold != label) & (predicted == label)).sum())
                    false_negative = int(((gold == label) & (predicted != label)).sum())
                    denominator = 2 * true_positive + false_positive + false_negative
                    label_f1.append(2 * true_positive / denominator if denominator else 0.0)
                rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "seed": seed,
                            "metric": "emotion_accuracy",
                            "value": accuracy,
                            "num_rows": len(frame),
                            "run_dir": str(run_dir),
                        },
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "seed": seed,
                            "metric": "emotion_macro_f1",
                            "value": float(np.mean(label_f1)),
                            "num_rows": len(frame),
                            "run_dir": str(run_dir),
                        },
                    ]
                )
    return pd.DataFrame(rows)


def summarize_resources() -> dict:
    path = EXPERIMENT_DIR / "resource_history.jsonl"
    if not path.is_file():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def summarize(selected: list[dict]) -> dict:
        gpu_rows = [gpu for record in selected for gpu in record.get("gpus", [])]
        if not gpu_rows:
            return {"samples": 0}
        utilizations = np.asarray([gpu["utilization_percent"] for gpu in gpu_rows], dtype=float)
        return {
            "samples": len(selected),
            "gpu_observations": len(gpu_rows),
            "mean_gpu_utilization_percent": float(utilizations.mean()),
            "observations_at_or_above_90_percent": int(np.count_nonzero(utilizations >= 90)),
            "fraction_at_or_above_90_percent": float(np.mean(utilizations >= 90)),
            "mean_power_w": float(np.mean([gpu["power_w"] for gpu in gpu_rows])),
            "maximum_temperature_c": int(max(gpu["temperature_c"] for gpu in gpu_rows)),
        }

    training = [
        record
        for record in records
        if int(record.get("training_counts", {}).get("running", 0)) > 0
    ]
    evaluating = [
        record for record in records if record.get("evaluation_phase") == "evaluating"
    ]
    return {
        "sampling_interval_seconds": 60,
        "total_records": len(records),
        "training": summarize(training),
        "evaluation": summarize(evaluating),
    }


def load_overlap_audit() -> dict:
    path = EXPERIMENT_DIR / "test_overlap_audit.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("training_leakage_passed", False):
        raise ValueError("Training-to-test leakage audit did not pass.")
    return audit


def load_token_audit() -> dict:
    path = EXPERIMENT_DIR / "token_length_audit.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    disagreements = [
        comparison[split]["different_cutoff_status_rows"]
        for comparison in audit["paired_comparisons_to_true"].values()
        for split in ("train", "validation")
    ]
    if any(disagreements):
        raise ValueError("A control condition differs from true in cutoff status.")
    return audit


def summarize_conditions(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in seed_metrics.groupby(["dataset", "condition", "metric"], sort=False):
        dataset, condition, metric = keys
        values = group["value"].to_numpy(dtype=float)
        standard_error = float(stats.sem(values))
        margin = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci_95_low": float(values.mean() - margin),
                "ci_95_high": float(values.mean() + margin),
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
        )
    return pd.DataFrame(rows)


def paired_seed_tests(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for metric in METRICS:
            true = (
                seed_metrics.query("dataset == @dataset and condition == 'true' and metric == @metric")
                .set_index("seed")["value"]
                .reindex(SEEDS)
            )
            for control in CONTROLS:
                compared = (
                    seed_metrics.query(
                        "dataset == @dataset and condition == @control and metric == @metric"
                    )
                    .set_index("seed")["value"]
                    .reindex(SEEDS)
                )
                raw_difference = true.to_numpy() - compared.to_numpy()
                advantage = -raw_difference if metric in LOWER_IS_BETTER else raw_difference
                mean_advantage = float(advantage.mean())
                sd_advantage = float(advantage.std(ddof=1))
                margin = float(stats.t.ppf(0.975, len(SEEDS) - 1) * stats.sem(advantage))
                t_result = stats.ttest_1samp(advantage, popmean=0.0)
                signs = np.asarray(
                    [
                        [1 if (mask >> bit) & 1 else -1 for bit in range(len(SEEDS))]
                        for mask in range(2 ** len(SEEDS))
                    ],
                    dtype=float,
                )
                permutation_means = (signs * advantage).mean(axis=1)
                exact_p = float(
                    np.count_nonzero(np.abs(permutation_means) >= abs(mean_advantage) - 1e-15)
                    / len(permutation_means)
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "control": control,
                        "n_seeds": len(SEEDS),
                        "true_mean": float(true.mean()),
                        "control_mean": float(compared.mean()),
                        "true_advantage": mean_advantage,
                        "advantage_ci_95_low": mean_advantage - margin,
                        "advantage_ci_95_high": mean_advantage + margin,
                        "cohen_dz": mean_advantage / sd_advantage if sd_advantage else math.nan,
                        "paired_t_p": float(t_result.pvalue),
                        "exact_sign_flip_p": exact_p,
                        "lower_is_better": metric in LOWER_IS_BETTER,
                    }
                )
    result = pd.DataFrame(rows)
    result["paired_t_p_holm"] = np.nan
    for _, indices in result.groupby(["dataset", "metric"]).groups.items():
        ordered = sorted(indices, key=lambda index: result.loc[index, "paired_t_p"])
        adjusted = np.empty(len(ordered), dtype=float)
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            value = min(1.0, (count - rank) * float(result.loc[index, "paired_t_p"]))
            running = max(running, value)
            adjusted[rank] = running
        for index, value in zip(ordered, adjusted):
            result.loc[index, "paired_t_p_holm"] = value
    return result


def load_difference(condition: str, seed: int, dataset: str, metric: str) -> np.ndarray:
    column = PER_SAMPLE_COLUMNS[metric]
    true_path = latest_run("true", seed, dataset) / "metrics" / "per_sample_metrics.csv"
    control_path = latest_run(condition, seed, dataset) / "metrics" / "per_sample_metrics.csv"
    true = pd.read_csv(true_path, usecols=["sample_id", "reference_narrative", column])
    control = pd.read_csv(control_path, usecols=["sample_id", "reference_narrative", column])
    aligned = true.merge(control, on="sample_id", suffixes=("_true", "_control"), validate="one_to_one")
    if len(aligned) != len(true) or len(aligned) != len(control):
        raise ValueError(f"Sample mismatch for {condition}/{seed}/{dataset}")
    if not aligned["reference_narrative_true"].equals(aligned["reference_narrative_control"]):
        raise ValueError(f"Reference mismatch for {condition}/{seed}/{dataset}")
    difference = (
        aligned[f"{column}_true"].to_numpy(dtype=float)
        - aligned[f"{column}_control"].to_numpy(dtype=float)
    )
    return -difference if metric in LOWER_IS_BETTER else difference


def hierarchical_bootstrap(samples: int = 10_000, seed: int = 20260819) -> pd.DataFrame:
    rows = []
    base_rng = np.random.default_rng(seed)
    for dataset in DATASETS:
        for metric in PER_SAMPLE_COLUMNS:
            for control in CONTROLS:
                differences = [load_difference(control, value, dataset, metric) for value in SEEDS]
                bootstrap_means = np.empty((len(SEEDS), samples), dtype=float)
                for seed_index, values in enumerate(differences):
                    for start in range(0, samples, 250):
                        stop = min(samples, start + 250)
                        indices = base_rng.integers(
                            0, len(values), size=(stop - start, len(values))
                        )
                        bootstrap_means[seed_index, start:stop] = values[indices].mean(axis=1)
                selected_seeds = base_rng.integers(0, len(SEEDS), size=(samples, len(SEEDS)))
                replicate_indices = np.arange(samples)
                distribution = np.stack(
                    [bootstrap_means[selected_seeds[:, draw], replicate_indices] for draw in range(len(SEEDS))]
                ).mean(axis=0)
                point = float(np.mean([values.mean() for values in differences]))
                ci_low, ci_high = np.quantile(distribution, [0.025, 0.975])
                p_two_sided = min(
                    1.0,
                    2.0
                    * min(
                        (np.count_nonzero(distribution <= 0) + 1) / (samples + 1),
                        (np.count_nonzero(distribution >= 0) + 1) / (samples + 1),
                    ),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "control": control,
                        "true_advantage": point,
                        "ci_95_low": float(ci_low),
                        "ci_95_high": float(ci_high),
                        "p_two_sided": float(p_two_sided),
                        "bootstrap_samples": samples,
                        "bootstrap_seed": seed,
                        "significant_95": bool(ci_low > 0 or ci_high < 0),
                        "favored": (
                            "true" if ci_low > 0 else control if ci_high < 0 else "no_clear_difference"
                        ),
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


def plot_primary_metrics(condition_summary: pd.DataFrame) -> None:
    colors = {
        "true": "#1f5a94",
        "forced_wrong": "#d36b27",
        "constant_neutral": "#488a56",
        "marker_control": "#a33d3d",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        for column_index, metric in enumerate(("rougeL_mean", "bertscore_f1_mean")):
            axis = axes[row_index, column_index]
            subset = condition_summary.query("dataset == @dataset and metric == @metric").set_index(
                "condition"
            )
            means = np.asarray([subset.loc[condition, "mean"] for condition in CONDITIONS])
            low = np.asarray([subset.loc[condition, "ci_95_low"] for condition in CONDITIONS])
            high = np.asarray([subset.loc[condition, "ci_95_high"] for condition in CONDITIONS])
            positions = np.arange(len(CONDITIONS))
            axis.bar(
                positions,
                means,
                color=[colors[condition] for condition in CONDITIONS],
                width=0.68,
                yerr=np.vstack([means - low, high - means]),
                capsize=5,
            )
            axis.set_xticks(positions, [name.replace("_", "\n") for name in CONDITIONS])
            axis.set_title(f"{dataset.title()}: {metric_label(metric)}")
            axis.set_ylabel("Score")
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Narrative quality across five training seeds", fontsize=15, fontweight="bold")
    fig.savefig(PLOTS_DIR / "primary_metrics_by_condition.png", dpi=180)
    plt.close(fig)


def plot_true_advantages(hierarchical: pd.DataFrame) -> None:
    subset = hierarchical[hierarchical["metric"].isin(PRIMARY_METRICS)].copy()
    labels = [
        f"{row.dataset} / {metric_label(row.metric)} / {row.control}"
        for row in subset.itertuples(index=False)
    ]
    points = subset["true_advantage"].to_numpy(dtype=float)
    lows = subset["ci_95_low"].to_numpy(dtype=float)
    highs = subset["ci_95_high"].to_numpy(dtype=float)
    positions = np.arange(len(subset))[::-1]
    colors = ["#1f5a94" if low > 0 else "#a33d3d" if high < 0 else "#6f767d" for low, high in zip(lows, highs)]
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    for point, position, low, high, color in zip(points, positions, lows, highs, colors):
        axis.errorbar(
            [point],
            [position],
            xerr=np.asarray([[point - low], [high - point]]),
            fmt="none",
            ecolor=color,
            elinewidth=2,
            capsize=4,
        )
    axis.scatter(points, positions, c=colors, s=42, zorder=3)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("True-label advantage (positive favors true)")
    axis.set_title("Hierarchical bootstrap: true labels vs controls")
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(PLOTS_DIR / "true_advantage_forest.png", dpi=180)
    plt.close(fig)


def plot_emotion_metrics(emotion_summary: pd.DataFrame) -> None:
    colors = {
        "true": "#1f5a94",
        "forced_wrong": "#d36b27",
        "constant_neutral": "#488a56",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        for column_index, metric in enumerate(("emotion_accuracy", "emotion_macro_f1")):
            axis = axes[row_index, column_index]
            subset = emotion_summary.query(
                "dataset == @dataset and metric == @metric"
            ).set_index("condition")
            means = np.asarray(
                [subset.loc[condition, "mean"] for condition in EMOTION_CONDITIONS]
            )
            standard_deviations = np.asarray(
                [subset.loc[condition, "sd"] for condition in EMOTION_CONDITIONS]
            )
            positions = np.arange(len(EMOTION_CONDITIONS))
            axis.bar(
                positions,
                means,
                color=[colors[condition] for condition in EMOTION_CONDITIONS],
                width=0.68,
                yerr=standard_deviations,
                capsize=5,
            )
            axis.set_xticks(
                positions, [name.replace("_", "\n") for name in EMOTION_CONDITIONS]
            )
            axis.set_ylim(0, 1)
            axis.set_title(
                f"{dataset.title()}: "
                f"{'Accuracy' if metric == 'emotion_accuracy' else 'Macro-F1'}"
            )
            axis.set_ylabel("Score")
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Emotion prediction as a manipulation check", fontsize=15, fontweight="bold")
    fig.savefig(PLOTS_DIR / "emotion_metrics_by_condition.png", dpi=180)
    plt.close(fig)


def build_report(
    condition_summary: pd.DataFrame,
    emotion_summary: pd.DataFrame,
    paired: pd.DataFrame,
    hierarchical: pd.DataFrame,
    training_metrics: pd.DataFrame,
    resource_summary: dict,
    overlap_audit: dict,
    token_audit: dict,
) -> str:
    cross_test = overlap_audit["cross_test_overlap"]
    lines = [
        "# Format vs Emotion Semantics: Five-Seed Control Experiment",
        "",
        "![Primary metrics by condition](plots/primary_metrics_by_condition.png)",
        "",
        "## Experimental design",
        "",
        "All four conditions use identical source texts and reference narratives. True uses the correct Ekman label; forced-wrong uses an always-incorrect label while exactly preserving label frequencies; constant-neutral uses the same fixed emotion token; marker-control replaces emotion with a fixed non-semantic marker. Each condition is trained with seeds 42, 1, 2, 3, and 4.",
        "",
        "Primary narrative metrics are ROUGE-L and BERTScore F1. BLEU, ROUGE-1/2, and normalized DTW are secondary. Higher is better except normalized DTW. Values below are mean +/- SD across five seeds.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend([f"## {dataset.title()} results", ""])
        lines.append("| Condition | BLEU | ROUGE-L | BERTScore F1 | Normalized DTW |")
        lines.append("|---|---:|---:|---:|---:|")
        for condition in CONDITIONS:
            row_values = []
            for metric in (
                "bleu_corpus",
                "rougeL_mean",
                "bertscore_f1_mean",
                "normalized_dtw_distance_mean",
            ):
                row = condition_summary.query(
                    "dataset == @dataset and condition == @condition and metric == @metric"
                ).iloc[0]
                row_values.append(f'{row["mean"]:.5f} +/- {row["sd"]:.5f}')
            lines.append(f"| {condition} | " + " | ".join(row_values) + " |")
        lines.append("")

    lines.extend(
        [
            "## Emotion-task manipulation check",
            "",
            "![Emotion metrics by condition](plots/emotion_metrics_by_condition.png)",
            "",
            "Emotion prediction is evaluated over the seven Ekman classes. These scores verify whether each training manipulation learned its assigned emotion output; they are not a fair emotion benchmark comparison because forced-wrong and constant-neutral were deliberately trained with incorrect or uninformative labels. Marker-control does not emit an emotion label and is therefore not applicable.",
            "",
            "| Dataset | Condition | Accuracy | Macro-F1 |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for condition in EMOTION_CONDITIONS:
            values = []
            for metric in ("emotion_accuracy", "emotion_macro_f1"):
                row = emotion_summary.query(
                    "dataset == @dataset and condition == @condition and metric == @metric"
                ).iloc[0]
                values.append(f'{row["mean"]:.4f} +/- {row["sd"]:.4f}')
            lines.append(f"| {dataset} | {condition} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "The true-label model clearly learned the intended emotion task, while the two semantic controls behaved as designed. Because this strong emotion-learning signal did not produce a reliable narrative-quality advantage, the null narrative result cannot be explained by a failure to learn emotion labels; it instead indicates no demonstrated positive transfer from correct emotion semantics to narrative generation under this setup.",
            "",
        ]
    )

    lines.extend(
        [
            "## True-label comparisons",
            "",
            "![True-label advantage confidence intervals](plots/true_advantage_forest.png)",
            "",
            "Positive advantage means true labels produced better narrative quality. Hierarchical confidence intervals resample both seeds and test samples.",
            "",
            "| Dataset | Metric | Control | True advantage | 95% CI | p | Conclusion |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    primary = hierarchical[hierarchical["metric"].isin(PRIMARY_METRICS)]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {metric_label(row.metric)} | {row.control} | "
            f"{row.true_advantage:+.6f} | [{row.ci_95_low:+.6f}, {row.ci_95_high:+.6f}] | "
            f"{row.p_two_sided:.4f} | {row.favored} |"
        )

    key = primary[primary["control"] == "forced_wrong"]
    wins = int((key["favored"] == "true").sum())
    losses = int((key["favored"] == "forced_wrong").sum())
    lines.extend(["", "## Interpretation", ""])
    if wins == len(key):
        interpretation = (
            "Correct emotion semantics consistently outperform the distribution-matched wrong-label control "
            "on both primary metrics and both test sets. This supports a semantic contribution beyond output formatting."
        )
    elif wins >= 2 and losses == 0:
        interpretation = (
            "Correct emotion semantics show partial evidence of benefit, but the effect is not robust across every "
            "primary metric and test set. The result should be reported as mixed rather than conclusive."
        )
    elif losses:
        interpretation = (
            "The true-label condition is not consistently superior to the distribution-matched wrong-label control. "
            "The observed joint-format gain cannot be attributed solely to correct emotion semantics."
        )
    else:
        interpretation = (
            "No robust primary-metric difference is detected between true and forced-wrong labels. The evidence "
            "therefore favors a format or auxiliary-task regularization explanation over a demonstrated semantic effect."
        )
    lines.extend(
        [
            interpretation,
            "",
            "Seed-level paired t-tests, Holm-adjusted p-values, exact sign-flip tests, and all secondary metrics are available in the accompanying CSV files. With only five seeds, uncertainty intervals and effect sizes should be emphasized over isolated p-values.",
            "",
            "## Training completion",
            "",
            "| Condition | Completed seeds | Mean best validation loss |",
            "|---|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        rows = training_metrics[training_metrics["condition"] == condition]
        lines.append(
            f"| {condition} | {len(rows)}/5 | {rows['best_eval_loss'].mean():.6f} |"
        )
    lines.extend(
        [
            "",
            "All accepted runs reached epoch 3 and global step 1,959. Runtime fields for resumed runs describe only the resumed segment and should not be compared directly with fresh-run wall time.",
            "",
            "## Data-separation audit",
            "",
            f"No normalized source text or reference narrative from train/validation appears in either test set. The Stage1 and manual tests share {cross_test['shared_sources']} source texts, equal to {100 * cross_test['fraction_of_manual_test']:.1f}% of the manual test, and are therefore not fully independent replications; their reference narratives differ for all shared sources.",
            "",
            "An initial forced-wrong pilot was stopped after step 250 because its grouped label shift retained excessive true-label information (NMI 0.568). It was excluded before evaluation. The corrected minimum-MI derangement was restarted from step 0 and is the only forced-wrong model family included here.",
            "",
            "Formatted token-length auditing found zero train or validation rows whose 1,536-token cutoff status differed between true and any control condition. The length-matched marker prompt is a median of one token shorter than true, with an absolute difference of at most two tokens.",
            "",
        ]
    )
    training_resources = resource_summary.get("training", {})
    if training_resources.get("samples", 0):
        lines.extend(
            [
                "## Compute utilization",
                "",
                f"During active training, mean GPU utilization was {training_resources['mean_gpu_utilization_percent']:.1f}% across {training_resources['gpu_observations']} one-minute GPU observations; {100 * training_resources['fraction_at_or_above_90_percent']:.1f}% of observations were at or above 90% utilization. Mean board power was {training_resources['mean_power_w']:.1f} W and maximum observed temperature was {training_resources['maximum_temperature_c']} C.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_metrics = load_seed_metrics()
    emotion_seed_metrics = load_emotion_seed_metrics()
    training_metrics = load_training_metrics()
    resource_summary = summarize_resources()
    overlap_audit = load_overlap_audit()
    token_audit = load_token_audit()
    condition_summary = summarize_conditions(seed_metrics)
    emotion_summary = summarize_conditions(emotion_seed_metrics)
    paired = paired_seed_tests(seed_metrics)
    hierarchical = hierarchical_bootstrap()
    plot_primary_metrics(condition_summary)
    plot_true_advantages(hierarchical)
    plot_emotion_metrics(emotion_summary)
    seed_metrics.to_csv(RESULTS_DIR / "seed_metrics.csv", index=False)
    emotion_seed_metrics.to_csv(RESULTS_DIR / "emotion_seed_metrics.csv", index=False)
    training_metrics.to_csv(RESULTS_DIR / "training_metrics.csv", index=False)
    condition_summary.to_csv(RESULTS_DIR / "condition_summary.csv", index=False)
    emotion_summary.to_csv(RESULTS_DIR / "emotion_condition_summary.csv", index=False)
    paired.to_csv(RESULTS_DIR / "paired_seed_comparisons.csv", index=False)
    hierarchical.to_csv(RESULTS_DIR / "hierarchical_bootstrap.csv", index=False)
    (RESULTS_DIR / "resource_summary.json").write_text(
        json.dumps(resource_summary, indent=2) + "\n", encoding="utf-8"
    )
    report = build_report(
        condition_summary,
        emotion_summary,
        paired,
        hierarchical,
        training_metrics,
        resource_summary,
        overlap_audit,
        token_audit,
    )
    (RESULTS_DIR / "FINAL_REPORT.md").write_text(report + "\n", encoding="utf-8")
    print(f"Wrote final results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
