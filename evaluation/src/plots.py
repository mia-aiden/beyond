from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_histogram(series: pd.Series, *, title: str, xlabel: str, output_path: Path) -> None:
    clean_series = series.dropna()
    if clean_series.empty:
        return
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(clean_series, bins=20, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_narrative_distributions(per_sample_df: pd.DataFrame, output_dir: Path) -> None:
    _save_histogram(
        per_sample_df["rouge1_fmeasure"],
        title="ROUGE-1 distribution",
        xlabel="ROUGE-1 F1",
        output_path=output_dir / "rouge_distribution.png",
    )
    _save_histogram(
        per_sample_df["bertscore_f1"],
        title="BERTScore F1 distribution",
        xlabel="BERTScore F1",
        output_path=output_dir / "bertscore_distribution.png",
    )
    _save_histogram(
        per_sample_df["normalized_dtw_distance"],
        title="Normalized DTW distribution",
        xlabel="Normalized DTW distance",
        output_path=output_dir / "dtw_distribution.png",
    )


def plot_length_distribution(eval_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = pd.DataFrame(
        {
            "reference_length": eval_df["reference_narrative"].str.split().map(len),
            "prediction_length": eval_df["predicted_narrative"].str.split().map(len),
        }
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df["reference_length"], color="steelblue", label="reference", bins=20, alpha=0.6)
    sns.histplot(plot_df["prediction_length"], color="darkorange", label="prediction", bins=20, alpha=0.6)
    plt.title("Reference vs prediction length distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_emotion_confusion_matrix(matrix_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Emotion confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
