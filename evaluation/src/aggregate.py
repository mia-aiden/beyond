from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Template

from io_utils import write_json


DEFAULT_REPORT_TEMPLATE = """# Evaluation Report

- Run timestamp: {{ summary.run_timestamp }}
- Gold file: `{{ summary.gold_path }}`
- Prediction file: `{{ summary.pred_path }}`
- Inference mode: {{ summary.inference_mode }}
{% if summary.inference_mode %}
- Inference backend: {{ summary.inference_backend }}
- Model path: `{{ summary.model_path }}`
- Generation failures: {{ summary.num_generation_failures }}
{% endif %}
- Gold rows: {{ summary.num_gold_rows }}
- Prediction rows: {{ summary.num_prediction_rows }}
- Aligned rows: {{ summary.num_aligned_rows }}
- Missing prediction rows: {{ summary.num_missing_prediction_rows }}

## Narrative Metrics

- BLEU (corpus): {{ summary.bleu_corpus }}
- ROUGE-1 (mean): {{ summary.rouge1_mean }}
- ROUGE-2 (mean): {{ summary.rouge2_mean }}
- ROUGE-L (mean): {{ summary.rougeL_mean }}
- BERTScore F1 (mean): {{ summary.bertscore_f1_mean }}
- Normalized DTW (mean): {{ summary.normalized_dtw_mean }}

## Emotion Metrics

- Accuracy: {{ summary.emotion_accuracy }}
- Macro-F1: {{ summary.emotion_macro_f1 }}

{% if summary.judge_enabled %}
## LLM Judge

- Relevance mean: {{ summary.judge_relevance_mean }}
- Faithfulness mean: {{ summary.judge_faithfulness_mean }}
- Coherence mean: {{ summary.judge_coherence_mean }}
- NLI distribution: {{ summary.nli_distribution }}
{% endif %}

{% if summary.manual_enabled %}
## Manual Evaluation

- Relevance mean: {{ summary.manual_relevance_mean }}
- Grammar mean: {{ summary.manual_grammar_mean }}
{% endif %}

{% if summary.argument_enabled %}
## Argument Mining

- Claim rate: {{ summary.argument_claim_rate }}
- Reason rate: {{ summary.argument_reason_rate }}
- Stance marker rate: {{ summary.argument_stance_rate }}
{% endif %}
"""


def merge_per_sample_metrics(base_df: pd.DataFrame, metric_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged = base_df.copy()
    for metric_df in metric_dfs:
        if metric_df is None or metric_df.empty:
            continue
        columns = [column for column in metric_df.columns if column != "sample_id"]
        merged = merged.merge(metric_df[["sample_id", *columns]], on="sample_id", how="left")
    return merged


def build_summary(
    *,
    run_timestamp: str,
    gold_path: Path,
    pred_path: Path,
    inference_mode: bool,
    inference_backend: str,
    model_path: str,
    num_generation_failures: int,
    num_gold_rows: int,
    num_prediction_rows: int,
    num_aligned_rows: int,
    num_missing_prediction_rows: int,
    automatic_summary_df: pd.DataFrame,
    emotion_summary_df: pd.DataFrame,
    judge_summary_df: pd.DataFrame,
    manual_summary_df: pd.DataFrame,
    argument_summary_df: pd.DataFrame,
    judge_enabled: bool,
    manual_enabled: bool,
    argument_enabled: bool,
) -> dict:
    summary = {
        "run_timestamp": run_timestamp,
        "gold_path": str(gold_path),
        "pred_path": str(pred_path),
        "inference_mode": inference_mode,
        "inference_backend": inference_backend,
        "model_path": model_path,
        "num_generation_failures": num_generation_failures,
        "num_gold_rows": num_gold_rows,
        "num_prediction_rows": num_prediction_rows,
        "num_aligned_rows": num_aligned_rows,
        "num_missing_prediction_rows": num_missing_prediction_rows,
        "judge_enabled": judge_enabled,
        "manual_enabled": manual_enabled,
        "argument_enabled": argument_enabled,
    }
    for frame in [automatic_summary_df, emotion_summary_df, judge_summary_df, manual_summary_df, argument_summary_df]:
        if frame is None or frame.empty:
            continue
        for row in frame.itertuples(index=False):
            summary[row.metric] = row.value

    if "accuracy" in summary and "emotion_accuracy" not in summary:
        summary["emotion_accuracy"] = summary["accuracy"]
    if "macro_f1" in summary and "emotion_macro_f1" not in summary:
        summary["emotion_macro_f1"] = summary["macro_f1"]
    if "normalized_dtw_distance_mean" in summary and "normalized_dtw_mean" not in summary:
        summary["normalized_dtw_mean"] = summary["normalized_dtw_distance_mean"]
    return summary


def write_summary_json(summary: dict, path: Path) -> None:
    write_json(path, summary)


def write_report(summary: dict, template_path: Path, output_path: Path) -> None:
    if template_path.exists():
        with template_path.open("r", encoding="utf-8") as infile:
            template = Template(infile.read())
    else:
        template = Template(DEFAULT_REPORT_TEMPLATE)
    output_path.write_text(template.render(summary=summary), encoding="utf-8")
