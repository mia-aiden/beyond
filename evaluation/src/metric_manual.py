from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_manual_eval_sheet(
    eval_df: pd.DataFrame,
    *,
    output_path: Path,
    sample_size: int,
    stratify_by_emotion: bool,
) -> pd.DataFrame:
    if stratify_by_emotion and "emotion_label" in eval_df.columns:
        groups = []
        labels = sorted(eval_df["emotion_label"].dropna().astype(str).unique().tolist())
        per_group = max(1, sample_size // max(1, len(labels)))
        for label in labels:
            group = eval_df[eval_df["emotion_label"].astype(str) == label].head(per_group)
            groups.append(group)
        sample_df = pd.concat(groups, ignore_index=True).head(sample_size).copy()
    else:
        sample_df = eval_df.head(sample_size).copy()
    sheet_df = sample_df[
        ["sample_id", "source_text", "reference_narrative", "predicted_narrative"]
    ].copy()
    sheet_df["manual_relevance"] = ""
    sheet_df["manual_grammar"] = ""
    sheet_df["notes"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return sheet_df


def aggregate_manual_eval(sheet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(sheet_path, encoding="utf-8-sig")
    scored_df = df.copy()
    scored_df["manual_relevance"] = pd.to_numeric(scored_df["manual_relevance"], errors="coerce")
    scored_df["manual_grammar"] = pd.to_numeric(scored_df["manual_grammar"], errors="coerce")

    summary_df = pd.DataFrame(
        [
            {"metric": "manual_relevance_mean", "value": float(scored_df["manual_relevance"].mean())},
            {"metric": "manual_grammar_mean", "value": float(scored_df["manual_grammar"].mean())},
        ]
    )
    return scored_df, summary_df
