from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from io_utils import normalize_text, read_csv


GOLD_REQUIRED_COLUMNS = ["source_text", "emotion_label", "reference_narrative"]
PRED_REQUIRED_COLUMNS = ["sample_id", "predicted_narrative", "predicted_emotion_label"]


class ValidationError(Exception):
    pass


@dataclass
class AlignmentResult:
    gold_df: pd.DataFrame
    pred_df: pd.DataFrame
    aligned_df: pd.DataFrame
    missing_rows_df: pd.DataFrame
    extra_prediction_rows_df: pd.DataFrame


def _require_columns(df: pd.DataFrame, required: list[str], csv_path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns in {csv_path}: {missing}")


def load_gold_with_sample_ids(csv_path: Path) -> pd.DataFrame:
    df = read_csv(csv_path)
    _require_columns(df, GOLD_REQUIRED_COLUMNS, csv_path)

    df = df.copy()
    df["sample_id"] = range(len(df))
    for column in GOLD_REQUIRED_COLUMNS:
        df[column] = df[column].map(normalize_text)
    return df


def load_prediction(csv_path: Path) -> pd.DataFrame:
    df = read_csv(csv_path)
    _require_columns(df, PRED_REQUIRED_COLUMNS, csv_path)

    df = df.copy()
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="raise").astype(int)
    if df["sample_id"].duplicated().any():
        duplicated = df[df["sample_id"].duplicated(keep=False)]["sample_id"].tolist()
        raise ValidationError(f"Duplicate sample_id values found in {csv_path}: {duplicated[:10]}")

    df["predicted_narrative"] = df["predicted_narrative"].map(normalize_text)
    df["predicted_emotion_label"] = df["predicted_emotion_label"].map(normalize_text)
    return df


def align_gold_and_prediction(gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> AlignmentResult:
    missing_rows_df = gold_df.loc[~gold_df["sample_id"].isin(pred_df["sample_id"])].copy()
    if not missing_rows_df.empty:
        missing_rows_df["status"] = "missing_prediction"

    extra_prediction_rows_df = pred_df.loc[~pred_df["sample_id"].isin(gold_df["sample_id"])].copy()
    if not extra_prediction_rows_df.empty:
        extra_prediction_rows_df["status"] = "extra_prediction"

    aligned_df = gold_df.merge(pred_df, on="sample_id", how="inner", validate="one_to_one")
    return AlignmentResult(
        gold_df=gold_df,
        pred_df=pred_df,
        aligned_df=aligned_df,
        missing_rows_df=missing_rows_df,
        extra_prediction_rows_df=extra_prediction_rows_df,
    )
