from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def _display_label(label: str) -> str:
    return label if label else "empty_prediction"


def compute_emotion_metrics(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gold = eval_df["emotion_label"].astype(str).tolist()
    pred = eval_df["predicted_emotion_label"].astype(str).tolist()

    accuracy = accuracy_score(gold, pred)
    macro_f1 = f1_score(gold, pred, average="macro")
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    labels = sorted(set(gold) | set(pred))
    matrix = confusion_matrix(gold, pred, labels=labels)

    summary_rows = [
        {"metric": "accuracy", "value": float(accuracy)},
        {"metric": "macro_f1", "value": float(macro_f1)},
    ]
    for label in labels:
        label_report = report.get(label, {})
        label_name = _display_label(label)
        summary_rows.append(
            {
                "metric": f"{label_name}_precision",
                "value": float(label_report.get("precision", 0.0)),
            }
        )
        summary_rows.append(
            {
                "metric": f"{label_name}_recall",
                "value": float(label_report.get("recall", 0.0)),
            }
        )
        summary_rows.append(
            {
                "metric": f"{label_name}_f1",
                "value": float(label_report.get("f1-score", 0.0)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    display_labels = [_display_label(label) for label in labels]
    matrix_df = pd.DataFrame(matrix, index=display_labels, columns=display_labels)
    per_sample_df = eval_df[["sample_id", "emotion_label", "predicted_emotion_label"]].copy()
    per_sample_df["emotion_correct"] = (
        per_sample_df["emotion_label"] == per_sample_df["predicted_emotion_label"]
    )
    return summary_df, matrix_df, per_sample_df
