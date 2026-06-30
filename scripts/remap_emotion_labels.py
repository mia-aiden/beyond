"""
Remap fine-grained emotion labels to the 8-label proposal taxonomy.

The original data mixes GoEmotions-style labels with a few manually added
labels. This script keeps the original columns and adds remapped columns by
default, so the source annotations remain auditable.


Anger: anger, annoyance, frustration
Anxiety/Fear: fear, nervousness, concern, pessimism
Sadness: sadness, disappointment, grief, remorse, embarrassment
Hope/Optimism: optimism, joy, excitement, relief, desire, amusement
Pride/Approval: approval, pride, admiration, gratitude, caring, love
Disgust: disgust, disapproval
Surprise: surprise, confusion, curiosity, realization
Neutral: neutral, skepticism


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROPOSAL_LABELS = (
    "Anger",
    "Anxiety/Fear",
    "Sadness",
    "Hope/Optimism",
    "Pride/Approval",
    "Disgust",
    "Surprise",
    "Neutral",
)

LABEL_MAP = {
    "admiration": "Pride/Approval",
    "amusement": "Hope/Optimism",
    "anger": "Anger",
    "annoyance": "Anger",
    "approval": "Pride/Approval",
    "caring": "Pride/Approval",
    "concern": "Anxiety/Fear",
    "confusion": "Surprise",
    "curiosity": "Surprise",
    "desire": "Hope/Optimism",
    "disappointment": "Sadness",
    "disapproval": "Disgust",
    "disgust": "Disgust",
    "embarrassment": "Sadness",
    "excitement": "Hope/Optimism",
    "fear": "Anxiety/Fear",
    "frustration": "Anger",
    "gratitude": "Pride/Approval",
    "grief": "Sadness",
    "joy": "Hope/Optimism",
    "love": "Pride/Approval",
    "nervousness": "Anxiety/Fear",
    "neutral": "Neutral",
    "optimism": "Hope/Optimism",
    "pessimism": "Anxiety/Fear",
    "pride": "Pride/Approval",
    "realization": "Surprise",
    "relief": "Hope/Optimism",
    "remorse": "Sadness",
    "sadness": "Sadness",
    "skepticism": "Neutral",
    "surprise": "Surprise",
}


def remap_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return LABEL_MAP.get(key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map emotion labels to the 8-label proposal taxonomy."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["Emotion_Label"],
        help="Emotion label columns to remap.",
    )
    parser.add_argument(
        "--suffix",
        default="_8",
        help="Suffix for new remapped columns, e.g. Emotion_Label_8.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace original columns instead of adding suffixed columns.",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not input_path.is_file():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)
    missing_columns = [column for column in args.columns if column not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    for column in args.columns:
        mapped = df[column].map(remap_label)
        unknown = sorted(
            {
                str(value).strip()
                for value, mapped_value in zip(df[column], mapped)
                if pd.notna(value) and str(value).strip() and pd.isna(mapped_value)
            }
        )
        if unknown:
            print(f"ERROR: Unmapped labels in {column}: {unknown}", file=sys.stderr)
            sys.exit(1)

        target_column = column if args.replace else f"{column}{args.suffix}"
        df[target_column] = mapped

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")

    report_columns = [
        column if args.replace else f"{column}{args.suffix}" for column in args.columns
    ]
    for column in report_columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).reindex(PROPOSAL_LABELS).fillna(0).astype(int))


if __name__ == "__main__":
    main()
