from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from paths import EXPERIMENTS_DIR, TRAIN_POOL


EKMAN_LABELS = (
    "Anger",
    "Disgust",
    "Fear",
    "Joy",
    "Neutral",
    "Sadness",
    "Surprise",
)

LABEL_SYSTEM_PROMPT = """You are an expert emotion annotation system.
Classify the author's dominant emotional stance in a Reddit-style political comment.
Label the author's own stance, not merely the topic or an emotion expressed by quoted text.
If several emotions are present, choose the dominant one.
Return exactly one label from this list and no other text:
Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise"""

LABEL_INSTRUCTION = "Determine the author's dominant Ekman emotion label."

NARRATIVE_SYSTEM_PROMPT = """You are an expert political narrative analyst.
Use only the provided source text.
Identify the political actors, issue, causal interpretation, responsibility attribution,
and political claim internally, but do not output your reasoning.
Return only a concise political narrative in neutral third-person prose.
The narrative must be 1-3 sentences, preserve the author's perspective, and must not add
external facts or correct the author."""

NARRATIVE_INSTRUCTION = "Extract the political narrative from the source text."


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_alpaca_rows(
    frame: pd.DataFrame,
    *,
    system_prompt: str,
    instruction: str,
    output_column: str,
) -> list[dict[str, str]]:
    return [
        {
            "system": system_prompt,
            "instruction": instruction,
            "input": row.text,
            "output": getattr(row, output_column),
        }
        for row in frame.itertuples(index=False)
    ]


def build_dataset_info() -> dict[str, dict[str, object]]:
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
    }
    return {
        "political_label_train": {
            "file_name": "label_train.jsonl",
            "formatting": "alpaca",
            "columns": columns,
        },
        "political_label_validation": {
            "file_name": "label_validation.jsonl",
            "formatting": "alpaca",
            "columns": columns,
        },
        "political_narrative_train": {
            "file_name": "narrative_train.jsonl",
            "formatting": "alpaca",
            "columns": columns,
        },
        "political_narrative_validation": {
            "file_name": "narrative_validation.jsonl",
            "formatting": "alpaca",
            "columns": columns,
        },
    }


def stratified_split_indices(
    labels: pd.Series,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    validation_indices: set[int] = set()
    grouped_indices: dict[str, list[int]] = {}
    for index, label in labels.items():
        grouped_indices.setdefault(label, []).append(int(index))

    for indices in grouped_indices.values():
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_indices.update(indices[:validation_count])

    train_indices = [int(index) for index in labels.index if index not in validation_indices]
    return train_indices, sorted(validation_indices)


def prepare_data(input_path: Path, output_dir: Path, validation_fraction: float, seed: int) -> None:
    frame = pd.read_csv(input_path)
    required = {"text", "llm_stage1", "Narrative_Summary"}
    missing = sorted(required - set(frame.columns))
    if missing:
        # The uploaded training file uses Emotion_Label for the stage-1 label.
        if "llm_stage1" in missing and "Emotion_Label" in frame.columns:
            frame = frame.rename(columns={"Emotion_Label": "llm_stage1"})
            missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")

    source_rows = len(frame)
    frame = frame.copy()
    frame["text"] = frame["text"].map(normalize_text)
    frame["llm_stage1"] = frame["llm_stage1"].map(normalize_text)
    frame["Narrative_Summary"] = frame["Narrative_Summary"].map(normalize_text)
    frame = frame[(frame["text"] != "") & (frame["llm_stage1"] != "")].copy()

    unknown_labels = sorted(set(frame["llm_stage1"]) - set(EKMAN_LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected Ekman labels: {unknown_labels}")

    before_deduplication = len(frame)
    frame["_normalized_text"] = frame["text"].str.casefold()
    frame = frame.drop_duplicates("_normalized_text", keep="first").drop(columns="_normalized_text")
    frame = frame.reset_index(drop=True)

    train_indices, validation_indices = stratified_split_indices(
        frame["llm_stage1"],
        validation_fraction=validation_fraction,
        seed=seed,
    )
    label_train = frame.loc[train_indices].sort_index().reset_index(drop=True)
    label_validation = frame.loc[validation_indices].sort_index().reset_index(drop=True)

    narrative_train = label_train[label_train["Narrative_Summary"] != ""].reset_index(drop=True)
    narrative_validation = label_validation[
        label_validation["Narrative_Summary"] != ""
    ].reset_index(drop=True)

    write_jsonl(
        to_alpaca_rows(
            label_train,
            system_prompt=LABEL_SYSTEM_PROMPT,
            instruction=LABEL_INSTRUCTION,
            output_column="llm_stage1",
        ),
        output_dir / "label_train.jsonl",
    )
    write_jsonl(
        to_alpaca_rows(
            label_validation,
            system_prompt=LABEL_SYSTEM_PROMPT,
            instruction=LABEL_INSTRUCTION,
            output_column="llm_stage1",
        ),
        output_dir / "label_validation.jsonl",
    )
    write_jsonl(
        to_alpaca_rows(
            narrative_train,
            system_prompt=NARRATIVE_SYSTEM_PROMPT,
            instruction=NARRATIVE_INSTRUCTION,
            output_column="Narrative_Summary",
        ),
        output_dir / "narrative_train.jsonl",
    )
    write_jsonl(
        to_alpaca_rows(
            narrative_validation,
            system_prompt=NARRATIVE_SYSTEM_PROMPT,
            instruction=NARRATIVE_INSTRUCTION,
            output_column="Narrative_Summary",
        ),
        output_dir / "narrative_validation.jsonl",
    )

    dataset_info_path = output_dir / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(build_dataset_info(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    split_summary = {
        "input_path": str(input_path),
        "source_rows": source_rows,
        "rows_after_required_field_filter": before_deduplication,
        "duplicate_text_rows_removed": before_deduplication - len(frame),
        "label_train_rows": len(label_train),
        "label_validation_rows": len(label_validation),
        "narrative_train_rows": len(narrative_train),
        "narrative_validation_rows": len(narrative_validation),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "label_train_distribution": label_train["llm_stage1"].value_counts().to_dict(),
        "label_validation_distribution": label_validation["llm_stage1"].value_counts().to_dict(),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare two Alpaca SFT datasets.")
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        default=TRAIN_POOL,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "data",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_data(**vars(parse_args()))
