from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/root/autodl-fs/sft_experiments/data")


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"system", "instruction", "input", "output"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(row)
    return rows


def normalized_source(text: str) -> str:
    return " ".join(text.split()).casefold()


def index_unique(rows: list[dict[str, str]], source_path: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = normalized_source(row["input"])
        if key in indexed:
            raise ValueError(f"Duplicate source text in {source_path}")
        indexed[key] = row
    return indexed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_split(data_root: Path, split: str, output_path: Path) -> dict[str, Any]:
    label_path = data_root / f"label_{split}.jsonl"
    narrative_path = data_root / f"narrative_{split}.jsonl"
    label_rows = read_jsonl(label_path)
    narrative_rows = read_jsonl(narrative_path)
    narrative_index = index_unique(narrative_rows, narrative_path)

    paired_rows = []
    missing_narratives = []
    for label_row in label_rows:
        key = normalized_source(label_row["input"])
        narrative_row = narrative_index.get(key)
        if narrative_row is None:
            missing_narratives.append(label_row["input"])
            continue
        paired_rows.append(
            {
                "source_text": label_row["input"],
                "emotion_label": label_row["output"],
                "reference_narrative": narrative_row["output"],
                "label_system": label_row["system"],
                "label_instruction": label_row["instruction"],
                "narrative_system": narrative_row["system"],
                "narrative_instruction": narrative_row["instruction"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for row in paired_rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "split": split,
        "label_rows": len(label_rows),
        "narrative_rows": len(narrative_rows),
        "paired_rows": len(paired_rows),
        "label_rows_without_narrative": len(missing_narratives),
        "emotion_distribution": dict(Counter(row["emotion_label"] for row in paired_rows)),
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired Model C train/validation data.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    summaries = [
        pair_split(args.data_root, split, args.data_root / f"joint_{split}.jsonl")
        for split in ("train", "validation")
    ]
    summary = {
        "method": "inner join of the existing Model A and Model B splits by normalized source text",
        "test_data_used": False,
        "splits": summaries,
    }
    summary_path = args.data_root / "joint_split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
