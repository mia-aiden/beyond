#!/usr/bin/env python3
"""Audit condition-specific chat lengths and cutoff-rate differences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from paths import EXPERIMENTS_DIR, BASE_MODEL


DATA_DIR = EXPERIMENTS_DIR / "data"
EXPERIMENT_DIR = EXPERIMENTS_DIR / "format_semantics_multiseed"
MODEL = Path(BASE_MODEL)
CONDITIONS = ("true", "forced_wrong", "constant_neutral", "marker_control")
CUTOFF = 1536


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def lengths(path: Path, tokenizer: AutoTokenizer) -> np.ndarray:
    values = []
    for row in read_jsonl(path):
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": f'{row["instruction"]}\n{row["input"]}'},
            {"role": "assistant", "content": row["output"]},
        ]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        try:
            token_ids = token_ids["input_ids"]
        except (KeyError, TypeError):
            pass
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        values.append(len(token_ids))
    return np.asarray(values, dtype=np.int64)


def summarize(values: np.ndarray) -> dict:
    return {
        "rows": int(len(values)),
        "minimum": int(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": int(values.max()),
        "over_cutoff": int(np.count_nonzero(values > CUTOFF)),
        "over_cutoff_fraction": float(np.mean(values > CUTOFF)),
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    values: dict[str, dict[str, np.ndarray]] = {}
    payload = {
        "model": str(MODEL),
        "cutoff_len": CUTOFF,
        "user_message_join": "instruction + newline + input",
        "conditions": {},
        "paired_comparisons_to_true": {},
    }
    for condition in CONDITIONS:
        values[condition] = {}
        payload["conditions"][condition] = {}
        for split in ("train", "validation"):
            current = lengths(
                DATA_DIR / f"format_semantics_{condition}_{split}.jsonl",
                tokenizer,
            )
            values[condition][split] = current
            payload["conditions"][condition][split] = summarize(current)

    for condition in CONDITIONS[1:]:
        payload["paired_comparisons_to_true"][condition] = {}
        for split in ("train", "validation"):
            baseline = values["true"][split]
            current = values[condition][split]
            delta = current - baseline
            payload["paired_comparisons_to_true"][condition][split] = {
                "minimum_token_delta": int(delta.min()),
                "median_token_delta": float(np.median(delta)),
                "maximum_token_delta": int(delta.max()),
                "different_cutoff_status_rows": int(
                    np.count_nonzero((baseline > CUTOFF) != (current > CUTOFF))
                ),
            }

    output_path = EXPERIMENT_DIR / "token_length_audit.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
