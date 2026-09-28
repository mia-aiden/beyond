#!/usr/bin/env python3
"""Verify exact source and narrative separation across train, validation, and tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from paths import EXPERIMENTS_DIR, STAGE1_TEST, MANUAL_TEST


DATA_DIR = EXPERIMENTS_DIR / "data"
EXPERIMENT_DIR = EXPERIMENTS_DIR / "format_semantics_multiseed"
TESTS = {
    "stage1": STAGE1_TEST,
    "manual": MANUAL_TEST,
}


def normalize(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def hash_values(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def summarize(rows: list[dict]) -> dict:
    sources = [normalize(row["source_text"]) for row in rows]
    narratives = [normalize(row["reference_narrative"]) for row in rows]
    return {
        "rows": len(rows),
        "unique_sources": len(set(sources)),
        "unique_narratives": len(set(narratives)),
        "source_sha256": hash_values(sources),
        "narrative_sha256": hash_values(narratives),
        "source_values": set(sources),
        "narrative_values": set(narratives),
    }


def public(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not key.endswith("_values")}


def main() -> None:
    splits = {
        "train": summarize(read_jsonl(DATA_DIR / "joint_train.jsonl")),
        "validation": summarize(read_jsonl(DATA_DIR / "joint_validation.jsonl")),
    }
    for name, path in TESTS.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        splits[name] = summarize(frame.to_dict("records"))

    comparisons = {}
    pairs = (
        ("train", "validation"),
        ("train", "stage1"),
        ("validation", "stage1"),
        ("train", "manual"),
        ("validation", "manual"),
        ("stage1", "manual"),
    )
    for left, right in pairs:
        comparisons[f"{left}_vs_{right}"] = {
            "source_overlap": len(splits[left]["source_values"] & splits[right]["source_values"]),
            "narrative_overlap": len(
                splits[left]["narrative_values"] & splits[right]["narrative_values"]
            ),
        }

    stage1 = pd.read_csv(TESTS["stage1"], encoding="utf-8-sig").reset_index(
        names="stage1_sample_id"
    )
    manual = pd.read_csv(TESTS["manual"], encoding="utf-8-sig").reset_index(
        names="manual_sample_id"
    )
    stage1["normalized_source"] = stage1["source_text"].map(normalize)
    manual["normalized_source"] = manual["source_text"].map(normalize)
    shared_tests = stage1.merge(
        manual,
        on="normalized_source",
        suffixes=("_stage1", "_manual"),
        validate="one_to_one",
    )

    leakage_comparisons = (
        "train_vs_stage1",
        "validation_vs_stage1",
        "train_vs_manual",
        "validation_vs_manual",
    )
    training_leakage_passed = all(
        comparisons[name]["source_overlap"] == 0
        and comparisons[name]["narrative_overlap"] == 0
        for name in leakage_comparisons
    ) and comparisons["train_vs_validation"]["source_overlap"] == 0

    payload = {
        "normalization": "whitespace collapse followed by Unicode casefold",
        "splits": {name: public(summary) for name, summary in splits.items()},
        "comparisons": comparisons,
        "cross_test_overlap": {
            "shared_sources": len(shared_tests),
            "fraction_of_manual_test": len(shared_tests) / len(manual),
            "emotion_label_agreements": int(
                (
                    shared_tests["emotion_label_stage1"]
                    == shared_tests["emotion_label_manual"]
                ).sum()
            ),
            "exact_narrative_agreements": int(
                (
                    shared_tests["reference_narrative_stage1"].map(normalize)
                    == shared_tests["reference_narrative_manual"].map(normalize)
                ).sum()
            ),
            "sample_id_pairs": shared_tests[
                ["stage1_sample_id", "manual_sample_id"]
            ].to_dict("records"),
        },
        "training_leakage_passed": training_leakage_passed,
        "warnings": [
            "Stage1 and manual tests partially overlap and are not fully independent.",
            "One reference narrative is duplicated across train and validation for different sources.",
        ],
    }
    output_path = EXPERIMENT_DIR / "test_overlap_audit.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["training_leakage_passed"]:
        raise SystemExit(f"Training-to-test leakage detected; inspect {output_path}")
    print(f"No exact training-to-test leakage. Wrote {output_path}")


if __name__ == "__main__":
    main()
