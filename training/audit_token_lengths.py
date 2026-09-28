from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def token_lengths(path: Path, tokenizer: AutoTokenizer) -> list[int]:
    lengths = []
    for row in read_jsonl(path):
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": f'{row["instruction"]}\n\n{row["input"]}'},
            {"role": "assistant", "content": row["output"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        try:
            token_ids = encoded["input_ids"]
        except (KeyError, TypeError):
            token_ids = encoded
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        lengths.append(len(token_ids))
    return lengths


def summarize(lengths: list[int]) -> dict[str, float | int]:
    values = np.asarray(lengths)
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_1024": int((values > 1024).sum()),
        "over_1536": int((values > 1536).sum()),
        "over_2048": int((values > 2048).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chat-formatted SFT token lengths.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/autodl-tmp/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/autodl-fs/sft_experiments/data"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    summary = {}
    for filename in ("label_train.jsonl", "narrative_train.jsonl"):
        summary[filename] = summarize(token_lengths(args.data_dir / filename, tokenizer))

    output_path = args.data_dir / "token_length_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
