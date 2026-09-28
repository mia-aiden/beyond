from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from paths import BASE_MODEL

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
EVAL_SRC = PROJECT_ROOT / "evaluation" / "src"
for import_path in (CURRENT_DIR, EVAL_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from prepare_sft_data import EKMAN_LABELS, NARRATIVE_INSTRUCTION, NARRATIVE_SYSTEM_PROMPT  # noqa: E402
from validate_inputs import load_gold_with_sample_ids  # noqa: E402

SEQ_SYSTEM = (
    "You are an expert political analyst. First classify the author's dominant "
    "Ekman emotion as exactly one of: Anger, Disgust, Fear, Joy, Neutral, Sadness, "
    "Surprise. Then extract the political narrative in neutral third-person prose, "
    "1-3 sentences, using only the source text and preserving the author's "
    "perspective. Output exactly two lines:\nEmotion: <label>\nNarrative: <text>"
)
SEQ_INSTRUCTION = "Classify the author's Ekman emotion, then extract the political narrative."


def batched(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


def truncate_source(text, tokenizer, max_source_tokens):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_source_tokens:
        return text
    return tokenizer.decode(ids[:max_source_tokens], skip_special_tokens=True)


def build_messages(mode, source_text, gold_emotion):
    if mode == "bgold":
        instruction = f"The author's dominant emotion is {gold_emotion}.\n" + NARRATIVE_INSTRUCTION
        return [
            {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n{source_text}"},
        ]
    return [
        {"role": "system", "content": SEQ_SYSTEM},
        {"role": "user", "content": f"{SEQ_INSTRUCTION}\n{source_text}"},
    ]


def parse_seqjoint(text):
    emotion = ""
    match = re.search(r"Emotion:\s*(.+)", text)
    if match:
        candidate = match.group(1).splitlines()[0].strip()
        for label in EKMAN_LABELS:
            if candidate.lower().startswith(label.lower()):
                emotion = label
                break
    idx = text.find("Narrative:")
    if idx >= 0:
        narrative = text[idx + len("Narrative:"):].strip()
    else:
        narrative = re.sub(r"^\s*Emotion:.*(?:\n|$)", "", text).strip()
    return emotion, narrative


def load_model(base_model_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, local_files_only=True, use_fast=True, clean_up_tokenization_spaces=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False, local_files_only=True)
    model.generation_config.max_length = None
    model.eval()
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bgold", "seqjoint"), required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path(BASE_MODEL))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-tokens", type=int, default=1300)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    max_new = args.max_new_tokens or (256 if args.mode == "bgold" else 320)
    gold_df = load_gold_with_sample_ids(args.gold)
    tokenizer, model = load_model(args.base_model, args.adapter)
    rows = [
        {
            "sample_id": int(r.sample_id),
            "source_text": truncate_source(str(r.source_text), tokenizer, args.max_source_tokens),
            "gold_emotion": str(r.emotion_label),
        }
        for r in gold_df.itertuples(index=False)
    ]

    predictions = []
    row_batches = batched(rows, max(1, args.batch_size))
    for bi, batch in enumerate(row_batches, start=1):
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(args.mode, it["source_text"], it["gold_emotion"]),
                tokenize=False, add_generation_prompt=True,
            )
            for it in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        for it, seq in zip(batch, out, strict=True):
            raw = tokenizer.decode(seq[width:], skip_special_tokens=True).strip()
            if args.mode == "seqjoint":
                emotion, narrative = parse_seqjoint(raw)
            else:
                emotion, narrative = "", raw
            predictions.append({
                "sample_id": it["sample_id"],
                "predicted_narrative": narrative,
                "predicted_emotion_label": emotion,
                "raw_generation": raw,
                "parse_success": bool(narrative),
                "request_error": "",
                "finish_reason": "stop",
                "model_name": args.adapter.name,
                "run_name": f"{args.mode}_transformers",
            })
        if bi % 25 == 0 or bi == len(row_batches):
            print(f"Generated {len(predictions)}/{len(rows)} rows", flush=True)

    df = pd.DataFrame(predictions).sort_values("sample_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} predictions to {args.output}")
    print(f"Empty narratives: {int((~df['parse_success']).sum())}")
    if args.mode == "seqjoint":
        print(f"Emotion parsed: {int((df['predicted_emotion_label'] != '').sum())}/{len(df)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
