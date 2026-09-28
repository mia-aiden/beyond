from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
EVAL_SRC = PROJECT_ROOT / "evaluation" / "src"
for import_path in (CURRENT_DIR, EVAL_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from prepare_sft_data import (  # noqa: E402
    EKMAN_LABELS,
    LABEL_INSTRUCTION,
    LABEL_SYSTEM_PROMPT,
    NARRATIVE_INSTRUCTION,
    NARRATIVE_SYSTEM_PROMPT,
)
from validate_inputs import load_gold_with_sample_ids  # noqa: E402


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def truncate_source(text: str, tokenizer: AutoTokenizer, max_source_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_source_tokens:
        return text
    return tokenizer.decode(token_ids[:max_source_tokens], skip_special_tokens=True)


def build_messages(task: str, source_text: str) -> list[dict[str, str]]:
    if task == "label":
        system_prompt = LABEL_SYSTEM_PROMPT
        instruction = LABEL_INSTRUCTION
    else:
        system_prompt = NARRATIVE_SYSTEM_PROMPT
        instruction = NARRATIVE_INSTRUCTION
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n{source_text}"},
    ]


def build_label_constraint(
    tokenizer: AutoTokenizer,
    prompt_width: int,
) -> Callable[[int, torch.Tensor], list[int]]:
    label_paths = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in EKMAN_LABELS
    ]
    stop_token_ids = {
        token_id
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )
        if token_id is not None and token_id >= 0
    }

    def allowed_tokens(_: int, input_ids: torch.Tensor) -> list[int]:
        generated = input_ids[prompt_width:].tolist()
        candidates: set[int] = set()
        for path in label_paths:
            if generated == path:
                candidates.update(stop_token_ids)
            elif len(generated) < len(path) and path[:len(generated)] == generated:
                candidates.add(path[len(generated)])
        return sorted(candidates or stop_token_ids)

    return allowed_tokens


def canonical_label(raw_generation: str) -> str:
    normalized = raw_generation.strip()
    if normalized in EKMAN_LABELS:
        return normalized
    first_line = normalized.splitlines()[0].strip() if normalized else ""
    return first_line if first_line in EKMAN_LABELS else ""


def load_model(
    base_model_path: Path,
    adapter_path: Path,
) -> tuple[AutoTokenizer, PeftModel]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
        local_files_only=True,
    )
    model.generation_config.max_length = None
    model.eval()
    return tokenizer, model


def generate_predictions(
    *,
    task: str,
    gold_path: Path,
    output_path: Path,
    base_model_path: Path,
    adapter_path: Path,
    batch_size: int,
    max_source_tokens: int,
) -> None:
    gold_df = load_gold_with_sample_ids(gold_path)
    tokenizer, model = load_model(base_model_path, adapter_path)
    rows = [
        {
            "sample_id": int(row.sample_id),
            "source_text": truncate_source(
                str(row.source_text),
                tokenizer,
                max_source_tokens,
            ),
        }
        for row in gold_df.itertuples(index=False)
    ]

    predictions: list[dict[str, Any]] = []
    row_batches = batched(rows, max(1, batch_size))
    for batch_index, batch in enumerate(row_batches, start=1):
        conversations = [
            build_messages(task, item["source_text"])
            for item in batch
        ]
        prompt_texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in conversations
        ]
        inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        prompt_width = int(inputs["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": 8 if task == "label" else 256,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if task == "label":
            generation_kwargs["prefix_allowed_tokens_fn"] = build_label_constraint(
                tokenizer,
                prompt_width,
            )

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        for item, sequence in zip(batch, output_ids, strict=True):
            raw_generation = tokenizer.decode(
                sequence[prompt_width:],
                skip_special_tokens=True,
            ).strip()
            predicted_label = canonical_label(raw_generation) if task == "label" else ""
            parse_success = bool(predicted_label) if task == "label" else bool(raw_generation)
            predictions.append(
                {
                    "sample_id": item["sample_id"],
                    "predicted_narrative": raw_generation if task == "narrative" else "",
                    "predicted_emotion_label": predicted_label,
                    "raw_generation": raw_generation,
                    "parse_success": parse_success,
                    "request_error": "",
                    "finish_reason": "stop",
                    "model_name": adapter_path.name,
                    "run_name": f"{task}_sft_transformers",
                }
            )

        if batch_index % 25 == 0 or batch_index == len(row_batches):
            print(f"Generated {len(predictions)}/{len(rows)} rows", flush=True)

    prediction_df = pd.DataFrame(predictions).sort_values("sample_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    failures = int((~prediction_df["parse_success"]).sum())
    print(f"Wrote {len(prediction_df)} predictions to {output_path}")
    print(f"Generation failures: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a task-specific LoRA adapter directly with Transformers and PEFT."
    )
    parser.add_argument("--task", choices=("label", "narrative"), required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/root/autodl-tmp/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-source-tokens", type=int, default=1300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_predictions(
        task=args.task,
        gold_path=args.gold,
        output_path=args.output,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        batch_size=args.batch_size,
        max_source_tokens=args.max_source_tokens,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
