from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer

from paths import BASE_MODEL


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


JOINT_SYSTEM_PROMPT = (
    "You are an expert political analyst. First classify the author's dominant "
    "Ekman emotion as exactly one of: Anger, Disgust, Fear, Joy, Neutral, Sadness, "
    "Surprise. Then extract the political narrative in neutral third-person prose, "
    "1-3 sentences, using only the source text and preserving the author's "
    "perspective. Output exactly two lines:\nEmotion: <label>\nNarrative: <text>"
)
JOINT_INSTRUCTION = "Classify the author's Ekman emotion, then extract the political narrative."
JOINT_OUTPUT_REGEX = (
    r"Emotion: (Anger|Disgust|Fear|Joy|Neutral|Sadness|Surprise)\nNarrative: [^\n]+"
)
MARKER_SYSTEM_PROMPT = (
    "You are an expert political analyst. First output the fixed non-semantic marker X. "
    "Then extract the political narrative in neutral third-person prose, 1-3 sentences, "
    "using only the source text and preserving the author's perspective. The marker is "
    "fixed and carries no "
    "semantic information. It is not inferred from the source and exists only to preserve "
    "a two-field response structure. Output exactly two lines:\nMarker: X\nNarrative: <text>"
)
MARKER_INSTRUCTION = "Output the fixed marker X, then extract the political narrative."
MARKER_OUTPUT_REGEX = r"Marker: X\nNarrative: [^\n]+"


def truncate_source(text: str, tokenizer: AutoTokenizer, max_source_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_source_tokens:
        return text
    return tokenizer.decode(token_ids[:max_source_tokens], skip_special_tokens=True)


def build_messages(task: str, source_text: str) -> list[dict[str, str]]:
    if task == "label":
        system_prompt = LABEL_SYSTEM_PROMPT
        instruction = LABEL_INSTRUCTION
    elif task in {"joint", "seqjoint_gold"}:
        system_prompt = JOINT_SYSTEM_PROMPT
        instruction = JOINT_INSTRUCTION
    elif task == "marker_joint":
        system_prompt = MARKER_SYSTEM_PROMPT
        instruction = MARKER_INSTRUCTION
    else:
        system_prompt = NARRATIVE_SYSTEM_PROMPT
        instruction = NARRATIVE_INSTRUCTION
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n{source_text}"},
    ]


def parse_joint_generation(text: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"Emotion: (Anger|Disgust|Fear|Joy|Neutral|Sadness|Surprise)\nNarrative: (.+)",
        text.strip(),
    )
    if not match:
        return "", ""
    return match.group(1), match.group(2).strip()


def parse_marker_generation(text: str) -> str:
    match = re.fullmatch(r"Marker: X\nNarrative: (.+)", text.strip())
    return match.group(1).strip() if match else ""


def generate_one(
    *,
    client: OpenAI,
    model_name: str,
    task: str,
    sample_id: int,
    source_text: str,
    gold_emotion: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> dict[str, Any]:
    extra_body: dict[str, Any] = {}
    if task == "label":
        extra_body["structured_outputs"] = {"choice": list(EKMAN_LABELS)}
    elif task == "joint":
        extra_body["structured_outputs"] = {"regex": JOINT_OUTPUT_REGEX}
    elif task == "marker_joint":
        extra_body["structured_outputs"] = {"regex": MARKER_OUTPUT_REGEX}

    try:
        if task == "seqjoint_gold":
            prompt = tokenizer.apply_chat_template(
                build_messages(task, source_text),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt += f"Emotion: {gold_emotion}\nNarrative: "
            completion = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            choice = completion.choices[0]
            raw_generation = choice.text.strip()
            predicted_emotion = gold_emotion
            predicted_narrative = raw_generation
            valid = bool(predicted_narrative)
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=build_messages(task, source_text),
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                extra_body=extra_body,
            )
            choice = completion.choices[0]
            raw_generation = (choice.message.content or "").strip()
        if task == "joint":
            predicted_emotion, predicted_narrative = parse_joint_generation(raw_generation)
            valid = bool(predicted_emotion and predicted_narrative)
        elif task == "marker_joint":
            predicted_emotion = ""
            predicted_narrative = parse_marker_generation(raw_generation)
            valid = bool(predicted_narrative)
        elif task == "label":
            valid = raw_generation in EKMAN_LABELS
            predicted_emotion = raw_generation if valid else ""
            predicted_narrative = ""
        elif task == "narrative":
            valid = bool(raw_generation)
            predicted_emotion = ""
            predicted_narrative = raw_generation
        return {
            "sample_id": sample_id,
            "predicted_narrative": predicted_narrative,
            "predicted_emotion_label": predicted_emotion,
            "raw_generation": raw_generation,
            "parse_success": valid,
            "request_error": "",
            "finish_reason": choice.finish_reason or "",
            "model_name": model_name,
            "run_name": f"{task}_sft_vllm",
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "predicted_narrative": "",
            "predicted_emotion_label": "",
            "raw_generation": "",
            "parse_success": False,
            "request_error": f"{type(exc).__name__}: {exc}",
            "finish_reason": "",
            "model_name": model_name,
            "run_name": f"{task}_sft_vllm",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task-specific SFT inference through vLLM.")
    parser.add_argument(
        "--task",
        choices=("label", "narrative", "joint", "marker_joint", "seqjoint_gold"),
        required=True,
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(BASE_MODEL),
    )
    parser.add_argument("--max-source-tokens", type=int, default=1300)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=180)
    args = parser.parse_args()

    gold_df = load_gold_with_sample_ids(args.gold)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    samples = {
        int(row.sample_id): {
            "source_text": truncate_source(
                row.source_text,
                tokenizer,
                args.max_source_tokens,
            ),
            "gold_emotion": str(row.emotion_label),
        }
        for row in gold_df.itertuples(index=False)
    }
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_retries=2,
    )
    available_models = {model.id for model in client.models.list().data}
    if args.model_name not in available_models:
        raise SystemExit(
            f"Model {args.model_name!r} is not served. Available models: {sorted(available_models)}"
        )

    max_tokens = 8 if args.task == "label" else (320 if args.task in {"joint", "marker_joint"} else 256)
    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures: dict[Future, int] = {}
        for sample_id, sample in samples.items():
            future = executor.submit(
                generate_one,
                client=client,
                model_name=args.model_name,
                task=args.task,
                sample_id=sample_id,
                source_text=sample["source_text"],
                gold_emotion=sample["gold_emotion"],
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            futures[future] = sample_id
        for future in as_completed(futures):
            completed[futures[future]] = future.result()

    prediction_df = pd.DataFrame(completed[sample_id] for sample_id in sorted(completed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    failures = int((~prediction_df["parse_success"]).sum())
    print(f"Wrote {len(prediction_df)} predictions to {args.output}")
    print(f"Generation failures: {failures}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
