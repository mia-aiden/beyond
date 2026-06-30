from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from io_utils import ensure_dir, load_yaml, render_template, timestamp_string
from parse_generation import parse_generation
from validate_inputs import ValidationError, load_gold_with_sample_ids


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = PROJECT_ROOT / "evaluation"


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(torch_dtype: str) -> torch.dtype | None:
    if torch_dtype == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(torch_dtype)


def load_local_model(config: dict[str, Any]) -> tuple[Any, Any, str]:
    device = _resolve_device(str(config.get("device", "auto")))
    dtype = _resolve_dtype(str(config.get("torch_dtype", "auto")))
    model_path = str(config["model_path"]).strip()
    if not model_path:
        raise ValueError("inference.yaml must set model_path before --inference can be used.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        torch_dtype=dtype,
    )
    if tokenizer.pad_token is None:
        if str(config.get("pad_token_fallback", "eos")) == "eos" and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) is not None:
        tokenizer.padding_side = "left"

    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_predictions(
    gold_df: pd.DataFrame,
    *,
    config: dict[str, Any],
    system_prompt_path: Path,
    user_prompt_path: Path,
) -> pd.DataFrame:
    tokenizer, model, device = load_local_model(config)
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []

    batch_size = max(1, int(config.get("batch_size", 1)))
    max_new_tokens = int(config.get("max_new_tokens", 256))
    temperature = float(config.get("temperature", 0.0))
    top_p = float(config.get("top_p", 1.0))
    do_sample = bool(config.get("do_sample", False))
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    prompt_rows = []
    for row in gold_df.itertuples(index=False):
        user_prompt = render_template(user_prompt_path, source_text=row.source_text)
        prompt_rows.append(
            {
                "sample_id": row.sample_id,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
            }
        )

    for batch in _batched(prompt_rows, batch_size):
        prompts = [item["prompt"] for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for batch_index, item in enumerate(batch):
            new_tokens = outputs[batch_index][int(prompt_lengths[batch_index]):]
            raw_generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = parse_generation(raw_generation)

            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "predicted_narrative": parsed["predicted_narrative"],
                    "predicted_emotion_label": parsed["predicted_emotion_label"],
                    "raw_generation": parsed["raw_generation"],
                    "parse_success": parsed["parse_success"],
                    "model_name": Path(str(config["model_path"])).name or str(config["model_path"]),
                    "run_name": "local_inference",
                }
            )

    return pd.DataFrame(rows)


def _resolve_path(cli_path: Path | None, configured_path: str) -> Path | None:
    if cli_path is not None:
        return cli_path
    configured_path = configured_path.strip()
    if configured_path:
        return Path(configured_path)
    return None


def _resolve_output_path(output_arg: Path | None) -> Path:
    if output_arg is None:
        output_dir = ensure_dir(EVAL_ROOT / "outputs" / f"inference_{timestamp_string()}")
        return output_dir / "generated_predictions.csv"

    if output_arg.exists() and output_arg.is_dir():
        return output_arg / "generated_predictions.csv"

    if output_arg.suffix.lower() != ".csv":
        output_arg.mkdir(parents=True, exist_ok=True)
        return output_arg / "generated_predictions.csv"

    output_arg.parent.mkdir(parents=True, exist_ok=True)
    return output_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Hugging Face inference only.")
    parser.add_argument("--gold", type=Path, help="Gold/reference CSV path.")
    parser.add_argument("--output", type=Path, help="Output CSV path for generated predictions.")
    parser.add_argument("--run-defaults", type=Path, default=EVAL_ROOT / "configs" / "run_defaults.yaml")
    parser.add_argument("--inference-config", type=Path, default=EVAL_ROOT / "configs" / "inference.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_defaults = load_yaml(args.run_defaults)
    inference_config = load_yaml(args.inference_config)
    gold_path = _resolve_path(args.gold, str(run_defaults.get("default_gold_path", "")))
    if gold_path is None:
        raise SystemExit("Gold path is required. Pass --gold or set default_gold_path in run_defaults.yaml.")
    output_path = _resolve_output_path(args.output)

    try:
        gold_df = load_gold_with_sample_ids(gold_path)
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc

    pred_df = generate_predictions(
        gold_df,
        config=inference_config,
        system_prompt_path=EVAL_ROOT / "prompts" / "inference_system.txt",
        user_prompt_path=EVAL_ROOT / "prompts" / "inference_user.txt",
    )
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote generated predictions to {output_path}")
    print(f"Rows: {len(pred_df)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
