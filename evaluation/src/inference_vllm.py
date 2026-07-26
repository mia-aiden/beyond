from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from io_utils import ensure_dir, load_yaml, render_template, timestamp_string
from prediction_schema import VLLM_RESPONSE_FORMAT, parse_structured_prediction
from validate_inputs import ValidationError, load_gold_with_sample_ids


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = PROJECT_ROOT / "evaluation"


def _build_client(config: dict[str, Any]) -> OpenAI:
    base_url = str(config.get("base_url", "")).strip()
    if not base_url:
        raise ValueError("vllm.yaml must define base_url.")

    return OpenAI(
        base_url=base_url,
        api_key=str(config.get("api_key", "EMPTY")),
        timeout=float(config.get("request_timeout_seconds", 180)),
        max_retries=max(0, int(config.get("max_retries", 2))),
    )


def _check_server(client: OpenAI, served_model_name: str) -> None:
    try:
        available_models = {model.id for model in client.models.list().data}
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to the vLLM server. Start serve_vllm.py first and "
            "check base_url/api_key in vllm.yaml."
        ) from exc

    if served_model_name not in available_models:
        raise RuntimeError(
            f"vLLM model {served_model_name!r} is not available. "
            f"Server models: {sorted(available_models)}"
        )


def _generate_one(
    *,
    client: OpenAI,
    served_model_name: str,
    sample_id: int,
    system_prompt: str,
    user_prompt: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=served_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=int(config.get("max_tokens", 384)),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            response_format=VLLM_RESPONSE_FORMAT,
        )
        choice = completion.choices[0]
        raw_generation = choice.message.content or ""
        parsed = parse_structured_prediction(raw_generation)
        return {
            "sample_id": sample_id,
            "predicted_narrative": parsed["predicted_narrative"],
            "predicted_emotion_label": parsed["predicted_emotion_label"],
            "raw_generation": parsed["raw_generation"],
            "parse_success": parsed["parse_success"],
            "parse_error": parsed["parse_error"],
            "request_error": "",
            "finish_reason": choice.finish_reason or "",
            "model_name": served_model_name,
            "run_name": "vllm_inference",
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "predicted_narrative": "",
            "predicted_emotion_label": "",
            "raw_generation": "",
            "parse_success": False,
            "parse_error": "",
            "request_error": f"{type(exc).__name__}: {exc}",
            "finish_reason": "",
            "model_name": served_model_name,
            "run_name": "vllm_inference",
        }


def generate_vllm_predictions(
    gold_df: pd.DataFrame,
    *,
    config: dict[str, Any],
    system_prompt_path: Path,
    user_prompt_path: Path,
) -> pd.DataFrame:
    client = _build_client(config)
    served_model_name = str(config.get("served_model_name", "")).strip()
    if not served_model_name:
        raise ValueError("vllm.yaml must define served_model_name.")
    _check_server(client, served_model_name)

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    requests: list[dict[str, Any]] = []
    for row in gold_df.itertuples(index=False):
        requests.append(
            {
                "sample_id": int(row.sample_id),
                "user_prompt": render_template(user_prompt_path, source_text=row.source_text),
            }
        )

    concurrency = max(1, int(config.get("concurrency", 8)))
    completed_rows: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[Future, int] = {}
        for request in requests:
            future = executor.submit(
                _generate_one,
                client=client,
                served_model_name=served_model_name,
                sample_id=request["sample_id"],
                system_prompt=system_prompt,
                user_prompt=request["user_prompt"],
                config=config,
            )
            futures[future] = request["sample_id"]

        for future in as_completed(futures):
            sample_id = futures[future]
            completed_rows[sample_id] = future.result()

    ordered_rows = [completed_rows[request["sample_id"]] for request in requests]
    return pd.DataFrame(ordered_rows)


def _resolve_path(cli_path: Path | None, configured_path: str) -> Path | None:
    if cli_path is not None:
        return cli_path
    configured_path = configured_path.strip()
    return Path(configured_path) if configured_path else None


def _resolve_output_path(output_arg: Path | None) -> Path:
    if output_arg is None:
        output_dir = ensure_dir(EVAL_ROOT / "outputs" / f"vllm_inference_{timestamp_string()}")
        return output_dir / "generated_predictions.csv"
    if output_arg.exists() and output_arg.is_dir():
        return output_arg / "generated_predictions.csv"
    if output_arg.suffix.lower() != ".csv":
        output_arg.mkdir(parents=True, exist_ok=True)
        return output_arg / "generated_predictions.csv"
    output_arg.parent.mkdir(parents=True, exist_ok=True)
    return output_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM structured-output inference only.")
    parser.add_argument("--gold", type=Path, help="Gold/reference CSV path.")
    parser.add_argument("--output", type=Path, help="Output CSV path.")
    parser.add_argument(
        "--run-defaults",
        type=Path,
        default=EVAL_ROOT / "configs" / "run_defaults.yaml",
    )
    parser.add_argument(
        "--vllm-config",
        type=Path,
        default=EVAL_ROOT / "configs" / "vllm.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_defaults = load_yaml(args.run_defaults)
    vllm_config = load_yaml(args.vllm_config)
    gold_path = _resolve_path(args.gold, str(run_defaults.get("default_gold_path", "")))
    if gold_path is None:
        raise SystemExit("Gold path is required. Pass --gold or set default_gold_path.")

    try:
        gold_df = load_gold_with_sample_ids(gold_path)
        predictions = generate_vllm_predictions(
            gold_df,
            config=vllm_config,
            system_prompt_path=EVAL_ROOT / "prompts" / "inference_system.txt",
            user_prompt_path=EVAL_ROOT / "prompts" / "inference_user.txt",
        )
    except (ValidationError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    output_path = _resolve_output_path(args.output)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    failures = int((~predictions["parse_success"].astype(bool)).sum())
    print(f"Wrote generated predictions to {output_path}")
    print(f"Rows: {len(predictions)}; failures: {failures}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
