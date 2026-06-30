from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

from io_utils import ensure_dir, render_template


def _cache_key(sample_id: int, model_name: str, prompt_version: str) -> str:
    key = f"{sample_id}:{model_name}:{prompt_version}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def run_llm_judge(
    eval_df: pd.DataFrame,
    *,
    config: dict,
    system_prompt_path: Path,
    user_prompt_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.get("enabled", False):
        return pd.DataFrame(), pd.DataFrame()

    client = OpenAI()
    cache_dir = ensure_dir(Path(config["cache_dir"]))
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    prompt_version = hashlib.sha256(
        (system_prompt + user_prompt_path.read_text(encoding="utf-8")).encode("utf-8")
    ).hexdigest()[:12]

    rows = []
    for row in eval_df.itertuples(index=False):
        cache_path = cache_dir / f"{_cache_key(row.sample_id, config['model'], prompt_version)}.json"
        if cache_path.exists():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            user_prompt = render_template(
                user_prompt_path,
                source_text=row.source_text,
                reference_narrative=row.reference_narrative,
                predicted_narrative=row.predicted_narrative,
            )
            last_error = None
            for _ in range(int(config.get("max_retries", 3))):
                try:
                    response = client.chat.completions.create(
                        model=config["model"],
                        temperature=config["temperature"],
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    result = json.loads(response.choices[0].message.content)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    result = None
            if result is None:
                raise RuntimeError(f"LLM judge failed for sample_id={row.sample_id}") from last_error
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        rows.append({"sample_id": row.sample_id, **result})

    per_sample_df = pd.DataFrame(rows)
    nli_distribution = (
        per_sample_df["nli_label"].value_counts(normalize=True).sort_index().round(4).to_dict()
        if "nli_label" in per_sample_df.columns
        else {}
    )
    summary_df = pd.DataFrame(
        [
            {"metric": "judge_relevance_mean", "value": float(per_sample_df["relevance_score"].mean())},
            {"metric": "judge_faithfulness_mean", "value": float(per_sample_df["faithfulness_score"].mean())},
            {"metric": "judge_coherence_mean", "value": float(per_sample_df["coherence_score"].mean())},
            {"metric": "nli_distribution", "value": json.dumps(nli_distribution, ensure_ascii=False)},
        ]
    )
    return per_sample_df, summary_df
