from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

from io_utils import render_template


def run_argument_mining(
    eval_df: pd.DataFrame,
    *,
    enabled: bool,
    model: str | None = None,
    system_prompt_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not enabled:
        return pd.DataFrame(), pd.DataFrame()

    if model is None or system_prompt_path is None:
        raise ValueError("Argument mining requires a model and system prompt path when enabled.")

    client = OpenAI()
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    rows = []
    for row in eval_df.itertuples(index=False):
        user_prompt = (
            f"Predicted narrative:\n{row.predicted_narrative}\n\n"
            "Return JSON with keys has_claim, has_reason, has_stance_marker, argumentative_strength."
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        result = json.loads(response.choices[0].message.content)
        rows.append({"sample_id": row.sample_id, **result})

    per_sample_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {"metric": "argument_claim_rate", "value": float(per_sample_df["has_claim"].astype(float).mean())},
            {"metric": "argument_reason_rate", "value": float(per_sample_df["has_reason"].astype(float).mean())},
            {
                "metric": "argument_stance_rate",
                "value": float(per_sample_df["has_stance_marker"].astype(float).mean()),
            },
        ]
    )
    return per_sample_df, summary_df
