from __future__ import annotations

import json
from typing import Any

from io_utils import normalize_text


EKMAN_LABELS = (
    "Anger",
    "Disgust",
    "Fear",
    "Joy",
    "Neutral",
    "Sadness",
    "Surprise",
)

PREDICTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "predicted_narrative": {
            "type": "string",
        },
        "predicted_emotion_label": {
            "type": "string",
            "enum": list(EKMAN_LABELS),
        },
    },
    "required": [
        "predicted_narrative",
        "predicted_emotion_label",
    ],
    "additionalProperties": False,
}

VLLM_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "narrative_emotion_prediction",
        "strict": True,
        "schema": PREDICTION_JSON_SCHEMA,
    },
}


def parse_structured_prediction(raw_text: Any) -> dict[str, Any]:
    raw_generation = "" if raw_text is None else str(raw_text)
    result = {
        "predicted_narrative": "",
        "predicted_emotion_label": "",
        "raw_generation": raw_generation,
        "parse_success": False,
        "parse_error": "",
    }

    try:
        value = json.loads(raw_generation)
    except (json.JSONDecodeError, TypeError) as exc:
        result["parse_error"] = f"invalid_json: {exc}"
        return result

    if not isinstance(value, dict):
        result["parse_error"] = "output_is_not_a_json_object"
        return result

    expected_keys = {"predicted_narrative", "predicted_emotion_label"}
    actual_keys = set(value)
    if actual_keys != expected_keys:
        result["parse_error"] = (
            f"unexpected_keys: expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
        return result

    if not isinstance(value["predicted_narrative"], str):
        result["parse_error"] = "predicted_narrative_is_not_a_string"
        return result
    if not isinstance(value["predicted_emotion_label"], str):
        result["parse_error"] = "predicted_emotion_label_is_not_a_string"
        return result

    narrative = normalize_text(value["predicted_narrative"])
    emotion = normalize_text(value["predicted_emotion_label"])
    if not narrative:
        result["parse_error"] = "predicted_narrative_is_empty"
        return result
    if emotion not in EKMAN_LABELS:
        result["parse_error"] = f"invalid_emotion_label: {emotion!r}"
        return result

    result["predicted_narrative"] = narrative
    result["predicted_emotion_label"] = emotion
    result["parse_success"] = True
    return result
