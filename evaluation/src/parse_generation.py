from __future__ import annotations

import json
import re
from typing import Any

from io_utils import normalize_text


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
NARRATIVE_RE = re.compile(r'"?predicted_narrative"?\s*[:=]\s*"?(.*?)"?(?:,|\n|$)', re.IGNORECASE | re.DOTALL)
EMOTION_RE = re.compile(r'"?predicted_emotion_label"?\s*[:=]\s*"?(.*?)"?(?:,|\n|$)', re.IGNORECASE | re.DOTALL)


def _clean_extracted_value(value: str) -> str:
    value = value.strip().strip(",").strip()
    if value.endswith("}"):
        value = value[:-1].rstrip()
    return normalize_text(value.strip('"').strip("'"))


def parse_generation(raw_text: Any) -> dict[str, Any]:
    raw_text = "" if raw_text is None else str(raw_text)
    parsed = {
        "predicted_narrative": "",
        "predicted_emotion_label": "",
        "raw_generation": raw_text,
        "parse_success": False,
    }

    json_match = JSON_OBJECT_RE.search(raw_text)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            parsed["predicted_narrative"] = normalize_text(obj.get("predicted_narrative", ""))
            parsed["predicted_emotion_label"] = normalize_text(obj.get("predicted_emotion_label", ""))
            parsed["parse_success"] = bool(
                parsed["predicted_narrative"] or parsed["predicted_emotion_label"]
            )
            return parsed
        except json.JSONDecodeError:
            pass

    narrative_match = NARRATIVE_RE.search(raw_text)
    if narrative_match:
        parsed["predicted_narrative"] = _clean_extracted_value(narrative_match.group(1))

    emotion_match = EMOTION_RE.search(raw_text)
    if emotion_match:
        parsed["predicted_emotion_label"] = _clean_extracted_value(emotion_match.group(1))

    parsed["parse_success"] = bool(
        parsed["predicted_narrative"] or parsed["predicted_emotion_label"]
    )
    return parsed
