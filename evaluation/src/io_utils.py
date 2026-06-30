from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Template


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as infile:
        data = yaml.safe_load(infile) or {}
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: Any) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = _SENTENCE_SPLIT_RE.split(normalized)
    return [part.strip() for part in parts if part.strip()]


def render_template(template_path: Path, **kwargs: Any) -> str:
    with template_path.open("r", encoding="utf-8") as infile:
        template = Template(infile.read())
    return template.render(**kwargs)
