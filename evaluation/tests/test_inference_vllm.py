from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_vllm import generate_vllm_predictions
from prediction_schema import VLLM_RESPONSE_FORMAT


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"predicted_narrative":"The policy is framed as harmful.",'
                            '"predicted_emotion_label":"Anger"}'
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="narrative-model")])
        )
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class GenerateVllmPredictionsTest(unittest.TestCase):
    def test_sends_schema_and_extracts_fields(self) -> None:
        fake_client = _FakeClient()
        gold_df = pd.DataFrame([{"sample_id": 0, "source_text": "A source comment."}])
        config = {
            "served_model_name": "narrative-model",
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "concurrency": 1,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            system_path = Path(temp_dir) / "system.txt"
            user_path = Path(temp_dir) / "user.txt"
            system_path.write_text("System instructions", encoding="utf-8")
            user_path.write_text("Source: {{ source_text }}", encoding="utf-8")

            with patch("inference_vllm._build_client", return_value=fake_client):
                predictions = generate_vllm_predictions(
                    gold_df,
                    config=config,
                    system_prompt_path=system_path,
                    user_prompt_path=user_path,
                )

        request = fake_client.chat.completions.last_request
        self.assertEqual(request["response_format"], VLLM_RESPONSE_FORMAT)
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertIn("A source comment.", request["messages"][1]["content"])
        self.assertTrue(predictions.loc[0, "parse_success"])
        self.assertEqual(predictions.loc[0, "predicted_emotion_label"], "Anger")
        self.assertEqual(predictions.loc[0, "finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
