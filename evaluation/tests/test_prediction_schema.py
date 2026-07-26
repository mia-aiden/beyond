from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prediction_schema import parse_structured_prediction


class ParseStructuredPredictionTest(unittest.TestCase):
    def test_valid_prediction(self) -> None:
        result = parse_structured_prediction(
            '{"predicted_narrative":"A political actor is held responsible.",'
            '"predicted_emotion_label":"Anger"}'
        )

        self.assertTrue(result["parse_success"])
        self.assertEqual(result["predicted_emotion_label"], "Anger")
        self.assertEqual(result["parse_error"], "")

    def test_rejects_malformed_json(self) -> None:
        result = parse_structured_prediction(
            '{"predicted_narrative":"Incomplete",'
            '"predicted_emotion_label":"Neutral"'
        )

        self.assertFalse(result["parse_success"])
        self.assertIn("invalid_json", result["parse_error"])

    def test_rejects_extra_fields(self) -> None:
        result = parse_structured_prediction(
            '{"predicted_narrative":"Narrative",'
            '"predicted_emotion_label":"Neutral","reason":"extra"}'
        )

        self.assertFalse(result["parse_success"])
        self.assertIn("unexpected_keys", result["parse_error"])

    def test_rejects_invalid_emotion(self) -> None:
        result = parse_structured_prediction(
            '{"predicted_narrative":"Narrative",'
            '"predicted_emotion_label":"Annoyance"}'
        )

        self.assertFalse(result["parse_success"])
        self.assertIn("invalid_emotion_label", result["parse_error"])

    def test_rejects_empty_narrative(self) -> None:
        result = parse_structured_prediction(
            '{"predicted_narrative":"  ","predicted_emotion_label":"Neutral"}'
        )

        self.assertFalse(result["parse_success"])
        self.assertEqual(result["parse_error"], "predicted_narrative_is_empty")

    def test_rejects_non_string_values(self) -> None:
        result = parse_structured_prediction(
            '{"predicted_narrative":42,"predicted_emotion_label":"Neutral"}'
        )

        self.assertFalse(result["parse_success"])
        self.assertEqual(result["parse_error"], "predicted_narrative_is_not_a_string")


if __name__ == "__main__":
    unittest.main()
