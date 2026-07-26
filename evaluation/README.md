# Evaluation

The evaluation system supports three input paths:

1. Evaluate an existing prediction CSV with `--pred`.
2. Generate with local Hugging Face Transformers and evaluate with `--inference`.
3. Generate through a deployed vLLM server with constrained JSON output and
   evaluate with `--vllm-inference`.

## Input schemas

Gold CSV:

```text
source_text,emotion_label,reference_narrative
```

The evaluator assigns `sample_id = 0, 1, 2, ...` from the gold row order.

Prediction CSV:

```text
sample_id,predicted_narrative,predicted_emotion_label
```

Emotion labels use the exact seven-class Ekman set:

```text
Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
```

## vLLM structured-output path

The vLLM server and evaluation client can run in separate environments. Install
vLLM in the Linux/CUDA server environment:

```bash
pip install -r evaluation/requirements-vllm.txt
```

Set at least `model_path` in `evaluation/configs/vllm.yaml`. The default setup
serves one model as `narrative-model` on port 8000. If the tokenizer does not
contain a chat template, set `chat_template` to a compatible template file.

Start the server:

```bash
python evaluation/src/serve_vllm.py
```

In another terminal, generate predictions and immediately evaluate them:

```bash
python evaluation/src/run_eval.py \
  --gold /path/to/test_gold.csv \
  --vllm-inference
```

To generate a prediction CSV without running metrics:

```bash
python evaluation/src/inference_vllm.py \
  --gold /path/to/test_gold.csv \
  --output /path/to/generated_predictions.csv
```

For a server on another machine, do not run `serve_vllm.py` locally. Set
`base_url` and `api_key` in `evaluation/configs/vllm.yaml` to the remote server.

The vLLM request uses JSON Schema constrained decoding. It requires exactly:

```json
{
  "predicted_narrative": "A non-empty narrative",
  "predicted_emotion_label": "Anger"
}
```

Extra properties are prohibited and the emotion field is constrained to the
seven valid labels. The client also parses and validates the returned JSON before
writing:

- `raw_generation`: complete model response content
- `parse_success`: whether strict client validation succeeded
- `parse_error`: JSON/schema validation failure
- `request_error`: server or network failure
- `finish_reason`: vLLM completion reason

Failed rows remain in `generated_predictions.csv` with empty prediction fields so
they can be audited. They are counted in the run report as generation failures.

## Other modes

Evaluate an existing prediction CSV:

```bash
python evaluation/src/run_eval.py \
  --gold /path/to/test_gold.csv \
  --pred /path/to/predictions.csv
```

Run the original Transformers backend and evaluate:

```bash
python evaluation/src/run_eval.py \
  --gold /path/to/test_gold.csv \
  --inference
```

The Transformers model path is configured in `evaluation/configs/inference.yaml`.
Default gold and prediction paths can be set in
`evaluation/configs/run_defaults.yaml`.
