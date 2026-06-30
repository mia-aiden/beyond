from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import pandas as pd

from aggregate import build_summary, merge_per_sample_metrics, write_report, write_summary_json
from inference_local import generate_predictions
from io_utils import ensure_dir, load_yaml, timestamp_string
from metric_argument import run_argument_mining
from metric_emotion import compute_emotion_metrics
from metric_judge import run_llm_judge
from metric_manual import aggregate_manual_eval, export_manual_eval_sheet
from metric_ngram import compute_ngram_metrics
from metric_semantic import compute_semantic_metrics
from plots import plot_emotion_confusion_matrix, plot_length_distribution, plot_narrative_distributions
from validate_inputs import ValidationError, align_gold_and_prediction, load_gold_with_sample_ids, load_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT = PROJECT_ROOT / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run narrative + emotion evaluation.")
    parser.add_argument("--gold", type=Path, help="Gold/reference CSV path.")
    parser.add_argument("--pred", type=Path, help="Prediction CSV path.")
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run local Hugging Face inference before evaluation.",
    )
    parser.add_argument("--output-root", type=Path, default=EVAL_ROOT / "outputs", help="Output root directory.")
    parser.add_argument("--run-defaults", type=Path, default=EVAL_ROOT / "configs" / "run_defaults.yaml")
    parser.add_argument("--metrics-config", type=Path, default=EVAL_ROOT / "configs" / "metrics.yaml")
    parser.add_argument("--judge-config", type=Path, default=EVAL_ROOT / "configs" / "judge.yaml")
    parser.add_argument("--manual-config", type=Path, default=EVAL_ROOT / "configs" / "manual_eval.yaml")
    parser.add_argument("--inference-config", type=Path, default=EVAL_ROOT / "configs" / "inference.yaml")
    return parser.parse_args()


def _concat_metric_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.concat(non_empty, ignore_index=True)


def _resolve_path(cli_path: Path | None, configured_path: str) -> Path | None:
    if cli_path is not None:
        return cli_path
    configured_path = configured_path.strip()
    if configured_path:
        return Path(configured_path)
    return None


def main() -> None:
    args = parse_args()
    run_defaults = load_yaml(args.run_defaults)
    metrics_config = load_yaml(args.metrics_config)
    judge_config = load_yaml(args.judge_config)
    manual_config = load_yaml(args.manual_config)
    inference_config = load_yaml(args.inference_config)
    gold_path = _resolve_path(args.gold, str(run_defaults.get("default_gold_path", "")))
    pred_input_path = _resolve_path(args.pred, str(run_defaults.get("default_prediction_path", "")))

    if gold_path is None:
        raise SystemExit("Gold path is required. Pass --gold or set default_gold_path in run_defaults.yaml.")
    if args.inference and args.pred is not None:
        raise SystemExit("Use either --pred or --inference, not both.")
    if not args.inference and pred_input_path is None:
        raise SystemExit(
            "Prediction path is required in evaluation mode. Pass --pred or set default_prediction_path in run_defaults.yaml."
        )

    run_timestamp = timestamp_string()
    run_dir = ensure_dir(args.output_root / f"run_{run_timestamp}")
    predictions_dir = ensure_dir(run_dir / "predictions")
    metrics_dir = ensure_dir(run_dir / "metrics")
    plots_dir = ensure_dir(run_dir / "plots")
    manual_dir = ensure_dir(run_dir / "manual")

    try:
        gold_df = load_gold_with_sample_ids(gold_path)
        if args.inference:
            pred_df = generate_predictions(
                gold_df,
                config=inference_config,
                system_prompt_path=EVAL_ROOT / "prompts" / "inference_system.txt",
                user_prompt_path=EVAL_ROOT / "prompts" / "inference_user.txt",
            )
            pred_path = predictions_dir / "generated_predictions.csv"
            pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
            pred_df = load_prediction(pred_path)
        else:
            pred_path = pred_input_path
            pred_df = load_prediction(pred_input_path)
        alignment = align_gold_and_prediction(gold_df, pred_df)
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc

    alignment.gold_df.to_csv(metrics_dir / "gold_with_sample_ids.csv", index=False, encoding="utf-8-sig")
    alignment.aligned_df.to_csv(metrics_dir / "aligned_rows.csv", index=False, encoding="utf-8-sig")

    missing_frames = []
    if not alignment.missing_rows_df.empty:
        missing_frames.append(alignment.missing_rows_df)
    if not alignment.extra_prediction_rows_df.empty:
        missing_frames.append(alignment.extra_prediction_rows_df)
    if missing_frames:
        pd.concat(missing_frames, ignore_index=True).to_csv(
            metrics_dir / "missing_rows.csv", index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=["sample_id", "status"]).to_csv(
            metrics_dir / "missing_rows.csv", index=False, encoding="utf-8-sig"
        )

    num_generation_failures = 0
    if args.inference and "parse_success" in pred_df.columns:
        num_generation_failures = int((~pred_df["parse_success"].astype(bool)).sum())

    automatic_per_sample_df = pd.DataFrame({"sample_id": alignment.aligned_df["sample_id"]})
    automatic_summary_df = pd.DataFrame(columns=["metric", "value"])
    if run_defaults.get("automatic_metrics", True):
        ngram_per_sample_df, ngram_summary_df = compute_ngram_metrics(
            alignment.aligned_df,
            rouge_types=metrics_config["rouge"]["types"],
            use_stemmer=metrics_config["rouge"]["use_stemmer"],
        )
        semantic_per_sample_df, semantic_summary_df = compute_semantic_metrics(
            alignment.aligned_df,
            bertscore_model_type=metrics_config["bertscore"]["model_type"],
            bertscore_lang=metrics_config["bertscore"]["lang"],
            bertscore_batch_size=metrics_config["bertscore"]["batch_size"],
            bertscore_rescale_with_baseline=metrics_config["bertscore"]["rescale_with_baseline"],
            dtw_sentence_model=metrics_config["dtw"]["sentence_model"],
        )
        automatic_per_sample_df = merge_per_sample_metrics(
            automatic_per_sample_df, [ngram_per_sample_df, semantic_per_sample_df]
        )
        automatic_summary_df = _concat_metric_frames(ngram_summary_df, semantic_summary_df)
        automatic_summary_df.to_csv(metrics_dir / "automatic_metrics.csv", index=False, encoding="utf-8-sig")

    emotion_summary_df = pd.DataFrame(columns=["metric", "value"])
    emotion_per_sample_df = pd.DataFrame({"sample_id": alignment.aligned_df["sample_id"]})
    emotion_matrix_df = pd.DataFrame()
    if run_defaults.get("emotion_metrics", True):
        emotion_summary_df, emotion_matrix_df, emotion_per_sample_df = compute_emotion_metrics(
            alignment.aligned_df
        )
        emotion_summary_df.to_csv(metrics_dir / "emotion_metrics.csv", index=False, encoding="utf-8-sig")
        emotion_per_sample_df.to_csv(
            metrics_dir / "emotion_per_sample.csv", index=False, encoding="utf-8-sig"
        )
        emotion_matrix_df.to_csv(
            metrics_dir / "emotion_confusion_matrix.csv", encoding="utf-8-sig"
        )

    judge_per_sample_df = pd.DataFrame()
    judge_summary_df = pd.DataFrame(columns=["metric", "value"])
    if run_defaults.get("judge_enabled", False) and judge_config.get("enabled", False):
        judge_per_sample_df, judge_summary_df = run_llm_judge(
            alignment.aligned_df,
            config=judge_config,
            system_prompt_path=EVAL_ROOT / "prompts" / "llm_judge_system.txt",
            user_prompt_path=EVAL_ROOT / "prompts" / "llm_judge_user.txt",
        )
        judge_per_sample_df.to_csv(metrics_dir / "judge_scores.csv", index=False, encoding="utf-8-sig")

    manual_summary_df = pd.DataFrame(columns=["metric", "value"])
    if run_defaults.get("manual_enabled", False) and manual_config.get("enabled", False):
        sheet_path = manual_dir / "manual_eval_batch.csv"
        export_manual_eval_sheet(
            alignment.aligned_df,
            output_path=sheet_path,
            sample_size=int(manual_config["sample_size"]),
            stratify_by_emotion=bool(manual_config.get("stratify_by_emotion", True)),
        )
        _, manual_summary_df = aggregate_manual_eval(sheet_path)
        manual_summary_df.to_csv(
            manual_dir / "manual_eval_summary.csv", index=False, encoding="utf-8-sig"
        )

    argument_per_sample_df = pd.DataFrame()
    argument_summary_df = pd.DataFrame(columns=["metric", "value"])
    if run_defaults.get("argument_enabled", False):
        argument_per_sample_df, argument_summary_df = run_argument_mining(
            alignment.aligned_df,
            enabled=True,
            model=judge_config.get("model"),
            system_prompt_path=EVAL_ROOT / "prompts" / "argument_mining_system.txt",
        )
        if not argument_per_sample_df.empty:
            argument_per_sample_df.to_csv(
                metrics_dir / "argument_results.csv", index=False, encoding="utf-8-sig"
            )

    base_per_sample_df = alignment.aligned_df[
        [
            "sample_id",
            "source_text",
            "emotion_label",
            "reference_narrative",
            "predicted_emotion_label",
            "predicted_narrative",
        ]
    ].copy()
    per_sample_metrics_df = merge_per_sample_metrics(
        base_per_sample_df,
        [automatic_per_sample_df, emotion_per_sample_df, judge_per_sample_df, argument_per_sample_df],
    )
    per_sample_metrics_df.to_csv(metrics_dir / "per_sample_metrics.csv", index=False, encoding="utf-8-sig")

    if not automatic_per_sample_df.empty:
        plot_narrative_distributions(automatic_per_sample_df, plots_dir)
    plot_length_distribution(alignment.aligned_df, plots_dir / "length_distribution.png")
    if not emotion_matrix_df.empty:
        plot_emotion_confusion_matrix(emotion_matrix_df, plots_dir / "emotion_confusion_matrix.png")

    summary = build_summary(
        run_timestamp=run_timestamp,
        gold_path=gold_path,
        pred_path=pred_path,
        inference_mode=bool(args.inference),
        model_path=str(inference_config.get("model_path", "")) if args.inference else "",
        num_generation_failures=num_generation_failures,
        num_gold_rows=len(alignment.gold_df),
        num_prediction_rows=len(alignment.pred_df),
        num_aligned_rows=len(alignment.aligned_df),
        num_missing_prediction_rows=len(alignment.missing_rows_df),
        automatic_summary_df=automatic_summary_df,
        emotion_summary_df=emotion_summary_df,
        judge_summary_df=judge_summary_df,
        manual_summary_df=manual_summary_df,
        argument_summary_df=argument_summary_df,
        judge_enabled=bool(run_defaults.get("judge_enabled", False) and judge_config.get("enabled", False)),
        manual_enabled=bool(run_defaults.get("manual_enabled", False) and manual_config.get("enabled", False)),
        argument_enabled=bool(run_defaults.get("argument_enabled", False)),
    )
    write_summary_json(summary, metrics_dir / "summary.json")
    write_report(summary, EVAL_ROOT / "templates" / "final_report.md.j2", run_dir / "report.md")

    print(f"Wrote evaluation run to {run_dir}")
    print(f"Aligned rows: {len(alignment.aligned_df)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
