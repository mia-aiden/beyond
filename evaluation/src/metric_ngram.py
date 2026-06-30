from __future__ import annotations

import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer


def compute_ngram_metrics(
    eval_df: pd.DataFrame,
    *,
    rouge_types: list[str],
    use_stemmer: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=use_stemmer)
    rows = []

    predictions = eval_df["predicted_narrative"].tolist()
    references = eval_df["reference_narrative"].tolist()
    bleu_corpus = sacrebleu.corpus_bleu(predictions, [references]).score

    for sample_id, prediction, reference in zip(
        eval_df["sample_id"], predictions, references, strict=True
    ):
        bleu_sentence = sacrebleu.sentence_bleu(prediction, [reference]).score
        rouge_scores = scorer.score(reference, prediction)
        row = {
            "sample_id": sample_id,
            "bleu_sentence": bleu_sentence,
        }
        for rouge_type in rouge_types:
            row[f"{rouge_type}_precision"] = rouge_scores[rouge_type].precision
            row[f"{rouge_type}_recall"] = rouge_scores[rouge_type].recall
            row[f"{rouge_type}_fmeasure"] = rouge_scores[rouge_type].fmeasure
        rows.append(row)

    per_sample_df = pd.DataFrame(rows)
    summary_rows = [{"metric": "bleu_corpus", "value": bleu_corpus}]
    summary_rows.append(
        {"metric": "bleu_sentence_mean", "value": float(per_sample_df["bleu_sentence"].mean())}
    )
    for rouge_type in rouge_types:
        summary_rows.append(
            {
                "metric": f"{rouge_type}_mean",
                "value": float(per_sample_df[f"{rouge_type}_fmeasure"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return per_sample_df, summary_df
