from __future__ import annotations

import math

import numpy as np
import pandas as pd
from bert_score import score as bertscore_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from io_utils import split_sentences


def _compute_dtw_distance(ref_embeddings: np.ndarray, pred_embeddings: np.ndarray) -> tuple[float, float]:
    if len(ref_embeddings) == 0 or len(pred_embeddings) == 0:
        return math.nan, math.nan

    cosine = cosine_similarity(ref_embeddings, pred_embeddings)
    distances = 1.0 - cosine
    n_rows, n_cols = distances.shape
    dp = np.full((n_rows + 1, n_cols + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            cost = distances[i - 1, j - 1]
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    raw_distance = float(dp[n_rows, n_cols])
    normalized = raw_distance / float(n_rows + n_cols)
    return raw_distance, normalized


def _encode_sentence_groups(
    sentence_model: SentenceTransformer,
    groups: list[list[str]],
) -> list[np.ndarray]:
    flat_sentences = [sentence for group in groups for sentence in group]
    if hasattr(sentence_model, "get_embedding_dimension"):
        embedding_dimension = sentence_model.get_embedding_dimension()
    else:
        embedding_dimension = sentence_model.get_sentence_embedding_dimension()
    if not flat_sentences:
        return [np.empty((0, embedding_dimension), dtype=np.float32) for _ in groups]

    flat_embeddings = sentence_model.encode(
        flat_sentences,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    encoded_groups = []
    offset = 0
    for group in groups:
        next_offset = offset + len(group)
        encoded_groups.append(flat_embeddings[offset:next_offset])
        offset = next_offset
    return encoded_groups


def compute_semantic_metrics(
    eval_df: pd.DataFrame,
    *,
    bertscore_model_type: str,
    bertscore_lang: str,
    bertscore_batch_size: int,
    bertscore_rescale_with_baseline: bool,
    dtw_sentence_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = eval_df["predicted_narrative"].fillna("").tolist()
    references = eval_df["reference_narrative"].fillna("").tolist()
    valid_mask = [bool(pred.strip()) and bool(ref.strip()) for pred, ref in zip(predictions, references, strict=True)]

    bert_p = np.full(len(eval_df), np.nan)
    bert_r = np.full(len(eval_df), np.nan)
    bert_f1 = np.full(len(eval_df), np.nan)

    if any(valid_mask):
        valid_predictions = [pred for pred, keep in zip(predictions, valid_mask, strict=True) if keep]
        valid_references = [ref for ref, keep in zip(references, valid_mask, strict=True) if keep]
        precision, recall, f1 = bertscore_score(
            valid_predictions,
            valid_references,
            model_type=bertscore_model_type,
            lang=bertscore_lang,
            batch_size=bertscore_batch_size,
            rescale_with_baseline=bertscore_rescale_with_baseline,
            verbose=False,
        )
        valid_indices = [index for index, keep in enumerate(valid_mask) if keep]
        for idx, p_score, r_score, f1_score in zip(valid_indices, precision, recall, f1, strict=True):
            bert_p[idx] = float(p_score)
            bert_r[idx] = float(r_score)
            bert_f1[idx] = float(f1_score)

    sentence_model = SentenceTransformer(dtw_sentence_model)
    reference_sentence_groups = [split_sentences(reference) for reference in references]
    prediction_sentence_groups = [split_sentences(prediction) for prediction in predictions]
    all_encoded_groups = _encode_sentence_groups(
        sentence_model,
        reference_sentence_groups + prediction_sentence_groups,
    )
    reference_embedding_groups = all_encoded_groups[: len(reference_sentence_groups)]
    prediction_embedding_groups = all_encoded_groups[len(reference_sentence_groups) :]

    for row_index, sample_id in enumerate(eval_df["sample_id"].tolist()):
        ref_embeddings = reference_embedding_groups[row_index]
        pred_embeddings = prediction_embedding_groups[row_index]
        if len(ref_embeddings) and len(pred_embeddings):
            dtw_distance, normalized_dtw_distance = _compute_dtw_distance(ref_embeddings, pred_embeddings)
        else:
            dtw_distance, normalized_dtw_distance = math.nan, math.nan

        rows.append(
            {
                "sample_id": sample_id,
                "bertscore_precision": bert_p[row_index],
                "bertscore_recall": bert_r[row_index],
                "bertscore_f1": bert_f1[row_index],
                "dtw_distance": dtw_distance,
                "normalized_dtw_distance": normalized_dtw_distance,
            }
        )

    per_sample_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {"metric": "bertscore_f1_mean", "value": float(per_sample_df["bertscore_f1"].mean())},
            {"metric": "dtw_distance_mean", "value": float(per_sample_df["dtw_distance"].mean())},
            {
                "metric": "normalized_dtw_distance_mean",
                "value": float(per_sample_df["normalized_dtw_distance"].mean()),
            },
        ]
    )
    return per_sample_df, summary_df
