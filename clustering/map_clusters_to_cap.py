import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from cap_topic_prototypes import CAP_TOPIC_PROTOTYPES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_SUMMARY_PATH = PROJECT_ROOT / "clustering" / "reddit_narratives_bertopic_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "clustering"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
TOP_K = 3
REP_DOC_COUNT = 3
REP_DOC_CHAR_LIMIT = 600


def truncate_text(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_cluster_text(cluster: dict) -> str:
    top_terms = ", ".join(cluster.get("top_terms", []))
    representative_docs = cluster.get("representative_docs", [])[:REP_DOC_COUNT]
    representative_docs_text = " ".join(
        truncate_text(doc, REP_DOC_CHAR_LIMIT) for doc in representative_docs
    )

    return (
        f"Cluster name: {cluster.get('name', '')}. "
        f"Top terms: {top_terms}. "
        f"Representative documents: {representative_docs_text}"
    )


def load_cluster_summaries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def build_cap_dataframe() -> pd.DataFrame:
    return pd.DataFrame(CAP_TOPIC_PROTOTYPES)


def rank_cap_candidates(
    cluster_summaries: list[dict],
    cap_df: pd.DataFrame,
    model_name: str,
) -> tuple[list[dict], pd.DataFrame]:
    model = SentenceTransformer(model_name)

    cluster_texts = [build_cluster_text(cluster) for cluster in cluster_summaries]
    cap_texts = cap_df["prototype_text"].tolist()

    cluster_embeddings = model.encode(cluster_texts, convert_to_numpy=True, show_progress_bar=True)
    cap_embeddings = model.encode(cap_texts, convert_to_numpy=True, show_progress_bar=False)

    similarity_matrix = cosine_similarity(cluster_embeddings, cap_embeddings)

    results = []
    review_rows = []

    for cluster, similarity_scores in zip(cluster_summaries, similarity_matrix):
        ranked_indices = similarity_scores.argsort()[::-1][:TOP_K]
        candidates = []

        for rank, cap_idx in enumerate(ranked_indices, start=1):
            cap_row = cap_df.iloc[cap_idx]
            candidates.append(
                {
                    "rank": rank,
                    "cap_topic_id": int(cap_row["topic_id"]),
                    "cap_topic_name": cap_row["name"],
                    "similarity": float(similarity_scores[cap_idx]),
                    "prototype_text": cap_row["prototype_text"],
                }
            )

        result = {
            "cluster_id": int(cluster["topic_id"]),
            "cluster_name": cluster["name"],
            "cluster_count": int(cluster["count"]),
            "top_terms": cluster.get("top_terms", []),
            "representative_docs": cluster.get("representative_docs", [])[:REP_DOC_COUNT],
            "cluster_text": build_cluster_text(cluster),
            "cap_candidates": candidates,
        }
        results.append(result)

        review_row = {
            "cluster_id": int(cluster["topic_id"]),
            "cluster_name": cluster["name"],
            "cluster_count": int(cluster["count"]),
            "top_terms": ", ".join(cluster.get("top_terms", [])),
            "candidate_1_id": candidates[0]["cap_topic_id"],
            "candidate_1_name": candidates[0]["cap_topic_name"],
            "candidate_1_similarity": round(candidates[0]["similarity"], 6),
            "candidate_2_id": candidates[1]["cap_topic_id"],
            "candidate_2_name": candidates[1]["cap_topic_name"],
            "candidate_2_similarity": round(candidates[1]["similarity"], 6),
            "candidate_3_id": candidates[2]["cap_topic_id"],
            "candidate_3_name": candidates[2]["cap_topic_name"],
            "candidate_3_similarity": round(candidates[2]["similarity"], 6),
            "manual_final_topic_id": "",
            "manual_final_topic_name": "",
            "manual_decision": "",
            "notes": "",
        }
        review_rows.append(review_row)

    review_df = pd.DataFrame(review_rows).sort_values(
        by=["candidate_1_similarity", "cluster_count"], ascending=[False, False]
    )
    return results, review_df


def write_outputs(results: list[dict], review_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "cluster_cap_top3_candidates.json"
    csv_path = output_dir / "cluster_cap_top3_candidates_review.csv"

    with json_path.open("w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)

    review_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved ranked CAP candidates to {json_path}")
    print(f"Saved review table to {csv_path}")


def main() -> None:
    cluster_summaries = load_cluster_summaries(CLUSTER_SUMMARY_PATH)
    cap_df = build_cap_dataframe()
    results, review_df = rank_cap_candidates(
        cluster_summaries=cluster_summaries,
        cap_df=cap_df,
        model_name=EMBEDDING_MODEL_NAME,
    )
    write_outputs(results, review_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
