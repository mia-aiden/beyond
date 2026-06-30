import json
import re
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fill this in on the server before running.
INPUT_PATH = Path("")

OUTPUT_DIR = PROJECT_ROOT / "data" / "clustering"
TEXT_COLUMN = "text"
MIN_WORD_COUNT = 80
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 120
HDBSCAN_MIN_SAMPLES = 30
TOP_N_WORDS = 10
NR_REPRESENTATIVE_DOCS = 5
RANDOM_STATE = 42


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents(input_path: Path) -> pd.DataFrame:
    if not str(input_path):
        raise ValueError("Please set INPUT_PATH in clustering/cluster_narratives.py before running.")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).map(normalize_text)
    df = df[df[TEXT_COLUMN].str.len() > 0].copy()

    if "word_count" in df.columns:
        df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0).astype(int)
        df = df[df["word_count"] > MIN_WORD_COUNT].copy()
    else:
        df["word_count"] = df[TEXT_COLUMN].str.split().map(len)
        df = df[df["word_count"] > MIN_WORD_COUNT].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check INPUT_PATH and MIN_WORD_COUNT.")

    return df.reset_index(drop=True)


def build_topic_model() -> BERTopic:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2),
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model


def write_outputs(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topics: list[int],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["topic_id"] = topics
    clustered_csv = output_dir / "reddit_narratives_bertopic_assignments.csv"
    df.to_csv(clustered_csv, index=False, encoding="utf-8-sig")

    topic_info = topic_model.get_topic_info()
    topic_info_csv = output_dir / "reddit_narratives_bertopic_topics.csv"
    topic_info.to_csv(topic_info_csv, index=False, encoding="utf-8-sig")

    summary_records = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        keywords = [word for word, _ in topic_model.get_topic(topic_id) or []]
        representative_docs = topic_model.get_representative_docs(topic_id) or []
        summary_records.append(
            {
                "topic_id": topic_id,
                "count": int(row["Count"]),
                "name": row.get("Name", ""),
                "top_terms": keywords,
                "representative_docs": representative_docs[:NR_REPRESENTATIVE_DOCS],
            }
        )

    summary_json = output_dir / "reddit_narratives_bertopic_summary.json"
    with summary_json.open("w", encoding="utf-8") as outfile:
        json.dump(summary_records, outfile, ensure_ascii=False, indent=2)

    model_dir = output_dir / "bertopic_model"
    topic_model.save(model_dir)

    print(f"Processed {len(df)} comments from {INPUT_PATH}")
    print(f"Saved assignments to {clustered_csv}")
    print(f"Saved topic table to {topic_info_csv}")
    print(f"Saved topic summary to {summary_json}")
    print(f"Saved BERTopic model to {model_dir}")


def main() -> None:
    df = load_documents(INPUT_PATH)
    documents = df[TEXT_COLUMN].tolist()

    topic_model = build_topic_model()
    topics, _ = topic_model.fit_transform(documents)

    write_outputs(df, topic_model, topics, OUTPUT_DIR)


if __name__ == "__main__":
    main()
