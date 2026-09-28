#!/usr/bin/env python3
"""Recompute the synthetic-vs-manual embedding audit with MPNet."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from paths import EXPERIMENTS_DIR, STAGE1_TEST, MANUAL_TEST, TRAIN_POOL  # noqa: E402


MODEL_NAME = "all-mpnet-base-v2"
AUTO_TRAIN_PATH = TRAIN_POOL
AUTO_TEST_PATH = STAGE1_TEST
MANUAL_PATH = MANUAL_TEST
OUTPUT_DIR = EXPERIMENTS_DIR / "synthetic_validation_mpnet"

BATCH_SIZE = 128
SEED = 42
TSNE_PERPLEXITY = 30.0
TSNE_MAX_ITER = 1_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_texts(series: pd.Series, source: str) -> tuple[list[str], int]:
    texts = series.fillna("").astype(str).str.strip()
    missing_count = int((texts == "").sum())
    if missing_count:
        print(f"Dropping {missing_count} empty narratives from {source}", flush=True)
    return texts[texts != ""].tolist(), missing_count


def load_narratives() -> tuple[list[str], list[str], dict[str, int]]:
    auto_train = pd.read_csv(AUTO_TRAIN_PATH)
    auto_test = pd.read_csv(AUTO_TEST_PATH)
    manual = pd.read_csv(MANUAL_PATH)

    synthetic, train_missing = clean_texts(
        auto_train["Narrative_Summary"], str(AUTO_TRAIN_PATH)
    )
    test_texts, test_missing = clean_texts(
        auto_test["reference_narrative"], str(AUTO_TEST_PATH)
    )
    synthetic.extend(test_texts)
    manual_texts, manual_missing = clean_texts(
        manual["reference_narrative"], str(MANUAL_PATH)
    )

    if len(synthetic) != 12_007 or len(manual_texts) != 200:
        raise ValueError(
            f"Unexpected corpus sizes: synthetic={len(synthetic)}, manual={len(manual_texts)}"
        )
    return synthetic, manual_texts, {
        "synthetic_input_count": len(auto_train) + len(auto_test),
        "synthetic_missing_narrative_count": train_missing + test_missing,
        "manual_input_count": len(manual),
        "manual_missing_narrative_count": manual_missing,
    }


def encode(
    model: SentenceTransformer,
    texts: list[str],
    label: str,
) -> np.ndarray:
    print(f"Encoding {len(texts):,} {label} narratives with {MODEL_NAME}", flush=True)
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)


def dtw_cosine(
    synthetic_embeddings: np.ndarray,
    manual_embeddings: np.ndarray,
) -> dict[str, float | int]:
    """Match the evaluation metric: cosine cost and (n + m) normalization."""
    n = len(synthetic_embeddings)
    m = len(manual_embeddings)
    previous_cost = np.full(m + 1, np.inf, dtype=np.float64)
    current_cost = np.full(m + 1, np.inf, dtype=np.float64)
    previous_length = np.zeros(m + 1, dtype=np.int32)
    current_length = np.zeros(m + 1, dtype=np.int32)
    previous_cost[0] = 0.0

    for index, embedding in enumerate(synthetic_embeddings, start=1):
        distances = np.clip(1.0 - manual_embeddings @ embedding, 0.0, 2.0)
        current_cost.fill(np.inf)
        current_length.fill(0)
        for j in range(1, m + 1):
            candidates = (
                (previous_cost[j], previous_length[j]),
                (current_cost[j - 1], current_length[j - 1]),
                (previous_cost[j - 1], previous_length[j - 1]),
            )
            best_cost, best_length = min(candidates, key=lambda item: item[0])
            current_cost[j] = best_cost + float(distances[j - 1])
            current_length[j] = best_length + 1
        previous_cost, current_cost = current_cost, previous_cost
        previous_length, current_length = current_length, previous_length
        if index % 1_000 == 0:
            print(f"DTW rows: {index:,}/{n:,}", flush=True)

    raw_distance = float(previous_cost[m])
    path_length = int(previous_length[m])
    return {
        "dtw_distance": raw_distance,
        "normalized_dtw_distance": raw_distance / (n + m),
        "path_normalized_dtw_distance": raw_distance / path_length,
        "dtw_path_length": path_length,
    }


def nearest_neighbor_stats(
    synthetic_embeddings: np.ndarray,
    manual_embeddings: np.ndarray,
) -> dict[str, float]:
    distances = 1.0 - manual_embeddings @ synthetic_embeddings.T
    nearest = np.min(np.clip(distances, 0.0, 2.0), axis=1)
    return {
        "manual_to_synthetic_cosine_distance_mean": float(nearest.mean()),
        "manual_to_synthetic_cosine_distance_median": float(np.median(nearest)),
        "manual_to_synthetic_cosine_distance_p95": float(np.quantile(nearest, 0.95)),
    }


def create_tsne(
    synthetic_embeddings: np.ndarray,
    manual_embeddings: np.ndarray,
) -> pd.DataFrame:
    embeddings = np.vstack([synthetic_embeddings, manual_embeddings])
    print("Reducing embeddings to 50 dimensions with PCA", flush=True)
    reduced = PCA(n_components=50, random_state=SEED).fit_transform(embeddings)
    print("Running t-SNE", flush=True)
    coordinates = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        init="pca",
        learning_rate="auto",
        max_iter=TSNE_MAX_ITER,
        random_state=SEED,
        verbose=1,
    ).fit_transform(reduced)
    groups = np.array(["Synthetic"] * len(synthetic_embeddings) + ["Manual"] * len(manual_embeddings))
    return pd.DataFrame({"tsne_x": coordinates[:, 0], "tsne_y": coordinates[:, 1], "group": groups})


def plot_tsne(frame: pd.DataFrame, output_path: Path) -> None:
    synthetic = frame[frame["group"] == "Synthetic"]
    manual = frame[frame["group"] == "Manual"]
    fig, axis = plt.subplots(figsize=(8, 6), dpi=180)
    axis.scatter(
        synthetic["tsne_x"],
        synthetic["tsne_y"],
        s=6,
        alpha=0.16,
        color="#40637a",
        linewidths=0,
        label="Synthetic",
    )
    axis.scatter(
        manual["tsne_x"],
        manual["tsne_y"],
        s=22,
        alpha=0.8,
        color="#c84b31",
        linewidths=0,
        label="Manual",
    )
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    synthetic, manual, data_audit = load_narratives()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
    synthetic_embeddings = encode(model, synthetic, "synthetic")
    manual_embeddings = encode(model, manual, "manual")

    np.save(OUTPUT_DIR / "synthetic_embeddings.npy", synthetic_embeddings)
    np.save(OUTPUT_DIR / "manual_embeddings.npy", manual_embeddings)

    summary: dict[str, object] = {
        "model_name": MODEL_NAME,
        "device": device,
        "seed": SEED,
        "synthetic_count": len(synthetic),
        "manual_count": len(manual),
        "data_audit": data_audit,
        "embedding_dimension": int(synthetic_embeddings.shape[1]),
        "dtw_local_distance": "1 - cosine_similarity",
        "dtw_normalization": "raw_distance / (synthetic_count + manual_count)",
        "input_files": {
            str(path): {"sha256": sha256(path)}
            for path in (AUTO_TRAIN_PATH, AUTO_TEST_PATH, MANUAL_PATH)
        },
    }
    summary.update(dtw_cosine(synthetic_embeddings, manual_embeddings))
    summary.update(nearest_neighbor_stats(synthetic_embeddings, manual_embeddings))

    tsne = create_tsne(synthetic_embeddings, manual_embeddings)
    tsne.to_csv(OUTPUT_DIR / "tsne_coordinates.csv", index=False)
    plot_tsne(tsne, OUTPUT_DIR / "narrative_tsne_mpnet.png")
    summary["tsne"] = {
        "pca_components": 50,
        "perplexity": TSNE_PERPLEXITY,
        "max_iter": TSNE_MAX_ITER,
        "init": "pca",
        "learning_rate": "auto",
    }

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
