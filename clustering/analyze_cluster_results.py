import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "clustering" / "analysis"

DATASETS = [
    {
        "name": "dataset1",
        "results_json": PROJECT_ROOT / "clustering" / "cluster_cap_top3_candidates.json",
        "assignments_csv": PROJECT_ROOT / "clustering" / "reddit_narratives_bertopic_assignments.csv",
        "political_topics_json": PROJECT_ROOT / "clustering" / "political_cluster_groups.json",
        "political_topics_key": "direct_political",
    },
    {
        "name": "dataset2",
        "results_json": PROJECT_ROOT / "results.json",
        "assignments_csv": PROJECT_ROOT / "reddit_narratives_bertopic_assignments_2.csv",
        "political_topics_json": PROJECT_ROOT / "clustering" / "results_direct_political_topics.json",
        "political_topics_key": "direct_political_topics",
    },
]


def load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def compute_average_top3_similarity(results: list[dict]) -> tuple[float, pd.DataFrame]:
    rows = []
    for item in results:
        similarities = [candidate["similarity"] for candidate in item.get("cap_candidates", [])[:3]]
        if not similarities:
            continue
        rows.append(
            {
                "cluster_id": item["cluster_id"],
                "cluster_name": item["cluster_name"],
                "cluster_count": item["cluster_count"],
                "avg_top3_similarity": sum(similarities) / len(similarities),
                "top1_similarity": similarities[0],
            }
        )

    df = pd.DataFrame(rows)
    overall_avg = float(df["avg_top3_similarity"].mean()) if not df.empty else float("nan")
    return overall_avg, df


def load_political_topics(path: Path, key: str) -> list[int]:
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    return list(data[key])


def plot_political_cluster_sizes(cluster_df: pd.DataFrame, output_path: Path, title: str) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = cluster_df.sort_values("cluster_count", ascending=False).copy()
    plot_df["cluster_id_label"] = plot_df["cluster_id"].astype(str)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_df, x="cluster_id_label", y="cluster_count", color="#4C78A8")
    plt.title(title)
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of comments")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def analyze_dataset(
    name: str,
    results_json: Path,
    assignments_csv: Path,
    political_topics_json: Path,
    political_topics_key: str,
) -> dict:
    results = load_results(results_json)
    overall_avg, cluster_df = compute_average_top3_similarity(results)
    political_topics = load_political_topics(political_topics_json, political_topics_key)
    political_cluster_df = cluster_df[cluster_df["cluster_id"].isin(political_topics)].copy()

    output_subdir = OUTPUT_DIR / name
    output_subdir.mkdir(parents=True, exist_ok=True)

    cluster_df.to_csv(output_subdir / "cluster_similarity_summary.csv", index=False, encoding="utf-8-sig")
    political_cluster_df.to_csv(
        output_subdir / "political_cluster_similarity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_political_cluster_sizes(
        cluster_df=political_cluster_df,
        output_path=output_subdir / "political_cluster_size_ranked.png",
        title=f"{name}: political clusters ranked by number of comments",
    )

    assignments_df = pd.read_csv(assignments_csv, encoding="utf-8-sig")
    assignment_counts = (
        assignments_df["topic_id"]
        .value_counts()
        .rename_axis("topic_id")
        .reset_index(name="comment_count")
        .sort_values("topic_id")
    )
    assignment_counts.to_csv(output_subdir / "assignment_topic_counts.csv", index=False, encoding="utf-8-sig")

    return {
        "dataset": name,
        "num_clusters_in_results": len(cluster_df),
        "num_political_clusters": len(political_cluster_df),
        "average_top3_cosine_similarity": overall_avg,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for dataset in DATASETS:
        summary_rows.append(
            analyze_dataset(
                name=dataset["name"],
                results_json=dataset["results_json"],
                assignments_csv=dataset["assignments_csv"],
                political_topics_json=dataset["political_topics_json"],
                political_topics_key=dataset["political_topics_key"],
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
