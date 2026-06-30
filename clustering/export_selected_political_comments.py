import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSIGNMENTS_PATH = PROJECT_ROOT / "reddit_narratives_bertopic_assignments_2.csv"
TOPICS_PATH = PROJECT_ROOT / "clustering" / "results_direct_political_topics.json"
OUTPUT_PATH = PROJECT_ROOT / "clustering" / "reddit_narratives_selected_political_comments.csv"


def load_selected_topics(topics_path: Path) -> list[int]:
    with topics_path.open("r", encoding="utf-8") as infile:
        groups = json.load(infile)

    return sorted(set(groups["direct_political_topics"]))


def export_selected_comments(assignments_path: Path, topics_path: Path, output_path: Path) -> None:
    selected_topics = load_selected_topics(topics_path)
    df = pd.read_csv(assignments_path, encoding="utf-8-sig")

    filtered_df = df[df["topic_id"].isin(selected_topics)].copy()
    filtered_df = filtered_df.sort_values(["topic_id"], ascending=[True])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Selected topics: {selected_topics}")
    print(f"Exported {len(filtered_df)} comments to {output_path}")


if __name__ == "__main__":
    export_selected_comments(ASSIGNMENTS_PATH, TOPICS_PATH, OUTPUT_PATH)
