import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "narratives.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cluster_2.csv"
SAMPLE_SIZE = 50000
MIN_WORD_COUNT = 80


def main(input_path, output_path) -> None:
    selected_rows: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header found in {input_path}")

        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue

            try:
                word_count = int(float(row.get("word_count") or 0))
            except ValueError:
                continue
            if word_count <= MIN_WORD_COUNT:
                continue

            selected_rows.append(row)
            if len(selected_rows) >= SAMPLE_SIZE:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Selected {len(selected_rows)} rows from {input_path}")
    print(f"Rule: word_count > {MIN_WORD_COUNT}, sequential scan")
    print(f"Saved sample to {output_path}")


if __name__ == "__main__":
    main(INPUT_PATH, OUTPUT_PATH)
