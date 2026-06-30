# Scripts — Narrative Extraction & Topic Tagging

Two lightweight pipelines that turn raw Reddit posts into topic-labeled narrative data.

## Pipeline Overview

```
Hugging Face: arrmlet/political-social-reddit-v1 (streaming)
        │
        ▼
┌──────────────────────────────┐
│  narrative_extraction.py     │  spaCy POS heuristics
└──────────────┬───────────────┘
               │  keeps first-person narrative posts
               ▼
        data/reddit_narratives.csv
               │
               ▼
┌──────────────────────────────┐
│  topic_extraction.py         │  regex multi-label tagging (Polars)
└──────────────┬───────────────┘
               ▼
        data/reddit_narratives_with_topics.csv
```

Run **narrative extraction first**, then **topic extraction** on its output.

---

## 1. `narrative_extraction.py`

### What It Does

1. Streams the `train` split of [`arrmlet/political-social-reddit-v1`](https://huggingface.co/datasets/arrmlet/political-social-reddit-v1) from Hugging Face.
2. Loads spaCy `en_core_web_sm` with NER and parser disabled (tokenizer + POS/morph only).
3. Truncates each `text` field to 5 000 characters before processing.
4. Skips empty strings and literal `[deleted]` / `[removed]` posts.
5. Retains a row only when **both** conditions are met:
   - **Past-tense verb ratio > 0.3** — past-tense verbs (`VBD`, `VBN`) as a fraction of all verbs.
   - **First-person pronoun density > 0.02** — first-person pronouns (`i`, `me`, `my`, `mine`, `we`, `us`, `our`, `ours`) as a fraction of all tokens.
6. Appends three metric columns: `past_verb_ratio`, `first_person_ratio`, `word_count`.
7. Writes results in batches (default 10 000 rows) to `data/reddit_narratives.csv`.

### Configuration (In-Script Constants)

| Constant | Default | Purpose |
|---|---|---|
| `MAX_ROWS_TO_PROCESS` | `1_000_000` | Max upstream rows to read before stopping. |
| `SAVE_BATCH_SIZE` | `10_000` | Rows buffered in memory before flushing to CSV. |
| `text_column` | `"text"` | Name of the body-text column in the dataset. |

No CLI arguments — edit the constants directly if needed.

### Dependencies

Provided by the project-level `requirements.txt`:

```
spacy, datasets, pandas, tqdm
```

First-time setup (spaCy language model):

```bash
python -m spacy download en_core_web_sm
```

### Usage

Run from the project root:

```bash
python scripts/narrative_extraction.py
```

Output: `data/reddit_narratives.csv`

---

## 2. `topic_extraction.py`

### What It Does

1. Reads a local CSV (default: `data/reddit_narratives.csv`) via Polars lazy scan.
2. Cleans the text column:
   - Decodes HTML entities (`&#x200B;`, `&amp;`, `&lt;`, `&gt;`).
   - Strips Reddit quote-line prefixes (`>`).
   - Removes bold markers (`**`).
3. Applies **six** regex-based topic rules for multi-label classification:
   - `healthcare`
   - `economy`
   - `education`
   - `social_civil_rights`
   - `system_law_enforcement`
   - `other_macro_politics`
4. Writes a comma-separated `topics` column (e.g. `economy,education`) to the output CSV.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | `data/reddit_narratives.csv` | Input CSV path. |
| `--output` | `data/reddit_narratives_with_topics.csv` | Output CSV path. |
| `--text-column` | `text` | Name of the body-text column. |
| `--require-topic` | off | If set, drops rows that match no topic. |

### Dependencies

Provided by the project-level `requirements.txt`:

```
polars
```

### Usage

```bash
# Default: read data/reddit_narratives.csv, write data/reddit_narratives_with_topics.csv
python scripts/topic_extraction.py

# Keep only rows with at least one topic
python scripts/topic_extraction.py --require-topic

# Custom input file
python scripts/topic_extraction.py --input path/to/other.csv
```

Output: `data/reddit_narratives_with_topics.csv`

---

## Quick Start (End-to-End)

```bash
# 1. Install dependencies (from project root)
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Extract narrative posts from Hugging Face → data/reddit_narratives.csv
python scripts/narrative_extraction.py

# 3. Tag topics → data/reddit_narratives_with_topics.csv
python scripts/topic_extraction.py --require-topic
```

All output CSVs are written to `data/` (git-ignored).
