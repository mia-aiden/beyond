# Beyond Political Stances

Code for the master's project *Beyond Political Stances*: a corpus of Reddit
political comments that contain personal narratives, silver-standard Ekman
emotion labels and narrative summaries, and LoRA fine-tuning of
Llama-3-8B-Instruct for emotion classification and narrative extraction, both
separately and jointly.

| Directory | Contents |
|---|---|
| `clustering/` | BERTopic clustering and mapping of the clusters to CAP major topics |
| `training/` | training data, Model A/B/C and Seq-joint training, vLLM and Transformers inference |
| `evaluation/` | evaluation runner and metrics (accuracy, macro-F1, BLEU, ROUGE, BERTScore, DTW) |
| `analysis/` | the statistical analyses reported in the thesis, including the 984-comment Stage1 filter |
| `scripts/` | label remapping helper |

Raw Reddit data, the annotated datasets, adapters, and evaluation outputs are
not included.

## Topic Selection

Topic selection starts from `reddit_narratives.csv`, which already contains comments that passed our narrative heuristic filter. We then sequentially scan this file, keep comments with `word_count > 80`, and take the first 50,000 comments as the working sample for topic discovery.

We run BERTopic on this 50,000-comment sample to obtain semantic clusters. Since the resulting clusters are not purely political, we do not use BERTopic labels directly as final topic labels. Instead, we compare each cluster against Comparative Agendas Project (CAP) major-topic prototypes. Each CAP prototype is written as a short semantic description built from the main topic and its representative subtopics.

For each BERTopic cluster, we construct a cluster text by combining its topic name, top terms, and representative documents. We encode both the cluster texts and the CAP prototypes with the same sentence-transformer model, compute cosine similarity, and keep the top 3 CAP candidates for each cluster. We then manually review these candidates and group clusters into `direct_political`, `borderline`, and `non_political`.

The final export (`clustering/export_selected_political_comments.py`) keeps the 33 clusters labeled `direct_political` out of the 71 non-noise clusters of the second BERTopic run. This yields `12,193` comments in `clustering/reddit_narratives_selected_political_comments.csv`, the corpus used for annotation and modeling; it includes the 200 comments of the Manual Gold Standard.

## Evaluation

The `evaluation/` subsystem evaluates political narrative generation and
seven-class Ekman emotion prediction. It accepts an existing prediction CSV,
local Transformers inference, or structured-output inference through a deployed
vLLM server.

See `evaluation/README.md` for data schemas, configuration, and commands.

## Training and analysis

`training/README.md` covers data preparation, training, and inference for all
models, and `analysis/README.md` maps each table of the thesis to the script
that produces it. Server paths are set in one place, `training/paths.py` and
`training/paths.sh`, and can be overridden with environment variables.
