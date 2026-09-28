#!/usr/bin/env python3
"""Prepare the multi-seed format-versus-semantics control experiment."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


DATA_DIR = Path("/root/autodl-fs/sft_experiments/data")
EXPERIMENT_DIR = Path("/root/autodl-fs/sft_experiments/format_semantics_multiseed")
CONFIG_DIR = EXPERIMENT_DIR / "configs"
MODEL_DIR = EXPERIMENT_DIR / "models"
BASE_MODEL = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
SEEDS = (42, 1, 2, 3, 4)

SEQ_SYSTEM = (
    "You are an expert political analyst. First classify the author's dominant "
    "Ekman emotion as exactly one of: Anger, Disgust, Fear, Joy, Neutral, Sadness, "
    "Surprise. Then extract the political narrative in neutral third-person prose, "
    "1-3 sentences, using only the source text and preserving the author's "
    "perspective. Output exactly two lines:\nEmotion: <label>\nNarrative: <text>"
)
SEQ_INSTRUCTION = "Classify the author's Ekman emotion, then extract the political narrative."

MARKER_SYSTEM = (
    "You are an expert political analyst. First output the fixed non-semantic marker X. "
    "Then extract the political narrative in neutral third-person prose, 1-3 sentences, "
    "using only the source text and preserving the author's perspective. The marker is "
    "fixed and carries no "
    "semantic information. It is not inferred from the source and exists only to preserve "
    "a two-field response structure. Output exactly two lines:\nMarker: X\nNarrative: <text>"
)
MARKER_INSTRUCTION = "Output the fixed marker X, then extract the political narrative."

CONDITIONS = ("true", "forced_wrong", "constant_neutral", "marker_control")
LEGACY_TRUE_MODELS = {
    42: Path("/root/autodl-fs/sft_experiments/models/seqjoint_lora_r8"),
    1: Path("/root/autodl-fs/sft_experiments/models/seqjoint_lora_r8_s1"),
    2: Path("/root/autodl-fs/sft_experiments/models/seqjoint_lora_r8_s2"),
    3: Path("/root/autodl-fs/sft_experiments/models/seqjoint_lora_r8_s3"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def mutual_information(labels: list[str], assigned: list[str]) -> tuple[float, float]:
    count = len(labels)
    left_counts = Counter(labels)
    right_counts = Counter(assigned)
    pair_counts = Counter(zip(labels, assigned))
    information = 0.0
    for (left, right), pair_count in pair_counts.items():
        p_pair = pair_count / count
        information += p_pair * math.log(
            p_pair / ((left_counts[left] / count) * (right_counts[right] / count))
        )
    left_entropy = -sum((value / count) * math.log(value / count) for value in left_counts.values())
    right_entropy = -sum((value / count) * math.log(value / count) for value in right_counts.values())
    normalized = information / math.sqrt(left_entropy * right_entropy)
    return information, normalized


def minimum_forced_wrong_information(labels: list[str]) -> tuple[float, float]:
    """Continuous minimum MI under equal marginals and a zero diagonal."""
    counts = Counter(labels)
    names = sorted(counts)
    probabilities = [counts[name] / len(labels) for name in names]
    size = len(names)
    matrix = [[0.0 if row == column else 1.0 for column in range(size)] for row in range(size)]
    total = sum(sum(row) for row in matrix)
    matrix = [[value / total for value in row] for row in matrix]
    for _ in range(10_000):
        previous = [row[:] for row in matrix]
        for row in range(size):
            scale = probabilities[row] / sum(matrix[row])
            matrix[row] = [value * scale for value in matrix[row]]
        for column in range(size):
            column_sum = sum(matrix[row][column] for row in range(size))
            scale = probabilities[column] / column_sum
            for row in range(size):
                matrix[row][column] *= scale
        if max(
            abs(matrix[row][column] - previous[row][column])
            for row in range(size)
            for column in range(size)
        ) < 1e-14:
            break
    information = sum(
        value * math.log(value / (probabilities[row] * probabilities[column]))
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
        if value > 0
    )
    entropy = -sum(value * math.log(value) for value in probabilities)
    return information, information / entropy


def _random_derangement(labels: list[str], rng: random.Random) -> list[str]:
    assigned = labels[:]
    rng.shuffle(assigned)
    conflicts = [index for index, (left, right) in enumerate(zip(labels, assigned)) if left == right]
    rng.shuffle(conflicts)
    all_indices = list(range(len(labels)))
    for index in conflicts:
        if assigned[index] != labels[index]:
            continue
        replacement_index = None
        for _ in range(100):
            candidate = rng.randrange(len(labels))
            if labels[candidate] != labels[index] and assigned[candidate] != labels[index]:
                replacement_index = candidate
                break
        if replacement_index is None:
            rng.shuffle(all_indices)
            replacement_index = next(
                candidate
                for candidate in all_indices
                if labels[candidate] != labels[index] and assigned[candidate] != labels[index]
            )
        assigned[index], assigned[replacement_index] = (
            assigned[replacement_index],
            assigned[index],
        )
    return assigned


def forced_wrong_labels(labels: list[str], seed: int, candidates: int = 128) -> list[str]:
    """Choose a low-MI random zero-match derangement with exact marginals."""
    counts = Counter(labels)
    if max(counts.values()) * 2 > len(labels):
        raise ValueError("An exact zero-match label derangement is impossible for this split.")

    rng = random.Random(seed)
    generated = [_random_derangement(labels, rng) for _ in range(candidates)]
    result = min(generated, key=lambda values: mutual_information(labels, values)[0])
    assert Counter(result) == counts
    assert all(original != replacement for original, replacement in zip(labels, result))
    return result


def build_rows(source_rows: list[dict], condition: str, split_seed: int) -> list[dict]:
    labels = [str(row["emotion_label"]).strip().title() for row in source_rows]
    replacements = forced_wrong_labels(labels, split_seed) if condition == "forced_wrong" else labels
    output_rows = []
    for row, label in zip(source_rows, replacements):
        narrative = str(row["reference_narrative"]).strip()
        if condition == "marker_control":
            system = MARKER_SYSTEM
            instruction = MARKER_INSTRUCTION
            response = f"Marker: X\nNarrative: {narrative}"
        else:
            system = SEQ_SYSTEM
            instruction = SEQ_INSTRUCTION
            emitted_label = "Neutral" if condition == "constant_neutral" else label
            response = f"Emotion: {emitted_label}\nNarrative: {narrative}"
        output_rows.append(
            {
                "system": system,
                "instruction": instruction,
                "input": row["source_text"],
                "output": response,
            }
        )
    return output_rows


def training_config(
    dataset: str,
    eval_dataset: str,
    output_dir: Path,
    seed: int,
    resume_checkpoint: Path | None = None,
) -> str:
    resume_line = f"resume_from_checkpoint: {resume_checkpoint}\n" if resume_checkpoint else ""
    overwrite = "false" if resume_checkpoint else "true"
    return f"""### model
model_name_or_path: {BASE_MODEL}
trust_remote_code: false

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all

### dataset
dataset_dir: {DATA_DIR}
dataset: {dataset}
eval_dataset: {eval_dataset}
template: llama3
cutoff_len: 1536
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 4

### output
output_dir: {output_dir}
logging_steps: 10
save_strategy: steps
save_steps: 250
save_total_limit: 2
plot_loss: true
overwrite_output_dir: {overwrite}
{resume_line}report_to: none

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
per_device_eval_batch_size: 1
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
eval_strategy: steps
eval_steps: 250
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
seed: {seed}
data_seed: {seed}
ddp_timeout: 180000000
"""


def prepare_data() -> dict:
    audit: dict[str, dict] = {}
    source_by_split = {
        split: read_jsonl(DATA_DIR / f"joint_{split}.jsonl")
        for split in ("train", "validation")
    }
    dataset_info_path = DATA_DIR / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    columns = {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}

    for condition in CONDITIONS:
        audit[condition] = {}
        for split, source_rows in source_by_split.items():
            rows = build_rows(source_rows, condition, 20260819 + (0 if split == "train" else 1))
            file_name = f"format_semantics_{condition}_{split}.jsonl"
            write_jsonl(rows, DATA_DIR / file_name)
            dataset_name = f"political_format_semantics_{condition}_{split}"
            dataset_info[dataset_name] = {
                "file_name": file_name,
                "formatting": "alpaca",
                "columns": columns,
            }
            audit[condition][split] = {
                "rows": len(rows),
                "source_sha256": stable_hash([str(row["input"]) for row in rows]),
                "narrative_sha256": stable_hash(
                    [str(row["output"]).split("Narrative: ", 1)[1] for row in rows]
                ),
            }

    for split, source_rows in source_by_split.items():
        true_labels = [str(row["emotion_label"]).strip().title() for row in source_rows]
        wrong_rows = read_jsonl(DATA_DIR / f"format_semantics_forced_wrong_{split}.jsonl")
        wrong_labels = [row["output"].splitlines()[0].split(":", 1)[1].strip() for row in wrong_rows]
        audit["forced_wrong"][split]["true_label_counts"] = dict(sorted(Counter(true_labels).items()))
        audit["forced_wrong"][split]["assigned_label_counts"] = dict(sorted(Counter(wrong_labels).items()))
        audit["forced_wrong"][split]["label_matches"] = sum(
            left == right for left, right in zip(true_labels, wrong_labels)
        )
        information, normalized_information = mutual_information(true_labels, wrong_labels)
        minimum_information, minimum_normalized_information = minimum_forced_wrong_information(
            true_labels
        )
        audit["forced_wrong"][split]["mutual_information_nats"] = information
        audit["forced_wrong"][split]["normalized_mutual_information"] = normalized_information
        audit["forced_wrong"][split][
            "continuous_minimum_mutual_information_nats"
        ] = minimum_information
        audit["forced_wrong"][split][
            "continuous_minimum_normalized_mutual_information"
        ] = minimum_normalized_information
        audit["forced_wrong"][split]["mi_above_continuous_minimum"] = (
            information - minimum_information
        )
        audit["forced_wrong"][split]["contingency_counts"] = {
            true_label: {
                assigned_label: sum(
                    left == true_label and right == assigned_label
                    for left, right in zip(true_labels, wrong_labels)
                )
                for assigned_label in sorted(set(wrong_labels))
            }
            for true_label in sorted(set(true_labels))
        }

    reference_hashes = {
        split: {
            key: audit[condition][split][key]
            for key in ("source_sha256", "narrative_sha256")
        }
        for split in ("train", "validation")
        for condition in (CONDITIONS[0],)
    }
    for condition in CONDITIONS[1:]:
        for split in ("train", "validation"):
            for key, expected in reference_hashes[split].items():
                assert audit[condition][split][key] == expected

    dataset_info_path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def prepare_configs() -> list[dict]:
    manifest = []
    for condition in CONDITIONS:
        if condition == "true":
            # Reuse the exact dataset registration used by the existing checkpoints.
            dataset = "political_seqjoint_train"
            eval_dataset = "political_seqjoint_validation"
        else:
            dataset = f"political_format_semantics_{condition}_train"
            eval_dataset = f"political_format_semantics_{condition}_validation"
        for seed in SEEDS:
            model_path = MODEL_DIR / condition / f"seed_{seed}"
            action = "train"
            resume_checkpoint = None
            if condition == "true" and seed in (42, 1):
                model_path = LEGACY_TRUE_MODELS[seed]
                action = "reuse"
            elif condition == "true" and seed in (2, 3):
                model_path = LEGACY_TRUE_MODELS[seed]
                resume_checkpoint = model_path / "checkpoint-250"
                action = "resume"

            config_path = CONFIG_DIR / f"{condition}_seed_{seed}.yaml"
            if action != "reuse":
                config_path.write_text(
                    training_config(dataset, eval_dataset, model_path, seed, resume_checkpoint),
                    encoding="utf-8",
                )
            manifest.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "action": action,
                    "config_path": str(config_path) if action != "reuse" else None,
                    "model_path": str(model_path),
                    "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                    "status": "reused" if action == "reuse" else "pending",
                }
            )
    return manifest


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    audit = prepare_data()
    manifest = prepare_configs()
    (EXPERIMENT_DIR / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (EXPERIMENT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(manifest)} experiment cells in {EXPERIMENT_DIR}")
    print("Training jobs:", sum(row["action"] != "reuse" for row in manifest))


if __name__ == "__main__":
    main()
