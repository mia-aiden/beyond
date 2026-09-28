import json
import random
from pathlib import Path

DATA = Path("/root/autodl-fs/sft_experiments/data")
CFG = Path("/root/autodl-fs/sft_experiments/configs")
MODEL = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"

SEQ_SYSTEM = (
    "You are an expert political analyst. First classify the author's dominant "
    "Ekman emotion as exactly one of: Anger, Disgust, Fear, Joy, Neutral, Sadness, "
    "Surprise. Then extract the political narrative in neutral third-person prose, "
    "1-3 sentences, using only the source text and preserving the author's "
    "perspective. Output exactly two lines:\nEmotion: <label>\nNarrative: <text>"
)
SEQ_INSTRUCTION = "Classify the author's Ekman emotion, then extract the political narrative."


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(rows, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- shuffle-emotion control: permute the emotion label, keep the true narrative ---
rng = random.Random(1234)
for split in ("train", "validation"):
    jrows = read_jsonl(DATA / f"joint_{split}.jsonl")
    emotions = [r["emotion_label"] for r in jrows]
    permuted = emotions[:]
    rng.shuffle(permuted)
    same = sum(a == b for a, b in zip(emotions, permuted))
    shuf = [
        {
            "system": SEQ_SYSTEM,
            "instruction": SEQ_INSTRUCTION,
            "input": r["source_text"],
            "output": f'Emotion: {permuted[i]}\nNarrative: {r["reference_narrative"]}',
        }
        for i, r in enumerate(jrows)
    ]
    write_jsonl(shuf, DATA / f"seqjoint_shuffle_{split}.jsonl")
    print(f"{split}: {len(shuf)} rows, emotion still-correct after shuffle = {same}/{len(shuf)}")

info = json.loads((DATA / "dataset_info.json").read_text())
cols = {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
for name, fn in [
    ("political_seqjoint_shuffle_train", "seqjoint_shuffle_train.jsonl"),
    ("political_seqjoint_shuffle_validation", "seqjoint_shuffle_validation.jsonl"),
]:
    info[name] = {"file_name": fn, "formatting": "alpaca", "columns": cols}
(DATA / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
print("registered shuffle datasets")


def cfg(dataset, evalset, outdir, seed):
    return f"""### model
model_name_or_path: {MODEL}
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
dataset_dir: {DATA}
dataset: {dataset}
eval_dataset: {evalset}
template: llama3
cutoff_len: 1536
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 4

### output
output_dir: {outdir}
logging_steps: 10
save_strategy: steps
save_steps: 250
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true
report_to: none

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


M = "/root/autodl-fs/sft_experiments/models"
plans = [
    ("seqjoint_shuffle_s42.yaml", "political_seqjoint_shuffle_train", "political_seqjoint_shuffle_validation", f"{M}/seqjoint_shuffle_lora_r8", 42),
    ("seqjoint_s1.yaml", "political_seqjoint_train", "political_seqjoint_validation", f"{M}/seqjoint_lora_r8_s1", 1),
    ("seqjoint_s2.yaml", "political_seqjoint_train", "political_seqjoint_validation", f"{M}/seqjoint_lora_r8_s2", 2),
    ("seqjoint_s3.yaml", "political_seqjoint_train", "political_seqjoint_validation", f"{M}/seqjoint_lora_r8_s3", 3),
]
for fn, ds, ev, out, seed in plans:
    (CFG / fn).write_text(cfg(ds, ev, out, seed))
    print("wrote", CFG / fn, "seed", seed)
