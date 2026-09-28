from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "joint_sft.yaml"
TASK_LABEL = 0
TASK_NARRATIVE = 1


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as infile:
        return yaml.safe_load(infile)


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "source_text",
                "emotion_label",
                "reference_narrative",
                "label_system",
                "label_instruction",
                "narrative_system",
                "narrative_instruction",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def truncate_source(text: str, tokenizer: Any, max_source_tokens: int) -> tuple[str, bool]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_source_tokens:
        return text, False
    return tokenizer.decode(token_ids[:max_source_tokens], skip_special_tokens=True), True


def encode_task(
    *,
    tokenizer: Any,
    system_prompt: str,
    instruction: str,
    source_text: str,
    target: str,
    task_id: int,
    cutoff_len: int,
) -> dict[str, Any]:
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n{source_text}"},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": target}]
    prompt_encoding = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_encoding = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = prompt_encoding["input_ids"]
    full_ids = full_encoding["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("The tokenizer chat template does not preserve the prompt prefix.")

    input_ids = full_ids[:cutoff_len]
    labels = [-100] * min(len(prompt_ids), len(input_ids))
    labels.extend(input_ids[len(labels):])
    target_tokens = sum(label != -100 for label in labels)
    if target_tokens == 0:
        raise ValueError("Encoded sample has no trainable target tokens.")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "task_id": task_id,
        "target_tokens": target_tokens,
        "sequence_tokens": len(input_ids),
    }


class PairedJointDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer: Any,
        *,
        cutoff_len: int,
        max_source_tokens: int,
    ) -> None:
        self.pairs = []
        self.truncated_sources = 0
        target_counts = {TASK_LABEL: [], TASK_NARRATIVE: []}
        sequence_counts = {TASK_LABEL: [], TASK_NARRATIVE: []}

        for row in rows:
            source_text, was_truncated = truncate_source(
                row["source_text"], tokenizer, max_source_tokens
            )
            self.truncated_sources += int(was_truncated)
            label_example = encode_task(
                tokenizer=tokenizer,
                system_prompt=row["label_system"],
                instruction=row["label_instruction"],
                source_text=source_text,
                target=row["emotion_label"],
                task_id=TASK_LABEL,
                cutoff_len=cutoff_len,
            )
            narrative_example = encode_task(
                tokenizer=tokenizer,
                system_prompt=row["narrative_system"],
                instruction=row["narrative_instruction"],
                source_text=source_text,
                target=row["reference_narrative"],
                task_id=TASK_NARRATIVE,
                cutoff_len=cutoff_len,
            )
            self.pairs.append((label_example, narrative_example))
            for example in (label_example, narrative_example):
                target_counts[example["task_id"]].append(example["target_tokens"])
                sequence_counts[example["task_id"]].append(example["sequence_tokens"])

        self.summary = {
            "paired_rows": len(self.pairs),
            "actual_sequences_per_epoch": len(self.pairs) * 2,
            "truncated_sources": self.truncated_sources,
            "label_target_tokens_mean": sum(target_counts[TASK_LABEL]) / len(self.pairs),
            "narrative_target_tokens_mean": sum(target_counts[TASK_NARRATIVE]) / len(self.pairs),
            "label_sequence_tokens_max": max(sequence_counts[TASK_LABEL]),
            "narrative_sequence_tokens_max": max(sequence_counts[TASK_NARRATIVE]),
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.pairs[index]


@dataclass
class PairedDataCollator:
    pad_token_id: int

    def __call__(
        self,
        features: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> dict[str, torch.Tensor]:
        examples = [example for pair in features for example in pair]
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_masks = []
        labels = []
        task_ids = []
        for example in examples:
            padding = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(example["attention_mask"] + [0] * padding)
            labels.append(example["labels"] + [-100] * padding)
            task_ids.append(example["task_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
        }


class WeightedMultiTaskTrainer(Trainer):
    def __init__(self, *args: Any, emotion_loss_weight: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if emotion_loss_weight < 0:
            raise ValueError("emotion_loss_weight must be non-negative.")
        self.emotion_loss_weight = emotion_loss_weight
        # This custom loss is already balanced by task, not by Trainer's global
        # num_items_in_batch. Let training_step apply gradient-accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        task_ids = inputs.pop("task_ids")
        labels = inputs.pop("labels")
        prediction_positions = labels[:, 1:].ne(-100).any(dim=0).nonzero().flatten()
        if prediction_positions.numel() == 0:
            raise ValueError("Batch has no trainable target tokens.")
        outputs = model(**inputs, logits_to_keep=prediction_positions)
        shift_logits = outputs.logits.contiguous()
        shift_labels = labels[:, prediction_positions + 1].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        valid_tokens = shift_labels.ne(-100)
        per_sample_loss = (token_losses * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1)

        label_loss = per_sample_loss[task_ids.eq(TASK_LABEL)].mean()
        narrative_loss = per_sample_loss[task_ids.eq(TASK_NARRATIVE)].mean()
        loss = (
            self.emotion_loss_weight * label_loss + narrative_loss
        ) / (self.emotion_loss_weight + 1.0)
        return (loss, outputs) if return_outputs else loss


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model C with a weighted joint objective.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--longest-first", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        local_files_only=True,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(Path(config["train_data_path"]))
    validation_rows = read_jsonl(Path(config["validation_data_path"]))
    if args.longest_first:
        train_rows.sort(key=lambda row: len(row["source_text"]), reverse=True)
        validation_rows.sort(key=lambda row: len(row["source_text"]), reverse=True)
    if args.limit_rows is not None:
        train_rows = train_rows[: args.limit_rows]
        validation_rows = validation_rows[: max(2, min(args.limit_rows, len(validation_rows)))]

    train_dataset = PairedJointDataset(
        train_rows,
        tokenizer,
        cutoff_len=int(config["cutoff_len"]),
        max_source_tokens=int(config["max_source_tokens"]),
    )
    validation_dataset = PairedJointDataset(
        validation_rows,
        tokenizer,
        cutoff_len=int(config["cutoff_len"]),
        max_source_tokens=int(config["max_source_tokens"]),
    )
    dataset_audit = {
        "train_dataset": train_dataset.summary,
        "validation_dataset": validation_dataset.summary,
    }
    audit_path = output_dir / "joint_dataset_audit.json"
    audit_path.write_text(
        json.dumps(dataset_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dataset_audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        return

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=list(config["lora_target"]),
            bias="none",
        ),
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    max_steps = args.max_steps if args.max_steps is not None else -1
    train_batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else int(config["per_device_train_batch_size"])
    )
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else int(config["gradient_accumulation_steps"])
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_dataset)
        / train_batch_size
        / gradient_accumulation_steps
    )
    planned_optimizer_steps = (
        max_steps
        if max_steps > 0
        else math.ceil(optimizer_steps_per_epoch * float(config["num_train_epochs"]))
    )
    warmup_steps = round(planned_optimizer_steps * float(config["warmup_ratio"]))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        max_steps=max_steps,
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        warmup_steps=warmup_steps,
        weight_decay=float(config["weight_decay"]),
        max_grad_norm=float(config["max_grad_norm"]),
        bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        eval_strategy="steps",
        eval_steps=int(config["eval_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        load_best_model_at_end=max_steps < 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(config["logging_steps"]),
        logging_first_step=True,
        report_to="none",
        seed=int(config["seed"]),
        data_seed=int(config["data_seed"]),
        dataloader_num_workers=int(config["dataloader_num_workers"]),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        prediction_loss_only=True,
        ddp_timeout=180000000,
    )
    trainer = WeightedMultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=PairedDataCollator(tokenizer.pad_token_id),
        emotion_loss_weight=float(config["emotion_loss_weight"]),
    )

    resume_checkpoint = None
    if bool(config.get("resume_from_checkpoint", True)) and max_steps < 0:
        resume_checkpoint = find_latest_checkpoint(output_dir)
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    summary = {
        "method": "paired weighted multi-task causal language modeling",
        "joint_loss": "(emotion_loss_weight * mean_label_token_loss + mean_narrative_token_loss) / (emotion_loss_weight + 1)",
        "emotion_loss_weight": float(config["emotion_loss_weight"]),
        "train_dataset": train_dataset.summary,
        "validation_dataset": validation_dataset.summary,
        "actual_sequences_per_optimizer_step": (
            train_batch_size
            * gradient_accumulation_steps
            * 2
        ),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "train_metrics": train_result.metrics,
        "config": config,
    }
    (output_dir / "joint_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(args.config, output_dir / "joint_sft.yaml")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
