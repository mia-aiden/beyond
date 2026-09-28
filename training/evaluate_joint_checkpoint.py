from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_joint_multitask import (
    DEFAULT_CONFIG,
    PairedDataCollator,
    PairedJointDataset,
    TASK_LABEL,
    TASK_NARRATIVE,
    read_config,
    read_jsonl,
)


def task_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[list[float], list[int]]:
    task_ids = batch.pop("task_ids")
    labels = batch.pop("labels")
    prediction_positions = labels[:, 1:].ne(-100).any(dim=0).nonzero().flatten()
    outputs = model(**batch, logits_to_keep=prediction_positions)
    shift_logits = outputs.logits.contiguous()
    shift_labels = labels[:, prediction_positions + 1].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    valid_tokens = shift_labels.ne(-100)
    per_sample = (token_losses * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1)
    return per_sample.float().cpu().tolist(), task_ids.cpu().tolist()


def evaluate(config: dict[str, Any], adapter: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], local_files_only=True, use_fast=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(Path(config["validation_data_path"]))
    dataset = PairedJointDataset(
        rows,
        tokenizer,
        cutoff_len=int(config["cutoff_len"]),
        max_source_tokens=int(config["max_source_tokens"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=PairedDataCollator(tokenizer.pad_token_id),
        num_workers=int(config["dataloader_num_workers"]),
        pin_memory=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter, local_files_only=True)
    model.eval()

    values = {TASK_LABEL: [], TASK_NARRATIVE: []}
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            losses, task_ids = task_losses(model, batch)
            for loss, task_id in zip(losses, task_ids, strict=True):
                values[task_id].append(loss)

    label_loss = sum(values[TASK_LABEL]) / len(values[TASK_LABEL])
    narrative_loss = sum(values[TASK_NARRATIVE]) / len(values[TASK_NARRATIVE])
    weight = float(config["emotion_loss_weight"])
    return {
        "adapter_path": str(adapter),
        "validation_pairs": len(dataset),
        "emotion_loss_weight": weight,
        "emotion_validation_loss": label_loss,
        "narrative_validation_loss": narrative_loss,
        "weighted_joint_validation_loss": (weight * label_loss + narrative_loss) / (weight + 1.0),
        "loss_definition": "mean per-target-token cross entropy within each task",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report Model C validation loss by task.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(read_config(args.config), args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
