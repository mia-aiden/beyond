from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from paths import PROJECT_ROOT, EXPERIMENTS_DIR


DEFAULT_ROOT = EXPERIMENTS_DIR
DEFAULT_CONFIG = PROJECT_ROOT / "training" / "configs" / "joint_sft.yaml"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(metric: str, value: float) -> str:
    return f"{value:.3f}" if metric == "bleu_corpus" else f"{value:.4f}"


def build_report(root: Path, config_path: Path) -> str:
    split = load_json(root / "data" / "joint_split_summary.json")
    model_dir = root / "models" / "joint_lora_r8"
    audit = load_json(model_dir / "joint_dataset_audit.json")
    finalization = load_json(model_dir / "joint_finalization_summary.json")
    components = load_json(model_dir / "joint_validation_components.json")
    ga_audit = load_json(root / "logs" / "joint_gradient_accumulation_audit.json")
    comparison = load_json(root / "comparison_model_abc" / "model_abc_comparison.json")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    train_split, validation_split = split["splits"]
    train_audit = audit["train_dataset"]
    validation_audit = audit["validation_dataset"]
    history = finalization["validation_history"]
    rows = comparison["rows"]

    lines = [
        "# Model C 联合多任务 SFT 实验报告",
        "",
        "## 1. 实验目标",
        "",
        "Model C 使用同一个 Llama 3 8B LoRA adapter 同时学习 Ekman 七分类和政治 narrative extraction。",
        "每个 source 构造成一条 emotion 序列和一条 narrative 序列，并在同一 paired batch 中训练。",
        "",
        "联合目标为：",
        "",
        "```text",
        "L_joint = (lambda * mean(L_emotion) + mean(L_narrative)) / (lambda + 1)",
        "```",
        "",
        "两项 loss 都先按每条样本的目标 token 数归一化，本实验 `lambda = "
        f"{config['emotion_loss_weight']}`。",
        "",
        "## 2. 数据处理与隔离",
        "",
        f"- Train：{train_split['paired_rows']:,} pairs，{train_audit['actual_sequences_per_epoch']:,} task sequences/epoch。",
        f"- Validation：{validation_split['paired_rows']:,} pairs。",
        "- Stage1 test：1,000 条；Manual test：200 条。",
        "- 两个测试集均未参与训练、验证或 checkpoint 选择。",
        "- 对规范化 source text 的独立检查确认 train、validation 和两个 test 集之间交集均为 0。",
        f"- 训练集中有 {train_split['label_rows_without_narrative']} 条 label-only 样本因 narrative 为空而未进入 paired 数据。",
        "",
        "目标长度审计：",
        "",
        "| Split | Emotion target mean | Narrative target mean | Truncated sources |",
        "|---|---:|---:|---:|",
        f"| Train | {train_audit['label_target_tokens_mean']:.2f} | {train_audit['narrative_target_tokens_mean']:.2f} | {train_audit['truncated_sources']} |",
        f"| Validation | {validation_audit['label_target_tokens_mean']:.2f} | {validation_audit['narrative_target_tokens_mean']:.2f} | {validation_audit['truncated_sources']} |",
        "",
        "## 3. 训练配置",
        "",
        f"- Base model：`{config['model_name_or_path']}`",
        f"- LoRA：rank {config['lora_rank']}，alpha {config['lora_alpha']}，dropout {config['lora_dropout']}，覆盖 7 个 projection modules。",
        f"- Paired batch size：{config['per_device_train_batch_size']}，每个 pair 实际展开为 2 条 task sequences。",
        f"- Gradient accumulation：{config['gradient_accumulation_steps']}，每次 optimizer step 实际处理 16 条 task sequences。",
        f"- Learning rate：{config['learning_rate']}，{config['lr_scheduler_type']} scheduler，warmup ratio {config['warmup_ratio']}。",
        f"- Precision：BF16；cutoff length：{config['cutoff_len']}；max source tokens：{config['max_source_tokens']}。",
        "- Hardware：NVIDIA RTX 3090 24 GB。",
        "",
        "梯度累积审计发现首个 run 的 custom loss 未向 Transformers 5.8 声明由 Trainer 负责 GA 归一化。",
        f"该 run 已归档为 `{ga_audit['invalid_run_path']}` 并排除；修复后从 base model 重新训练。",
        f"前 250 steps 的平均 logged loss 从 {ga_audit['invalid_first_250_logged_loss_mean']:.3f} 降至 "
        f"{ga_audit['fixed_first_250_logged_loss_mean']:.3f}，平均 grad norm 从 "
        f"{ga_audit['invalid_first_250_grad_norm_mean']:.3f} 降至 "
        f"{ga_audit['fixed_first_250_grad_norm_mean']:.3f}，均验证了约 8 倍的缩放修正。",
        "",
        "## 4. Checkpoint 选择",
        "",
        "checkpoint 仅按 validation weighted joint loss 选择，测试指标不参与选择。",
        "",
        "| Step | Epoch | Validation loss |",
        "|---:|---:|---:|",
    ]
    for entry in history:
        lines.append(
            f"| {entry['step']} | {float(entry['epoch']):.3f} | {entry['eval_loss']:.6f} |"
        )

    lines.extend(
        [
            "",
            f"最终选择 step **{finalization['selected_step']}**，validation loss "
            f"**{finalization['selected_validation_loss']:.6f}**。",
            "",
            "最终 checkpoint 的分任务复算结果：",
            "",
            "| Emotion validation loss | Narrative validation loss | Weighted joint loss |",
            "|---:|---:|---:|",
            f"| {components['emotion_validation_loss']:.6f} | {components['narrative_validation_loss']:.6f} | {components['weighted_joint_validation_loss']:.6f} |",
            "",
            "## 5. 测试方法",
            "",
            "Model C 与 A/B specialist adapter 均使用 Transformers + PEFT direct inference、相同 prompts、greedy decoding、source truncation 和 gold files。",
            "Emotion 使用 constrained decoding，只允许七个 Ekman 标签；narrative 直接生成纯文本。",
            "Emotion 报告 Accuracy 和 Macro-F1；narrative 报告 BLEU、ROUGE、BERTScore F1 和 normalized DTW distance。",
            "",
            "## 6. Model A/B/C 结果",
            "",
            "正的 improvement 始终表示 Model C 更好；normalized DTW 越低越好，因此差值方向已反转。",
            "",
            "| Task | Dataset | Metric | Specialist | Specialist value | Model C | C improvement |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['dataset']} | {row['metric']} | "
            f"{row['specialist_model']} | {format_metric(row['metric'], row['specialist_value'])} | "
            f"{format_metric(row['metric'], row['model_c_value'])} | "
            f"{row['model_c_improvement']:+.4f} |"
        )

    improved = sum(float(row["model_c_improvement"]) > 0 for row in rows)
    tied = sum(abs(float(row["model_c_improvement"])) < 1e-12 for row in rows)
    lines.extend(
        [
            "",
            "## 7. 结论与限制",
            "",
            f"Model C 在 {len(rows)} 个 task-dataset-metric 对比中改善 {improved} 项，持平 {tied} 项。",
            "是否支持 joint learning hypothesis 应分别观察 A vs C 的 emotion 指标和 B vs C 的 narrative 指标，不能跨任务合并为单一总分。",
            "Manual test 仅 200 条，且 Disgust 等类别 support 很小，类别级结果应谨慎解释。",
            "当前结果只对应 lambda=1 的单次 seed=42 实验；正式论文若要论证稳定性，应追加不同随机种子或 lambda ablation。",
            "",
            "## 8. 产物位置",
            "",
            f"- Final adapter：`{finalization['adapter_path']}`",
            f"- Adapter SHA-256：`{finalization['adapter_sha256']}`",
            f"- Model C evaluation：`{root / 'evaluation_joint'}`",
            f"- A/B/C comparison：`{root / 'comparison_model_abc'}`",
            f"- Training log：`{root / 'logs' / 'joint_sft.log'}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final Model C experiment report.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "MODEL_C_EXPERIMENT_REPORT.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
