# Two-task SFT

This directory fine-tunes separate LoRA rank-8 adapters for:

1. Ekman label prediction, where the target is exactly one of seven labels.
2. Political narrative extraction, where the target contains only the narrative.

It also supports Model C, a single joint LoRA adapter trained on paired label
and narrative examples. Each source appears twice in a paired batch, once per
task. The joint loss averages per-target-token loss within each task and then
combines the task losses with the configured emotion weight.

The source CSV is deduplicated by normalized source text and split into a
stratified 95% training set and 5% validation set. The two held-out evaluation
files are never used for training or validation.

On the cloud server:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory
export OMP_NUM_THREADS=8

python /root/autodl-tmp/beyond/training/prepare_sft_data.py
python /root/autodl-tmp/beyond/training/audit_token_lengths.py

cd /root/autodl-tmp/LLaMA-Factory
llamafactory-cli train /root/autodl-tmp/beyond/training/configs/label_sft.yaml
llamafactory-cli train /root/autodl-tmp/beyond/training/configs/narrative_sft.yaml
```

All generated datasets, adapters, logs, predictions, and metrics belong under
`/root/autodl-fs/sft_experiments/`.

Prepare and train Model C with:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8

python /root/autodl-tmp/beyond/training/prepare_joint_data.py
python /root/autodl-tmp/beyond/training/train_joint_multitask.py
```

The validated Model C environment uses PyTorch `2.13.0+cu130`, Transformers
`5.8.0`, PEFT `0.18.1`, and LLaMA-Factory `0.9.6.dev0` on one RTX 3090 24 GB.
This is intentionally documented separately from the repository's older root
requirements because Model C uses the Transformers 5 tensor `logits_to_keep`
interface to avoid materializing full-vocabulary logits for prompt positions.

After training has stopped, finalize the checkpoint with the lowest validation
loss and evaluate either task through the same adapter:

```bash
python /root/autodl-tmp/beyond/training/finalize_joint_model.py

/root/autodl-tmp/beyond/training/run_joint_eval.sh \
  label \
  /root/autodl-fs/train_set_stage1_test_eval.csv \
  /root/autodl-fs/sft_experiments/evaluation_joint/label/stage1 \
  8
```

To run both tasks on both held-out test sets and then build the A/B/C comparison:

```bash
/root/autodl-tmp/beyond/training/run_joint_evaluation_suite.sh 8
```

During a formal run, the external finish pipeline applies the fixed early-stop
policy (two validation checks without improvement, or one 10% degradation),
finalizes the best checkpoint, runs all evaluations, builds the report, and
archives exact code hashes:

```bash
screen -dmS model_c_finish \
  /root/autodl-tmp/beyond/training/finish_joint_pipeline.sh
```

The completed experiment report is stored at
`/root/autodl-fs/sft_experiments/MODEL_C_EXPERIMENT_REPORT.md` and mirrored in
this directory as `MODEL_C_EXPERIMENT_REPORT.md`.

For evaluation, start the task adapter server in one terminal:

```bash
/root/autodl-tmp/beyond/training/serve_sft_adapter.sh label
```

Then evaluate either the base model or adapter from another terminal:

```bash
/root/autodl-tmp/beyond/training/run_task_eval.sh \
  label label \
  /root/autodl-fs/train_set_stage1_test_eval.csv \
  /root/autodl-fs/sft_experiments/evaluation/label/stage1_sft
```

The vLLM server exposes the base model and the named task adapter. Label
generation uses a structured choice constraint, so every successful response is
exactly one valid Ekman label. Narrative generation returns only narrative text.

To evaluate an adapter directly with Transformers and PEFT, without starting
vLLM, run:

```bash
/root/autodl-tmp/beyond/training/run_direct_sft_eval.sh \
  label \
  /root/autodl-fs/train_set_stage1_test_eval.csv \
  /root/autodl-fs/sft_experiments/evaluation_direct/label/stage1_sft \
  8
```

Replace `label` with `narrative` for the narrative adapter. The final argument
is the inference batch size. The direct path uses the same prompts, greedy
decoding, source truncation, gold alignment, and metric implementations as the
vLLM path. It loads the base model and LoRA adapter directly into Transformers;
the label task uses constrained decoding to permit only the seven Ekman labels.
