# Training and inference

This directory fine-tunes LoRA rank-8 adapters on Llama-3-8B-Instruct for Ekman
emotion classification and political narrative extraction, separately (Model A
and Model B) and with one shared adapter (Model C, Seq-joint), and runs the vLLM
and Transformers inference for all experiments in Chapter 5. The statistics
built from the resulting runs are in `../analysis`.

## Paths

All scripts ran on one AutoDL server. Python scripts take their paths from
`paths.py` and shell scripts from `paths.sh`. The defaults are the server
layout, and each path can be overridden with an environment variable of the
same name:

| Variable | Default | Holds |
|---|---|---|
| `EXPERIMENTS_DIR` | `/root/autodl-fs/sft_experiments` | datasets, adapters, predictions, metrics |
| `DATA_DIR` | `/root/autodl-fs` | development pool and the two test CSVs |
| `BASE_MODEL` | `/root/autodl-tmp/Meta-Llama-3-8B-Instruct` | base model |
| `LLAMAFACTORY_DIR` | `/root/autodl-tmp/LLaMA-Factory` | LLaMA-Factory checkout |
| `LLAMAFACTORY_PYTHON`, `LLAMAFACTORY_CLI` | `llamafactory` conda env | training |
| `VLLM_PYTHON`, `VLLM_BIN` | `vllm-eval` conda env | vLLM serving and evaluation |
| `CONDA_SH` | `/root/miniconda3/etc/profile.d/conda.sh` | conda activation in shell scripts |

The YAML files in `configs/` still contain the server paths (`model_name_or_path`,
`dataset_dir`, `output_dir`, ...); edit them or override the keys on the
`llamafactory-cli train` command line. The commands below are run from the
repository root after `source training/paths.sh`.

The scripts for the Model B multi-seed, prompt-cross, and format-semantics
evaluations pointed to a separate vLLM environment (`/root/autodl-tmp/vllm-cu128`)
that no longer exists on the server; they now default to `vllm-eval`.

## Model A and Model B

The development pool is deduplicated by normalized source text and split into a
stratified 95% training set and 5% validation set. The two held-out test files
are never used for training or validation.

```bash
source "$CONDA_SH" && conda activate llamafactory
export OMP_NUM_THREADS=8

python training/prepare_sft_data.py
python training/audit_token_lengths.py

cd "$LLAMAFACTORY_DIR"
llamafactory-cli train "$PROJECT_ROOT/training/configs/label_sft.yaml"
llamafactory-cli train "$PROJECT_ROOT/training/configs/narrative_sft.yaml"
```

Given the PID of a running label training job, `run_full_sft_pipeline.sh` waits
for it, evaluates the base model and the label adapter with vLLM, and then trains
and evaluates the narrative adapter.

## Model C

Model C trains one joint LoRA adapter on paired label and narrative examples.
Each source appears twice in a paired batch, once per task. The joint loss
averages the per-target-token loss within each task and combines the two task
losses with the configured emotion weight.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
python training/prepare_joint_data.py
python training/train_joint_multitask.py
```

The validated Model C environment uses PyTorch `2.13.0+cu130`, Transformers
`5.8.0`, PEFT `0.18.1`, and LLaMA-Factory `0.9.6.dev0` on one RTX 3090 24 GB.
Model C relies on the Transformers 5 tensor `logits_to_keep` interface to avoid
materializing full-vocabulary logits for prompt positions, which is why this
environment differs from the root requirements.

`finish_joint_pipeline.sh` applies the fixed early-stop policy (two validation
checks without improvement, or one 10% degradation), finalizes the checkpoint
with the lowest validation loss (`finalize_joint_model.py`), evaluates both
tasks on both test sets, and builds the A/B/C comparison. The λ ablation
(0.3, 0.5, 0.8) is started with `launch_joint_ablations.sh`.

## Evaluation

vLLM path: start the adapter server, then evaluate the base model or the adapter.
Label generation uses a structured choice constraint, so every response is
exactly one Ekman label; narrative generation returns only the narrative text.

```bash
training/serve_sft_adapter.sh label
training/run_task_eval.sh label label "$STAGE1_TEST" "$EXPERIMENTS_DIR/evaluation/label/stage1_sft"
```

Direct path: load the base model and LoRA adapter with Transformers and PEFT,
without a server. The label task uses constrained decoding over the seven labels.

```bash
training/run_direct_sft_eval.sh label "$STAGE1_TEST" "$EXPERIMENTS_DIR/evaluation_direct/label/stage1_sft" 8
```

Replace `label` with `narrative` for the narrative adapter; the last argument
is the batch size. Both paths use the same prompts, greedy decoding, source
truncation, gold alignment, and metrics. Model C is evaluated with
`run_joint_eval.sh` and `run_joint_evaluation_suite.sh`.

## Sequential-output experiments (Section 5.8)

- `setup_controls.py` writes the Seq-joint configs for seeds 1 to 3 and a
  shuffled-label control; copies are in `configs/seqjoint_*.yaml`.
- `setup_format_semantics_multiseed.py` builds the true, forced-wrong,
  constant-neutral, and marker datasets and configs for five seeds.
  `supervise_format_semantics_pipeline.py` trains them
  (`run_format_semantics_queue.py`) and evaluates them
  (`run_format_semantics_eval_queue.py`).
- `run_model_b_multiseed_eval.py` evaluates Model B seeds 42, 1, and 2.
- `run_prompt_cross_3seed.py` runs the two mismatched cells of the
  model-by-prompt experiment.
- `launch_mentor_followup.sh` trains Model A and λ=0.8 with seeds 1 and 2 and a
  λ=0 run; `run_mentor_followup_eval.py` evaluates them.

## Exploratory runs

Single-seed runs that came before Section 5.8 and are not reported as results:
gold-emotion conditioning (`infer_conditioned.py`, `run_conditioned_infer.sh`,
`finish_conditioned_eval.sh`, `eval_cgold_bgold.sh`), the shuffled-label and
seed controls (`eval_controls_now.sh`, `finish_controls_eval.sh`), and a soft
prompt on top of Model B (`train_soft_prompt.py`, `infer_soft_prompt.py`,
`finish_soft_eval.sh`, `finish_soft_eval2.sh`). `run_train_queue.sh` runs a list
of LLaMA-Factory configs on one GPU.
