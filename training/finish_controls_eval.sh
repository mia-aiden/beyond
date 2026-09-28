#!/usr/bin/env bash
set -uo pipefail
CE=/root/autodl-fs/sft_experiments/controls_eval
M=/root/autodl-fs/sft_experiments/models
GOLD_S1=/root/autodl-fs/train_set_stage1_test_eval.csv
GOLD_MAN=/root/autodl-fs/manual_test_ekman_eval.csv
MB_S1=/root/autodl-fs/sft_experiments/evaluation_direct/narrative/stage1_sft/evaluation/run_20260727_164624
MB_MAN=/root/autodl-fs/sft_experiments/evaluation_direct/narrative/manual_sft/evaluation/run_20260727_164858
PR=/root/autodl-tmp/beyond
TAGS="shuffle seqjoint_s1 seqjoint_s2 seqjoint_s3"
source /root/miniconda3/etc/profile.d/conda.sh

adapter_dir () {
  case "$1" in
    shuffle) echo "$M/seqjoint_shuffle_lora_r8";;
    seqjoint_s1) echo "$M/seqjoint_lora_r8_s1";;
    seqjoint_s2) echo "$M/seqjoint_lora_r8_s2";;
    seqjoint_s3) echo "$M/seqjoint_lora_r8_s3";;
  esac
}

echo "[ctrl] waiting for training queues to finish..."
while screen -ls | grep -qE '\.(q0|q1)[[:space:]]'; do sleep 60; done
for t in $TAGS; do
  while [ ! -s "$(adapter_dir "$t")/adapter_model.safetensors" ]; do sleep 60; done
done
echo "[ctrl] all adapters ready; running inference on 2 GPUs..."

infer_pair () {  # gpu tag1 tag2
  local gpu="$1"; shift
  conda activate llamafactory
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$gpu"
  cd "$PR/training"
  for t in "$@"; do
    python infer_conditioned.py --mode seqjoint --adapter "$(adapter_dir "$t")" --gold "$GOLD_S1" --output "$CE/$t/stage1/predictions/pred.csv"
    python infer_conditioned.py --mode seqjoint --adapter "$(adapter_dir "$t")" --gold "$GOLD_MAN" --output "$CE/$t/manual/predictions/pred.csv"
  done
}
( infer_pair 0 shuffle seqjoint_s2 ) &
( infer_pair 1 seqjoint_s1 seqjoint_s3 ) &
wait
echo "[ctrl] inference done; metrics + bootstrap..."

conda activate vllm-eval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for t in $TAGS; do
  for ds in stage1 manual; do
    gold="$GOLD_S1"; mb="$MB_S1"
    [ "$ds" = manual ] && gold="$GOLD_MAN" && mb="$MB_MAN"
    python "$PR/evaluation/src/run_eval.py" --gold "$gold" --pred "$CE/$t/$ds/predictions/pred.csv" \
      --run-defaults "$PR/training/configs/run_narrative_eval.yaml" --output-root "$CE/$t/$ds/evaluation"
    arun=$(ls -d "$CE/$t/$ds/evaluation"/run_* 2>/dev/null | tail -1)
    python "$PR/training/paired_bootstrap_narrative.py" --model-a-run "$arun" --model-b-run "$mb" \
      --model-a-name "$t" --model-b-name model_b --dataset "$ds" --output-dir "$CE/bootstrap/${t}_vs_b_${ds}"
  done
done

{
  echo "# Controls & multi-seed — seq-joint variants vs Model B (narrative metrics)"
  echo "shuffle = emotion label permuted (uninformative). diff = candidate - Model B; DTW lower is better."
  for t in $TAGS; do for ds in stage1 manual; do
    echo; echo "## ${t}_vs_b_${ds}"; cat "$CE/bootstrap/${t}_vs_b_${ds}/PAIRED_BOOTSTRAP_REPORT.md" 2>&1
  done; done
} > "$CE/CONTROL_RESULTS.md"
echo "[ctrl] ALL DONE -> $CE/CONTROL_RESULTS.md"
