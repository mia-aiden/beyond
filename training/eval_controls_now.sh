#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
CE=$EXPERIMENTS_DIR/controls_eval
M=$EXPERIMENTS_DIR/models
GOLD_S1=$STAGE1_TEST
GOLD_MAN=$MANUAL_TEST
MB_S1=$EXPERIMENTS_DIR/evaluation_direct/narrative/stage1_sft/evaluation/run_20260727_164624
MB_MAN=$EXPERIMENTS_DIR/evaluation_direct/narrative/manual_sft/evaluation/run_20260727_164858
PR=$PROJECT_ROOT
TAGS="shuffle seqjoint_s1"
source "$CONDA_SH"

adapter_dir () {
  case "$1" in
    shuffle) echo "$M/seqjoint_shuffle_lora_r8";;
    seqjoint_s1) echo "$M/seqjoint_lora_r8_s1";;
  esac
}

echo "[eval] inference on 2 GPUs..."
infer_one () {  # gpu tag
  local gpu="$1" t="$2"
  conda activate llamafactory
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$gpu"
  cd "$PR/training"
  python infer_conditioned.py --mode seqjoint --adapter "$(adapter_dir "$t")" --gold "$GOLD_S1" --output "$CE/$t/stage1/predictions/pred.csv"
  python infer_conditioned.py --mode seqjoint --adapter "$(adapter_dir "$t")" --gold "$GOLD_MAN" --output "$CE/$t/manual/predictions/pred.csv"
}
( infer_one 0 shuffle ) &
( infer_one 1 seqjoint_s1 ) &
wait
echo "[eval] metrics + bootstrap..."

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
  echo "# Shuffle control + seed-1 — vs Model B (narrative)"
  echo "shuffle = emotion label permuted (uninformative). diff = candidate - Model B; DTW lower is better."
  for t in $TAGS; do for ds in stage1 manual; do
    echo; echo "## ${t}_vs_b_${ds}"; cat "$CE/bootstrap/${t}_vs_b_${ds}/PAIRED_BOOTSTRAP_REPORT.md" 2>&1
  done; done
} > "$CE/CONTROL_RESULTS.md"
echo "[eval] DONE -> $CE/CONTROL_RESULTS.md"
