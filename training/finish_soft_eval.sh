#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
CE=$EXPERIMENTS_DIR/soft_eval
GOLD_S1=$STAGE1_TEST
GOLD_MAN=$MANUAL_TEST
PR=$PROJECT_ROOT
FS=$EXPERIMENTS_DIR
MB_S1=$FS/evaluation_direct/narrative/stage1_sft/evaluation/run_20260727_164624
MB_MAN=$FS/evaluation_direct/narrative/manual_sft/evaluation/run_20260727_164858
source "$CONDA_SH"

echo "[soft] inference..."
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
cd "$PR/training"
infer () {  # variant softdir ds gold
  local out="$CE/$1/$3/predictions/pred.csv"
  [ -s "$out" ] && { echo "skip $1/$3 (exists)"; return; }
  python infer_soft_prompt.py --variant "$1" --soft "$FS/models/$2/soft_prompt.pt" --gold "$4" --output "$out"
}
infer emotion soft_emotion stage1 "$GOLD_S1"
infer emotion soft_emotion manual "$GOLD_MAN"
infer shared  soft_shared  stage1 "$GOLD_S1"
infer shared  soft_shared  manual "$GOLD_MAN"
echo "[soft] metrics..."

conda activate vllm-eval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for v in soft_emotion soft_shared; do for ds in stage1 manual; do
  gold="$GOLD_S1"; [ "$ds" = manual ] && gold="$GOLD_MAN"
  python "$PR/evaluation/src/run_eval.py" --gold "$gold" --pred "$CE/$v/$ds/predictions/pred.csv" \
    --run-defaults "$PR/training/configs/run_narrative_eval.yaml" --output-root "$CE/$v/$ds/evaluation"
done; done

rundir(){ ls -d "$1"/run_* 2>/dev/null | tail -1; }
boot(){ [ -n "$1" ] && [ -n "$2" ] && python "$PR/analysis/paired_bootstrap_narrative.py" \
  --model-a-run "$1" --model-b-run "$2" --model-a-name "$3" --model-b-name "$4" --dataset "$5" --output-dir "$6" >/dev/null 2>&1; }
echo "[soft] bootstrap..."
for ds in stage1 manual; do
  E=$(rundir "$CE/soft_emotion/$ds/evaluation"); S=$(rundir "$CE/soft_shared/$ds/evaluation")
  MB="$MB_S1"; [ "$ds" = manual ] && MB="$MB_MAN"
  boot "$E" "$S"  soft_emotion soft_shared "$ds" "$CE/bootstrap/emotion_vs_shared_$ds"
  boot "$E" "$MB" soft_emotion model_b     "$ds" "$CE/bootstrap/emotion_vs_B_$ds"
  boot "$S" "$MB" soft_shared  model_b     "$ds" "$CE/bootstrap/shared_vs_B_$ds"
done

{
  echo "# Emotion soft-prompt on frozen Model B (narrative)"
  echo "emotion = per-emotion vector (7); shared = single emotion-agnostic vector (control, same capacity). diff = A - B; DTW lower better."
  for ds in stage1 manual; do
    echo; echo "## ${ds}: emotion-soft vs shared-soft   [KEY — isolates emotion content]"
    cat "$CE/bootstrap/emotion_vs_shared_$ds/PAIRED_BOOTSTRAP_REPORT.md" 2>/dev/null
    echo; echo "## ${ds}: emotion-soft vs Model B"
    cat "$CE/bootstrap/emotion_vs_B_$ds/PAIRED_BOOTSTRAP_REPORT.md" 2>/dev/null
    echo; echo "## ${ds}: shared-soft vs Model B"
    cat "$CE/bootstrap/shared_vs_B_$ds/PAIRED_BOOTSTRAP_REPORT.md" 2>/dev/null
  done
} > "$CE/SOFT_RESULTS.md"
echo "[soft] DONE -> $CE/SOFT_RESULTS.md"
