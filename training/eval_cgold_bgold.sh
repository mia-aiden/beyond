#!/usr/bin/env bash
set -uo pipefail
CE=/root/autodl-fs/sft_experiments/cgold_bgold_eval
GOLD_S1=/root/autodl-fs/train_set_stage1_test_eval.csv
GOLD_MAN=/root/autodl-fs/manual_test_ekman_eval.csv
PR=/root/autodl-tmp/beyond
FS=/root/autodl-fs/sft_experiments
B_ADP=$FS/models/narrative_lora_r8
C_ADP=$FS/models/joint_lora_r8          # Model C, lambda=1.0
source /root/miniconda3/etc/profile.d/conda.sh

echo "[cgbg] inference (sequential, single GPU)..."
conda activate llamafactory
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
cd "$PR/training"
run_infer () {  # tag adapter
  python infer_conditioned.py --mode bgold --adapter "$2" --gold "$GOLD_S1" --output "$CE/$1/stage1/predictions/pred.csv"
  python infer_conditioned.py --mode bgold --adapter "$2" --gold "$GOLD_MAN" --output "$CE/$1/manual/predictions/pred.csv"
}
run_infer bgold_inf "$B_ADP"
run_infer cgold_inf "$C_ADP"
echo "[cgbg] metrics..."

conda activate vllm-eval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
for tag in bgold_inf cgold_inf; do for ds in stage1 manual; do
  gold="$GOLD_S1"; [ "$ds" = manual ] && gold="$GOLD_MAN"
  python "$PR/evaluation/src/run_eval.py" --gold "$gold" --pred "$CE/$tag/$ds/predictions/pred.csv" \
    --run-defaults "$PR/training/configs/run_narrative_eval.yaml" --output-root "$CE/$tag/$ds/evaluation"
done; done

rundir () { ls -d "$1"/run_* 2>/dev/null | tail -1; }
boot () {
  [ -n "$1" ] && [ -n "$2" ] && python "$PR/training/paired_bootstrap_narrative.py" \
    --model-a-run "$1" --model-b-run "$2" --model-a-name "$3" --model-b-name "$4" \
    --dataset "$5" --output-dir "$6" >/dev/null 2>&1
}
echo "[cgbg] bootstrap..."
for ds in stage1 manual; do
  CG=$(rundir "$CE/cgold_inf/$ds/evaluation"); BG=$(rundir "$CE/bgold_inf/$ds/evaluation")
  if [ "$ds" = stage1 ]; then
    Bbase=$(rundir "$FS/evaluation_direct/narrative/stage1_sft/evaluation"); Cbase=$(rundir "$FS/evaluation_joint/narrative/stage1/evaluation")
  else
    Bbase=$(rundir "$FS/evaluation_direct/narrative/manual_sft/evaluation"); Cbase=$(rundir "$FS/evaluation_joint/narrative/manual/evaluation")
  fi
  boot "$CG" "$BG" cgold bgold "$ds" "$CE/bootstrap/cgold_vs_bgold_$ds"
  boot "$BG" "$Bbase" bgold_inf B_noemo "$ds" "$CE/bootstrap/bgold_vs_B_$ds"
  boot "$CG" "$Cbase" cgold_inf C_noemo "$ds" "$CE/bootstrap/cgold_vs_C_$ds"
done

{
  echo "# C+gold vs B+gold (gold emotion injected in prompt at inference)"
  echo "B = narrative_lora_r8, C = joint_lora_r8 (lambda=1.0). Neither was trained with emotion in the narrative prompt."
  echo
  for ds in stage1 manual; do
    echo "## ${ds}: C+gold vs B+gold  (requested; diff = C+gold - B+gold)"
    cat "$CE/bootstrap/cgold_vs_bgold_$ds/PAIRED_BOOTSTRAP_REPORT.md" 2>/dev/null; echo
    echo "## ${ds}: does gold emotion help B?  (B+gold - B_noemo)"
    cat "$CE/bootstrap/bgold_vs_B_$ds/PAIRED_BOOTSTRAP_REPORT.md" 2>/dev/null; echo
    echo "## ${ds}: does gold emotion help C?  (C+gold - C_noemo)"
    cat "$CE/bootstrap/cgold_vs_C_$ds/PAIRED_BOOTSTRAP_REPORT.md" 2>/dev/null; echo
  done
} > "$CE/CGOLD_BGOLD_RESULTS.md"
echo "[cgbg] DONE -> $CE/CGOLD_BGOLD_RESULTS.md"
