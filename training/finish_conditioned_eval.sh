#!/usr/bin/env bash
set -uo pipefail
CE=/root/autodl-fs/sft_experiments/conditioned_eval
GOLD_S1=/root/autodl-fs/train_set_stage1_test_eval.csv
GOLD_MAN=/root/autodl-fs/manual_test_ekman_eval.csv
MB_S1=/root/autodl-fs/sft_experiments/evaluation_direct/narrative/stage1_sft/evaluation/run_20260727_164624
MB_MAN=/root/autodl-fs/sft_experiments/evaluation_direct/narrative/manual_sft/evaluation/run_20260727_164858
PR=/root/autodl-tmp/beyond

echo "[finish] waiting for prediction files..."
for f in bgold/stage1 bgold/manual seqjoint/stage1 seqjoint/manual; do
  while [ ! -s "$CE/$f/predictions/pred.csv" ]; do sleep 20; done
done
while screen -ls | grep -qE 'inf_bgold|inf_seqjoint'; do sleep 20; done
echo "[finish] predictions ready; computing metrics..."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-eval
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8

run_metrics () {
  python "$PR/evaluation/src/run_eval.py" \
    --gold "$3" --pred "$CE/$1/$2/predictions/pred.csv" \
    --run-defaults "$PR/training/configs/run_narrative_eval.yaml" \
    --output-root "$CE/$1/$2/evaluation"
}
run_metrics bgold stage1 "$GOLD_S1"
run_metrics bgold manual "$GOLD_MAN"
run_metrics seqjoint stage1 "$GOLD_S1"
run_metrics seqjoint manual "$GOLD_MAN"
echo "[finish] metrics done; running paired bootstrap vs Model B..."

boot () {
  local arun; arun=$(ls -d "$CE/$1/$2/evaluation"/run_* 2>/dev/null | tail -1)
  python "$PR/training/paired_bootstrap_narrative.py" \
    --model-a-run "$arun" --model-b-run "$3" \
    --model-a-name "$1" --model-b-name model_b --dataset "$2" \
    --output-dir "$CE/bootstrap/${1}_vs_b_${2}"
}
boot bgold stage1 "$MB_S1"
boot bgold manual "$MB_MAN"
boot seqjoint stage1 "$MB_S1"
boot seqjoint manual "$MB_MAN"

{
  echo "# Conditioned eval — B+gold and seq-joint vs Model B (narrative metrics)"
  echo "difference = candidate - Model B; DTW lower is better."
  for c in bgold_vs_b_stage1 bgold_vs_b_manual seqjoint_vs_b_stage1 seqjoint_vs_b_manual; do
    echo; echo "## $c"; cat "$CE/bootstrap/$c/PAIRED_BOOTSTRAP_REPORT.md" 2>&1
  done
} > "$CE/CONDITIONED_RESULTS.md"
echo "[finish] ALL DONE -> $CE/CONDITIONED_RESULTS.md"
