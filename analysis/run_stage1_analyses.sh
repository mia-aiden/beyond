#!/usr/bin/env bash
# Rerun every analysis behind the Chapter 5 tables on one experiment tree.
#   run_stage1_analyses.sh TREE OUT
# TREE is the experiment directory; use the mirror written by
# filter_stage1_overlap.py to get the 984-comment Stage1 results reported in
# the thesis. OUT receives all outputs, so TREE is only read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../training/paths.sh"

TREE=$1
OUT=$2
PY=$VLLM_PYTHON
export EXPERIMENTS_DIR=$TREE
mkdir -p "$OUT"

"$PY" "$HERE/summarize_model_b_seqjoint_3seed.py" --output-dir "$OUT/model_b_seqjoint"
"$PY" "$HERE/summarize_prompt_cross_3seed.py" --output-dir "$OUT/prompt_cross"
"$PY" "$HERE/summarize_format_semantics_multiseed.py" --output-dir "$OUT/format_semantics"
"$PY" "$HERE/emotion_significance.py" --output "$OUT/emotion/emotion_significance.json"
"$PY" "$HERE/compare_model_abc.py" --root "$TREE" --output-dir "$OUT/model_abc" > /dev/null
"$PY" "$HERE/backend_consistency.py" --output-dir "$OUT/backend_consistency"

# lambda=0.8 and lambda=0.3 against Model B, both on the direct Transformers path
B_STAGE1=$TREE/evaluation_direct/narrative/stage1_sft/evaluation/run_20260727_164624
B_MANUAL=$TREE/evaluation_direct/narrative/manual_sft/evaluation/run_20260727_164858
boot() {
  "$PY" "$HERE/paired_bootstrap_narrative.py" --model-a-run "$3" --model-b-run "$4" \
    --model-a-name "$1" --model-b-name model_b --dataset "$2" \
    --output-dir "$OUT/paired_${1}_$2" --samples 10000 --seed 42 > /dev/null
}
boot lambda_0.8 stage1 "$TREE/ablations/lambda_0_8/evaluation/narrative/stage1/evaluation/run_20260803_160831" "$B_STAGE1"
boot lambda_0.8 manual "$TREE/ablations/lambda_0_8/evaluation/narrative/manual/evaluation/run_20260803_161041" "$B_MANUAL"
boot lambda_0.3 stage1 "$TREE/ablations/lambda_0_3/evaluation/narrative/stage1/evaluation/run_20260803_144758" "$B_STAGE1"
boot lambda_0.3 manual "$TREE/ablations/lambda_0_3/evaluation/narrative/manual/evaluation/run_20260803_145008" "$B_MANUAL"
echo "wrote $OUT"
