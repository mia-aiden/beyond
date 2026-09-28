#!/usr/bin/env python3
"""Agreement between the vLLM and the direct Transformers path (Sections 5.5 and 5.6).

For both test sets: label agreement and metrics of the emotion adapter, exact
match and metric differences of the narrative adapter, and the length of the
narratives generated on the direct path.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from paths import EXPERIMENTS_DIR  # noqa: E402

NARRATIVE_METRICS = ["bleu_corpus", "rouge1_mean", "rouge2_mean", "rougeL_mean", "bertscore_f1_mean",
                     "normalized_dtw_distance_mean"]


def latest_run(sub):
    runs = sorted(glob.glob(str(EXPERIMENTS_DIR / sub / "run_*")) + glob.glob(str(EXPERIMENTS_DIR / sub / "evaluation" / "run_*")))
    if not runs:
        sys.exit(f"no run under {EXPERIMENTS_DIR / sub}")
    return Path(runs[-1]) / "metrics"


def per_sample(sub, name):
    return pd.read_csv(latest_run(sub) / name, encoding="utf-8-sig", keep_default_na=False)


def summary(sub):
    return json.loads((latest_run(sub) / "summary.json").read_text(encoding="utf-8"))


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def compare(ds):
    direct, vllm = f"evaluation_direct/label/{ds}_sft", f"evaluation/label/{ds}_label"
    df = per_sample(direct, "emotion_per_sample.csv").merge(per_sample(vllm, "emotion_per_sample.csv"), on="sample_id")
    same = int((df["predicted_emotion_label_x"] == df["predicted_emotion_label_y"]).sum())
    d, v = summary(direct), summary(vllm)
    out = {"n": len(df), "label_exact_match": same / len(df), "label_differences": len(df) - same,
           "direct_accuracy": d["accuracy"], "vllm_accuracy": v["accuracy"],
           "direct_macro_f1": d["macro_f1"], "vllm_macro_f1": v["macro_f1"]}

    direct, vllm = f"evaluation_direct/narrative/{ds}_sft", f"evaluation/narrative/{ds}_narrative"
    dn = per_sample(direct, "per_sample_metrics.csv")
    df = dn.merge(per_sample(vllm, "per_sample_metrics.csv"), on="sample_id")
    out["narrative_exact_match"] = float((df["predicted_narrative_x"].str.strip() == df["predicted_narrative_y"].str.strip()).mean())
    d, v = summary(direct), summary(vllm)
    out["narrative_abs_differences"] = {m: abs(d[m] - v[m]) for m in NARRATIVE_METRICS}

    words = np.array([len(t.split()) for t in dn["predicted_narrative"]])
    out["direct_narrative_words_mean"] = float(words.mean())
    out["direct_narrative_words_p95"] = float(np.percentile(words, 95))
    out["direct_narrative_max_sentences"] = max(len(sentences(t)) for t in dn["predicted_narrative"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "backend_consistency")
    args = parser.parse_args()
    results = {ds: compare(ds) for ds in ("stage1", "manual")}
    for ds, r in results.items():
        diffs = r["narrative_abs_differences"]
        print(f"{ds} (n={r['n']}): labels {r['label_exact_match'] * 100:.1f}% identical ({r['label_differences']} differ), "
              f"accuracy {r['direct_accuracy']:.3f}/{r['vllm_accuracy']:.3f}, macro-F1 {r['direct_macro_f1']:.3f}/{r['vllm_macro_f1']:.3f}")
        print(f"  narratives {r['narrative_exact_match'] * 100:.1f}% identical, BLEU difference {diffs['bleu_corpus']:.3f}, "
              f"largest other difference {max(v for k, v in diffs.items() if k != 'bleu_corpus'):.4f}")
        print(f"  direct narratives: mean {r['direct_narrative_words_mean']:.1f} words, P95 {r['direct_narrative_words_p95']:.0f}, "
              f"at most {r['direct_narrative_max_sentences']} sentences")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "backend_consistency.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
