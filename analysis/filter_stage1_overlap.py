#!/usr/bin/env python3
"""Drop the Stage1 comments that also belong to the Manual Gold Standard.

Sixteen of the 1,000 reserved Stage1 comments are also in the 200-comment
Manual Gold Standard. The thesis reports every Stage1 result on the remaining
984 comments. This script builds a mirror of the experiment directory in which
every file is a symlink to the original, then rewrites the per-sample and
summary files of each Stage1 run without the overlapping rows. The original
results are never modified.

Pass 1 recomputes every Stage1 summary.json from its unfiltered per-sample files
and stops if any value differs from the stored one; pass 2 writes the filtered
files. Run the analyses on the mirror with run_stage1_analyses.sh.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from paths import EXPERIMENTS_DIR, MANUAL_TEST, STAGE1_TEST  # noqa: E402

EMOTIONS = {"Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"}


def norm(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def narrative_summary(ps):
    preds = ps["predicted_narrative"].fillna("").astype(str).tolist()
    refs = ps["reference_narrative"].fillna("").astype(str).tolist()
    return {
        "bleu_corpus": float(sacrebleu.corpus_bleu(preds, [refs]).score),
        "bleu_sentence_mean": float(ps["bleu_sentence"].mean()),
        "rouge1_mean": float(ps["rouge1_fmeasure"].mean()),
        "rouge2_mean": float(ps["rouge2_fmeasure"].mean()),
        "rougeL_mean": float(ps["rougeL_fmeasure"].mean()),
        "bertscore_f1_mean": float(ps["bertscore_f1"].mean()),
        "dtw_distance_mean": float(ps["dtw_distance"].mean()),
        "normalized_dtw_distance_mean": float(ps["normalized_dtw_distance"].mean()),
        "normalized_dtw_mean": float(ps["normalized_dtw_distance"].mean()),
    }


def emotion_summary(ep):
    gold = ep["emotion_label"].fillna("").astype(str)
    pred = ep["predicted_emotion_label"].fillna("").astype(str)
    out = {"accuracy": float(accuracy_score(gold, pred)), "macro_f1": float(f1_score(gold, pred, average="macro"))}
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    for label, values in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        out[f"{label}_precision"] = float(values["precision"])
        out[f"{label}_recall"] = float(values["recall"])
        out[f"{label}_f1"] = float(values["f1-score"])
    out["emotion_accuracy"] = out["accuracy"]
    out["emotion_macro_f1"] = out["macro_f1"]
    return out


def build_mirror(src, dst):
    for root, dirs, files in os.walk(src):
        target = dst / Path(root).relative_to(src)
        target.mkdir(parents=True, exist_ok=True)
        for d in list(dirs):
            if (Path(root) / d).is_symlink():
                os.symlink(os.path.realpath(Path(root) / d), target / d)
                dirs.remove(d)
        for f in files:
            os.symlink(Path(root) / f, target / f)


def stage1_runs(dst):
    for root, _, files in os.walk(dst):
        if "summary.json" in files:
            path = Path(root) / "summary.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and Path(str(meta.get("gold_path", ""))).name == STAGE1_TEST.name:
                yield path


def analyse(summary_path, overlap):
    """Return (original summary, sample_ids to drop, recomputed full, recomputed filtered)."""
    real = Path(os.path.realpath(summary_path)).parent
    orig = json.loads(summary_path.read_text(encoding="utf-8"))
    gold = read(real / "gold_with_sample_ids.csv")
    drop = set(gold.loc[gold["source_text"].map(norm).isin(overlap), "sample_id"].astype(str))
    full, filt = {}, {}
    if "bleu_corpus" in orig:
        ps = read(real / "per_sample_metrics.csv")
        full.update(narrative_summary(ps))
        filt.update(narrative_summary(ps[~ps["sample_id"].astype(str).isin(drop)]))
    if "accuracy" in orig:
        ep = read(real / "emotion_per_sample.csv")
        full.update(emotion_summary(ep))
        filt.update(emotion_summary(ep[~ep["sample_id"].astype(str).isin(drop)]))
    return orig, drop, full, filt


def replace_file(path, write):
    # remove the symlink first so the original file is never written through
    if os.path.lexists(path):
        os.unlink(path)
    write(path)


def write_filtered(summary_path, orig, drop, filt):
    mdir = summary_path.parent
    new = {k: v for k, v in orig.items() if k.split("_")[0] not in EMOTIONS}
    new.update(filt)
    for key in ("num_gold_rows", "num_prediction_rows", "num_aligned_rows"):
        new[key] = orig[key] - len(drop)
    new["stage1_overlap_filter"] = f"removed the {len(drop)} Stage1 comments that also belong to the Manual Gold Standard"
    for path in mdir.glob("*.csv"):
        frame = pd.read_csv(os.path.realpath(path), encoding="utf-8-sig", keep_default_na=False)
        if "sample_id" in frame.columns:
            kept = frame[~frame["sample_id"].astype(str).isin(drop)]
            replace_file(path, lambda p, k=kept: k.to_csv(p, index=False, encoding="utf-8-sig"))
        elif list(frame.columns[:2]) == ["metric", "value"]:
            frame["value"] = [new.get(m, v) for m, v in zip(frame["metric"], frame["value"])]
            replace_file(path, lambda p, f=frame: f.to_csv(p, index=False, encoding="utf-8-sig"))
    replace_file(summary_path, lambda p: p.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--dst", type=Path, required=True, help="new mirror directory (must not exist)")
    args = parser.parse_args()
    if args.dst.exists():
        sys.exit(f"{args.dst} already exists")
    if args.src.resolve() in args.dst.resolve().parents:
        sys.exit("the mirror must not be inside the experiment directory")

    stage1 = pd.read_csv(STAGE1_TEST, encoding="utf-8-sig")
    manual = pd.read_csv(MANUAL_TEST, encoding="utf-8-sig")
    overlap = set(stage1["source_text"].map(norm)) & set(manual["source_text"].map(norm))
    print(f"{len(overlap)} of {len(stage1)} Stage1 comments are also Manual comments")

    build_mirror(args.src.resolve(), args.dst)
    runs, maxdiff = [], 0.0
    for path in stage1_runs(args.dst):
        orig, drop, full, filt = analyse(path, overlap)
        if len(drop) != len(overlap) or orig.get("num_gold_rows") != len(stage1):
            sys.exit(f"unexpected Stage1 run {path}: {len(drop)} overlapping rows, {orig.get('num_gold_rows')} gold rows")
        for key, value in full.items():
            if isinstance(orig.get(key), (int, float)):
                maxdiff = max(maxdiff, abs(value - orig[key]))
        runs.append((path, orig, drop, filt))
    print(f"pass 1: {len(runs)} Stage1 runs, max |recomputed - stored| on the full data = {maxdiff:.1e}")
    if not runs or maxdiff > 1e-9:
        sys.exit("validation failed; nothing written")
    for path, orig, drop, filt in runs:
        write_filtered(path, orig, drop, filt)
    print(f"pass 2: wrote filtered files for {len(runs)} runs to {args.dst}")


if __name__ == "__main__":
    main()
