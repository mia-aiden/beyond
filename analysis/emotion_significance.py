#!/usr/bin/env python3
"""Emotion significance of the lambda=0.8 joint model against Model A (Section 5.7.3).

McNemar exact test and a paired bootstrap on seed 42, and a hierarchical
bootstrap over seeds 42, 1, and 2 that resamples seeds and then samples.
Reads the vLLM label runs under EXPERIMENTS_DIR/mentor_followup/evaluation.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from paths import EXPERIMENTS_DIR  # noqa: E402

SEEDS = [42, 1, 2]
DATASETS = ["stage1", "manual"]
LABELS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
LMAP = {label: i for i, label in enumerate(LABELS)}
K = len(LABELS)
NBOOT = 10000
RNG = np.random.default_rng(42)


def load(base, model_dir, dataset):
    path = sorted(glob.glob(os.path.join(base, model_dir, "label", dataset, "evaluation", "*", "metrics",
                                         "emotion_per_sample.csv")))[-1]
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["sample_id"])] = (r["emotion_label"].strip(), r["predicted_emotion_label"].strip())
    return rows


def enc(labels):
    # labels outside the seven classes map to K and always count as wrong
    return np.array([LMAP.get(x, K) for x in labels], dtype=np.int64)


def macro_f1_idx(g, p):
    cm = np.bincount(g * (K + 1) + p, minlength=(K + 1) * (K + 1)).reshape(K + 1, K + 1)[:K, :]
    tp = np.array([cm[c, c] for c in range(K)], dtype=float)
    fn = cm.sum(1) - tp
    fp = np.bincount(p, minlength=K + 1)[:K].astype(float) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return f1.mean()


def acc_idx(g, p):
    return np.mean(g == p)


def mcnemar_exact(a_corr, c_corr):
    b = int(np.sum(a_corr & ~c_corr))
    c = int(np.sum(~a_corr & c_corr))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * (0.5 ** n)
    return b, c, min(p, 1.0)


def paired_boot(g, a, c, fn):
    n = len(g)
    base = fn(g, c) - fn(g, a)
    d = np.empty(NBOOT)
    for i in range(NBOOT):
        idx = RNG.integers(0, n, n)
        d[i] = fn(g[idx], c[idx]) - fn(g[idx], a[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return base, lo, hi, min(2.0 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0)


def hier_boot(ps, fn):
    seeds = list(ps.keys())
    point = np.mean([fn(g, c) - fn(g, a) for (g, a, c) in ps.values()])
    d = np.empty(NBOOT)
    for i in range(NBOOT):
        chosen = RNG.choice(seeds, len(seeds), replace=True)
        diffs = []
        for s in chosen:
            g, a, c = ps[s]
            idx = RNG.integers(0, len(g), len(g))
            diffs.append(fn(g[idx], c[idx]) - fn(g[idx], a[idx]))
        d[i] = np.mean(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi), float(min(2.0 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPERIMENTS_DIR / "mentor_followup" / "emotion_significance.json")
    args = parser.parse_args()
    base = str(EXPERIMENTS_DIR / "mentor_followup" / "evaluation")

    results = {}
    for ds in DATASETS:
        print(f"\n== {ds}")
        ps = {}
        for s in SEEDS:
            A, C = load(base, f"model_a_seed{s}", ds), load(base, f"lambda_08_seed{s}", ds)
            ids = sorted(set(A) & set(C))
            gold = [A[i][0] for i in ids]
            assert gold == [C[i][0] for i in ids]
            g, a, c = enc(gold), enc([A[i][1] for i in ids]), enc([C[i][1] for i in ids])
            ps[s] = (g, a, c)
            print(f"  seed {s}: n={len(ids)}  acc A={acc_idx(g, a):.4f} C={acc_idx(g, c):.4f}  "
                  f"macro-F1 A={macro_f1_idx(g, a):.4f} C={macro_f1_idx(g, c):.4f}")
            if s == 42:
                b, cc, p = mcnemar_exact(g == a, g == c)
                print(f"    McNemar (seed 42, accuracy): b={b} c={cc} p={p:.4f}")
                for name, fn in (("acc", acc_idx), ("macro-F1", macro_f1_idx)):
                    delta, lo, hi, p = paired_boot(g, a, c, fn)
                    print(f"    paired bootstrap (seed 42) {name}: {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}")
        for name, fn in (("accuracy", acc_idx), ("macro-F1", macro_f1_idx)):
            a_vals = [fn(g, a) for g, a, c in ps.values()]
            c_vals = [fn(g, c) for g, a, c in ps.values()]
            print(f"  {name} over seeds: Model A {np.mean(a_vals):.4f} +- {np.std(a_vals, ddof=1):.4f}, "
                  f"lambda=0.8 {np.mean(c_vals):.4f} +- {np.std(c_vals, ddof=1):.4f}")
        results[ds] = {}
        for name, fn in (("Acc", acc_idx), ("MacroF1", macro_f1_idx)):
            delta, lo, hi, p = hier_boot(ps, fn)
            print(f"  three-seed hierarchical bootstrap {name}: {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}")
            results[ds][name] = {"delta": delta, "ci": [lo, hi], "p": p}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
