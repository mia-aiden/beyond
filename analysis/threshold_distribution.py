#!/usr/bin/env python3
"""Distribution of the two narrative-filter ratios on an unfiltered sample (Appendix B).

Reads the first 12,000 usable comments of the raw Reddit parquet, tags them with
spaCy, and plots the past-tense verb ratio and the first-person pronoun ratio
against the thresholds used in Section 4.1.1.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from paths import EXPERIMENTS_DIR, RAW_PARQUET  # noqa: E402

N_TARGET = 12000
PAST_THR, FP_THR = 0.30, 0.02
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
PAST_TAGS = {"VBD", "VBN"}


def load_texts(parquet, n=N_TARGET):
    # stream in small batches: a whole row group does not fit in 2 GB of memory
    texts = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1000, columns=["text"]):
        for t in batch.column("text").to_pylist():
            if not isinstance(t, str):
                continue
            t = t.strip()
            if t and t not in ("[deleted]", "[removed]"):
                texts.append(t[:5000])
        del batch
        if len(texts) >= n:
            break
    gc.collect()
    return texts[:n]


def filter_ratios(texts):
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])
    past_ratios, fp_ratios = [], []
    for doc in nlp.pipe(texts, batch_size=32):
        if not len(doc):
            continue
        verbs = past = fp = 0
        for tok in doc:
            if tok.tag_ in VERB_TAGS:
                verbs += 1
                if tok.tag_ in PAST_TAGS:
                    past += 1
            if tok.lower_ in FIRST_PERSON:
                fp += 1
        past_ratios.append(past / verbs if verbs else 0.0)
        fp_ratios.append(fp / len(doc))
    return np.array(past_ratios), np.array(fp_ratios)


def plot(past, fp, keep_both, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(past, bins=40, range=(0, 1), color="#4C72B0", alpha=0.85)
    ax[0].axvline(PAST_THR, color="#C44E52", ls="--", lw=1.6, label=f"threshold = {PAST_THR}")
    ax[0].set_xlabel("Past-tense verb ratio (VBD/VBN over all verbs)")
    ax[0].set_ylabel("Number of comments")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"(a) retained if > {PAST_THR}", fontsize=9)
    ax[1].hist(fp, bins=40, range=(0, 0.2), color="#55A868", alpha=0.85)
    ax[1].axvline(FP_THR, color="#C44E52", ls="--", lw=1.6, label=f"threshold = {FP_THR}")
    ax[1].set_xlabel("First-person pronoun ratio (over all tokens)")
    ax[1].set_ylabel("Number of comments")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"(b) retained if > {FP_THR}", fontsize=9)
    fig.suptitle(f"Pre-filter distribution on {len(past):,} sampled Reddit comments "
                 f"(retained by both filters: {keep_both.mean() * 100:.1f}%)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=RAW_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "mentor_followup")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args.parquet)
    print(f"collected {len(texts)} usable comments")
    past, fp = filter_ratios(texts)
    np.savez(args.output_dir / "threshold_ratios.npz", past=past, fp=fp)

    keep_past, keep_fp = past > PAST_THR, fp > FP_THR
    keep_both = keep_past & keep_fp
    print(f"n = {len(past)}")
    print(f"  pass past > {PAST_THR}: {keep_past.mean() * 100:.1f}%")
    print(f"  pass first-person > {FP_THR}: {keep_fp.mean() * 100:.1f}%")
    print(f"  pass both: {keep_both.mean() * 100:.1f}% (n={int(keep_both.sum())})")
    print(f"  past median {np.median(past):.3f}, first-person median {np.median(fp):.4f}")

    plot(past, fp, keep_both, args.output_dir / "threshold_distribution.png")
    summary = {"n": len(past), "pass_past": float(keep_past.mean()), "pass_fp": float(keep_fp.mean()),
               "pass_both": float(keep_both.mean()), "past_median": float(np.median(past)),
               "past_mean": float(past.mean()), "fp_median": float(np.median(fp)), "fp_mean": float(fp.mean())}
    (args.output_dir / "threshold_distribution.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
