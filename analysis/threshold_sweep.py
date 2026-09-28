#!/usr/bin/env python3
"""Retention of the narrative filter as each threshold moves around 0.3 / 0.02.

Uses the ratios cached by threshold_distribution.py when they exist. This
sensitivity check was run for the supervisor but is not reported in the thesis.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from threshold_distribution import FP_THR, PAST_THR, filter_ratios, load_texts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from paths import EXPERIMENTS_DIR, RAW_PARQUET  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=RAW_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR / "mentor_followup")
    args = parser.parse_args()
    cache = args.output_dir / "threshold_ratios.npz"
    if cache.exists():
        d = np.load(cache)
        past, fp = d["past"], d["fp"]
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        past, fp = filter_ratios(load_texts(args.parquet))
        np.savez(cache, past=past, fp=fp)

    def joint(pt, ft):
        return float(((past > pt) & (fp > ft)).mean())

    past_grid = [round(x, 3) for x in np.arange(0.15, 0.4501, 0.025)]
    fp_grid = [round(x, 4) for x in np.arange(0.005, 0.0401, 0.0025)]
    vary_past = [joint(pt, FP_THR) for pt in past_grid]
    vary_fp = [joint(PAST_THR, ft) for ft in fp_grid]
    base = joint(PAST_THR, FP_THR)

    print(f"n = {len(past)}; retained at (0.30, 0.02): {base * 100:.1f}%")
    for pt, r in zip(past_grid, vary_past):
        print(f"  past > {pt:<5}  first-person > 0.02: {r * 100:5.1f}%")
    for ft, r in zip(fp_grid, vary_fp):
        print(f"  past > 0.30  first-person > {ft:<6}: {r * 100:5.1f}%")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, grid, values, thr, shown, label, color, dx in (
        (ax[0], past_grid, vary_past, PAST_THR, "0.30", "Past-tense verb ratio threshold", "#4C72B0", 0.03),
        (ax[1], fp_grid, vary_fp, FP_THR, "0.02", "First-person pronoun ratio threshold", "#55A868", 0.004),
    ):
        a.plot(grid, [r * 100 for r in values], "-o", color=color, ms=3)
        a.axvline(thr, color="#C44E52", ls="--", lw=1.5)
        a.annotate(f"chosen {shown}\n({base * 100:.1f}%)", xy=(thr, base * 100), xytext=(thr + dx, base * 100 + 6),
                   fontsize=8, arrowprops=dict(arrowstyle="->", color="#C44E52"))
        a.set_xlabel(label)
        a.set_ylabel("Retention: pass both (%)")
        a.grid(alpha=0.3)
    ax[0].set_title("(a) vary past-tense threshold (first-person = 0.02)", fontsize=9)
    ax[1].set_title("(b) vary first-person threshold (past-tense = 0.30)", fontsize=9)
    fig.suptitle(f"Threshold sensitivity: corpus retention on {len(past):,} sampled comments", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.output_dir / "threshold_sensitivity.png", dpi=150)
    summary = {"n": len(past), "chosen_retention": base, "past_grid": past_grid, "retention_vary_past": vary_past,
               "fp_grid": fp_grid, "retention_vary_fp": vary_fp}
    (args.output_dir / "threshold_sensitivity.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
