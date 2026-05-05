"""Diagnose the diff channel quality of a 4-channel dataset.

Quick sanity-check to find out *why* training rg-yolo gives near-zero mAP:
the most common cause is that the per-pixel diff channel is uninformative
(either ORB alignment failed silently, or the captures are simply too
similar / too different from the golden).

Usage::

    python data/inspect_4ch.py --root dataset_rg

Prints aggregate statistics and saves a per-file CSV at
``<root>/diff_stats.csv`` so you can spot outliers.

A healthy diff channel for PCB inspection should:
  - have mean ≈ 5–25  (mostly black background with sparse defects)
  - have stdev ≈ 15–40 (some local hot spots)
  - have <10 % of pixels above 60 (sparse defects)
  - have a clear bimodal distribution (background vs defect)

Symptoms that indicate poor alignment / corrupted diff:
  - mean > 50          → too much "noise"; alignment is bad
  - stdev < 5          → diff is almost flat; capture ≈ golden everywhere
  - frac>60 > 0.5      → most of the image looks like a defect; useless for training
  - identical stats across all files → a single failure mode
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Root of the 4-channel dataset (parent of train/, val/, test/).")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    out_csv = args.root / "diff_stats.csv"
    rows: list[dict] = []

    for split in args.splits:
        d = args.root / split / "images"
        if not d.is_dir():
            print(f"[skip] {d}")
            continue
        npys = sorted(d.glob("*.npy"))
        if not npys:
            print(f"[skip] no .npy in {d}")
            continue
        diff_mean = []
        diff_std = []
        diff_hot = []  # frac of pixels > 60
        for f in npys:
            a = np.load(f)
            if a.ndim != 3 or a.shape[2] != 4:
                continue
            # Channels in saved npy: [D, B, G, R] (per build_4ch_dataset.py).
            d_ch = a[..., 0].astype(np.float32)
            mn = float(d_ch.mean())
            sd = float(d_ch.std())
            hot = float((d_ch > 60).mean())
            diff_mean.append(mn)
            diff_std.append(sd)
            diff_hot.append(hot)
            rows.append(dict(split=split, file=f.name, mean=mn, std=sd, frac_hot=hot))
        n = len(diff_mean)
        if not n:
            continue
        print(f"\n[{split}] {n} npy files:")
        print(f"  diff mean  : mean={np.mean(diff_mean):6.1f}  std={np.std(diff_mean):6.1f}")
        print(f"  diff std   : mean={np.mean(diff_std):6.1f}  std={np.std(diff_std):6.1f}")
        print(f"  frac > 60  : mean={np.mean(diff_hot):6.2%} std={np.std(diff_hot):6.2%}")

        # Verdict
        m = np.mean(diff_mean)
        s = np.mean(diff_std)
        h = np.mean(diff_hot)
        if m > 50 or h > 0.5:
            verdict = "BAD (likely alignment failure or wrong golden)"
        elif s < 5:
            verdict = "WEAK (capture ≈ golden everywhere — no signal)"
        elif 5 <= m <= 25 and s >= 15 and h < 0.10:
            verdict = "HEALTHY"
        else:
            verdict = "QUESTIONABLE"
        print(f"  verdict    : {verdict}")

    if rows:
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-file stats saved to {out_csv}")


if __name__ == "__main__":
    main()
