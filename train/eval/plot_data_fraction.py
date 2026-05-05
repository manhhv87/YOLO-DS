"""Generate Figure: mAP@0.5 vs training-set fraction (H3 study).

Reads test_results.json produced by train.py / train_ds.py after each run,
plots two curves (RGB baseline vs DS-YOLO) at 25 / 50 / 100 % training
fraction, and saves the figure as ``paper/figures/data_fraction.pdf``.

Usage
-----
    cd train
    python eval/plot_data_fraction.py

    # Override output path
    python eval/plot_data_fraction.py --out ../paper/figures/data_fraction.pdf

    # Print numbers only (no plot)
    python eval/plot_data_fraction.py --print-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent.parent.resolve()   # train/
PAPER_FIGURES = HERE.parent / "paper" / "figures"

# ---------------------------------------------------------------------------
# Where each run lives (relative to train/)
# ---------------------------------------------------------------------------
_RUNS: dict[str, list[tuple[float, Path]]] = {
    "RGB baseline": [
        (0.25, HERE / "runs" / "detect" / "rgb_25pct" / "test_results.json"),
        (0.50, HERE / "runs" / "detect" / "rgb_50pct" / "test_results.json"),
        (1.00, HERE / "runs" / "detect" / "rgb"       / "test_results.json"),
    ],
    "DS-YOLO (ours)": [
        (0.25, HERE / "runs" / "fraction" / "ds_yolo_25pct" / "test_results.json"),
        (0.50, HERE / "runs" / "fraction" / "ds_yolo_50pct" / "test_results.json"),
        (1.00, HERE / "runs" / "ds_yolo"  / "ds_yolo"       / "test_results.json"),
    ],
}

# Marker styles per curve
_STYLES = {
    "RGB baseline":  dict(color="#555555", marker="o", linestyle="--", linewidth=1.5, markersize=7),
    "DS-YOLO (ours)": dict(color="#1f77b4", marker="s", linestyle="-",  linewidth=2.0, markersize=7),
}


def _load(path: Path) -> float | None:
    """Return mAP@0.5 from test_results.json, or None if missing."""
    if not path.is_file():
        return None
    with path.open() as f:
        d = json.load(f)
    return float(d.get("map50", d.get("mAP50", float("nan"))))


def collect() -> dict[str, list[tuple[float, float]]]:
    """Return {label: [(fraction, map50), ...]} for available runs."""
    out: dict[str, list[tuple[float, float]]] = {}
    for label, runs in _RUNS.items():
        pts = []
        for frac, path in runs:
            v = _load(path)
            if v is not None and v == v:  # not nan
                pts.append((frac, v))
            else:
                print(f"[warn] missing: {path}")
        if pts:
            out[label] = pts
    return out


def plot(data: dict[str, list[tuple[float, float]]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for label, pts in data.items():
        xs = [int(f * 100) for f, _ in pts]
        ys = [m * 100 for _, m in pts]       # convert to percentage points
        ax.plot(xs, ys, label=label, **_STYLES[label])

    ax.set_xlabel("Training-set fraction (%)", fontsize=10)
    ax.set_ylabel("mAP@0.5 (%)", fontsize=10)
    ax.set_xticks([25, 50, 100])
    ax.set_xticklabels(["25", "50", "100"])
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(15, 108)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(PAPER_FIGURES / "data_fraction.pdf"),
                   help="Output path (default: paper/figures/data_fraction.pdf)")
    p.add_argument("--print-only", action="store_true",
                   help="Print numbers without generating a figure.")
    args = p.parse_args()

    data = collect()
    if not data:
        print("[error] No test_results.json found. Run training first.")
        return

    print("\nData-fraction results:")
    for label, pts in data.items():
        for frac, map50 in pts:
            print(f"  {label:20s}  {int(frac*100):3d}%  mAP@0.5 = {map50:.4f}")

    if not args.print_only:
        plot(data, Path(args.out))


if __name__ == "__main__":
    main()
