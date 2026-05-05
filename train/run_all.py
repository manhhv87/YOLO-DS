"""Master experiment runner for the DS-YOLO paper.

Executes all steps from the PLAN.md in the correct order.
Each step is idempotent: if the expected output already exists it is skipped
(use --force to re-run).

Usage
-----
    cd train

    # Run everything from scratch
    python run_all.py

    # Run only specific steps
    python run_all.py --steps resplit train_all

    # Re-run even if outputs exist
    python run_all.py --force

    # Dry-run: print commands without executing
    python run_all.py --dry-run

Available steps (run in this order)
-------------------------------------
    resplit       Re-split in-house dataset by source image level
    train_all     Train 3 Ultralytics variants (rgb, diff-only, stack6)
    train_ds      Train DS-YOLO with train_ds.py
    fraction      Data-fraction study (H3): train rgb + ds-yolo at 25/50/100 %
    robustness    Robustness study: eval rgb vs ds-yolo under alignment perturbation
    soldef_val    Train RGB baseline on SolDef_A for external validation
    tables        Aggregate all results into LaTeX table rows
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Paths — all relative to train/
# ---------------------------------------------------------------------------
# In-house dataset (1 board, production cell)
INHOUSE_LEAKY    = HERE / "datasets" / "inhouse" / "leaky"   # original Roboflow export (DO NOT use for experiments)
GOLDEN           = HERE / "datasets" / "inhouse" / "golden" / "golden_inhouse.jpg"

# SolDef_A public dataset
SOLDEF_DATA      = HERE / "datasets" / "soldef" / "data.yaml"

# Generated datasets (created by scripts below)
DATASET_FIXED    = HERE / "dataset_fixed"               # resplit_inhouse.py output

RUNS_DETECT      = HERE / "runs" / "detect"
RUNS_DS          = HERE / "runs" / "ds_yolo"
RUNS_ROBUST      = HERE / "runs" / "robustness"
RUNS_FRACTION    = HERE / "runs" / "fraction"

# Training hyper-parameters (match paper Section VI-B)
EPOCHS   = 60
IMGSZ    = 640
BATCH    = 4
SIZE     = "m"


# ---------------------------------------------------------------------------
# Shell helper
# ---------------------------------------------------------------------------

def run(cmd: list[str], dry: bool = False) -> None:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n[run] {cmd_str}")
    if not dry:
        result = subprocess.run(cmd, cwd=str(HERE))
        if result.returncode != 0:
            print(f"[ERROR] command failed with code {result.returncode}")
            sys.exit(result.returncode)


def exists(path: Path) -> bool:
    return path.exists()


def skip(label: str, check: Path) -> bool:
    if check.exists():
        print(f"[skip] {label} — output already exists: {check}")
        return True
    return False


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def step_resplit(dry: bool, force: bool) -> None:
    out = DATASET_FIXED / "data.yaml"
    if not force and exists(out):
        print(f"[skip] resplit — {out} exists")
        return
    run([sys.executable, "data/resplit_inhouse.py",
         "--src", str(INHOUSE_LEAKY),
         "--dst", str(DATASET_FIXED),
         "--val-frac", "0.15",
         "--test-frac", "0.15",
         "--seed", "0"], dry=dry)


def _train_variant(variant: str, data: Path, name: str | None = None,
                   fraction: float = 1.0, extra: list | None = None,
                   dry: bool = False) -> None:
    name = name or variant
    out  = RUNS_DETECT / name / "weights" / "best.pt"
    # Do not skip here — caller decides.
    cmd = [sys.executable, "train.py",
           "--variant",  variant,
           "--size",     SIZE,
           "--data",     str(data),
           "--epochs",   str(EPOCHS),
           "--imgsz",    str(IMGSZ),
           "--batch",    str(BATCH),
           "--name",     name]
    if fraction < 1.0:
        cmd += ["--data-fraction", str(fraction)]
    if extra:
        cmd += extra
    run(cmd, dry=dry)


def step_train_all(dry: bool, force: bool) -> None:
    data_rgb = DATASET_FIXED / "data.yaml"

    variants_data = [
        ("rgb",       data_rgb),
        ("diff-only", data_rgb),
        ("stack6",    data_rgb),
    ]
    for variant, data in variants_data:
        out = RUNS_DETECT / variant / "weights" / "best.pt"
        if not force and exists(out):
            print(f"[skip] train {variant} — {out} exists")
            continue
        _train_variant(variant, data, dry=dry)


def _train_ds(name: str, fusion: str, dry: bool,
              extra: list | None = None) -> None:
    """Train one DS-YOLO variant (fusion=crfm or sub) via train_ds.py."""
    out = RUNS_DS / name / "weights" / "best.pt"
    cmd = [sys.executable, "train_ds.py",
           "--data",     str(DATASET_FIXED / "data.yaml"),
           "--golden",   str(GOLDEN),
           "--variant",  SIZE,
           "--fusion",   fusion,
           "--epochs",   str(EPOCHS),
           "--imgsz",    str(IMGSZ),
           "--batch",    str(BATCH),
           "--name",     name,
           "--save-dir", str(RUNS_DS)]
    if extra:
        cmd += extra
    run(cmd, dry=dry)


def step_train_ds(dry: bool, force: bool) -> None:
    # DS-YOLO (CRFM fusion — main contribution)
    if force or not exists(RUNS_DS / "ds_yolo" / "weights" / "best.pt"):
        _train_ds("ds_yolo", fusion="crfm", dry=dry)
    else:
        print(f"[skip] train ds_yolo — output exists")

    # F-Sub (feature subtraction ablation)
    if force or not exists(RUNS_DS / "f_sub" / "weights" / "best.pt"):
        _train_ds("f_sub", fusion="sub", dry=dry)
    else:
        print(f"[skip] train f_sub — output exists")


def step_fraction(dry: bool, force: bool) -> None:
    """Data-fraction study (H3): train rgb + ds-yolo at 25 / 50 / 100 %."""
    data_rgb = DATASET_FIXED / "data.yaml"

    # 100% is already covered by step_train_all (rgb) and step_train_ds (ds_yolo)
    # plot_data_fraction.py reads runs/detect/rgb/ and runs/ds_yolo/ds_yolo/ for 100%
    for frac in [0.25, 0.50]:
        frac_tag = str(int(frac * 100))

        # RGB — Ultralytics always saves to runs/detect/<name>/
        name_rgb = f"rgb_{frac_tag}pct"
        out_rgb  = RUNS_DETECT / name_rgb / "weights" / "best.pt"
        if force or not exists(out_rgb):
            _train_variant("rgb", data_rgb, name=name_rgb, fraction=frac, dry=dry)
        else:
            print(f"[skip] fraction rgb {frac_tag}% — {out_rgb} exists")

        # DS-YOLO
        name_ds = f"ds_yolo_{frac_tag}pct"
        out_ds  = RUNS_FRACTION / name_ds / "weights" / "best.pt"
        if force or not exists(out_ds):
            run([sys.executable, "train_ds.py",
                 "--data",          str(data_rgb),
                 "--golden",        str(GOLDEN),
                 "--variant",       SIZE,
                 "--epochs",        str(EPOCHS),
                 "--imgsz",         str(IMGSZ),
                 "--batch",         str(BATCH),
                 "--name",          name_ds,
                 "--save-dir",      str(RUNS_FRACTION),
                 "--data-fraction", str(frac)], dry=dry)
        else:
            print(f"[skip] fraction ds-yolo {frac_tag}% — {out_ds} exists")


def step_robustness(dry: bool, force: bool) -> None:
    """Robustness study: evaluate rgb vs ds-yolo under alignment perturbation (Table V)."""
    pairs = [
        ("rgb",     RUNS_DETECT / "rgb"    / "weights" / "best.pt",
                    DATASET_FIXED / "data.yaml",
                    RUNS_ROBUST / "rgb.csv",
                    False),
        ("ds-yolo", RUNS_DS / "ds_yolo" / "weights" / "best.pt",
                    DATASET_FIXED / "data.yaml",
                    RUNS_ROBUST / "ds_yolo.csv",
                    True),
    ]
    for variant, weights, data, out, needs_golden in pairs:
        if not force and exists(out):
            print(f"[skip] robustness {variant} — {out} exists")
            continue
        if not exists(weights):
            print(f"[skip] robustness {variant} — weights not found: {weights}")
            continue
        cmd = [sys.executable, "eval/eval_robustness.py",
               "--variant", variant,
               "--weights", str(weights),
               "--data",    str(data),
               "--imgsz",   str(IMGSZ),
               "--out",     str(out)]
        if needs_golden:
            cmd += ["--golden", str(GOLDEN)]
        run(cmd, dry=dry)


def step_soldef_val(dry: bool, force: bool) -> None:
    """Train RGB baseline on SolDef_A (external generalizability check)."""
    name = "rgb_soldef"
    out  = RUNS_DETECT / name / "weights" / "best.pt"
    if not force and exists(out):
        print(f"[skip] soldef_val — {out} exists")
        return
    if not SOLDEF_DATA.exists():
        print(f"[skip] soldef_val — SolDef data.yaml not found: {SOLDEF_DATA}")
        return
    _train_variant("rgb", SOLDEF_DATA, name=name, dry=dry)


def step_tables(dry: bool, _force: bool) -> None:
    """Print all LaTeX table rows from aggregated results."""
    print("\n" + "=" * 60)
    print("TABLE II — Main results")
    print("=" * 60)
    # Order must match Table II in paper: RGB, Diff-only, Stack6, F-Sub, DS-YOLO
    run([sys.executable, "eval/aggregate_results.py", "main",
         "--runs",     str(RUNS_DETECT),
         "--ds-runs",  str(RUNS_DS),
         "--variants", "rgb", "diff-only", "stack6", "f-sub", "ds-yolo"],
        dry=dry)

    print("\n" + "=" * 60)
    print("TABLE III — Per-class mAP")
    print("=" * 60)
    run([sys.executable, "eval/aggregate_results.py", "perclass",
         "--runs",     str(RUNS_DETECT),
         "--ds-runs",  str(RUNS_DS),
         "--variants", "rgb", "ds-yolo",
         "--data",     str(DATASET_FIXED / "data.yaml"),
         "--golden",   str(GOLDEN),
         "--imgsz",    str(IMGSZ),
         "--split",    "test"],
        dry=dry)

    print("\n" + "=" * 60)
    print("TABLE IV — Latency")
    print("=" * 60)
    rgb_weights = RUNS_DETECT / "rgb" / "weights" / "best.pt"
    if exists(rgb_weights):
        run([sys.executable, "eval/aggregate_results.py", "latency",
             "--weights",    str(rgb_weights),
             "--imgsz",      str(IMGSZ),
             "--channels",   "3",
             "--runs-count", "200"],
            dry=dry)
    else:
        print("[skip] latency rgb — weights not found")

    print("\n" + "=" * 60)
    print("TABLE V — Robustness")
    print("=" * 60)
    # Table V compares RGB baseline vs DS-YOLO under alignment perturbation.
    robust_csvs = [
        (RUNS_ROBUST / "rgb.csv",      "RGB baseline"),
        (RUNS_ROBUST / "ds_yolo.csv",  r"DS-YOLO (ours)"),
    ]
    avail = [(str(p), lbl) for p, lbl in robust_csvs if p.exists()]
    if len(avail) >= 2:
        run([sys.executable, "eval/aggregate_results.py", "robust",
             "--csvs",   *[p for p, _ in avail],
             "--labels", *[lbl for _, lbl in avail]],
            dry=dry)
    else:
        print("[skip] robustness CSVs not found yet")


def step_figures(dry: bool, _force: bool) -> None:
    """Generate paper figures (requires trained weights)."""
    data_yaml = DATASET_FIXED / "data.yaml"
    rgb_w     = RUNS_DETECT / "rgb"    / "weights" / "best.pt"
    ds_w      = RUNS_DS     / "ds_yolo"/ "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("FIGURE 2 — Data fraction (H3)")
    print("=" * 60)
    run([sys.executable, "eval/plot_data_fraction.py"], dry=dry)

    print("\n" + "=" * 60)
    print("FIGURE 3 — Golden / Aligned / Diff")
    print("=" * 60)
    cmd_fig3 = [sys.executable, "eval/plot_qual_figures.py",
                "--data",     str(data_yaml),
                "--golden",   str(GOLDEN),
                "--imgsz",    str(IMGSZ),
                "--fig3-only"]
    run(cmd_fig3, dry=dry)

    print("\n" + "=" * 60)
    print("FIGURE 4 — RGB vs DS-YOLO detection")
    print("=" * 60)
    if exists(rgb_w) and exists(ds_w):
        cmd_fig4 = [sys.executable, "eval/plot_qual_figures.py",
                    "--data",     str(data_yaml),
                    "--golden",   str(GOLDEN),
                    "--imgsz",    str(IMGSZ),
                    "--rgb",      str(rgb_w),
                    "--ds-yolo",  str(ds_w)]
        run(cmd_fig4, dry=dry)
    else:
        print(f"[skip] Figure 4 — weights not found (need rgb + ds_yolo from train_ds)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_STEPS = [
    "resplit", "train_all", "train_ds",
    "fraction", "robustness", "soldef_val", "tables", "figures",
]

STEP_FN = {
    "resplit":    step_resplit,
    "train_all":  step_train_all,
    "train_ds":   step_train_ds,
    "fraction":   step_fraction,
    "robustness": step_robustness,
    "soldef_val": step_soldef_val,
    "tables":     step_tables,
    "figures":    step_figures,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps",   nargs="+", default=ALL_STEPS, choices=ALL_STEPS,
                   help="Which steps to run (default: all).")
    p.add_argument("--force",   action="store_true",
                   help="Re-run even if outputs already exist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    # Hyper-parameters (override the constants at top of file)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz",  type=int, default=None)
    p.add_argument("--batch",  type=int, default=None)
    p.add_argument("--size",   default=None, choices=["n", "s", "m", "l", "x"])
    args = p.parse_args()

    # Apply overrides to module-level constants so all step functions pick them up.
    global EPOCHS, IMGSZ, BATCH, SIZE
    if args.epochs is not None: EPOCHS = args.epochs
    if args.imgsz  is not None: IMGSZ  = args.imgsz
    if args.batch  is not None: BATCH  = args.batch
    if args.size   is not None: SIZE   = args.size

    print(f"[run_all] steps: {args.steps}")
    print(f"[run_all] epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}  size={SIZE}")
    print(f"[run_all] force={args.force}  dry={args.dry_run}")

    for step in args.steps:
        print(f"\n{'='*60}")
        print(f"STEP: {step}")
        print(f"{'='*60}")
        STEP_FN[step](dry=args.dry_run, force=args.force)

    print("\n[run_all] DONE")


if __name__ == "__main__":
    main()
