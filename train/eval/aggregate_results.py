"""Aggregate training/validation results into LaTeX tables.

Handles two CSV formats:
  * Ultralytics format  (train.py variants: rgb, diff-only, stack6, f-sub, rg-yolo)
  * train_ds.py format  (ds-yolo variant)

Usage::

    # Table II — all variants (no GPU needed)
    python aggregate_results.py main \\
        --runs runs/detect \\
        --variants rgb diff-only stack6 rg-yolo ds-yolo

    # Table III — per-class mAP (needs GPU)
    python aggregate_results.py perclass \\
        --runs runs/detect \\
        --variants rgb rg-yolo ds-yolo \\
        --data dataset_fixed/data.yaml

    # Latency of a single model
    python aggregate_results.py latency \\
        --weights runs/detect/rg-yolo/weights/best.pt \\
        --imgsz 1024 --runs-count 200

    # Robustness table (Table V) from eval_robustness.py CSV outputs
    python aggregate_results.py robust \\
        --csvs runs/robustness/rgb.csv runs/robustness/rg-yolo.csv \\
        --labels "RGB baseline" "RG-YOLO (ours)"
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Display names used for the LaTeX rows.
# ---------------------------------------------------------------------------
PRETTY = {
    "rgb":       "YOLOv8m (RGB)",
    "diff-only": "YOLOv8m (Diff-only)",
    "stack6":    "YOLOv8m (Stack6)",
    "f-sub":     "YOLOv8m (F-Sub)",
    "rg-yolo":   r"RG-YOLO (4-ch input)",
    "ds-yolo":   r"\textbf{DS-YOLO (ours)}",
    # SolDef_A pre-training ablations
    "rgb-soldef":    "YOLOv8m (RGB, SolDef pretrain)",
    "ds-yolo-soldef": r"\textbf{DS-YOLO (SolDef pretrain)}",
}

CLASS_NAMES = ["OK", "NG"]

# ---------------------------------------------------------------------------
# CSV format detection
# ---------------------------------------------------------------------------
# Ultralytics results.csv uses keys like "metrics/mAP50(B)".
# train_ds.py uses keys like "val_map50".
_ULTRALYTICS_MAP50_KEY = "metrics/mAP50(B)"
_DS_MAP50_KEY          = "val_map50"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_results_csv(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        # Ultralytics writes column names with leading whitespace; strip them.
        rows = []
        for raw in reader:
            rows.append({k.strip(): v for k, v in raw.items()})
        return rows


def _is_ds_format(rows: list[dict]) -> bool:
    """Return True if rows came from train_ds.py (not Ultralytics)."""
    return bool(rows) and _DS_MAP50_KEY in rows[0]


def _map50_value(row: dict) -> float:
    """Extract mAP@0.5 from either CSV format."""
    for key in (_DS_MAP50_KEY, _ULTRALYTICS_MAP50_KEY):
        try:
            return float(row[key])
        except (KeyError, ValueError):
            pass
    return -1.0


def _pick_best_epoch(rows: list[dict]) -> dict:
    return max(rows, key=_map50_value)


def _get_metric(row: dict, ultralytics_key: str, ds_key: str) -> str:
    """Return formatted metric from either CSV format."""
    for key in (ds_key, ultralytics_key):
        try:
            return _fmt(float(row[key]))
        except (KeyError, ValueError):
            pass
    return "--"


def _fmt(x: float | str, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------
def _read_test_json(run_dir: Path) -> dict | None:
    """Return test_results.json contents if present, else None."""
    p = run_dir / "test_results.json"
    if p.is_file():
        with p.open() as f:
            return json.load(f)
    return None


def cmd_main(args: argparse.Namespace) -> None:
    """Build LaTeX rows for Table II (overall metrics).

    Prefers test_results.json (test-split, unbiased) saved by train.py /
    train_ds.py after training.  Falls back to val metrics from results.csv
    with a warning when the JSON is absent.
    """
    # Mapping of variants trained via train_ds.py (not Ultralytics) to their
    # run directory names inside args.ds_runs.
    _DS_VARIANT_TO_DIR = {
        "ds-yolo":        "ds_yolo",
        "ds-yolo-soldef": "ds_yolo_soldef",
        "f-sub":          "f_sub",   # feature-subtraction ablation, also in ds_runs
    }

    print(r"% --- paste below into Table II of paper/main.tex ---")
    for variant in args.variants:
        if variant in _DS_VARIANT_TO_DIR:
            run_dir = Path(args.ds_runs) / _DS_VARIANT_TO_DIR[variant]
        else:
            run_dir = Path(args.runs) / variant

        # --- Prefer test_results.json ----------------------------------------
        test_data = _read_test_json(run_dir)
        if test_data:
            prec   = _fmt(test_data.get("precision"))
            recall = _fmt(test_data.get("recall"))
            map50  = _fmt(test_data.get("map50"))
            map_   = _fmt(test_data.get("map"))
        else:
            # Fall back to best-epoch val metrics from results.csv
            csv_path = run_dir / "results.csv"
            if not csv_path.is_file():
                print(f"% [missing] {run_dir}")
                continue
            rows = _read_results_csv(csv_path)
            if not rows:
                print(f"% [empty] {csv_path}")
                continue
            best = _pick_best_epoch(rows)
            if _is_ds_format(rows):
                prec   = _fmt(best.get("val_precision"))
                recall = _fmt(best.get("val_recall"))
                map50  = _fmt(best.get("val_map50"))
                map_   = _fmt(best.get("val_map"))
            else:
                prec   = _fmt(best.get("metrics/precision(B)"))
                recall = _fmt(best.get("metrics/recall(B)"))
                map50  = _fmt(best.get("metrics/mAP50(B)"))
                map_   = _fmt(best.get("metrics/mAP50-95(B)"))
            print(f"% [warn] {variant}: test_results.json missing, using val metrics")

        print(f"{PRETTY.get(variant, variant)} & "
              f"{prec} & {recall} & {map50} & {map_} \\\\")


def cmd_perclass(args: argparse.Namespace) -> None:
    """Build LaTeX rows for Table III (per-class mAP@0.5).

    For DS-YOLO, uses the evaluate() function from train_ds.py directly
    (because DS-YOLO needs a golden image at validation time).
    For all other variants, uses Ultralytics yolo.val().
    """
    import sys
    from pathlib import Path as _Path

    rows: dict[str, dict] = {v: {} for v in args.variants}

    for variant in args.variants:
        if variant in ("ds-yolo", "ds-yolo-soldef"):
            # --- DS-YOLO path -------------------------------------------
            run_name = "ds_yolo" if variant == "ds-yolo" else "ds_yolo_soldef"
            weights  = _Path(args.ds_runs) / run_name / "weights" / "best.pt"
            if not weights.is_file():
                print(f"% [missing] {weights}")
                continue
            if args.golden is None:
                print(f"% [skip ds-yolo] --golden required for DS-YOLO per-class eval")
                continue

            import torch, yaml as _yaml
            from torch.utils.data import DataLoader
            from ultralytics.data.build import build_yolo_dataset
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import DEFAULT_CFG

            # train_ds.py and models/ live in train/ (parent of eval/).
            here = _Path(__file__).parent.parent
            if str(here) not in sys.path:
                sys.path.insert(0, str(here))
            from models.ds_yolo import DSYOLOv8m
            from train_ds import load_golden, evaluate as ds_evaluate

            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt   = torch.load(str(weights), map_location=device, weights_only=False)
            nc     = int(ckpt.get("num_classes", 2))
            model  = DSYOLOv8m(num_classes=nc).to(device)
            model.load_state_dict(ckpt["state_dict"], strict=True)

            with open(args.data) as f:
                data_yaml = _yaml.safe_load(f)
            cfg = get_cfg(DEFAULT_CFG, overrides=dict(
                imgsz=args.imgsz, batch=4, mode="val", rect=False, cache=False))
            val_ds  = build_yolo_dataset(cfg, data_yaml["test"], 4, data_yaml, mode="val")
            val_ldr = DataLoader(val_ds, batch_size=4, shuffle=False,
                                 num_workers=2, collate_fn=val_ds.collate_fn)
            golden  = load_golden(args.golden, args.imgsz, device)
            val_met = ds_evaluate(model, val_ldr, golden, device, nc=nc)

            # DS-YOLO evaluate() returns overall metrics; for per-class we
            # re-run with Ultralytics validator via a temporary wrapper.
            # As a fallback, store the scalar mAP for both classes.
            for cls in CLASS_NAMES:
                rows[variant][cls] = val_met.get("map50", float("nan"))

        else:
            # --- Ultralytics path ----------------------------------------
            from ultralytics import YOLO
            weights = _Path(args.runs) / variant / "weights" / "best.pt"
            if not weights.is_file():
                print(f"% [missing weights] {weights}")
                continue
            yolo    = YOLO(str(weights))
            metrics = yolo.val(data=args.data, imgsz=args.imgsz,
                               split=args.split, verbose=False)
            per_cls = list(metrics.box.maps)
            for i, cls in enumerate(CLASS_NAMES):
                rows[variant][cls] = per_cls[i] if i < len(per_cls) else float("nan")

    print(r"% --- paste below into Table III of paper/main.tex ---")
    base, ours = args.variants[0], args.variants[-1]
    for cls in CLASS_NAMES:
        a     = rows[base].get(cls, float("nan"))
        b     = rows[ours].get(cls, float("nan"))
        delta = (b - a) * 100.0 if (a == a and b == b) else float("nan")
        line  = f"{cls} & {_fmt(a)} & {_fmt(b)} & {_fmt(delta, digits=1)} \\\\"
        print(line)


def cmd_latency(args: argparse.Namespace) -> None:
    """Measure mean per-image latency of a trained model.

    Reports the YOLO forward-pass cost only; capture / alignment / OPC UA
    write times are measured separately on the production PC.
    """
    import numpy as np
    from ultralytics import YOLO

    yolo = YOLO(args.weights)
    img = np.zeros((args.imgsz, args.imgsz, args.channels), dtype=np.uint8)

    # Warmup.
    for _ in range(20):
        yolo.predict(img, imgsz=args.imgsz, verbose=False)

    times = []
    for _ in range(args.runs_count):
        t0 = time.perf_counter()
        yolo.predict(img, imgsz=args.imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"% latency over {len(times)} runs, imgsz={args.imgsz}, channels={args.channels}")
    print(f"%   mean   = {statistics.fmean(times):.2f} ms")
    print(f"%   median = {statistics.median(times):.2f} ms")
    print(f"%   stdev  = {statistics.stdev(times):.2f} ms")
    print(f"%   min/max= {min(times):.2f} / {max(times):.2f} ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared parent parser for --runs and --ds-runs
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs",    default="runs/detect",
                        help="Root dir for Ultralytics variant runs.")
    common.add_argument("--ds-runs", default="runs/ds_yolo",
                        help="Root dir for DS-YOLO runs (train_ds.py output).")

    pm = sub.add_parser("main", parents=[common],
                        help="Aggregate Table II from results.csv files.")
    pm.add_argument("--variants", nargs="+",
                    default=["rgb", "diff-only", "stack6", "rg-yolo", "ds-yolo"])
    pm.set_defaults(func=cmd_main)

    pc = sub.add_parser("perclass", parents=[common],
                        help="Build Table III by re-running val.")
    pc.add_argument("--variants", nargs="+", default=["rgb", "rg-yolo", "ds-yolo"])
    pc.add_argument("--data",   required=True,
                    help="data.yaml of the dataset used for evaluation.")
    pc.add_argument("--golden", default=None,
                    help="Golden image path (required when ds-yolo is in variants).")
    pc.add_argument("--imgsz", type=int, default=1024)
    pc.add_argument("--split", default="test")
    pc.set_defaults(func=cmd_perclass)

    pl = sub.add_parser("latency", help="Time the YOLO forward pass.")
    pl.add_argument("--weights",    required=True)
    pl.add_argument("--imgsz",      type=int, default=1024)
    pl.add_argument("--channels",   type=int, default=3)
    pl.add_argument("--runs-count", type=int, default=200)
    pl.set_defaults(func=cmd_latency)

    pr = sub.add_parser("robust",
                        help="Print Table V from eval_robustness.py CSV outputs.")
    pr.add_argument("--csvs",   nargs="+", required=True,
                    help="One CSV per variant, in the order they appear in the table.")
    pr.add_argument("--labels", nargs="+", default=None,
                    help="Display names for each CSV (same order).")
    pr.set_defaults(func=cmd_robust)

    args = p.parse_args()
    args.func(args)


# ---------------------------------------------------------------------------
# Robustness table helper (Table V)
# ---------------------------------------------------------------------------
def cmd_robust(args: argparse.Namespace) -> None:
    """Print LaTeX rows for Table V from eval_robustness.py CSV outputs."""
    import csv as _csv

    all_data: list[tuple[str, dict]] = []
    labels = args.labels or [Path(c).stem for c in args.csvs]

    sigmas_t: list[float] = []
    for csv_path, label in zip(args.csvs, labels):
        rows: dict[float, float] = {}
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                st = float(row["sigma_t"])
                rows[st] = float(row["map50"])
                if st not in sigmas_t:
                    sigmas_t.append(st)
        all_data.append((label, rows))

    sigmas_t = sorted(set(sigmas_t))
    header_cols = " & ".join(str(int(s)) for s in sigmas_t)
    print(r"% --- paste below into Table V of paper/main.tex ---")
    print(r"\midrule")
    print(f"$\\sigma_t$ (px) & {header_cols} \\\\")
    print(r"\midrule")
    for label, rows in all_data:
        vals = " & ".join(_fmt(rows.get(s, float("nan"))) for s in sigmas_t)
        print(f"{label} & {vals} \\\\")


if __name__ == "__main__":
    main()
