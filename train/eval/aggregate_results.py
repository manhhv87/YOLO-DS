"""Aggregate training/validation results into LaTeX tables.

Handles two CSV formats:
  * Ultralytics format  (train.py: rgb baseline)
  * train_ds.py format  (ds-yolo)

Usage::

    # Results table (no GPU needed)
    python aggregate_results.py main \\
        --runs    runs/detect \\
        --ds-runs runs/ds_yolo \\
        --variants rgb ds-yolo

    # Table III — per-class mAP (needs GPU)
    python aggregate_results.py perclass \\
        --runs    runs/detect \\
        --ds-runs runs/ds_yolo \\
        --variants rgb ds-yolo \\
        --data    dataset_fixed/data.yaml \\
        --golden  datasets/inhouse/golden/golden_inhouse.jpg \\
        --imgsz   640 --split test

    # Latency of a single model
    python aggregate_results.py latency \\
        --weights runs/detect/rgb/weights/best.pt \\
        --imgsz 1024 --runs-count 200

    # Robustness table (Table V) from eval_robustness.py CSV outputs
    python aggregate_results.py robust \\
        --csvs runs/robustness/rgb.csv runs/robustness/ds_yolo.csv \\
        --labels "RGB baseline" "DS-YOLO (ours)"
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
    "rgb":       "YOLOv8s (RGB)",
    "diff-only": "YOLOv8s (Diff-only)",
    "stack6":    "YOLOv8s (Stack6)",
    "f-sub":     "YOLOv8s (F-Sub)",
    "ds-yolo":   r"\textbf{DS-YOLO (ours)}",
    # SolDef_A pre-training ablations (step_soldef_finetune)
    "rgb-soldef-pre":     "YOLOv8s (RGB, SolDef pretrain)",
    "ds-yolo-soldef-pre": r"\textbf{DS-YOLO (SolDef pretrain)}",
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
    # Variants trained via train_ds.py → directory name inside args.ds_runs.
    _DS_VARIANT_TO_DIR = {
        "ds-yolo":            "ds_yolo",
        "ds-yolo-soldef-pre": "ds_yolo_soldef_pre",
        "f-sub":              "f_sub",
    }
    # Ultralytics variants whose directory name differs from the variant tag
    # (hyphens in tag vs underscores on disk, or custom --name used in training).
    _DETECT_VARIANT_TO_DIR = {
        "rgb-soldef-pre": "rgb_soldef_pre",
    }

    print(r"% --- paste below into Table II of paper/main.tex ---")
    # Integrity guard: detect two variants that report byte-identical metrics,
    # which almost always means a test_results.json was copied between run dirs
    # (the 'diff-only' == RGB bug). Surfacing it prevents shipping a fake row.
    _seen_sigs: dict = {}
    for variant in args.variants:
        if variant in _DS_VARIANT_TO_DIR:
            run_dir = Path(args.ds_runs) / _DS_VARIANT_TO_DIR[variant]
        elif variant in _DETECT_VARIANT_TO_DIR:
            run_dir = Path(args.runs) / _DETECT_VARIANT_TO_DIR[variant]
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

        _sig = (prec, recall, map50, map_)
        if "--" not in _sig and _sig in _seen_sigs:
            print(f"% [INTEGRITY WARNING] '{variant}' has metrics identical to "
                  f"'{_seen_sigs[_sig]}' {_sig} — almost certainly a copied "
                  f"test_results.json, not a distinct run. Verify before submitting.")
        elif "--" not in _sig:
            _seen_sigs[_sig] = variant

        print(f"{PRETTY.get(variant, variant)} & "
              f"{prec} & {recall} & {map50} & {map_} \\\\")


# Variant -> on-disk run-dir base name (shared by cmd_main and cmd_mainstats).
_DS_VARIANT_TO_DIR = {
    "ds-yolo":            "ds_yolo",
    "ds-yolo-soldef-pre": "ds_yolo_soldef_pre",
    "f-sub":              "f_sub",
}
_DETECT_VARIANT_TO_DIR = {
    "rgb-soldef-pre": "rgb_soldef_pre",
}


def cmd_mainstats(args: argparse.Namespace) -> None:
    """Build Table II rows as mean$\\pm$std over multiple seeds.

    Reads each seed run's test_results.json (canonical name for seed 0, with an
    ``_s{seed}`` suffix for seed>0, matching run_all.py's naming) and reports
    mean$\\pm$std for precision / recall / mAP@0.5 / mAP@0.5:0.95. A +0.7pp gap
    on an 11-image test set is only meaningful with this variance.
    """
    def _dir_for(variant: str, seed: int) -> Path:
        if variant in _DS_VARIANT_TO_DIR:
            base, root = _DS_VARIANT_TO_DIR[variant], Path(args.ds_runs)
        elif variant in _DETECT_VARIANT_TO_DIR:
            base, root = _DETECT_VARIANT_TO_DIR[variant], Path(args.runs)
        else:
            base, root = variant, Path(args.runs)
        name = base if seed == 0 else f"{base}_s{seed}"
        return root / name

    def _ms(xs: list[float]) -> str:
        if not xs:
            return "--"
        if len(xs) == 1:
            return f"{xs[0]:.3f}"
        return f"{statistics.fmean(xs):.3f}$\\pm${statistics.stdev(xs):.3f}"

    print(r"% --- Table II (mean$\pm$std over seeds) — paste into main.tex ---")
    print(f"% seeds = {args.seeds}")
    for variant in args.variants:
        vals: dict[str, list[float]] = {"precision": [], "recall": [], "map50": [], "map": []}
        n_found = 0
        for seed in args.seeds:
            td = _read_test_json(_dir_for(variant, seed))
            if not td:
                continue
            n_found += 1
            for k in vals:
                try:
                    vals[k].append(float(td[k]))
                except (KeyError, TypeError, ValueError):
                    pass
        if n_found == 0:
            print(f"% [missing] {variant}: no seed runs found "
                  f"(looked for {[str(_dir_for(variant, s)) for s in args.seeds]})")
            continue
        print(f"{PRETTY.get(variant, variant)} & {_ms(vals['precision'])} & "
              f"{_ms(vals['recall'])} & {_ms(vals['map50'])} & {_ms(vals['map'])} "
              f"\\\\  % n={n_found} seeds")


_ABLATION_PRETTY = {
    "ds_yolo":            r"DS-YOLO (P3+P4+P5, gate, learn.\ $\alpha$)",
    "ds_crfm_p4p5":       r"\quad fuse P4+P5 only",
    "ds_crfm_p5":         r"\quad fuse P5 only",
    "ds_crfm_nogate":     r"\quad no gate (additive diff)",
    "ds_crfm_fixedalpha": r"\quad fixed $\alpha{=}1$",
}


def cmd_ablation(args: argparse.Namespace) -> None:
    """Print the CRFM design-ablation table from each run's test_results.json."""
    print(r"% --- CRFM design ablation (paste into main.tex) ---")
    _seen: dict = {}
    for name in args.names:
        td = _read_test_json(Path(args.ds_runs) / name)
        if not td:
            print(f"% [missing] {name}")
            continue
        sig = (_fmt(td.get("precision")), _fmt(td.get("recall")),
               _fmt(td.get("map50")), _fmt(td.get("map")))
        if "--" not in sig and sig in _seen:
            print(f"% [INTEGRITY WARNING] {name} has metrics identical to "
                  f"{_seen[sig]} — likely a duplicate/copied run.")
        elif "--" not in sig:
            _seen[sig] = name
        label = _ABLATION_PRETTY.get(name, name)
        print(f"{label} & {sig[0]} & {sig[1]} & {sig[2]} & {sig[3]} \\\\")


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
        if variant in ("ds-yolo", "ds-yolo-soldef-pre"):
            # --- DS-YOLO path -------------------------------------------
            _ds_perclass_dir = {"ds-yolo": "ds_yolo", "ds-yolo-soldef-pre": "ds_yolo_soldef_pre"}
            run_name = _ds_perclass_dir[variant]
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
            # Reads variant/fusion from the checkpoint metadata and filters
            # thop bookkeeping buffers automatically.
            model  = DSYOLOv8m.from_checkpoint(str(weights), device=device)
            nc     = int(model.nc)

            with open(args.data) as f:
                data_yaml = _yaml.safe_load(f)
            # Resolve test path: use data.yaml's "path" key as root (Ultralytics
            # convention), falling back to the data.yaml's own directory.
            _data_root = _Path(data_yaml.get("path", _Path(args.data).parent))
            _test_rel  = data_yaml["test"]
            _test_abs  = str(_data_root / _test_rel) if not _Path(_test_rel).is_absolute() else _test_rel
            cfg = get_cfg(DEFAULT_CFG, overrides=dict(
                imgsz=args.imgsz, batch=4, mode="val", rect=False, cache=False))
            val_ds  = build_yolo_dataset(cfg, _test_abs, 4, data_yaml, mode="val")
            val_ldr = DataLoader(val_ds, batch_size=4, shuffle=False,
                                 num_workers=2, collate_fn=val_ds.collate_fn)
            golden  = load_golden(args.golden, args.imgsz, device)
            val_met = ds_evaluate(model, val_ldr, golden, device, nc=nc)

            # evaluate() already computes genuine per-class AP@0.5 in
            # per_class['ap50'] aligned to per_class['classes'] (the class
            # indices that actually appeared).  Map those to OK/NG instead of
            # duplicating the scalar overall mAP across both classes (the old
            # stub made Table III's DS-YOLO column and its delta meaningless).
            pc = val_met.get("per_class", {}) or {}
            ap50_by_cls = {int(c): float(v)
                           for c, v in zip(pc.get("classes", []), pc.get("ap50", []))}
            for i, cls in enumerate(CLASS_NAMES):
                rows[variant][cls] = ap50_by_cls.get(i, float("nan"))

        else:
            # --- Ultralytics path ----------------------------------------
            from ultralytics import YOLO
            weights = _Path(args.runs) / variant / "weights" / "best.pt"
            if not weights.is_file():
                print(f"% [missing weights] {weights}")
                continue
            import tempfile
            yolo    = YOLO(str(weights))
            with tempfile.TemporaryDirectory() as _tmp_val:
                metrics = yolo.val(data=args.data, imgsz=args.imgsz,
                                   split=args.split, verbose=False,
                                   project=_tmp_val, name="val")
            # ap50: per-class AP@IoU=0.5, aligned to ap_class_index
            ap50_by_cls = {int(c): float(v) for c, v in
                           zip(metrics.box.ap_class_index, metrics.box.ap50)}
            for i, cls in enumerate(CLASS_NAMES):
                rows[variant][cls] = ap50_by_cls.get(i, float("nan"))

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

    Reports Ultralytics' internal preprocess / inference / postprocess
    breakdown (averaged over --runs-count runs) for Table IV of the paper.
    Capture / alignment / OPC UA write times are measured separately on the
    production PC.
    """
    import numpy as np
    import torch
    from ultralytics import YOLO

    yolo  = YOLO(args.weights)
    img   = np.zeros((args.imgsz, args.imgsz, args.channels), dtype=np.uint8)
    cuda  = torch.cuda.is_available()

    def _sync():
        if cuda:
            torch.cuda.synchronize()

    # Warmup (not timed).
    for _ in range(20):
        yolo.predict(img, imgsz=args.imgsz, verbose=False)
    _sync()

    pre_ms, inf_ms, post_ms, wall_ms = [], [], [], []
    for _ in range(args.runs_count):
        _sync()
        t0 = time.perf_counter()
        results = yolo.predict(img, imgsz=args.imgsz, verbose=False)
        _sync()
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        spd = results[0].speed          # dict: preprocess / inference / postprocess
        pre_ms.append(spd.get("preprocess",  0.0))
        inf_ms.append(spd.get("inference",   0.0))
        post_ms.append(spd.get("postprocess", 0.0))

    def _stats(lst: list) -> str:
        return f"{statistics.fmean(lst):.2f} ± {statistics.stdev(lst):.2f} ms"

    print(f"% latency over {len(wall_ms)} runs  "
          f"imgsz={args.imgsz}  channels={args.channels}  "
          f"device={'cuda' if cuda else 'cpu'}")
    print(f"%   preprocess  = {_stats(pre_ms)}")
    print(f"%   inference   = {_stats(inf_ms)}")
    print(f"%   postprocess = {_stats(post_ms)}")
    print(f"%   wall total  = {_stats(wall_ms)}")
    total_mean = statistics.fmean(pre_ms) + statistics.fmean(inf_ms) + statistics.fmean(post_ms)
    print(f"%   stage sum   = {total_mean:.2f} ms")
    print(f"% --- Table IV row ---")
    print(f"YOLO inference & "
          f"{statistics.fmean(pre_ms):.1f} & "
          f"{statistics.fmean(inf_ms):.1f} & "
          f"{statistics.fmean(post_ms):.1f} & "
          f"{total_mean:.1f} \\\\")


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
                    default=["rgb", "ds-yolo"])
    pm.set_defaults(func=cmd_main)

    pms = sub.add_parser("mainstats", parents=[common],
                         help="Table II as mean+/-std over multiple seeds.")
    pms.add_argument("--variants", nargs="+",
                     default=["rgb", "ds-yolo"])
    pms.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    pms.set_defaults(func=cmd_mainstats)

    pa = sub.add_parser("ablation", parents=[common],
                        help="CRFM design-ablation table from test_results.json.")
    pa.add_argument("--names", nargs="+", required=True,
                    help="DS run-dir names under --ds-runs (ds_yolo ds_crfm_p5 ...).")
    pa.set_defaults(func=cmd_ablation)

    pc = sub.add_parser("perclass", parents=[common],
                        help="Build Table III by re-running val.")
    pc.add_argument("--variants", nargs="+", default=["rgb", "ds-yolo"])
    pc.add_argument("--data",   required=True,
                    help="data.yaml of the dataset used for evaluation.")
    pc.add_argument("--golden", default=None,
                    help="Golden image path (required when ds-yolo is in variants).")
    pc.add_argument("--imgsz", type=int, default=640)
    pc.add_argument("--split", default="test")
    pc.set_defaults(func=cmd_perclass)

    pl = sub.add_parser("latency", help="Time the YOLO forward pass.")
    pl.add_argument("--weights",    required=True)
    pl.add_argument("--imgsz",      type=int, default=640)
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
