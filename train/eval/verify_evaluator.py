"""Cross-check: do the two mAP implementations agree?

Table II mixes metrics from two evaluators — Ultralytics ``yolo.val()`` for the
single-stream baselines and the hand-rolled ``train_ds.evaluate()`` for DS-YOLO.
A reviewer cannot trust a cross-variant delta unless the two evaluators agree on
the SAME model. This script scores one plain RGB checkpoint with BOTH and prints
the mAP@0.5 / mAP@0.5:0.95 difference. They should match to well under 0.1pp.

It works by wrapping the Ultralytics DetectionModel in a tiny adapter that
ignores the golden argument, so the DS-YOLO-oriented ``evaluate()`` can score it.

Usage
-----
    python eval/verify_evaluator.py \\
        --weights runs/detect/rgb/weights/best.pt \\
        --data    dataset_fixed/data.yaml \\
        --imgsz 640 --split test

Run this on the machine that has the dataset + weights + a GPU. A pass (|delta|
< 0.005) lets you report in the paper that the custom evaluator reproduces
Ultralytics' mAP, making Table II's cross-variant comparison sound.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG


class _RGBEvalAdapter(nn.Module):
    """Expose a plain Ultralytics DetectionModel through the DS-YOLO calling
    convention ``model(capture, golden)`` by ignoring ``golden``."""

    def __init__(self, det_model: nn.Module, nc: int) -> None:
        super().__init__()
        self.det = det_model
        self.nc = nc

    def forward(self, capture: torch.Tensor, golden=None, **_kw):  # noqa: D401
        return self.det(capture)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="A plain RGB best.pt.")
    p.add_argument("--data", required=True, help="data.yaml.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="test")
    p.add_argument("--tol", type=float, default=0.002,
                   help="Max allowed |delta| (on BOTH mAP@0.5 and mAP@0.5:0.95) "
                        "to count as a pass. Must be TIGHTER than the headline "
                        "DS-YOLO-vs-baseline gap (~0.007) for the check to be "
                        "meaningful.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train_ds.evaluate lives in train/ (parent of eval/).
    here = Path(__file__).parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from train_ds import evaluate as ds_evaluate

    with open(args.data) as f:
        data_yaml = yaml.safe_load(f)
    nc = int(data_yaml.get("nc", len(data_yaml.get("names", [0, 1]))))

    # --- (a) Ultralytics validator ---
    yolo = YOLO(args.weights)
    m = yolo.val(data=args.data, imgsz=args.imgsz, split=args.split, verbose=False)
    map50_ultra = float(m.box.map50)
    map_ultra = float(m.box.map)

    # --- (b) custom evaluate() on the same model + split ---
    data_root = Path(data_yaml.get("path", str(Path(args.data).parent)))
    if not data_root.is_absolute():
        data_root = Path(args.data).parent / data_root
    split_rel = data_yaml.get(args.split, f"{args.split}/images")
    split_abs = str(data_root / split_rel) if not Path(split_rel).is_absolute() else split_rel

    # rect=True matches Ultralytics yolo.val()'s default letterboxing; with a
    # mismatch the cross-check can report a spurious non-zero delta on
    # non-square data and wrongly blame NMS.
    cfg = get_cfg(DEFAULT_CFG, overrides=dict(
        imgsz=args.imgsz, batch=4, mode="val", rect=True, cache=False))
    ds = build_yolo_dataset(cfg, split_abs, 4, data_yaml, mode="val")
    ldr = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                     collate_fn=ds.collate_fn)

    adapter = _RGBEvalAdapter(yolo.model.to(device).eval(), nc).to(device).eval()
    met = ds_evaluate(adapter, ldr, None, device, nc=nc)
    map50_custom = float(met["map50"])
    map_custom = float(met["map"])

    d50 = map50_custom - map50_ultra
    d = map_custom - map_ultra
    print("\n================ evaluator cross-check ================")
    print(f"  mAP@0.5      ultralytics={map50_ultra:.4f}  custom={map50_custom:.4f}  "
          f"delta={d50:+.4f}")
    print(f"  mAP@0.5:0.95 ultralytics={map_ultra:.4f}  custom={map_custom:.4f}  "
          f"delta={d:+.4f}")
    # Gate on BOTH mAP@0.5 and mAP@0.5:0.95 — both are Table II columns.
    ok = (abs(d50) <= args.tol) and (abs(d) <= args.tol)
    print(f"  {'PASS' if ok else 'FAIL'}: max(|delta50|,|delta50-95|)="
          f"{max(abs(d50), abs(d)):.4f} {'<=' if ok else '>'} tol={args.tol}")
    print("=======================================================")
    if not ok:
        print("[hint] A delta this large means the two evaluators are NOT "
              "comparable; Table II cross-variant deltas are then unreliable. "
              "Ensure evaluate() uses the SAME ultralytics NMS as yolo.val() "
              "(from ultralytics.utils.nms, multi_label=True), and aligned "
              "conf/iou/max_det/letterbox.")
        sys.exit(1)


if __name__ == "__main__":
    main()
