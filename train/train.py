"""Train one of the variants reported in the paper.

Examples
--------
Baseline (RGB)::

    python train.py --variant rgb       --data dataset_fixed/data.yaml   --epochs 60

RGB on SolDef_A (external validation)::

    python train.py --variant rgb       --data soldef/yolo/data.yaml     --epochs 60 --name rgb_soldef

The script writes Ultralytics' results.csv into ``runs/detect/<variant>/``.
That CSV is then aggregated by ``aggregate_results.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

VARIANT_CHANNELS = {
    "rgb": 3,
    # 'diff-only' / 'stack6' removed: neither produced a real model through the
    # Ultralytics pipeline (diff-only had no diff input; the 6-ch stack6 patch is
    # discarded when yolo.train() rebuilds the 3-ch model), so both were just
    # duplicate RGB runs. f-sub / ds-yolo are trained via train_ds.py.
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANT_CHANNELS), required=True)
    p.add_argument("--data", required=True, help="Path to the dataset YAML.")
    p.add_argument("--size",    default="s", choices=["n", "s", "m", "l", "x"],
                   help="YOLOv8 backbone size (n/s/m/l/x). Default: s "
                        "(matches the paper and run_all.py).")
    p.add_argument("--weights", default=None,
                   help="Pretrained weights path. Default: yolov8{size}.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr0", type=float, default=0.01,
                   help="Initial LR. MUST match train_ds.py (SGD@0.01) so the "
                        "baseline-vs-DS-YOLO comparison isolates the architecture.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (vary it for the multi-seed study).")
    p.add_argument("--name", default=None)
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="Fraction of training data to use (for the H3 study).")
    p.add_argument("--mosaic", type=float, default=0.0,
                   help="Mosaic augmentation probability (0..1). DEFAULT 0.0 "
                        "because mosaic shrinks already-small SMD components by 4x, "
                        "and for single-board datasets it scrambles the fixed layout. "
                        "Override for diverse multi-board datasets.")
    # Geometric augmentations.  For the in-house single-board scenario these
    # are kept at 0.0 so capture↔golden alignment is preserved.  For datasets
    # with multiple board layouts (e.g. SolDef_A) and no golden reference
    # they should be enabled — pass ``--full-aug`` from run_all.py to flip
    # them all to the standard YOLOv8 defaults in one shot.
    p.add_argument("--degrees",     type=float, default=0.0)
    p.add_argument("--translate",   type=float, default=0.0)
    p.add_argument("--scale",       type=float, default=0.0)
    p.add_argument("--shear",       type=float, default=0.0)
    p.add_argument("--perspective", type=float, default=0.0)
    p.add_argument("--full-aug", action="store_true",
                   help="Use Ultralytics default augmentation (mosaic=1.0, "
                        "translate=0.1, scale=0.5).  Only for datasets with "
                        "multiple board layouts and no golden reference (e.g. "
                        "SolDef_A external validation).")
    args = p.parse_args()
    if args.full_aug:
        # Override only flags the user did NOT explicitly pass.  Standard
        # YOLOv8 defaults from ultralytics/cfg/default.yaml.
        if args.mosaic    == 0.0: args.mosaic    = 1.0
        if args.translate == 0.0: args.translate = 0.1
        if args.scale     == 0.0: args.scale     = 0.5

    weights = args.weights or f"yolov8{args.size}.pt"
    yolo = YOLO(weights)
    mosaic_p = args.mosaic
    hsv_h, hsv_s, hsv_v = 0.015, 0.7, 0.4

    yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name or args.variant,
        fraction=args.data_fraction,
        # --- Unified optimisation recipe: MUST match train_ds.py exactly so the
        # DS-YOLO-vs-baseline comparison isolates the CRFM architecture, not the
        # optimiser. Ultralytics' default optimizer='auto' silently selects
        # AdamW@~1e-3 with a LINEAR schedule on small datasets, which differed
        # from DS-YOLO's SGD@1e-2 + cosine and confounded the headline result. ---
        optimizer="SGD",
        lr0=args.lr0,
        lrf=0.01,              # final LR fraction -> matches custom cosine min
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        cos_lr=True,           # cosine LR (Ultralytics default is linear)
        seed=args.seed,
        deterministic=True,
        # ----------------------------------------------------------------------
        # Geometric augmentations (default 0.0 for the in-house alignment
        # scenario; --full-aug raises them to YOLOv8 defaults for SolDef_A).
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        # Photometric jitter only on plain RGB; otherwise corrupts diff/golden.
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        fliplr=0.5,
        mosaic=mosaic_p,
    )

    # ---- Test-set evaluation (unbiased, for paper reporting) ---------------
    save_dir = Path(yolo.trainer.save_dir) if hasattr(yolo, "trainer") else None
    try:
        run_name = args.name or args.variant
        m = yolo.val(data=args.data, imgsz=args.imgsz, split="test", verbose=False,
                     name=f"val_{run_name}")
        result = dict(
            precision=float(m.box.mp),
            recall=float(m.box.mr),
            map50=float(m.box.map50),
            map=float(m.box.map),
            split="test",
        )
        if save_dir:
            out = save_dir / "test_results.json"
            with out.open("w") as f:
                json.dump(result, f, indent=2)
            print(f"Test results saved to {out}")
    except Exception as e:
        print(f"[warn] test-set evaluation failed: {e}")


if __name__ == "__main__":
    main()
