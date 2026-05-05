"""Train one of the variants reported in the paper.

Examples
--------
Baseline (RGB)::

    python train.py --variant rgb       --data dataset_fixed/data.yaml   --epochs 60

Stack6 (RGB + Golden, 6-ch)::

    python train.py --variant stack6    --data dataset_fixed/data.yaml   --epochs 60

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

from models.model_4ch import patch_first_conv

VARIANT_CHANNELS = {
    "rgb": 3,
    "diff-only": 3,
    "stack6": 6,
    # f-sub / ds-yolo are trained via train_ds.py (dual-stream, not Ultralytics pipeline)
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANT_CHANNELS), required=True)
    p.add_argument("--data", required=True, help="Path to the dataset YAML.")
    p.add_argument("--size",    default="m", choices=["n", "s", "m", "l", "x"],
                   help="YOLOv8 backbone size (n/s/m/l/x). Default: m.")
    p.add_argument("--weights", default=None,
                   help="Pretrained weights path. Default: yolov8{size}.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--name", default=None)
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="Fraction of training data to use (for the H3 study).")
    p.add_argument("--mosaic", type=float, default=0.0,
                   help="Mosaic augmentation probability (0..1). DEFAULT 0.0 "
                        "because mosaic shrinks already-small SMD components by 4x, "
                        "and for single-board datasets it scrambles the fixed layout. "
                        "Override for diverse multi-board datasets.")
    args = p.parse_args()

    weights = args.weights or f"yolov8{args.size}.pt"
    in_channels = VARIANT_CHANNELS[args.variant]
    yolo = YOLO(weights)
    if in_channels != 3:
        patch_first_conv(yolo.model, in_channels=in_channels)

    # 4-channel and 6-channel variants store information beyond pixel intensity
    # in the extra channels (diff map, golden RGB).  Photometric augmentation
    # (HSV jitter) and the default fill value 114 used by mosaic / letterbox
    # would silently corrupt those extra channels — for the diff channel,
    # fill=114 looks like a strong fake defect everywhere.  Disable HSV for
    # multi-channel variants and force mosaic=0 so the diff signal stays
    # clean.
    multi_channel = in_channels != 3
    if multi_channel and args.mosaic > 0:
        print(f"[warn] --mosaic={args.mosaic} requested but variant has "
              f"{in_channels} channels: forcing mosaic=0.0 to preserve the "
              f"extra channel(s).")
        mosaic_p = 0.0
    else:
        mosaic_p = args.mosaic
    hsv_h = 0.0 if multi_channel else 0.015
    hsv_s = 0.0 if multi_channel else 0.7
    hsv_v = 0.0 if multi_channel else 0.4

    yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name or args.variant,
        fraction=args.data_fraction,
        # Disable strong geometric augmentations: they would force invariance
        # to exactly the cue we want to exploit.
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
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
