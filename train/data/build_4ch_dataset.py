"""Pre-compute 4-channel (BGR + Diff) images for a YOLO dataset.

Input layout (standard YOLO dataset):

    dataset/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt
        golden.jpg

Output layout (saved as 4-channel PNG, plus an updated YAML):

    dataset_rg/
        images/{train,val,test}/*.png   # (H, W, 4) uint8  channels: BGR + diff
        labels/{train,val,test}/*.txt   # copied verbatim
        rg-yolo.yaml                    # includes channels: 4

Channel order: BGR (OpenCV convention) + grayscale diff as 4th channel.
Ultralytics loads with cv2.IMREAD_UNCHANGED (4 channels) and converts
BGR→RGB internally, leaving the diff channel at index 3.

Run::

    python data/build_4ch_dataset.py \
        --src    dataset_fixed \
        --dst    dataset_rg \
        --golden datasets/inhouse/golden/golden_ok.bmp
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from align import align_to_golden, difference_map


def process_split(src_split: Path, dst_split: Path, golden: np.ndarray) -> tuple[int, int]:
    dst_split.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    for img_path in sorted(src_split.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        cap = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if cap is None:
            n_fail += 1
            continue
        try:
            aligned, _, _ = align_to_golden(cap, golden)
        except RuntimeError:
            # Fallback: keep the unaligned capture; difference will be
            # noisier but training does not crash.
            aligned = cv2.resize(cap, (golden.shape[1], golden.shape[0]))
        diff = difference_map(aligned, golden, grayscale=True)
        # Keep BGR order (OpenCV convention); Ultralytics converts BGR→RGB internally.
        # Diff map is appended as 4th channel and is not affected by the BGR→RGB swap.
        bgrd = np.dstack([aligned, diff]).astype(np.uint8)   # (H, W, 4)
        cv2.imwrite(str(dst_split / (img_path.stem + ".png")), bgrd)
        n_ok += 1
    return n_ok, n_fail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    golden = cv2.imread(str(args.golden), cv2.IMREAD_COLOR)
    if golden is None:
        raise SystemExit(f"Cannot read golden image: {args.golden}")

    for split in args.splits:
        src_imgs = args.src / split / "images"
        dst_imgs = args.dst / split / "images"
        if not src_imgs.is_dir():
            print(f"[skip] {src_imgs} does not exist")
            continue
        n_ok, n_fail = process_split(src_imgs, dst_imgs, golden)
        print(f"[{split}] {n_ok} processed, {n_fail} failed")

        # Copy labels verbatim.
        src_lbl = args.src / split / "labels"
        dst_lbl = args.dst / split / "labels"
        if src_lbl.is_dir():
            dst_lbl.mkdir(parents=True, exist_ok=True)
            for lbl in src_lbl.glob("*.txt"):
                shutil.copy(lbl, dst_lbl / lbl.name)

    args.dst.mkdir(parents=True, exist_ok=True)
    yaml_out = args.dst / "rg-yolo.yaml"
    with yaml_out.open("w") as f:
        yaml.safe_dump(
            {
                "path": str(args.dst.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "OK", 1: "NG"},
                "channels": 4,
            },
            f,
            sort_keys=False,
        )
    print(f"[done] wrote {yaml_out}")


if __name__ == "__main__":
    main()
