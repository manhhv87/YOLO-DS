"""Pre-compute 4-channel (RGB + Diff) tensors for a YOLO dataset.

Input layout (standard YOLO dataset):

    dataset/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt
        golden.jpg

Output layout:

    dataset_rg/
        images/{train,val,test}/*.jpg   # original source jpg (copied verbatim)
        images/{train,val,test}/*.npy   # (H, W, 4) uint8  channels: RGB + diff
        labels/{train,val,test}/*.txt   # copied verbatim
        rg-yolo.yaml

How Ultralytics loads 4-channel data:
  BaseDataset builds self.npy_files = [Path(f).with_suffix(".npy") for f in im_files].
  When loading image i, if the .npy file exists it is loaded with np.load() instead
  of cv2.imread(), giving all 4 channels directly.  The source .jpg must also be
  present so that get_img_files() includes the image in the dataset.

Channel order in the .npy: RGB + grayscale diff (index 3).
This matches YOLOv8 pretrained weight expectations (Ultralytics converts BGR→RGB
for standard images, so pretrained first-conv weights expect RGB order).

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
    n_ok = n_fail = n_align_fail = 0
    align_inliers = []
    for img_path in sorted(src_split.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        cap = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if cap is None:
            n_fail += 1
            continue
        try:
            aligned, _, inlier_ratio = align_to_golden(cap, golden)
            align_inliers.append(float(inlier_ratio))
        except RuntimeError:
            # Fallback: keep the unaligned capture.  Track these so the user
            # knows whether the diff channel is reliable.
            aligned = cv2.resize(cap, (golden.shape[1], golden.shape[0]))
            n_align_fail += 1
        diff = difference_map(aligned, golden, grayscale=True)
        # Channel order in npy: [D, B, G, R]
        # Ultralytics applies BGR→RGB reversal (im[::-1] on CHW) to loaded npy,
        # turning [D, B, G, R] → [R, G, B, D] which is what the model expects.
        diff_ch = diff[..., np.newaxis] if diff.ndim == 2 else diff  # (H, W, 1)
        dbgr = np.dstack([diff_ch, aligned]).astype(np.uint8)        # (H, W, 4) D+BGR
        # Save 4-channel data as .npy — Ultralytics loads this automatically
        # when a .npy with the same stem exists alongside the source image.
        np.save(dst_split / (img_path.stem + ".npy"), dbgr)
        # Also copy the source jpg so get_img_files() can discover the image.
        shutil.copy(img_path, dst_split / img_path.name)
        n_ok += 1
    if align_inliers:
        m = float(np.mean(align_inliers))
        lo = float(np.min(align_inliers))
        print(f"  alignment inlier-ratio: mean={m:.2f}  min={lo:.2f}  "
              f"(<0.4 = poor, diff channel is noise)")
    if n_align_fail:
        print(f"  WARNING: {n_align_fail} images fell back to no-align "
              f"(diff channel is unreliable on those)")
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
