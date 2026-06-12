"""Synthesise controlled NG samples by injecting placement defects.

The in-house board is photographed in its known-good state, so almost every
labelled component is OK -> the NG class is tiny and the mAP saturates. This
tool enlarges and diversifies the NG population on the SAME board by applying
realistic placement defects to randomly chosen components and relabelling those
boxes as NG. It turns "recognise this memorised board" into the intended task:
"detect where the board differs from the golden reference".

Defect operations (per chosen component box):
  * remove : inpaint the component away (missing component)
  * shift  : move the component patch by a few % of its size (mis-placement /
             tombstone), inpaint the original location, and move its box
  * rotate : rotate the patch 90/180 deg in place (mis-orientation)

Inputs are YOLO-format images + labels; outputs are new image/label pairs with
the affected boxes set to the NG class. Non-affected boxes keep their original
class, so a single synthesised image realistically contains both OK and NG
components.

Usage
-----
    python data/inject_defects.py \\
        --images dataset_fixed/train/images \\
        --out    dataset_fixed/train_injected \\
        --copies 3 --per-image 2 --ops remove shift rotate --seed 0

Then point training at the merged set (originals + injected), keeping the
source-level split intact (only inject within each split separately).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _read_labels(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if lbl_path.is_file():
        for line in lbl_path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                out.append((int(float(p[0])), *map(float, p[1:5])))
    return out


def _to_px(box, W, H):
    _, cx, cy, w, h = box
    x1 = int(round((cx - w / 2) * W)); y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W)); y2 = int(round((cy + h / 2) * H))
    x1, x2 = max(0, min(x1, W - 1)), max(1, min(x2, W))
    y1, y2 = max(0, min(y1, H - 1)), max(1, min(y2, H))
    return x1, y1, x2, y2


def _to_norm(x1, y1, x2, y2, W, H):
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def op_remove(img, box_px):
    x1, y1, x2, y2 = box_px
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = 255
    out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return out, None  # box position unchanged


def op_rotate(img, box_px, rng):
    x1, y1, x2, y2 = box_px
    patch = img[y1:y2, x1:x2].copy()
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return img, None
    k = rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE])
    rot = cv2.rotate(patch, k)
    rot = cv2.resize(rot, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    img[y1:y2, x1:x2] = rot
    return img, None


def op_shift(img, box_px, rng, max_frac=0.4):
    x1, y1, x2, y2 = box_px
    bw, bh = x2 - x1, y2 - y1
    if bw < 3 or bh < 3:
        return img, None
    H, W = img.shape[:2]
    patch = img[y1:y2, x1:x2].copy()
    # remove component from original location
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = 255
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    # pick a non-trivial shift
    for _ in range(8):
        dx = int(rng.uniform(-max_frac, max_frac) * bw)
        dy = int(rng.uniform(-max_frac, max_frac) * bh)
        if abs(dx) >= 2 or abs(dy) >= 2:
            break
    nx1 = int(np.clip(x1 + dx, 0, W - bw)); ny1 = int(np.clip(y1 + dy, 0, H - bh))
    nx2, ny2 = nx1 + bw, ny1 + bh
    img[ny1:ny2, nx1:nx2] = patch
    return img, (nx1, ny1, nx2, ny2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="Dir of source images (YOLO).")
    p.add_argument("--labels", default=None,
                   help="Dir of YOLO labels (default: sibling 'labels' of --images).")
    p.add_argument("--out", required=True, help="Output dir (creates images/ + labels/).")
    p.add_argument("--copies", type=int, default=2,
                   help="Synthetic NG copies per source image.")
    p.add_argument("--per-image", type=int, default=2,
                   help="Number of components to corrupt per copy.")
    p.add_argument("--ops", nargs="+", default=["remove", "shift", "rotate"],
                   choices=["remove", "shift", "rotate"])
    p.add_argument("--ng-class", type=int, default=1, help="Class index for NG.")
    p.add_argument("--max-shift", type=float, default=0.4)
    p.add_argument("--include-original", action="store_true",
                   help="Also copy the originals into --out for a self-contained set.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels) if args.labels else img_dir.parent / "labels"
    out_img = Path(args.out) / "images"; out_img.mkdir(parents=True, exist_ok=True)
    out_lbl = Path(args.out) / "labels"; out_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted(q for q in img_dir.iterdir() if q.suffix.lower() in _IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No images under {img_dir}")

    n_written = 0
    n_ng_boxes = 0
    for ip in imgs:
        boxes = _read_labels(lbl_dir / (ip.stem + ".txt"))
        if not boxes:
            continue
        img0 = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img0 is None:
            continue
        H, W = img0.shape[:2]

        if args.include_original:
            cv2.imwrite(str(out_img / ip.name), img0)
            (out_lbl / (ip.stem + ".txt")).write_text(
                "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                          for c, cx, cy, w, h in boxes) + "\n", encoding="utf-8")

        for ci in range(args.copies):
            img = img0.copy()
            new_boxes = [list(b) for b in boxes]
            k = min(args.per_image, len(new_boxes))
            victims = rng.choice(len(new_boxes), size=k, replace=False)
            for vi in victims:
                op = random.choice(args.ops)
                box_px = _to_px(new_boxes[vi], W, H)
                if op == "remove":
                    img, moved = op_remove(img, box_px)
                elif op == "rotate":
                    img, moved = op_rotate(img, box_px, rng)
                else:
                    img, moved = op_shift(img, box_px, rng, args.max_shift)
                # relabel as NG and update geometry if the box moved
                new_boxes[vi][0] = args.ng_class
                if moved is not None:
                    cx, cy, w, h = _to_norm(*moved, W, H)
                    new_boxes[vi][1:5] = [cx, cy, w, h]
                n_ng_boxes += 1

            stem = f"{ip.stem}_inj{ci}"
            cv2.imwrite(str(out_img / f"{stem}{ip.suffix}"), img)
            (out_lbl / f"{stem}.txt").write_text(
                "\n".join(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                          for c, cx, cy, w, h in new_boxes) + "\n", encoding="utf-8")
            n_written += 1

    print(f"[inject_defects] wrote {n_written} synthetic images, "
          f"{n_ng_boxes} injected NG boxes -> {args.out}")
    print(f"[inject_defects] ops={args.ops} copies={args.copies} "
          f"per_image={args.per_image} ng_class={args.ng_class}")


if __name__ == "__main__":
    main()
