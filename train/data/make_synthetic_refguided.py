"""Synthesise a REFERENCE-GUIDED detection benchmark from a single board.

WHY: On a single fixed board the golden reference is redundant — a capture-only
model memorises the one layout, so CRFM never learns (alpha stays 0). The golden
only becomes NECESSARY when the correct layout VARIES per sample and is revealed
ONLY by the golden. This script manufactures exactly that condition from the real
board images, so we can test whether CRFM can learn when the reference is required.

PER SAMPLE:
  1. Take a real board image + its component boxes.
  2. Randomly choose a PRESENT subset of components (random dropout) -> this is the
     sample's layout. The dropped components are inpainted out.
  3. golden = that layout, all components correct.
  4. capture = golden + defects on k present components:
        - 'missing' : component inpainted out of the capture (present in golden) ->
                      labelled NG at the golden position. A capture-only model sees
                      an empty slot indistinguishable from a non-slot UNLESS it
                      compares to the golden -> forces the reference.
        - 'shift'   : component moved -> labelled NG at the shifted position.
  5. Non-defective present components -> OK.

OUTPUT (per-sample golden, so train_ds must use --golden-dir):
    out/
      images/ <stem>.jpg
      labels/ <stem>.txt      (class cx cy w h ; 0=OK, 1=NG)
      golden/ <stem>.jpg      (the matching reference for THIS sample)
      data.yaml               (train/val/test split + names; 'golden: golden')

Usage:
    python data/make_synthetic_refguided.py \\
        --images datasets/inhouse/leaky/train/images \\
        --labels datasets/inhouse/leaky/train/labels \\
        --out dataset_synth --n 600 --keep 0.7 --defect-rate 0.25 --seed 0
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _source_of(stem: str) -> str:
    """Original capture name, stripping any Roboflow augmentation suffix.

    ``g1_png.rf.2be1...`` -> ``g1``, so augmented copies of one physical board
    stay together when splitting (no board leaks across train/val/test).
    """
    return stem.split("_png.rf.")[0] if "_png.rf." in stem else stem


def _read_boxes(lbl: Path):
    out = []
    if lbl.is_file():
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                out.append([float(x) for x in p[1:5]])  # cx,cy,w,h (drop class)
    return out


def _px(b, W, H):
    cx, cy, w, h = b
    x1 = int(round((cx - w / 2) * W)); y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W)); y2 = int(round((cy + h / 2) * H))
    return (max(0, min(x1, W - 1)), max(0, min(y1, H - 1)),
            max(1, min(x2, W)), max(1, min(y2, H)))


def _norm(x1, y1, x2, y2, W, H):
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def _inpaint(img, box_px):
    x1, y1, x2, y2 = box_px
    m = np.zeros(img.shape[:2], np.uint8)
    m[y1:y2, x1:x2] = 255
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)


def _shift(img, box_px, rng, max_frac=0.6):
    x1, y1, x2, y2 = box_px
    bw, bh = x2 - x1, y2 - y1
    if bw < 3 or bh < 3:
        return img, None
    H, W = img.shape[:2]
    patch = img[y1:y2, x1:x2].copy()
    img = _inpaint(img, box_px)
    for _ in range(8):
        dx = int(rng.uniform(-max_frac, max_frac) * bw)
        dy = int(rng.uniform(-max_frac, max_frac) * bh)
        if abs(dx) >= 3 or abs(dy) >= 3:
            break
    nx1 = int(np.clip(x1 + dx, 0, W - bw)); ny1 = int(np.clip(y1 + dy, 0, H - bh))
    img[ny1:ny1 + bh, nx1:nx1 + bw] = patch
    return img, (nx1, ny1, nx1 + bw, ny1 + bh)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=600, help="Total synthetic samples.")
    p.add_argument("--keep", type=float, default=0.7,
                   help="Fraction of components kept PRESENT per sample (rest dropped).")
    p.add_argument("--defect-rate", type=float, default=0.25,
                   help="Fraction of present components made defective (NG).")
    p.add_argument("--ops", nargs="+", default=["missing", "shift"],
                   choices=["missing", "shift"])
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels) if args.labels else img_dir.parent / "labels"
    imgs = sorted(q for q in img_dir.iterdir() if q.suffix.lower() in _IMG_EXTS)
    imgs = [q for q in imgs if _read_boxes(lbl_dir / (q.stem + ".txt"))]
    if not imgs:
        raise FileNotFoundError(f"No labelled images under {img_dir}")

    out = Path(args.out)
    splits = {"train": 1 - args.val_frac - args.test_frac,
              "val": args.val_frac, "test": args.test_frac}
    # Remove any stale split dirs first, so re-running (e.g. with a different
    # --n/--seed) cannot leave orphaned samples from a previous run.
    for s in splits:
        if (out / s).exists():
            shutil.rmtree(out / s)
    for s in splits:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
        (out / s / "golden").mkdir(parents=True, exist_ok=True)

    # BOARD-LEVEL split: group augmented copies by their ORIGINAL source capture,
    # then assign each source to exactly ONE split, so no physical board (and none
    # of its augmentations) appears in more than one of train/val/test.
    by_source: dict[str, list[Path]] = {}
    for q in imgs:
        by_source.setdefault(_source_of(q.stem), []).append(q)
    sources = sorted(by_source)
    order_src = list(range(len(sources)))
    rng.shuffle(order_src)
    n_test = max(1, int(round(args.test_frac * len(sources))))
    n_val = max(1, int(round(args.val_frac * len(sources))))
    img_by_split = {"train": [], "val": [], "test": []}
    for rank, k in enumerate(order_src):
        s = "test" if rank < n_test else ("val" if rank < n_test + n_val else "train")
        img_by_split[s].extend(by_source[sources[k]])
    if not img_by_split["train"]:
        raise ValueError("Too few labelled source captures for a board-level "
                         "split; lower --val-frac/--test-frac or add images.")
    # Assign each sample to a split (proportional); its base image is drawn only
    # from that split's disjoint pool of source augmentations.
    sample_split = []
    for s in ("train", "val", "test"):
        sample_split += [s] * max(1, int(round(splits[s] * args.n)))
    rng.shuffle(sample_split)

    n_ng = n_ok = 0
    manifest = []
    for i in range(len(sample_split)):
        split = sample_split[i]
        pool = img_by_split[split]
        src = pool[int(rng.integers(len(pool)))]
        img0 = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img0 is None:
            continue
        img0 = cv2.resize(img0, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        H, W = img0.shape[:2]
        boxes = _read_boxes(lbl_dir / (src.stem + ".txt"))
        order = list(range(len(boxes)))
        rng.shuffle(order)
        k_present = max(2, int(round(args.keep * len(boxes))))
        present = order[:k_present]
        absent = order[k_present:]

        # golden = drop the absent components (random per-sample layout)
        golden = img0.copy()
        for j in absent:
            golden = _inpaint(golden, _px(boxes[j], W, H))

        # capture = golden + defects on k present components
        cap = golden.copy()
        n_def = max(1, int(round(args.defect_rate * len(present))))
        rng.shuffle(present)
        defective = set(present[:n_def])
        lines = []
        for j in present:
            bpx = _px(boxes[j], W, H)
            if j in defective:
                op = random.choice(args.ops)
                if op == "missing":
                    cap = _inpaint(cap, bpx)
                    nb = _norm(*bpx, W, H)             # box stays at golden position
                else:
                    cap, moved = _shift(cap, bpx, rng)
                    nb = _norm(*moved, W, H) if moved else _norm(*bpx, W, H)
                lines.append(f"1 {nb[0]:.6f} {nb[1]:.6f} {nb[2]:.6f} {nb[3]:.6f}")
                n_ng += 1
            else:
                nb = _norm(*bpx, W, H)
                lines.append(f"0 {nb[0]:.6f} {nb[1]:.6f} {nb[2]:.6f} {nb[3]:.6f}")
                n_ok += 1

        stem = f"s{i:05d}"
        cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), cap)
        cv2.imwrite(str(out / split / "golden" / f"{stem}.jpg"), golden)
        (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n",
                                                             encoding="utf-8")
        manifest.append((stem, split, _source_of(src.stem)))

    data = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "train/images", "val": "val/images", "test": "test/images",
        "golden": "golden",      # per-sample golden dir (sibling of images/)
        "names": {0: "OK", 1: "NG"}, "nc": 2,
    }
    with (out / "data.yaml").open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    # Audit trail: which original board capture each sample came from, and its
    # split (lets you verify board-level disjointness across train/val/test).
    (out / "split_manifest.csv").write_text(
        "stem,split,source\n"
        + "\n".join(f"{s},{sp},{src}" for s, sp, src in manifest) + "\n",
        encoding="utf-8")

    print(f"[make_synthetic_refguided] {args.n} samples -> {out}")
    print(f"[make_synthetic_refguided] OK boxes={n_ok}  NG boxes={n_ng}  "
          f"(keep={args.keep}, defect_rate={args.defect_rate}, ops={args.ops})")
    print(f"[make_synthetic_refguided] per-sample golden in <split>/golden/ — "
          f"train with train_ds.py --golden-dir")


if __name__ == "__main__":
    main()
