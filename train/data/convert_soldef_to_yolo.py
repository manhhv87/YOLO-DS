"""Convert the SolDef_A dataset (LabelMe polygon JSON) to a YOLO bbox dataset.

SolDef_A is provided with LabelMe-style polygon annotations:

    Labeled/
        WIN_20220329_14_30_32_Pro.jpg
        WIN_20220329_14_30_32_Pro.json   # polygons + labels
        ...

This script writes a Ultralytics YOLO-compatible folder:

    soldef_yolo/
        data.yaml
        train/{images,labels}/
        val/{images,labels}/
        test/{images,labels}/
        class_counts.json

Each polygon's axis-aligned bounding box is written as one YOLO line
(``cls_id cx cy w h``, all normalised to ``[0,1]``).

Two label modes:
  * ``--mode binary``  (default) -- ``good`` -> 0 (OK), every other label -> 1 (NG)
  * ``--mode multi``           -- 5 classes: good / exc_solder / spike / no_good / poor_solder

Usage::

    python convert_soldef_to_yolo.py \
        --src soldef/raw/Labeled \
        --dst soldef/yolo \
        --mode binary --val-frac 0.15 --test-frac 0.15 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
BINARY_MAP = {
    "good": 0,
    "exc_solder": 1,
    "spike": 1,
    "no_good": 1,
    "poor_solder": 1,
}
BINARY_NAMES = {0: "OK", 1: "NG"}

MULTI_MAP = {
    "good":        0,
    "exc_solder":  1,
    "spike":       2,
    "no_good":     3,
    "poor_solder": 4,
}
MULTI_NAMES = {v: k for k, v in MULTI_MAP.items()}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def polygon_to_yolo_bbox(points: list[list[float]], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """AABB of polygon -> (cx, cy, w, h) normalised."""
    xs = [p[0] for p in points if len(p) >= 2]
    ys = [p[1] for p in points if len(p) >= 2]
    if not xs or not ys:
        return None
    x1, x2 = max(min(xs), 0.0), min(max(xs), float(img_w))
    y1, y2 = max(min(ys), 0.0), min(max(ys), float(img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def convert_one(json_path: Path, label_map: dict[str, int]) -> tuple[Path, list[str], Counter]:
    """Convert one LabelMe JSON to YOLO label lines + return image path + per-class counts."""
    with json_path.open(encoding="utf-8") as f:
        d = json.load(f)
    img_w = int(d["imageWidth"])
    img_h = int(d["imageHeight"])
    img_path = json_path.with_name(d.get("imagePath", json_path.stem + ".jpg"))
    if not img_path.is_file():
        # LabelMe sometimes stores absolute paths; fall back to <stem>.jpg next to the json.
        img_path = json_path.with_suffix(".jpg")

    counts: Counter = Counter()
    lines: list[str] = []
    for shape in d.get("shapes", []):
        if shape.get("shape_type", "polygon") not in ("polygon", "rectangle"):
            continue
        label = shape.get("label", "?")
        if label not in label_map:
            continue
        cls = label_map[label]
        bbox = polygon_to_yolo_bbox(shape.get("points", []), img_w, img_h)
        if bbox is None:
            continue
        cx, cy, w, h = bbox
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[cls] += 1

    return img_path, lines, counts


def write_split(pairs: list[tuple[Path, list[str]]], out_root: Path, split: str) -> int:
    img_out = out_root / split / "images"
    lbl_out = out_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_src, lines in pairs:
        if not img_src.is_file():
            continue
        shutil.copy(img_src, img_out / img_src.name)
        (lbl_out / (img_src.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Path to SolDef_AI/Labeled.")
    p.add_argument("--dst", type=Path, required=True, help="Output YOLO dataset root.")
    p.add_argument("--mode", choices=("binary", "multi"), default="binary")
    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    label_map = BINARY_MAP if args.mode == "binary" else MULTI_MAP
    names = BINARY_NAMES if args.mode == "binary" else MULTI_NAMES

    json_files = sorted(args.src.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files under {args.src}")

    # Convert every json. Drop ones that have no usable boxes.
    items: list[tuple[Path, list[str]]] = []
    total_counts: Counter = Counter()
    for jp in json_files:
        try:
            img_src, lines, counts = convert_one(jp, label_map)
        except Exception as e:
            print(f"[skip] {jp.name}: {e}")
            continue
        if not lines:
            continue
        items.append((img_src, lines))
        total_counts += counts

    # Shuffle + split.
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n = len(items)
    n_test = int(round(n * args.test_frac))
    n_val  = int(round(n * args.val_frac))
    n_train = n - n_val - n_test
    train, val, test = items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    n_train_w = write_split(train, args.dst, "train")
    n_val_w   = write_split(val,   args.dst, "val")
    n_test_w  = write_split(test,  args.dst, "test")

    # Write data.yaml.
    data_yaml = dict(
        path=str(args.dst.resolve()).replace("\\", "/"),
        train="train/images",
        val="val/images",
        test="test/images",
        names={int(k): v for k, v in names.items()},
        nc=len(names),
    )
    with (args.dst / "data.yaml").open("w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    summary = dict(
        mode=args.mode,
        n_train=n_train_w, n_val=n_val_w, n_test=n_test_w, n_total=n,
        instance_counts={names[c]: int(v) for c, v in total_counts.items()},
        seed=args.seed,
    )
    (args.dst / "class_counts.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote YOLO dataset to {args.dst}")


if __name__ == "__main__":
    main()
