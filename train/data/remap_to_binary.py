"""Remap the 8-class Roboflow dataset to a binary OK/NG taxonomy and
re-split it into train/val/test with a sensible ratio.

Source layout (input)::

    dataset/src_8class/
        data.yaml                    -- 8 classes (cap_NG, cap_OK, ..., res_OK)
        train/{images,labels}/       -- 204 images (the bulk)
        valid/{images,labels}/       -- 4 images (too small to be useful)
        test/{images,labels}/        -- 3 images (idem)

Destination layout (output)::

    dataset/
        data.yaml                    -- 2 classes: OK (id=0), NG (id=1)
        train/{images,labels}/
        val/{images,labels}/
        test/{images,labels}/
        binary_class_counts.json     -- per-split, per-class instance counts

Class mapping
-------------
The 8 source classes encode (component_type, status). For inspection the
component type is incidental; what matters is whether the placement is OK
or NG. We therefore merge::

    cap_OK, cd4017_OK, ne555_OK, res_OK   ->  OK  (binary id 0)
    cap_NG, cd4017_NG, ne555_NG, res_NG   ->  NG  (binary id 1)

Source class IDs (from the source ``data.yaml``)::

    0: cap_NG    1: cap_OK    2: cd4017_NG    3: cd4017_OK
    4: ne555_NG  5: ne555_OK  6: res_NG       7: res_OK

so the mapping is ``{0:1, 1:0, 2:1, 3:0, 4:1, 5:0, 6:1, 7:0}``.

Usage::

    python remap_to_binary.py \
        --src   dataset/src_8class \
        --dst   dataset \
        --val-frac 0.15 --test-frac 0.15 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml


SRC_ID_TO_BINARY = {
    0: 1,   # cap_NG     -> NG
    1: 0,   # cap_OK     -> OK
    2: 1,   # cd4017_NG  -> NG
    3: 0,   # cd4017_OK  -> OK
    4: 1,   # ne555_NG   -> NG
    5: 0,   # ne555_OK   -> OK
    6: 1,   # res_NG     -> NG
    7: 0,   # res_OK     -> OK
}
BINARY_NAMES = {0: "OK", 1: "NG"}


def collect_pairs(src_root: Path) -> list[tuple[Path, Path]]:
    """Return all ``(image_path, label_path)`` pairs across every src split."""
    pairs: list[tuple[Path, Path]] = []
    for split in ("train", "valid", "val", "test"):
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.is_file():
                pairs.append((img, lbl))
    return pairs


def remap_label_text(text: str) -> tuple[str, Counter]:
    """Remap one YOLO label file's contents to binary ids; return new text + per-class counts."""
    out_lines: list[str] = []
    counts: Counter = Counter()
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        if cls_id not in SRC_ID_TO_BINARY:
            # Drop unknown classes silently rather than crash.
            continue
        new_id = SRC_ID_TO_BINARY[cls_id]
        out_lines.append(" ".join([str(new_id), *parts[1:]]))
        counts[new_id] += 1
    return "\n".join(out_lines) + ("\n" if out_lines else ""), counts


def write_pair(img_src: Path, lbl_src: Path, dst_split: Path) -> Counter:
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy(img_src, dst_split / "images" / img_src.name)
    new_text, counts = remap_label_text(lbl_src.read_text(encoding="utf-8"))
    (dst_split / "labels" / (lbl_src.stem + ".txt")).write_text(new_text, encoding="utf-8")
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True,
                   help="Source dataset root (contains train/valid/test).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Destination dataset root.")
    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pairs = collect_pairs(args.src)
    if not pairs:
        raise SystemExit(f"No (image, label) pairs found under {args.src}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_test  = int(round(n_total * args.test_frac))
    n_val   = int(round(n_total * args.val_frac))
    n_train = n_total - n_test - n_val

    test_pairs  = pairs[:n_test]
    val_pairs   = pairs[n_test:n_test + n_val]
    train_pairs = pairs[n_test + n_val:]

    # Sanity-check: every split has at least one image.
    for split, ps in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        if not ps:
            raise SystemExit(f"split '{split}' is empty -- adjust the fractions")

    # Write the splits.
    counts: dict[str, Counter] = {}
    for split, ps in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        c = Counter()
        for img, lbl in ps:
            c += write_pair(img, lbl, args.dst / split)
        counts[split] = c

    # Write data.yaml at args.dst.
    data_yaml = {
        "path":  str(args.dst.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "names": {0: "OK", 1: "NG"},
        "nc":    2,
    }
    with (args.dst / "data.yaml").open("w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    summary = {
        "n_train": len(train_pairs),
        "n_val":   len(val_pairs),
        "n_test":  len(test_pairs),
        "instance_counts": {
            split: {BINARY_NAMES[k]: int(v) for k, v in c.items()}
            for split, c in counts.items()
        },
        "seed": args.seed,
    }
    (args.dst / "binary_class_counts.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote dataset to {args.dst}")


if __name__ == "__main__":
    main()
