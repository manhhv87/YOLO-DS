"""Fix data leakage: re-split in-house dataset at SOURCE image level.

Problem
-------
The original Roboflow export shuffled AUGMENTED copies of the same source
image across train/val/test, so 46 of 75 source images leaked into multiple
splits.  This script:

  1. Collects every (image, label) pair from the three current splits.
  2. Groups them by SOURCE image name (strips the ``_png.rf.<hash>`` suffix).
  3. Shuffles the 75 SOURCE names (seed-controlled) then splits 70 / 15 / 15.
  4. For TRAIN  : copies ALL augmented copies of each assigned source.
  5. For VAL/TEST: copies only ONE copy per source (first, alphabetically)
     to avoid artificially inflating validation/test metrics.
  6. Writes the result to ``dataset_fixed/`` with a fresh ``data.yaml``.

Usage
-----
    cd train
    python data/resplit_inhouse.py
    python data/resplit_inhouse.py --src datasets/inhouse/leaky --dst dataset_fixed --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_source(stem: str) -> str:
    """Return the source image name, stripping any Roboflow augmentation hash.

    Examples
    --------
    ``g1_png.rf.2be11ebc097aa...``  ->  ``g1``
    ``Image__2025-10-31__23-39-49_png.rf.abc...``  ->  ``Image__2025-10-31__23-39-49``
    ``some_plain_image``  ->  ``some_plain_image``
    """
    if "_png.rf." in stem:
        return stem.split("_png.rf.")[0]
    return stem


def collect_all_pairs(src_root: Path) -> defaultdict:
    """Return {source_name: [(img_path, lbl_path), ...]} across all splits."""
    src_to_pairs: defaultdict[str, list[tuple[Path, Path | None]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            src = extract_source(img.stem)
            src_to_pairs[src].append((img, lbl if lbl.is_file() else None))
    return src_to_pairs


def copy_pair(img_src: Path, lbl_src: Path | None, dst_split: Path) -> None:
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, dst_split / "images" / img_src.name)
    if lbl_src is not None:
        shutil.copy2(lbl_src, dst_split / "labels" / (img_src.stem + ".txt"))
    else:
        # Write empty label file so YOLO doesn't complain about missing labels.
        (dst_split / "labels" / (img_src.stem + ".txt")).write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src",        type=Path, default=Path("datasets/inhouse/leaky"),
                   help="Current dataset root (default: datasets/inhouse/leaky)")
    p.add_argument("--dst",        type=Path, default=Path("dataset_fixed"),
                   help="Output dataset root (default: dataset_fixed)")
    p.add_argument("--golden-src", type=Path, default=Path("datasets/inhouse/golden"),
                   help="Golden images folder to copy into dst/ (default: datasets/inhouse/golden)")
    p.add_argument("--val-frac",   type=float, default=0.15,
                   help="Fraction of SOURCES for val (default: 0.15)")
    p.add_argument("--test-frac",  type=float, default=0.15,
                   help="Fraction of SOURCES for test (default: 0.15)")
    p.add_argument("--seed",       type=int,   default=0)
    args = p.parse_args()

    print(f"[resplit] src={args.src}  dst={args.dst}  seed={args.seed}")

    # -- Step 1: collect all pairs grouped by source -------------------------
    src_to_pairs = collect_all_pairs(args.src)
    all_sources  = sorted(src_to_pairs.keys())

    print(f"[resplit] {len(all_sources)} unique source images, "
          f"{sum(len(v) for v in src_to_pairs.values())} total augmented files")

    # -- Step 2: split sources -----------------------------------------------
    rng = random.Random(args.seed)
    sources = all_sources[:]
    rng.shuffle(sources)

    n_total = len(sources)
    n_test  = max(1, round(n_total * args.test_frac))
    n_val   = max(1, round(n_total * args.val_frac))
    n_train = n_total - n_val - n_test

    if n_train <= 0:
        raise SystemExit("val-frac + test-frac >= 1.0 — not enough sources for training.")

    train_srcs = set(sources[:n_train])
    val_srcs   = set(sources[n_train:n_train + n_val])
    test_srcs  = set(sources[n_train + n_val:])

    print(f"[resplit] source-level split: "
          f"train={len(train_srcs)}  val={len(val_srcs)}  test={len(test_srcs)}")

    # -- Step 3: verify no leakage -------------------------------------------
    assert not (train_srcs & val_srcs),  "BUG: train ∩ val non-empty"
    assert not (train_srcs & test_srcs), "BUG: train ∩ test non-empty"
    assert not (val_srcs   & test_srcs), "BUG: val ∩ test non-empty"

    # -- Step 4: copy files --------------------------------------------------
    stats = {"train": 0, "val": 0, "test": 0}

    for src, pairs in src_to_pairs.items():
        # Sort pairs so the "first" copy is deterministic.
        pairs_sorted = sorted(pairs, key=lambda t: t[0].name)

        if src in train_srcs:
            # Keep ALL augmented copies for training.
            for img, lbl in pairs_sorted:
                copy_pair(img, lbl, args.dst / "train")
                stats["train"] += 1

        elif src in val_srcs:
            # Keep only the FIRST copy for clean validation.
            img, lbl = pairs_sorted[0]
            copy_pair(img, lbl, args.dst / "val")
            stats["val"] += 1

        else:  # test_srcs
            # Keep only the FIRST copy for clean testing.
            img, lbl = pairs_sorted[0]
            copy_pair(img, lbl, args.dst / "test")
            stats["test"] += 1

    # -- Step 5: copy golden images ------------------------------------------
    golden_src = args.golden_src
    if golden_src.is_dir():
        golden_dst = args.dst / "golden"
        if golden_dst.exists():
            shutil.rmtree(golden_dst)
        shutil.copytree(golden_src, golden_dst)
        print(f"[resplit] copied golden/ from {golden_src}")
    else:
        print(f"[resplit] WARNING: golden dir not found at {golden_src}")

    # -- Step 6: write data.yaml ---------------------------------------------
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

    # -- Step 7: write summary -----------------------------------------------
    summary = {
        "seed":          args.seed,
        "n_sources":     n_total,
        "source_split":  {"train": len(train_srcs), "val": len(val_srcs), "test": len(test_srcs)},
        "image_count":   stats,
        "note": (
            "val and test each contain exactly 1 image per source "
            "(no augmented duplicates). train keeps all augmented copies."
        ),
    }
    (args.dst / "split_info.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[resplit] images written: "
          f"train={stats['train']}  val={stats['val']}  test={stats['test']}")
    print(f"[resplit] done -> {args.dst}")
    print(f"[resplit] summary saved to {args.dst / 'split_info.json'}")


if __name__ == "__main__":
    main()
