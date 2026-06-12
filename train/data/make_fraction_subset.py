"""Generate a SHARED, seed-fixed training subset for the data-fraction (H3) study.

Both the Ultralytics baseline (``train.py``) and DS-YOLO (``train_ds.py``) must
train on the IDENTICAL images at each fraction; otherwise the H3 comparison is
confounded by different subset selections. (The original bug: ``train.py`` used
Ultralytics' deterministic *first-N* ``fraction`` while ``train_ds.py`` used a
seeded ``randperm`` — so at 25% the two models saw different ~36-image subsets,
which alone can explain the headline +40.4pp gap.)

This script samples ``round(fraction * N)`` training images with a fixed seed,
writes their ABSOLUTE paths to a ``.txt`` list, and emits a ``data.yaml`` whose
``train:`` points to that list. Both the Ultralytics pipeline and
``build_yolo_dataset`` accept a ``.txt`` of image paths, so the emitted yaml can
be fed to BOTH trainers with ``fraction``/``--data-fraction`` left at ``1.0``.

Usage
-----
    python data/make_fraction_subset.py \\
        --data dataset_fixed/data.yaml \\
        --fraction 0.25 --seed 0 \\
        --out-yaml dataset_fixed/data_frac25_seed0.yaml

Then:
    python train.py    --variant rgb --data dataset_fixed/data_frac25_seed0.yaml ...
    python train_ds.py --data dataset_fixed/data_frac25_seed0.yaml ...           # same subset
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import yaml

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_train_dir(data_yaml_path: Path, d: dict) -> Path:
    """Return the absolute directory holding the training images."""
    root = Path(d.get("path", data_yaml_path.parent))
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    train_rel = d["train"]
    train_path = Path(train_rel)
    if not train_path.is_absolute():
        train_path = root / train_path
    return train_path


def _source_of(stem: str) -> str:
    """Strip Roboflow augmentation suffix so a source's copies group together.

    Mirrors resplit_inhouse.extract_source: 'IMG123_png.rf.<hash>' -> 'IMG123'.
    """
    return re.split(r"_png\.rf\.|\.rf\.", stem)[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Source dataset data.yaml.")
    p.add_argument("--fraction", type=float, required=True, help="Subset fraction in (0,1].")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-yaml", required=True, help="Path for the emitted subset data.yaml.")
    p.add_argument("--by-source", action="store_true",
                   help="Sample whole sources (all augmented copies kept together) "
                        "instead of individual images. Default: image-level.")
    args = p.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        p.error("--fraction must be in (0, 1].")

    data_yaml_path = Path(args.data).resolve()
    with data_yaml_path.open() as f:
        d = yaml.safe_load(f)

    train_dir = _resolve_train_dir(data_yaml_path, d)
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Train image dir not found: {train_dir} (from {data_yaml_path}). "
            f"This script must run where the dataset exists on disk.")

    imgs = sorted(q for q in train_dir.iterdir() if q.suffix.lower() in _IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No images found under {train_dir}.")

    rng = np.random.default_rng(args.seed)

    if args.by_source:
        # Group by source, sample a fraction of sources, keep all their copies.
        by_src: dict[str, list[Path]] = {}
        for q in imgs:
            by_src.setdefault(_source_of(q.stem), []).append(q)
        srcs = sorted(by_src)
        k = max(1, round(len(srcs) * args.fraction))
        pick_srcs = rng.choice(len(srcs), size=k, replace=False)
        chosen = sorted(q for i in pick_srcs for q in by_src[srcs[int(i)]])
        unit = f"{k}/{len(srcs)} sources"
    else:
        k = max(1, round(len(imgs) * args.fraction))
        pick = rng.choice(len(imgs), size=k, replace=False)
        chosen = sorted(imgs[int(i)] for i in pick)
        unit = f"{len(chosen)}/{len(imgs)} images"

    out_yaml = Path(args.out_yaml).resolve()
    list_txt = out_yaml.with_suffix(".txt")
    list_txt.write_text(
        "\n".join(str(q.resolve()).replace("\\", "/") for q in chosen) + "\n",
        encoding="utf-8",
    )

    # Emit a yaml identical to the source except train -> the .txt list.
    out = dict(d)
    out["path"] = str(train_dir.parent.parent.resolve()).replace("\\", "/") \
        if "path" not in d else str(Path(d["path"]).resolve()).replace("\\", "/")
    out["train"] = str(list_txt).replace("\\", "/")
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"[make_fraction_subset] fraction={args.fraction} seed={args.seed} "
          f"-> {unit}")
    print(f"[make_fraction_subset] list  -> {list_txt}")
    print(f"[make_fraction_subset] yaml  -> {out_yaml}")
    print(f"[make_fraction_subset] Feed this yaml to BOTH train.py and "
          f"train_ds.py with fraction/--data-fraction = 1.0")


if __name__ == "__main__":
    main()
