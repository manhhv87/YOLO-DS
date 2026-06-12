"""Robustness study: re-evaluate a trained detector under synthetic
homography perturbations.

For each translation sigma (sigma_t in pixels) the script:
  1. Loads every test image (already aligned to golden in the dataset).
  2. Applies a random rigid perturbation (translation + rotation) on top of
     the existing alignment, simulating residual mechanical drift.
  3. For the ``rgb`` variant   : saves the perturbed image as a .jpg and
     calls yolo.val() on the temporary dataset.
  4. For the ``ds-yolo`` variant: saves the perturbed image and evaluates
     DS-YOLO with the ORIGINAL (unperturbed) golden — this measures
     DS-YOLO's robustness advantage from CRFM feature-level fusion.
  5. Reports mAP@0.5 per perturbation level -> Table V of the paper.

Run
---
    # RGB baseline robustness (sigmas 0..20 px)
    python eval/eval_robustness.py \\
        --variant rgb \\
        --weights runs/detect/rgb/weights/best.pt \\
        --data    dataset_fixed/data.yaml \\
        --out     runs/robustness/rgb.csv

    # DS-YOLO robustness
    python eval/eval_robustness.py \\
        --variant ds-yolo \\
        --weights runs/ds_yolo/ds_yolo/weights/best.pt \\
        --data    dataset_fixed/data.yaml \\
        --golden  datasets/inhouse/golden/golden_inhouse.jpg \\
        --out     runs/robustness/ds_yolo.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# Perturbation grid (Table V of the paper).
SIGMAS_T = [0, 2, 5, 10, 20]   # translation noise, pixels
SIGMAS_R = [0.0]                # keep rotation=0 for the table; extend if needed


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def make_perturb_matrix(sigma_t: float, sigma_r: float,
                        rng: np.random.Generator,
                        center: tuple[int, int]) -> np.ndarray:
    """Return a 3x3 homography that is an identity + small noise."""
    tx, ty = rng.normal(0.0, sigma_t, size=2) if sigma_t > 0 else (0.0, 0.0)
    theta  = np.deg2rad(rng.normal(0.0, sigma_r)) if sigma_r > 0 else 0.0
    cx, cy = float(center[0]), float(center[1])

    c, s = np.cos(theta), np.sin(theta)
    M = np.array([
        [c, -s, cx * (1 - c) + cy * s + tx],
        [s,  c, cy * (1 - c) - cx * s + ty],
        [0,  0, 1.0],
    ], dtype=np.float64)
    return M


def perturb_image(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Build temp perturbed datasets
# ---------------------------------------------------------------------------

def _load_golden_bgr(golden_path: str, imgsz: int) -> np.ndarray:
    golden = cv2.imread(golden_path, cv2.IMREAD_COLOR)
    if golden is None:
        raise FileNotFoundError(f"Cannot read golden: {golden_path}")
    return cv2.resize(golden, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)


def transform_yolo_labels(lbl_path: Path, out_path: Path,
                          M: np.ndarray, imgsz: int) -> None:
    """Transform YOLO-format bounding boxes by homography M.

    YOLO boxes are stored as normalized (cx, cy, w, h).  We project all four
    corners of each box through M and recompute the axis-aligned bounding box
    of the projected corners so the labels stay consistent with the warped image.
    Boxes that land entirely outside the image are dropped.
    """
    lines = lbl_path.read_text().splitlines() if lbl_path.is_file() else []
    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        cx, cy, w, h = map(float, parts[1:5])
        # Convert normalised -> pixel corners (top-left, top-right, …)
        x1 = (cx - w / 2) * imgsz
        y1 = (cy - h / 2) * imgsz
        x2 = (cx + w / 2) * imgsz
        y2 = (cy + h / 2) * imgsz
        corners = np.array([[x1, y1, 1], [x2, y1, 1],
                             [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
        projected = (M @ corners.T).T          # shape (4, 3)
        projected /= projected[:, 2:3]         # homogeneous divide
        px, py = projected[:, 0], projected[:, 1]
        nx1, ny1 = px.min(), py.min()
        nx2, ny2 = px.max(), py.max()
        # Clip to image bounds
        nx1, nx2 = np.clip([nx1, nx2], 0, imgsz)
        ny1, ny2 = np.clip([ny1, ny2], 0, imgsz)
        if nx2 - nx1 < 1 or ny2 - ny1 < 1:
            continue                            # box outside image, drop
        ncx = ((nx1 + nx2) / 2) / imgsz
        ncy = ((ny1 + ny2) / 2) / imgsz
        nw  = (nx2 - nx1) / imgsz
        nh  = (ny2 - ny1) / imgsz
        out_lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
    out_path.write_text("\n".join(out_lines), encoding="utf-8")


def build_perturbed_rgb(test_img_dir: Path,
                        test_lbl_dir: Path,
                        tmp_dir: Path,
                        sigma_t: float,
                        sigma_r: float,
                        rng: np.random.Generator,
                        imgsz: int) -> None:
    """Write perturbed .jpg images + transformed labels under
    ``{tmp_dir}/test/images`` and ``{tmp_dir}/test/labels`` so the layout
    matches what Ultralytics expects for ``mode='val'`` with
    ``test: test/images`` in the data.yaml.

    Labels are projected through the same homography as the images so that
    ground-truth boxes remain aligned with the warped image content.  This
    isolates the model's detection quality under misalignment rather than
    penalising both models equally for a label-position mismatch.
    """
    img_out = tmp_dir / "test" / "images"
    lbl_out = tmp_dir / "test" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(test_img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        M = make_perturb_matrix(sigma_t, sigma_r, rng, (imgsz // 2, imgsz // 2))
        warped = perturb_image(img, M)

        cv2.imwrite(str(img_out / img_path.name), warped)

        lbl = test_lbl_dir / (img_path.stem + ".txt")
        transform_yolo_labels(lbl, lbl_out / (img_path.stem + ".txt"), M, imgsz)


def write_tmp_yaml(tmp_dir: Path, src_yaml: dict) -> Path:
    """Write a minimal data.yaml pointing only to the perturbed test split.

    Layout written by ``build_perturbed_rgb``:

        {tmp_dir}/
          test/
            images/*.jpg
            labels/*.txt
    """
    d = {
        "path":  str(tmp_dir.resolve()).replace("\\", "/"),
        "train": "test/images",   # unused but required by Ultralytics
        "val":   "test/images",   # unused but required
        "test":  "test/images",
        "names": src_yaml.get("names", {0: "OK", 1: "NG"}),
        "nc":    src_yaml.get("nc", 2),
    }
    out = tmp_dir / "data.yaml"
    with out.open("w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
    return out


# ---------------------------------------------------------------------------
# DS-YOLO evaluation on a temporary perturbed dataset
# ---------------------------------------------------------------------------

def _eval_ds_on_tmp(tmp_yaml: Path, ds_context: dict, imgsz: int) -> float:
    """Evaluate DS-YOLO on the perturbed images using the fixed golden."""
    import torch
    from torch.utils.data import DataLoader
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    here = Path(__file__).parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from train_ds import evaluate as ds_evaluate

    import yaml as _yaml
    with open(tmp_yaml) as f:
        data_yaml = _yaml.safe_load(f)

    # build_yolo_dataset expects an absolute path for the image directory.
    # The tmp yaml uses relative "images" under "path", so resolve it here.
    data_path = Path(data_yaml.get("path", str(tmp_yaml.parent)))
    test_img_abs = str(data_path / data_yaml["test"])

    cfg = get_cfg(DEFAULT_CFG, overrides=dict(
        imgsz=imgsz, batch=4, mode="val", rect=False, cache=False))
    val_ds  = build_yolo_dataset(cfg, test_img_abs, 4, data_yaml, mode="val")
    val_ldr = DataLoader(val_ds, batch_size=4, shuffle=False,
                         num_workers=0, collate_fn=val_ds.collate_fn)

    val_met = ds_evaluate(
        ds_context["model"],
        val_ldr,
        ds_context["golden"],
        ds_context["device"],
        nc=ds_context["nc"],
    )
    return float(val_met.get("map50", 0.0))


# ---------------------------------------------------------------------------
# Evaluate one (sigma_t, sigma_r) point
# ---------------------------------------------------------------------------

def _resolve_data_root(data_yaml_path: Path, src_yaml: dict) -> Path:
    """Pick the dataset root that actually exists on disk.

    The data.yaml emitted by Colab runs hard-codes ``path:`` to a
    ``/content/...`` absolute path that is invalid on any other machine.
    We try, in order:

      1. ``src_yaml["path"]`` if it exists,
      2. the directory containing the data.yaml itself,
      3. ``data.yaml's parent.parent / data_yaml.stem`` (rare, kept for safety).

    The first candidate whose ``test/images`` subfolder exists wins.
    """
    test_rel = src_yaml.get("test", "test/images")
    candidates: list[Path] = []
    if "path" in src_yaml:
        candidates.append(Path(src_yaml["path"]))
    candidates.append(data_yaml_path.parent)

    for root in candidates:
        if (root / test_rel).is_dir():
            return root

    # Loud failure with all candidates in the message — much easier to debug
    # than a generic FileNotFoundError per sigma.
    raise FileNotFoundError(
        f"Could not locate test images for {data_yaml_path}.\n"
        f"  yaml path key   : {src_yaml.get('path', '<unset>')}\n"
        f"  yaml test key   : {test_rel}\n"
        f"  candidates tried: {[str(c / test_rel) for c in candidates]}\n"
        f"Fix: edit {data_yaml_path} so 'path:' points to the local dataset "
        f"root (or remove it to default to the yaml's own directory)."
    )


def eval_one(yolo,
             variant: str,
             sigma_t: float,
             sigma_r: float,
             data_yaml_path: Path,
             imgsz: int,
             rng: np.random.Generator,
             ds_context: dict | None = None) -> float:
    """Return mAP@0.5 for the given perturbation level."""

    with open(data_yaml_path) as f:
        src_yaml = yaml.safe_load(f)

    base = _resolve_data_root(data_yaml_path, src_yaml)
    test_rel = src_yaml.get("test", "test/images")
    test_img_dir = base / test_rel
    test_lbl_dir = base / test_rel.replace("images", "labels")

    with tempfile.TemporaryDirectory(prefix="robustness_") as tmp:
        tmp_dir = Path(tmp)
        build_perturbed_rgb(test_img_dir, test_lbl_dir, tmp_dir,
                            sigma_t, sigma_r, rng, imgsz)
        tmp_yaml = write_tmp_yaml(tmp_dir, src_yaml)

        if variant == "ds-yolo":
            return _eval_ds_on_tmp(tmp_yaml, ds_context, imgsz)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as _tmp_val:
                metrics = yolo.val(data=str(tmp_yaml), split="test",
                                   imgsz=imgsz, verbose=False,
                                   project=_tmp_val, name="val")
            return float(metrics.box.map50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant",  required=True, choices=["rgb", "ds-yolo"],
                   help="Model variant to evaluate.")
    p.add_argument("--weights",  required=True,
                   help="Path to best.pt checkpoint.")
    p.add_argument("--data",     required=True,
                   help="data.yaml of the dataset the model was trained on.")
    p.add_argument("--golden",   default=None,
                   help="Golden reference image (required for ds-yolo).")
    p.add_argument("--out",      required=True,
                   help="Output CSV path.")
    p.add_argument("--imgsz",    type=int, default=640,
                   help="Eval resolution. MUST match the model's training "
                        "imgsz (640 in the paper); a mismatch breaks DS-YOLO "
                        "golden/CRFM feature alignment and yields NaN/garbage.")
    p.add_argument("--sigmas-t", type=float, nargs="+", default=SIGMAS_T,
                   help="Translation noise std-dev values (px).")
    p.add_argument("--sigmas-r", type=float, nargs="+", default=SIGMAS_R,
                   help="Rotation noise std-dev values (degrees).")
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()

    if args.variant == "ds-yolo" and args.golden is None:
        p.error("--golden is required for variant=ds-yolo")

    rng = np.random.default_rng(args.seed)

    # Load the model.
    yolo       = None
    ds_context = None

    if args.variant == "ds-yolo":
        import torch
        here = Path(__file__).parent.parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from models.ds_yolo import DSYOLOv8m
        from train_ds import load_golden as ds_load_golden

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # `from_checkpoint` reads variant + fusion from the checkpoint and
        # filters thop's bookkeeping buffers — works regardless of which
        # YOLOv8 size (n/s/m/l/x) the model was trained with.
        model = DSYOLOv8m.from_checkpoint(args.weights, device=device)
        nc = int(model.nc)
        golden_tensor = ds_load_golden(args.golden, args.imgsz, device)
        ds_context = dict(model=model, golden=golden_tensor, device=device, nc=nc)
    else:
        yolo = YOLO(args.weights)

    data_yaml_path = Path(args.data)
    rows: list[dict] = []
    combos = list(product(args.sigmas_t, args.sigmas_r))
    print(f"[robustness] variant={args.variant}  "
          f"{len(combos)} perturbation levels")

    # Fail-fast pre-flight: surface yaml/path errors BEFORE running 5
    # perturbation loops that would otherwise all NaN-out silently.
    with open(data_yaml_path) as _f:
        _src_yaml = yaml.safe_load(_f)
    _resolve_data_root(data_yaml_path, _src_yaml)  # raises with full diagnostic

    for sigma_t, sigma_r in combos:
        # Let exceptions propagate loudly.  A per-sigma failure is almost always
        # systematic (imgsz mismatch, bad path, empty labels), not data-dependent
        # — the old `except Exception -> nan` silently masked it and produced the
        # committed CSV of all-NaNs.
        map50 = eval_one(yolo, args.variant, sigma_t, sigma_r,
                         data_yaml_path, args.imgsz, rng,
                         ds_context=ds_context)

        # Sanity gate: zero perturbation MUST reproduce the clean test mAP.  If it
        # is NaN/zero the evaluation pipeline is broken (commonly an imgsz
        # mismatch vs the training resolution) — abort instead of emitting a
        # misleading table.
        if sigma_t == 0 and sigma_r == 0 and (map50 != map50 or map50 <= 0.0):
            raise RuntimeError(
                f"sigma_t=0 produced mAP@0.5={map50}; it must reproduce the clean "
                f"test mAP (>0). Verify --imgsz ({args.imgsz}) matches the model's "
                f"training resolution and that test labels are present. Aborting "
                f"rather than writing a CSV of NaNs.")

        rows.append(dict(sigma_t=sigma_t, sigma_r=sigma_r, map50=map50))
        print(f"  sigma_t={sigma_t:>4}px  sigma_r={sigma_r}deg  "
              f"mAP@0.5={map50:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sigma_t", "sigma_r", "map50"])
        w.writeheader()
        w.writerows(rows)

    print(f"[robustness] saved -> {out_path}")


if __name__ == "__main__":
    main()
