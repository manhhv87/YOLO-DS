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


def build_perturbed_rgb(test_img_dir: Path,
                        test_lbl_dir: Path,
                        tmp_dir: Path,
                        sigma_t: float,
                        sigma_r: float,
                        rng: np.random.Generator,
                        imgsz: int) -> None:
    """Write perturbed .jpg images + copied labels into tmp_dir."""
    img_out = tmp_dir / "images"
    lbl_out = tmp_dir / "labels"
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
        if lbl.is_file():
            shutil.copy2(lbl, lbl_out / lbl.name)
        else:
            (lbl_out / (img_path.stem + ".txt")).write_text("", encoding="utf-8")


def write_tmp_yaml(tmp_dir: Path, src_yaml: dict) -> Path:
    """Write a minimal data.yaml pointing only to the perturbed test split."""
    d = {
        "path":  str(tmp_dir.resolve()).replace("\\", "/"),
        "train": "images",   # unused but required by Ultralytics
        "val":   "images",   # unused but required
        "test":  "images",
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

    cfg = get_cfg(DEFAULT_CFG, overrides=dict(
        imgsz=imgsz, batch=4, mode="val", rect=False, cache=False))
    val_ds  = build_yolo_dataset(cfg, data_yaml["test"], 4, data_yaml, mode="val")
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

    base = Path(src_yaml.get("path", str(data_yaml_path.parent)))
    test_rel = src_yaml.get("test", "test/images")
    test_img_dir = base / test_rel
    test_lbl_dir = base / test_rel.replace("images", "labels")

    if not test_img_dir.is_dir():
        raise FileNotFoundError(f"Test image dir not found: {test_img_dir}")

    with tempfile.TemporaryDirectory(prefix="robustness_") as tmp:
        tmp_dir = Path(tmp)
        build_perturbed_rgb(test_img_dir, test_lbl_dir, tmp_dir,
                            sigma_t, sigma_r, rng, imgsz)
        tmp_yaml = write_tmp_yaml(tmp_dir, src_yaml)

        if variant == "ds-yolo":
            return _eval_ds_on_tmp(tmp_yaml, ds_context, imgsz)
        else:
            metrics = yolo.val(data=str(tmp_yaml), split="test",
                               imgsz=imgsz, verbose=False)
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
    p.add_argument("--imgsz",    type=int, default=1024)
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
        ckpt   = torch.load(args.weights, map_location=device, weights_only=False)
        nc     = int(ckpt.get("num_classes", 2))
        model  = DSYOLOv8m(num_classes=nc).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        golden_tensor = ds_load_golden(args.golden, args.imgsz, device)
        ds_context = dict(model=model, golden=golden_tensor, device=device, nc=nc)
    else:
        yolo = YOLO(args.weights)

    data_yaml_path = Path(args.data)
    rows: list[dict] = []
    combos = list(product(args.sigmas_t, args.sigmas_r))
    print(f"[robustness] variant={args.variant}  "
          f"{len(combos)} perturbation levels")

    for sigma_t, sigma_r in combos:
        try:
            map50 = eval_one(yolo, args.variant, sigma_t, sigma_r,
                             data_yaml_path, args.imgsz, rng,
                             ds_context=ds_context)
        except Exception as e:
            print(f"  [WARN] sigma_t={sigma_t} sigma_r={sigma_r} failed: {e}")
            map50 = float("nan")

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
