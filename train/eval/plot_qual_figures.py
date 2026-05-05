"""Generate qualitative figures for the paper.

Figure 3 — qual_diff.pdf
    Three panels side-by-side: golden reference | aligned capture | diff map.
    Picks one test image that has at least one NG annotation.

Figure 4 — qual_detect.pdf
    Two columns: RGB baseline detections | DS-YOLO detections on the same board.
    Bounding boxes are drawn with the model's predicted class and confidence.

Usage
-----
    cd train

    # Both figures (requires trained weights)
    python eval/plot_qual_figures.py \
        --data    dataset_fixed/data.yaml \
        --golden  datasets/inhouse/golden/golden_ok.bmp \
        --rgb     runs/detect/rgb/weights/best.pt \
        --ds-yolo runs/ds_yolo/ds_yolo/weights/best.pt \
        --imgsz   640

    # Only Figure 3 (no model weights needed)
    python eval/plot_qual_figures.py \
        --data   dataset_fixed/data.yaml \
        --golden datasets/inhouse/golden/golden_ok.bmp \
        --fig3-only

    # Specify a particular test image
    python eval/plot_qual_figures.py ... --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).parent.parent.resolve()   # train/
PAPER_FIGURES = HERE.parent / "paper" / "figures"

# Class names and colours for bounding-box drawing.
_CLASS_COLORS = {
    0: (34, 139, 34),    # OK  — green
    1: (220, 20,  60),   # NG  — crimson
}
_CLASS_NAMES = {0: "OK", 1: "NG"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_golden(golden_path: str, imgsz: int) -> np.ndarray:
    """Return golden image as RGB uint8 array resized to imgsz×imgsz."""
    img = cv2.imread(golden_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read golden: {golden_path}")
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _align_to_golden(capture_bgr: np.ndarray, golden_bgr: np.ndarray,
                     max_features: int = 2000) -> np.ndarray:
    """ORB+RANSAC homography align; fallback to identity on failure."""
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(capture_bgr, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(golden_bgr,  cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return capture_bgr
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return capture_bgr
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        return capture_bgr
    h, w = golden_bgr.shape[:2]
    return cv2.warpPerspective(capture_bgr, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _pick_test_image(data_yaml: str, prefer_ng: bool = True) -> Path | None:
    """Return path to a test image, preferring one with NG annotations."""
    with open(data_yaml) as f:
        dy = yaml.safe_load(f)
    base = Path(dy.get("path", str(Path(data_yaml).parent)))
    if not base.is_absolute():
        base = Path(data_yaml).parent / base
    test_img_dir = base / dy.get("test", "test/images")
    test_lbl_dir = base / str(dy.get("test", "test/images")).replace("images", "labels")

    candidates = sorted(test_img_dir.glob("*"))
    candidates = [p for p in candidates
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not candidates:
        return None

    if prefer_ng:
        for img_path in candidates:
            lbl = test_lbl_dir / (img_path.stem + ".txt")
            if lbl.is_file():
                lines = lbl.read_text().strip().splitlines()
                if any(line.startswith("1 ") for line in lines):
                    return img_path

    return candidates[0]


def _draw_boxes(img_rgb: np.ndarray, boxes_xyxy: list, labels: list,
                confs: list, thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes on a copy of img_rgb."""
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2), cls, conf in zip(boxes_xyxy, labels, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = _CLASS_COLORS.get(int(cls), (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        tag = f"{_CLASS_NAMES.get(int(cls), str(int(cls)))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(out, tag, (x1 + 1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — golden | aligned | diff
# ---------------------------------------------------------------------------

def make_fig3(golden_rgb: np.ndarray, capture_path: Path,
              imgsz: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap_bgr = cv2.imread(str(capture_path), cv2.IMREAD_COLOR)
    if cap_bgr is None:
        print(f"[warn] cannot read {capture_path}")
        return
    cap_bgr  = cv2.resize(cap_bgr,  (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    gold_bgr = cv2.cvtColor(golden_rgb, cv2.COLOR_RGB2BGR)

    aligned_bgr = _align_to_golden(cap_bgr, gold_bgr)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    diff = cv2.absdiff(aligned_bgr, gold_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Amplify for visibility
    diff_vis = cv2.applyColorMap(
        cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_HOT,
    )
    diff_vis = cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB)

    panels = [
        (golden_rgb,  r"Golden reference $\mathbf{I}_g$"),
        (aligned_rgb, r"Aligned capture $\tilde{\mathbf{I}}_c$"),
        (diff_vis,    r"Difference map $\mathbf{D}$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Figure 3 saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — RGB vs DS-YOLO detection comparison
# ---------------------------------------------------------------------------

def _infer_rgb(weights_path: str, img_rgb: np.ndarray,
               imgsz: int) -> tuple[list, list, list]:
    """Run Ultralytics YOLO on img_rgb. Return (boxes_xyxy, classes, confs)."""
    from ultralytics import YOLO
    import torch
    model = YOLO(weights_path)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, imgsz=imgsz, conf=0.25, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], [], []
    boxes  = r.boxes.xyxy.cpu().numpy().tolist()
    labels = r.boxes.cls.cpu().numpy().astype(int).tolist()
    confs  = r.boxes.conf.cpu().numpy().tolist()
    return boxes, labels, confs


def _infer_dsyolo(weights_path: str, img_rgb: np.ndarray,
                  golden_rgb: np.ndarray, imgsz: int,
                  device: str) -> tuple[list, list, list]:
    """Run DS-YOLO on img_rgb with golden. Return (boxes_xyxy, classes, confs)."""
    import torch
    sys.path.insert(0, str(HERE))
    from models.ds_yolo import DSYOLOv8m
    try:
        from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
    except ImportError:
        import torchvision

        def xywh2xyxy(x):
            y = x.clone()
            y[..., 0] = x[..., 0] - x[..., 2] / 2
            y[..., 1] = x[..., 1] - x[..., 3] / 2
            y[..., 2] = x[..., 0] + x[..., 2] / 2
            y[..., 3] = x[..., 1] + x[..., 3] / 2
            return y

        def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=300, **_):
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]:
                pred = pred.transpose(1, 2)
            output = []
            for x in pred:
                boxes = xywh2xyxy(x[:, :4])
                cls_scores = x[:, 4:]
                conf, cls_idx = cls_scores.max(dim=1)
                mask = conf > conf_thres
                boxes, conf, cls_idx = boxes[mask], conf[mask], cls_idx[mask]
                if boxes.shape[0] == 0:
                    output.append(torch.zeros(0, 6, device=x.device))
                    continue
                keep = torchvision.ops.nms(boxes.float(), conf.float(), iou_thres)[:max_det]
                output.append(torch.cat(
                    [boxes[keep], conf[keep, None], cls_idx[keep, None].float()], dim=1))
            return output

    ckpt  = torch.load(weights_path, map_location=device, weights_only=False)
    nc    = int(ckpt.get("num_classes", 2))
    model = DSYOLOv8m(num_classes=nc).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def _to_tensor(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device)

    with torch.no_grad():
        cap_t  = _to_tensor(cv2.resize(img_rgb,    (imgsz, imgsz)))
        gold_t = _to_tensor(cv2.resize(golden_rgb, (imgsz, imgsz)))
        preds = model(cap_t, gold_t)
        decoded = preds[0] if isinstance(preds, (tuple, list)) else preds
        outputs = non_max_suppression(decoded, conf_thres=0.25)

    if not outputs or len(outputs[0]) == 0:
        return [], [], []

    det = outputs[0].cpu().numpy()
    # Scale from imgsz back to original image size
    scale_x = img_rgb.shape[1] / imgsz
    scale_y = img_rgb.shape[0] / imgsz
    boxes  = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y] for b in det]
    labels = det[:, 5].astype(int).tolist()
    confs  = det[:, 4].tolist()
    return boxes, labels, confs


def make_fig4(golden_rgb: np.ndarray, capture_path: Path,
              rgb_weights: str, ds_weights: str,
              imgsz: int, out_path: Path, device: str = "cpu") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap_bgr = cv2.imread(str(capture_path), cv2.IMREAD_COLOR)
    if cap_bgr is None:
        print(f"[warn] cannot read {capture_path}")
        return

    gold_bgr    = cv2.cvtColor(golden_rgb, cv2.COLOR_RGB2BGR)
    aligned_bgr = _align_to_golden(cap_bgr, gold_bgr)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input
    display_rgb = cv2.resize(aligned_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    # --- RGB detections ---
    boxes_rgb, lbl_rgb, conf_rgb = _infer_rgb(rgb_weights, display_rgb, imgsz)
    img_rgb_det = _draw_boxes(display_rgb, boxes_rgb, lbl_rgb, conf_rgb)

    # --- DS-YOLO detections ---
    gold_disp = cv2.resize(golden_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    boxes_ds, lbl_ds, conf_ds = _infer_dsyolo(ds_weights, display_rgb,
                                               gold_disp, imgsz, device)
    img_ds_det = _draw_boxes(display_rgb, boxes_ds, lbl_ds, conf_ds)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    axes[0].imshow(img_rgb_det)
    axes[0].set_title("YOLOv8m (RGB baseline)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(img_ds_det)
    axes[1].set_title("DS-YOLO (ours)", fontsize=9)
    axes[1].axis("off")

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Figure 4 saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import torch
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",       required=True, help="dataset_fixed/data.yaml")
    p.add_argument("--golden",     required=True, help="Path to golden_ok.bmp")
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--image",      default=None,
                   help="Specific test image to use (default: auto-pick NG image).")
    p.add_argument("--rgb",        default=None,
                   help="RGB model weights (required for Figure 4).")
    p.add_argument("--ds-yolo",    default=None,
                   help="DS-YOLO weights (required for Figure 4).")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fig3-only",  action="store_true",
                   help="Generate only Figure 3 (no model weights needed).")
    p.add_argument("--out-dir",    default=str(PAPER_FIGURES),
                   help="Output directory (default: paper/figures/)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    golden_rgb = _load_golden(args.golden, args.imgsz)

    # Pick test image
    if args.image:
        img_path = Path(args.image)
    else:
        img_path = _pick_test_image(args.data, prefer_ng=True)
        if img_path is None:
            print("[error] No test images found. Check --data path.")
            return
    print(f"[info] Using test image: {img_path}")

    # --- Figure 3 ---
    make_fig3(golden_rgb, img_path, args.imgsz, out_dir / "qual_diff.pdf")

    # --- Figure 4 ---
    if not args.fig3_only:
        if args.rgb is None or args.ds_yolo is None:
            print("[skip] Figure 4: --rgb and --ds-yolo weights required.")
        else:
            make_fig4(golden_rgb, img_path,
                      args.rgb, args.ds_yolo,
                      args.imgsz, out_dir / "qual_detect.pdf",
                      device=args.device)


if __name__ == "__main__":
    main()
