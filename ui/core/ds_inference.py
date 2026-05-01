"""Inference helper for DS-YOLO checkpoints saved by ``train/train_ds.py``.

Use this from ``pipeline.py`` (or directly from ``window_camera.py``) to
run a captured frame through DS-YOLO and obtain the same kind of
detection list that ``yolo_infer`` previously produced.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics.utils.ops import non_max_suppression

# Make the train/models package importable from the UI process. The
# DSYOLOv8m class is shared between training and inference.
_TRAIN_DIR = Path(__file__).resolve().parents[2] / "train"
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from models.ds_yolo import DSYOLOv8m  # noqa: E402


# ---------------------------------------------------------------------------
# Loader (caches model + golden across calls)
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_ds_yolo(
    weights_path: str,
    golden_path: str,
    imgsz: int = 1024,
    device: Optional[str] = None,
) -> Tuple[DSYOLOv8m, torch.Tensor, dict]:
    """Load a DS-YOLO checkpoint and the golden image once, cache afterwards.

    Returns ``(model, golden_tensor, meta)`` where ``meta`` contains keys
    such as ``num_classes`` and ``names`` if the checkpoint stored them.
    """
    device = device or _device()
    key = (str(Path(weights_path).resolve()), str(Path(golden_path).resolve()), imgsz, device)
    if key in _CACHE:
        return _CACHE[key]

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    nc = int(ckpt.get("num_classes", 2))
    model = DSYOLOv8m(num_classes=nc).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    img = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read golden image: {golden_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    golden = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0

    meta = dict(num_classes=nc, names=ckpt.get("names"), epoch=ckpt.get("epoch"))
    _CACHE[key] = (model, golden, meta)
    return model, golden, meta


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------
@torch.no_grad()
def ds_infer(
    weights_path: str,
    img_path: str,
    golden_path: str,
    *,
    imgsz: int = 1024,
    conf_th: float = 0.25,
    iou_th: float = 0.7,
    max_det: int = 300,
    device: Optional[str] = None,
) -> Tuple[List[List[float]], np.ndarray]:
    """Run DS-YOLO on an image saved at ``img_path``.

    Returns
    -------
    detections : list of [x1, y1, x2, y2, conf, cls_id]
        Coordinates are in **pixels of the resized ``imgsz`` image**, not
        the original capture (matching what the model sees).
    annotated  : (H, W, 3) BGR image with bounding boxes drawn.
    """
    device = device or _device()
    model, golden, _ = load_ds_yolo(weights_path, golden_path, imgsz=imgsz, device=device)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read input image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    inp = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    preds = model(inp, golden)
    decoded = preds[0] if isinstance(preds, (tuple, list)) else preds
    nms_out = non_max_suppression(decoded, conf_th, iou_th, max_det=max_det)[0]

    detections = nms_out.cpu().numpy().tolist() if nms_out is not None else []
    annotated = cv2.cvtColor(rgb_resized.copy(), cv2.COLOR_RGB2BGR)
    for x1, y1, x2, y2, conf, cls in detections:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated, f"{int(cls)}:{conf:.2f}", (int(x1), max(int(y1) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return detections, annotated
