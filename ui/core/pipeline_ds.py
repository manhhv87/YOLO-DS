"""DS-YOLO inspection pipeline for the AVI cell.

Drop-in replacement for ``main_pipeline`` in :mod:`core.pipeline`, but
uses the Dual-Stream YOLO model (with Cross-Reference Fusion) trained
under ``../train/`` instead of the legacy 8-class YOLOv8 detector.

The DS-YOLO detector already classifies each component as ``OK`` or
``NG`` directly, so the explicit shift-checking step from the legacy
pipeline is no longer needed: any geometric defect (shifted, rotated,
missing) shows up as a class-1 (``NG``) detection.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import cv2

from core.algorithm import (
    apply_clahe_bgr,
    crop_pcb_from_blue,
    orb_align_images,
    paste_crop_to_canvas,
)
from core.ds_inference import ds_infer


# ---------------------------------------------------------------------------
# Defaults — match the layout under ui/. Override via function args.
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_DEFAULT_WEIGHTS = os.path.join(_BASE_DIR, "runs", "ds_yolo", "weights", "best.pt")
# In-domain golden (1024x1024, same camera/lighting as the captures used for
# training).  Studio shots like the legacy golden_ok.bmp / golden2.bmp differ
# in lighting and resolution and produce a noisy diff channel — do NOT use
# them as model input.
_DEFAULT_GOLDEN = os.path.join(_BASE_DIR, "stored_images", "golden", "golden_inhouse.jpg")
_DEFAULT_GOLDEN_ALIGN = _DEFAULT_GOLDEN

CLASS_NAMES = ["OK", "NG"]


def main_pipeline_ds(
    image_path: str,
    *,
    weights_path: str = _DEFAULT_WEIGHTS,
    golden_path_for_align: str = _DEFAULT_GOLDEN_ALIGN,
    golden_path_for_model: str = _DEFAULT_GOLDEN,
    imgsz: int = 1024,
    conf_th: float = 0.25,
) -> Dict:
    """Run a single inspection cycle with DS-YOLO.

    Returns a dict with keys ``verdict`` (``"OK"`` / ``"NG"``),
    ``ng_components`` (list of bounding boxes flagged as NG),
    ``annotated_path`` (path to the saved annotated image), and
    ``raw_detections`` for diagnostics.
    """
    # 1) Load raw image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot read raw image: {image_path}")

    # 2) Align to golden (homography), then CLAHE for visual consistency only.
    golden_align = cv2.imread(golden_path_for_align)
    if golden_align is None:
        raise FileNotFoundError(f"Cannot read golden alignment image: {golden_path_for_align}")
    raw_clahe = apply_clahe_bgr(raw_img, clip=2.0, grid=(8, 8))
    golden_clahe = apply_clahe_bgr(golden_align, clip=2.0, grid=(8, 8))
    aligned_img, _ = orb_align_images(golden_clahe, raw_clahe, n_features=2000, draw_matches=False)

    # 3) Crop PCB (legacy step, keeps the input geometry consistent with the
    #    golden_ok canvas the model was trained on).
    cropped = crop_pcb_from_blue(aligned_img, debug=False)
    final_img = paste_crop_to_canvas(cropped, debug=False)

    # 4) Save the final pre-processed image so DS-YOLO can read it back.
    fname = os.path.basename(image_path)
    final_dir = os.path.join(_BASE_DIR, "stored_images", "final_images")
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, fname)
    cv2.imwrite(final_path, final_img)

    # 5) DS-YOLO inference. ds_infer() lazily loads weights and golden once
    #    and caches them across calls; subsequent inferences are fast.
    detections, annotated = ds_infer(
        weights_path=weights_path,
        img_path=final_path,
        golden_path=golden_path_for_model,
        imgsz=imgsz,
        conf_th=conf_th,
    )

    # 6) Decision: NG if any detection's class id is 1.
    ng_components: List[Dict] = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if int(cls_id) == 1:  # NG
            ng_components.append(
                dict(bbox=[int(x1), int(y1), int(x2), int(y2)],
                     conf=float(conf), cls=CLASS_NAMES[int(cls_id)])
            )

    verdict = "NG" if ng_components else "OK"

    # 7) Save annotated visualisation.
    result_dir = os.path.join(_BASE_DIR, "stored_images", "result")
    os.makedirs(result_dir, exist_ok=True)
    annotated_path = os.path.join(result_dir, fname)
    cv2.imwrite(annotated_path, annotated)

    return dict(
        verdict=verdict,
        ng_components=ng_components,
        annotated_path=annotated_path,
        raw_detections=detections,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m core.pipeline_ds <image_path>")
        sys.exit(1)
    res = main_pipeline_ds(sys.argv[1])
    print(json.dumps(res, indent=2, ensure_ascii=False))
