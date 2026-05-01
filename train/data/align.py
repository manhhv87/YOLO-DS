"""ORB + RANSAC homography alignment used by RG-YOLO.

Given a captured frame and a fixed golden reference, returns the warped
capture in the reference frame, the homography matrix and the inlier ratio.
Used both at training time (precomputing diff maps) and at inference time
(on the production line).
"""
from __future__ import annotations

import cv2
import numpy as np


def align_to_golden(
    capture_bgr: np.ndarray,
    golden_bgr: np.ndarray,
    *,
    n_features: int = 2000,
    ratio: float = 0.75,
    ransac_thresh: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align ``capture_bgr`` to ``golden_bgr`` using ORB + RANSAC.

    Returns
    -------
    warped : (H, W, 3) uint8
        Capture warped into the golden frame.
    H : (3, 3) float64
        Estimated homography ``warped = H @ capture``.
    inlier_ratio : float
        Fraction of matches kept by RANSAC.
    """
    if capture_bgr.shape != golden_bgr.shape:
        capture_bgr = cv2.resize(capture_bgr, (golden_bgr.shape[1], golden_bgr.shape[0]))

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_c, des_c = orb.detectAndCompute(cv2.cvtColor(capture_bgr, cv2.COLOR_BGR2GRAY), None)
    kp_g, des_g = orb.detectAndCompute(cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY), None)

    if des_c is None or des_g is None or len(kp_c) < 4 or len(kp_g) < 4:
        raise RuntimeError("Not enough ORB features for alignment.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des_c, des_g, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 4:
        raise RuntimeError(f"Only {len(good)} good matches; need >= 4.")

    src = np.float32([kp_c[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_g[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    inlier_ratio = float(mask.sum()) / float(len(mask)) if mask is not None else 0.0

    h, w = golden_bgr.shape[:2]
    warped = cv2.warpPerspective(capture_bgr, H, (w, h))
    return warped, H, inlier_ratio


def difference_map(
    aligned_bgr: np.ndarray,
    golden_bgr: np.ndarray,
    *,
    grayscale: bool = True,
) -> np.ndarray:
    """Per-pixel absolute difference between aligned capture and golden.

    With ``grayscale=True`` returns a single-channel ``(H, W)`` map; otherwise
    a 3-channel BGR map.
    """
    diff = cv2.absdiff(aligned_bgr, golden_bgr)
    if grayscale:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return diff
