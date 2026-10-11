"""
====================================================================
FASAL SURAKSHA - utils/segmentation.py
OpenCV-based leaf segmentation, background removal, and
Disease Severity Index (DSI) computation.
====================================================================
"""

import cv2
import numpy as np
from typing import Tuple, Dict
from loguru import logger

# ─────────────────────────────────────────────
# Severity thresholds (percent infected pixels)
# ─────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "Mild":     (0.0,  20.0),
    "Moderate": (20.0, 50.0),
    "Severe":   (50.0, 100.0),
}


def remove_background(image_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the leaf from the image using GrabCut + HSV masking.
    Returns:
        BGR image with background replaced by white pixels.
    """
    h, w = image_bgr.shape[:2]

    # Step 1: GrabCut
    mask = np.zeros((h, w), np.uint8)
    rect = (10, 10, w - 20, h - 20)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        grab_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    except Exception as exc:
        logger.warning(f"GrabCut fallback to HSV: {exc}")
        grab_mask = np.ones((h, w), np.uint8)

    # Step 2: HSV green + yellow-green masking
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green  = np.array([25,  30,  30])
    upper_green  = np.array([90, 255, 255])
    lower_yellow = np.array([15,  30,  30])
    upper_yellow = np.array([35, 255, 255])
    green_mask  = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    hsv_mask = cv2.bitwise_or(green_mask, yellow_mask) // 255

    combined = (grab_mask & hsv_mask).astype(np.uint8)

    # Step 3: Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, kernel, iterations=1)

    bg_white = np.full_like(image_bgr, 255)
    result = np.where(combined[:, :, np.newaxis] == 1, image_bgr, bg_white)
    return result.astype(np.uint8)


def extract_largest_contour(mask: np.ndarray) -> np.ndarray:
    """Return a binary mask keeping only the largest connected contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean


def compute_severity(
    segmented_bgr: np.ndarray,
    leaf_mask: np.ndarray,
) -> Dict[str, float | str]:
    """
    Compute Disease Severity Index (DSI) from segmented leaf.
    Disease pixels = brown / yellow-brown / dark necrotic discolorations.
    """
    hsv = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2HSV)

    ranges = [
        (np.array([5,  40,  40]), np.array([20, 255, 200])),  # Brown / tan
        (np.array([0,  60,  20]), np.array([10, 255, 100])),  # Dark brown
        (np.array([20, 30,  80]), np.array([35, 255, 255])),  # Yellow stress
        (np.array([0,   0,   0]), np.array([180, 255,  50])),  # Black necrosis
    ]
    disease_mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        disease_mask = cv2.bitwise_or(disease_mask, cv2.inRange(hsv, lo, hi))

    disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    leaf_pixels     = int(np.sum(leaf_mask > 0))
    diseased_pixels = int(np.sum(disease_mask > 0))

    if leaf_pixels == 0:
        return {
            "severity_percent":    0.0,
            "category":            "Mild",
            "leaf_pixel_count":    0,
            "diseased_pixel_count": 0,
        }

    pct = min(100.0, (diseased_pixels / leaf_pixels) * 100.0)

    category = "Mild"
    for cat, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= pct < hi:
            category = cat
            break
    if pct >= 50.0:
        category = "Severe"

    return {
        "severity_percent":    round(pct, 2),
        "category":            category,
        "leaf_pixel_count":    leaf_pixels,
        "diseased_pixel_count": diseased_pixels,
    }


def full_leaf_pipeline(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Full segmentation pipeline returning segmented leaf, mask, and severity dict."""
    target_size = (512, 512)
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
    segmented = remove_background(resized)

    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, leaf_mask_raw = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    leaf_mask = extract_largest_contour(leaf_mask_raw)

    severity = compute_severity(segmented, leaf_mask)
    return segmented, leaf_mask, severity


def draw_severity_overlay(
    original_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    severity_info: Dict,
) -> np.ndarray:
    """Renders colored contour and severity metric overlay on the leaf photo."""
    overlay = original_bgr.copy()
    color_map = {
        "Mild":     (0, 200, 0),
        "Moderate": (0, 165, 255),
        "Severe":   (0, 0, 220),
    }
    cat = severity_info.get("category", "Mild")
    color = color_map.get(cat, (0, 200, 0))

    resized_mask = cv2.resize(leaf_mask, (original_bgr.shape[1], original_bgr.shape[0]))
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 3)

    label = f"{cat}: {severity_info['severity_percent']:.1f}%"
    cv2.putText(overlay, label, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    return overlay
