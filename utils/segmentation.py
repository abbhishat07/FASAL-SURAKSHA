"""
====================================================================
FASAL SURAKSHA - utils/segmentation.py   (revised)
OpenCV leaf segmentation, background removal and Disease Severity
Index (DSI) computation.

What changed vs. the previous version
-------------------------------------
* The leaf mask no longer excludes diseased tissue.  Before, brown
  lesions were whitened out by the green/yellow colour filter, so the
  largest lesions were counted in the leaf area but could never be
  counted as disease -> severity was biased LOW for severe infections.
  Now the leaf is found from a green "core" + GrabCut (seeded from that
  core, not from a frame-sized rectangle), holes are filled, and lesions
  are measured on the ORIGINAL pixels inside the mask.
* Compound leaves keep every large leaflet (not just the largest one).
* "Healthy" category added; "no leaf found" is reported explicitly
  instead of silently returning 0 % / Mild.
* Disease colour rules are stricter so shadows, veins and light-green
  healthy tissue are not counted as disease.

LIMITATIONS (be honest in the UI):
  DSI is a colour-based lesion-area estimate.  It is sensitive to
  lighting and camera, and it is NOT yield loss.  Calibrate it against
  expert-scored leaves before relying on the numbers.
====================================================================
"""
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from loguru import logger

# ─────────────────────────────────────────────
# Severity categories (percent of leaf area showing disease colours)
# ─────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "Healthy":  (0.0,   2.0),
    "Mild":     (2.0,  20.0),
    "Moderate": (20.0, 50.0),
    "Severe":   (50.0, 100.01),
}

MAX_SIDE = 512                 # working resolution (aspect ratio preserved)
MIN_LEAF_FRACTION = 0.02       # leaf must cover >= 2 % of the frame
COMPONENT_KEEP_RATIO = 0.15    # keep leaflets >= 15 % of the biggest one


def _resize_keep_aspect(image_bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = max_side / float(max(h, w))
    if abs(scale - 1.0) < 1e-6:
        return image_bgr
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)


# ─────────────────────────────────────────────
# Leaf mask
# ─────────────────────────────────────────────
def _green_core(hsv: np.ndarray) -> np.ndarray:
    """Confident healthy-leaf pixels (OpenCV hue: 0-179)."""
    lower = np.array([30, 40, 40], np.uint8)
    upper = np.array([90, 255, 255], np.uint8)
    core = cv2.inRange(hsv, lower, upper)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return core


def _fill_holes_keep_leaflets(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (lesions) and keep all sizeable components."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    areas = [cv2.contourArea(c) for c in contours]
    biggest = max(areas)
    out = np.zeros_like(mask)
    for c, a in zip(contours, areas):
        if a >= COMPONENT_KEEP_RATIO * biggest:
            cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    return out



def _plant_like(hsv: np.ndarray) -> np.ndarray:
    """Coloured (green / yellow / brown) pixels.  White, grey and shadow-grey
    backgrounds have low saturation and are excluded; lesions keep their colour."""
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return (((s >= 45) & (v >= 30)) * 255).astype(np.uint8)


def _plain_background(plant_like: np.ndarray) -> bool:
    """True when the outer frame is mostly NOT plant-coloured (white / grey / paper backdrop)."""
    h, w = plant_like.shape
    b = max(4, int(0.05 * min(h, w)))
    border = np.concatenate([
        plant_like[:b, :].ravel(), plant_like[-b:, :].ravel(),
        plant_like[:, :b].ravel(), plant_like[:, -b:].ravel(),
    ])
    return float(np.count_nonzero(border)) / border.size < 0.15


def _segment(image_bgr: np.ndarray):
    """Return (mask, mode).  mode in {'plain', 'complex', 'none'}."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    core = _green_core(hsv)
    if np.count_nonzero(core) < MIN_LEAF_FRACTION * h * w:
        return np.zeros((h, w), np.uint8), "none"

    plant = _plant_like(hsv)
    if _plain_background(plant):
        # Plain backdrop: leaf = every coloured pixel, INCLUDING brown/yellow lesions.
        mask = cv2.morphologyEx(plant, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        mask = _fill_holes_keep_leaflets(mask)
        if np.count_nonzero(mask) >= MIN_LEAF_FRACTION * h * w:
            return mask, "plain"

    return _grabcut_from_core(image_bgr, core), "complex"


def _grabcut_from_core(image_bgr: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Cluttered background (soil, other plants): GrabCut seeded from the green core.
    LESS RELIABLE - GrabCut may treat large brown lesions as background, which
    under-estimates severity.  Callers should flag results from this mode."""
    h, w = image_bgr.shape[:2]

    # Seed GrabCut from the green core instead of a frame-sized rectangle.
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    sure_fg = cv2.erode(core, k_small)
    probable_fg = cv2.dilate(core, k_big)

    gc_mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc_mask[probable_fg > 0] = cv2.GC_PR_BGD
    gc_mask[cv2.dilate(core, k_small) > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(image_bgr, gc_mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        grab_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error as exc:
        logger.warning(f"GrabCut failed, using dilated green core only: {exc}")
        grab_fg = cv2.dilate(core, k_small)

    mask = cv2.bitwise_or(grab_fg, core)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    mask = _fill_holes_keep_leaflets(mask)

    if np.count_nonzero(mask) < MIN_LEAF_FRACTION * h * w:
        return np.zeros((h, w), np.uint8)
    return mask


def get_leaf_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Leaf mask (255 = leaf) for an image at working resolution; all-zero if no leaf."""
    return _segment(image_bgr)[0]


def extract_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    Kept for backward compatibility.  Fills holes and keeps the largest
    component plus any other sizeable leaflets (compound leaves).
    """
    return _fill_holes_keep_leaflets(mask)


def remove_background(image_bgr: np.ndarray) -> np.ndarray:
    """Return the image with everything outside the leaf set to white."""
    resized = _resize_keep_aspect(image_bgr)
    mask = get_leaf_mask(resized)
    white = np.full_like(resized, 255)
    return np.where(mask[:, :, None] > 0, resized, white).astype(np.uint8)


# ─────────────────────────────────────────────
# Disease Severity Index
# ─────────────────────────────────────────────
def _disease_mask(hsv: np.ndarray) -> np.ndarray:
    """
    Pixels that look diseased (OpenCV hue 0-179):
      * brown / tan lesions
      * yellow chlorosis (strongly yellow only, so light-green healthy
        tissue is NOT counted)
      * dark necrosis, excluding dark GREEN pixels (= shadow / veins)
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    brown = ((h <= 19) | (h >= 170)) & (s >= 40) & (v >= 25) & (v <= 235)
    chlorosis = (h >= 20) & (h <= 28) & (s >= 70) & (v >= 100)
    dark_green = (h >= 30) & (h <= 90) & (s >= 60)
    necrosis = (v < 50) & ~dark_green

    mask = (brown | chlorosis | necrosis).astype(np.uint8) * 255
    # drop speckle noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def compute_severity(image_bgr: np.ndarray, leaf_mask: np.ndarray) -> Dict[str, Any]:
    """
    Disease Severity Index = diseased pixels / leaf pixels * 100.

    `image_bgr` must contain the ORIGINAL pixels inside the leaf (the
    output of `full_leaf_pipeline` does).  A thin border band is excluded
    from both numerator and denominator to avoid segmentation-edge artefacts.
    """
    if leaf_mask is None or np.count_nonzero(leaf_mask) == 0:
        return {
            "severity_percent": 0.0,
            "category": "No leaf detected",
            "leaf_detected": False,
            "leaf_pixel_count": 0,
            "diseased_pixel_count": 0,
        }

    inner = cv2.erode(leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if np.count_nonzero(inner) == 0:
        inner = leaf_mask

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    diseased = cv2.bitwise_and(_disease_mask(hsv), inner)

    leaf_pixels = int(np.count_nonzero(inner))
    diseased_pixels = int(np.count_nonzero(diseased))
    pct = min(100.0, 100.0 * diseased_pixels / leaf_pixels)

    category = "Severe"
    for cat, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= pct < hi:
            category = cat
            break

    return {
        "severity_percent": round(pct, 2),
        "category": category,
        "leaf_detected": True,
        "leaf_pixel_count": leaf_pixels,
        "diseased_pixel_count": diseased_pixels,
    }


def full_leaf_pipeline(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns (segmented_bgr, leaf_mask, severity_dict).
    segmented_bgr = leaf on white background (original pixels kept inside).
    """
    resized = _resize_keep_aspect(image_bgr)
    leaf_mask, mode = _segment(resized)
    segmented = np.where(leaf_mask[:, :, None] > 0, resized, np.full_like(resized, 255)).astype(np.uint8)
    severity = compute_severity(resized, leaf_mask)
    severity["segmentation_mode"] = mode
    # Large brown lesions can be lost on cluttered backgrounds -> flag, don't hide.
    severity["severity_reliable"] = mode == "plain"
    if mode == "plain":
        severity["capture_advice"] = None
    elif mode == "none":
        severity["capture_advice"] = "No leaf found. Fill the frame with one leaf on a plain white background and retake."
    else:
        severity["capture_advice"] = (
            "Background is cluttered: severity may be under-estimated. Hold the leaf against plain white paper and retake."
        )
    return segmented, leaf_mask, severity


def draw_severity_overlay(
    original_bgr: np.ndarray,
    leaf_mask: np.ndarray,
    severity_info: Dict[str, Any],
) -> np.ndarray:
    """Coloured leaf contour + severity label drawn on the original photo."""
    overlay = original_bgr.copy()
    color_map = {
        "Healthy":  (0, 200, 0),
        "Mild":     (0, 200, 0),
        "Moderate": (0, 165, 255),
        "Severe":   (0, 0, 220),
    }
    cat = severity_info.get("category", "Mild")
    color = color_map.get(cat, (160, 160, 160))

    if leaf_mask is not None and np.count_nonzero(leaf_mask) > 0:
        resized_mask = cv2.resize(
            leaf_mask, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 3)

    if severity_info.get("leaf_detected", True):
        label = f"{cat}: {severity_info.get('severity_percent', 0.0):.1f}%"
    else:
        label = "No leaf detected - retake photo"
    cv2.putText(overlay, label, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)
    return overlay
