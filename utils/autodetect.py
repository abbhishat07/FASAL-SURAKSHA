"""
====================================================================
FASAL SURAKSHA - utils/autodetect.py (Optimized & Enhanced)
Unified Auto-Detect Pipeline for Foliar Pathology & Morphology.

Features:
- Pre-flight photo validation: blocking size limits (<120px)
- Foliar quality assessment: sharpness (Laplacian var) & exposure INSIDE leaf mask
- Crop-to-leaf bounding box optimization: eliminates background noise to maximize CNN resolution
- Botanical morphology analysis: palmately-lobed (Grape), compound (Tomato/Potato),
  monocot linear blade (Corn/Rice/Wheat), simple ovate (Apple/Pepper)
- MobileNetV3 / ONNX predictive inference with top-3 candidate distribution
- Dual-signal consistency verification: cross-checks CNN pathology with physical OpenCV lesion severity
- Grad-CAM heatmap visualization focused on the leaf ROI
- Structured export for PDF generation and batch CSV auditing
====================================================================
"""
import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from utils.gradcam import FasalGradCAM, ONNXInference, assess_reliability, parse_plantvillage_label, top_k
from utils.segmentation import draw_severity_overlay, full_leaf_pipeline

try:
    from train_cnn import CROP_DISEASE_CLASSES
except (ImportError, ModuleNotFoundError):
    CROP_DISEASE_CLASSES = []

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ─────────────────────────────────────────────────────────────────
# 1. Photo Quality & Pre-Flight Checks
# ─────────────────────────────────────────────────────────────────
def check_image_size(image_bgr: np.ndarray) -> List[str]:
    """Blocking problems that make the photo unusable for pathology."""
    if image_bgr is None or image_bgr.size == 0:
        return ["Invalid or empty image array."]
    h, w = image_bgr.shape[:2]
    if min(h, w) < 120:
        return ["Photo is too small. Move closer and use the normal camera resolution."]
    return []


def check_leaf_quality(
    image_bgr: np.ndarray, leaf_mask: np.ndarray, blur_threshold: float = 40.0
) -> Tuple[List[str], Dict[str, float]]:
    """
    Sharpness and brightness measured INSIDE the leaf only (a plain white or grey
    backdrop would otherwise look 'blurry' or 'overexposed').
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    img = (
        cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1
        else image_bgr
    )
    m = cv2.resize(leaf_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Drop the leaf edge, which always looks sharp due to background boundary contrast
    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if np.count_nonzero(inner) < 500:
        inner = m

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sel = inner > 0
    if not np.any(sel):
        return ["Unable to isolate leaf surface for quality check."], {"sharpness": 0.0, "brightness": 0.0}

    sharp = float(cv2.Laplacian(gray, cv2.CV_64F)[sel].var())
    bright = float(gray[sel].mean())
    warnings = []
    if sharp < blur_threshold:
        warnings.append("Photo looks blurry. Hold the phone steady and tap to focus on the leaf.")
    if bright < 40:
        warnings.append("Leaf is too dark. Take the photo in daylight.")
    elif bright > 225:
        warnings.append("Leaf is overexposed. Avoid harsh direct sun on the leaf.")
    return warnings, {"sharpness": round(sharp, 1), "brightness": round(bright, 1)}


def crop_to_leaf(image_bgr: np.ndarray, leaf_mask: np.ndarray, margin: float = 0.06) -> np.ndarray:
    """
    Crop the ORIGINAL high-resolution photo to the leaf's bounding box.
    This eliminates wasted background pixels and ensures the CNN classifier
    focuses 100% of its receptive field on leaf pathology.
    """
    ys, xs = np.where(leaf_mask > 0)
    if len(xs) == 0:
        return image_bgr
    mh, mw = leaf_mask.shape[:2]
    h, w = image_bgr.shape[:2]
    sx, sy = w / float(mw), h / float(mh)
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1 = int(max(0, x0 - mx)), int(min(w, x1 + mx))
    y0, y1 = int(max(0, y0 - my)), int(min(h, y1 + my))
    crop = image_bgr[y0:y1, x0:x1]
    return crop if crop.size and min(crop.shape[:2]) >= 32 else image_bgr


# ─────────────────────────────────────────────────────────────────
# 2. Botanical Foliar Morphology Analyzer
# ─────────────────────────────────────────────────────────────────
def analyze_foliar_morphology(
    leaf_crop_bgr: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extracts geometric contour, aspect ratio, solidity, and convexity-defect features
    to provide botanical constraints for Auto-Detect mode:
      - Palmately lobed with sinuses: Grape
      - Multi-leaflet compound foliage: Tomato, Potato
      - Slender linear monocot blades (high aspect ratio >= 2.0): Corn, Rice, Wheat
      - Simple ovate/elliptical high-solidity blades: Apple, Bell Pepper, Peach
    """
    gray = cv2.cvtColor(leaf_crop_bgr, cv2.COLOR_BGR2GRAY)
    if leaf_mask is not None and np.count_nonzero(leaf_mask) > 100:
        bin_mask = cv2.resize(leaf_mask, (leaf_crop_bgr.shape[1], leaf_crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        _, bin_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"aspect_ratio": 1.0, "solidity": 1.0, "deep_sinuses": 0, "morphology_hint": "unknown"}

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    total_area = sum(cv2.contourArea(cnt) for cnt in contours)
    if total_area < 50:
        return {"aspect_ratio": 1.0, "solidity": 1.0, "deep_sinuses": 0, "morphology_hint": "unknown"}

    # Bounding aspect ratio
    _, _, cw, ch = cv2.boundingRect(c)
    aspect_ratio = max(cw, ch) / max(1.0, float(min(cw, ch)))
    c_area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(c_area) / max(1.0, hull_area)

    # Convexity defects for sinus/lobing analysis
    deep_sinuses = 0
    dim_ref = float(min(cw, ch))
    hull_idx = cv2.convexHull(c, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3 and len(c) > 3:
        try:
            defects = cv2.convexityDefects(c, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    d = defects[i, 0, 3] if defects.ndim == 3 else defects[i, 3]
                    depth = float(d) / 256.0
                    if depth >= 0.05 * dim_ref:
                        deep_sinuses += 1
        except Exception:
            pass

    # Determine botanical hint
    morph_hint = "generic"
    if aspect_ratio >= 2.2:
        morph_hint = "monocot_linear"  # Corn, Rice, Wheat
    elif deep_sinuses >= 3 and 0.8 <= aspect_ratio <= 1.6:
        morph_hint = "palmate_lobed"   # Grape
    elif solidity >= 0.85 and aspect_ratio < 1.9:
        morph_hint = "simple_ovate"    # Apple, Pepper, Peach
    elif len(contours) > 2 and total_area > 500:
        morph_hint = "compound_pinnate"  # Tomato, Potato

    return {
        "aspect_ratio": round(aspect_ratio, 2),
        "solidity": round(solidity, 3),
        "deep_sinuses": deep_sinuses,
        "morphology_hint": morph_hint
    }


# ─────────────────────────────────────────────────────────────────
# 3. FasalDetector Main Class
# ─────────────────────────────────────────────────────────────────
class FasalDetector:
    """
    End-to-end autonomous detector from raw image to pathology report:
    Validation -> Segmentation -> Crop-to-Leaf -> Morphology -> Model Inference ->
    Reliability Gate -> Consistency Verification -> Grad-CAM Heatmap.
    """

    def __init__(
        self,
        weights: str = "models/fasal_suraksha_mobilenet.pth",
        class_names_path: Optional[str] = None,
        min_confidence: float = 60.0,
        min_margin: float = 20.0,
        crop_to_leaf_box: bool = True,
        blur_threshold: float = 40.0,
        allow_untrained: bool = False
    ):
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.crop_to_leaf_box = crop_to_leaf_box
        self.blur_threshold = blur_threshold

        # Find best available weights if default does not exist
        resolved_weights = weights
        if not os.path.exists(resolved_weights):
            candidates = [
                "models/fasal_suraksha_mobilenet.pth",
                "models/fasal_suraksha_cnn.onnx",
                "models/best.pt"
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_weights = cand
                    break

        self.class_names = self._load_class_names(resolved_weights, class_names_path)

        if resolved_weights.lower().endswith(".onnx"):
            self.model = ONNXInference(resolved_weights)
            self.can_heatmap = False
        else:
            self.model = FasalGradCAM(
                model_path=resolved_weights if os.path.exists(resolved_weights) else None,
                num_classes=len(self.class_names),
                allow_untrained=allow_untrained
            )
            self.can_heatmap = True

        logger.info(
            f"FasalDetector ready: {len(self.class_names)} classes | "
            f"weights={os.path.basename(resolved_weights)} | "
            f"heatmap={'on' if self.can_heatmap else 'off (ONNX)'}"
        )

    @staticmethod
    def _load_class_names(weights: str, explicit: Optional[str]) -> List[str]:
        """Loads class names from explicit path, checkpoint dictionary, sidecar JSON, or CROP_DISEASE_CLASSES."""
        if explicit and os.path.exists(explicit):
            with open(explicit, "r", encoding="utf-8") as f:
                return json.load(f)

        if os.path.exists(weights) and not weights.lower().endswith(".onnx"):
            try:
                import torch
                ck = torch.load(weights, map_location="cpu", weights_only=False)
                if isinstance(ck, dict):
                    if ck.get("classes"):
                        return list(ck["classes"])
                    if ck.get("class_names"):
                        return list(ck["class_names"])
            except Exception as e:
                logger.debug(f"Could not read classes from torch checkpoint: {e}")

        side = os.path.join(os.path.dirname(os.path.abspath(weights)), "class_names.json")
        if os.path.exists(side):
            with open(side, "r", encoding="utf-8") as f:
                return json.load(f)

        if CROP_DISEASE_CLASSES:
            return list(CROP_DISEASE_CLASSES)

        raise FileNotFoundError(
            f"Class names not found for {weights}. Provide class_names.json or pass explicit path."
        )

    # ─────────────────────────────────────────────────────────────
    # Main Detection Workflow
    # ─────────────────────────────────────────────────────────────
    def detect(
        self,
        image_bgr: np.ndarray,
        with_heatmap: bool = True,
        target_crop_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs complete auto-detection pipeline on a single leaf image.
        Returns a rich diagnostic payload with status, labels, metrics, and visualization overlays.
        """
        res: Dict[str, Any] = {
            "status": "ok",
            "message": "",
            "crop": None,
            "disease": None,
            "label": None,
            "confidence_percent": None,
            "top3": [],
            "reliable": False,
            "reliability_note": "",
            "severity": None,
            "quality": {},
            "morphology": {},
            "warnings": [],
            "images": {},
        }

        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            res.update(status="bad_image", message="Could not read the photo (empty or corrupt buffer).")
            return res

        # 1. Size & Pre-Flight Check
        blockers = check_image_size(image_bgr)
        if blockers:
            res.update(status="bad_image", message=" ".join(blockers))
            return res

        # 2. Leaf Segmentation & Disease Severity
        segmented, mask, severity = full_leaf_pipeline(image_bgr)
        res["severity"] = severity
        res["images"]["segmented"] = segmented
        res["images"]["severity_overlay"] = draw_severity_overlay(image_bgr, mask, severity)

        if not severity.get("leaf_detected"):
            msg = severity.get("capture_advice") or "No leaf found in the photo. Please center an infected leaf and retake."
            if float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean()) < 50:
                msg = "Photo is too dark to isolate the leaf. Please take the photo in clear daylight."
            res.update(status="no_leaf", message=msg)
            return res

        # 3. In-Mask Quality Audit (Sharpness & Exposure)
        warns, qm = check_leaf_quality(image_bgr, mask, self.blur_threshold)
        res["quality"] = qm
        res["warnings"] = list(warns)
        if severity.get("capture_advice"):
            res["warnings"].append(severity["capture_advice"])

        # 4. Crop to Leaf ROI
        leaf = crop_to_leaf(image_bgr, mask) if self.crop_to_leaf_box else image_bgr
        res["images"]["leaf_crop"] = leaf

        # 5. Botanical Foliar Morphology Analysis
        morph = analyze_foliar_morphology(leaf, mask)
        res["morphology"] = morph

        # 6. Deep Learning Inference
        idx, conf, probs = self.model.predict(leaf)

        # Apply target crop constraint or morphology reranking if hint is specified
        if target_crop_hint and target_crop_hint != "🔍 Auto-Detect (AI Computer Vision)":
            clean_hint = target_crop_hint.split(" (")[0].strip().lower()
            matching_indices = [
                i for i, c in enumerate(self.class_names)
                if clean_hint in c.lower().replace("_", " ")
            ]
            if matching_indices and probs is not None:
                sub_probs = probs[matching_indices]
                best_sub_idx = matching_indices[int(np.argmax(sub_probs))]
                idx = best_sub_idx
                conf = float(probs[idx]) * 100.0

        ok, note = assess_reliability(probs, self.min_confidence, self.min_margin)
        label_str = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
        crop_name, disease = parse_plantvillage_label(label_str)

        res.update(
            label=label_str,
            crop=crop_name,
            disease=disease,
            confidence_percent=round(conf, 1),
            top3=top_k(probs, 3, self.class_names) if probs is not None else [],
            reliable=ok,
            reliability_note=note
        )
        res["is_healthy"] = "healthy" in disease.lower()

        if not ok:
            res.update(status="unsure", message=note)

        # 7. Dual-Signal Cross Consistency Check
        sev_pct = float(severity.get("severity_percent", 0.0))
        if severity.get("severity_reliable"):
            if res["is_healthy"] and sev_pct >= 20.0:
                res["warnings"].append("The model predicted 'healthy' but visible lesions cover >= 20% of the leaf. Please verify.")
            elif not res["is_healthy"] and sev_pct < 1.5:
                res["warnings"].append("The model diagnosed a disease but physical lesion severity is minimal (< 1.5%). Could be early-stage or viral chlorosis.")

        # 8. Grad-CAM Model Attention Heatmap
        if with_heatmap and self.can_heatmap:
            try:
                _, _, cam, overlay = self.model.generate_heatmap(leaf, class_idx=idx)
                res["images"]["heatmap"] = overlay
            except Exception as e:
                logger.warning(f"Grad-CAM overlay generation notice: {e}")
                res["images"]["heatmap"] = leaf

        return res

    # ─────────────────────────────────────────────────────────────
    # Helper Utilities
    # ─────────────────────────────────────────────────────────────
    def to_pdf_data(self, res: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Maps detector response into a schema formatted for utils/pdf_generator.py."""
        sev = res.get("severity") or {}
        name = res.get("disease")
        if res.get("status") == "unsure" and name:
            name = f"Uncertain (best guess: {name})"
        data = {
            "crop": res.get("crop"),
            "disease_name": name,
            "confidence_percent": res.get("confidence_percent"),
            "severity_percent": sev.get("severity_percent") if sev.get("leaf_detected") else None,
            "severity_category": sev.get("category"),
            "reliability_note": res.get("reliability_note"),
        }
        data.update(extra or {})
        return data

    @staticmethod
    def summary(res: Dict[str, Any]) -> Dict[str, Any]:
        """Returns JSON-safe summary without large numpy image buffers."""
        return {k: v for k, v in res.items() if k != "images"}

    def detect_folder(self, folder: str, out_csv: Optional[str] = None) -> List[Dict[str, Any]]:
        """Batch detects all foliar images in a folder and optionally outputs a CSV audit report."""
        rows = []
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            r = self.detect(cv2.imread(os.path.join(folder, fn)), with_heatmap=False)
            sev = r.get("severity") or {}
            rows.append({
                "file": fn,
                "status": r["status"],
                "label": r["label"],
                "confidence_percent": r["confidence_percent"],
                "severity_percent": sev.get("severity_percent"),
                "severity_category": sev.get("category"),
                "message": r["message"],
                "warnings": " | ".join(r["warnings"])
            })
        if out_csv and rows:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        return rows


# ─────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────
def _main():
    ap = argparse.ArgumentParser(description="FASAL SURAKSHA - Automatic Leaf Disease Detection")
    ap.add_argument("--weights", default="models/fasal_suraksha_mobilenet.pth")
    ap.add_argument("--classes", default=None)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Single leaf image path")
    src.add_argument("--folder", help="Directory containing leaf images")
    ap.add_argument("--out_dir", default="detect_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    det = FasalDetector(args.weights, args.classes, allow_untrained=True)

    if args.folder:
        rows = det.detect_folder(args.folder, os.path.join(args.out_dir, "results.csv"))
        for r in rows:
            print(f"{r['file']:30s} {r['status']:9s} {str(r['label']):40s} {r['confidence_percent']}%  severity {r['severity_percent']}")
        print(f"\n{len(rows)} photos analyzed -> {os.path.join(args.out_dir, 'results.csv')}")
        return

    res = det.detect(cv2.imread(args.image))
    print(json.dumps(det.summary(res), indent=2, default=str))
    stem = os.path.splitext(os.path.basename(args.image))[0]
    for k, img in res["images"].items():
        if isinstance(img, np.ndarray) and img.size > 0:
            cv2.imwrite(os.path.join(args.out_dir, f"{stem}_{k}.jpg"), img)
    print(f"Artifacts saved to {args.out_dir}/")


if __name__ == "__main__":
    _main()
