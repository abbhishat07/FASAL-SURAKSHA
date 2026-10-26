"""
====================================================================
FASAL SURAKSHA - utils/gradcam.py   (revised)
Grad-CAM + inference for MobileNetV3-Small (PyTorch) and an ONNX
runtime path.

What changed vs. the previous version
-------------------------------------
* Weights are loaded with strict=True.  The old strict=False could
  leave the classifier head randomly initialised with only a log line.
  Now a bad / missing checkpoint raises a clear error.  Both common
  head layouts (plain Linear and Dropout+Linear) are detected from the
  checkpoint, and num_classes is inferred from it.
* No gradients during plain prediction; ONE forward pass for heatmaps.
* Grad-CAM uses torch.autograd.grad on a forward-hook activation
  (no full-backward hook, no state shared between calls) and runs under
  a lock, so a cached instance is safe in Streamlit.
* Heatmap is scaled by its max (not min-max) and clipped; it shows where
  the model looked.  It is NOT proof that the diagnosis is correct.
* ONNX path: numerically stable softmax, detects models that already
  output probabilities, and uses the SAME preprocessing as PyTorch.
* Both predict() methods now return (class_idx, confidence_percent,
  PROBABILITIES).  (PyTorch used to return logits.)
* Optional torch import, so an ONNX-only deployment does not need torch.
* Helpers: top_k(), assess_reliability() (low-confidence / unsure gate),
  parse_plantvillage_label().
====================================================================
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
except ImportError:  # ONNX-only deployment
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    models = None  # type: ignore

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = (224, 224)


# ─────────────────────────────────────────────
# Shared preprocessing (identical for PyTorch and ONNX)
# ─────────────────────────────────────────────
def preprocess_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 image -> float32 array shaped (1, 3, 224, 224), ImageNet-normalised.

    Uses PIL bilinear resize, matching torchvision's Resize on PIL images
    (the usual training pipeline).  If you trained with a different
    resize / crop, change it HERE so both inference paths stay in sync.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to preprocess_bgr")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


# ─────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────
def top_k(probs: np.ndarray, k: int = 3, class_names: Optional[List[str]] = None) -> List[Dict]:
    idx = np.argsort(probs)[::-1][:k]
    return [
        {
            "class_idx": int(i),
            "label": class_names[i] if class_names and i < len(class_names) else str(int(i)),
            "probability_percent": round(float(probs[i]) * 100.0, 2),
        }
        for i in idx
    ]


def assess_reliability(
    probs: np.ndarray,
    min_confidence: float = 60.0,
    min_margin: float = 20.0,
) -> Tuple[bool, str]:
    """
    Simple "is this prediction trustworthy enough to show?" gate.

    Softmax confidence is NOT calibrated and a classifier trained on lab
    images (e.g. PlantVillage) is often over-confident on field photos or
    non-leaf images.  Treat a False result as "ask for a clearer photo /
    expert check", and tune the thresholds on real field images.
    """
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]]) * 100.0
    top2 = float(probs[order[1]]) * 100.0 if len(order) > 1 else 0.0
    if top1 < min_confidence:
        return False, f"Low confidence ({top1:.1f}%). Retake the photo or consult an expert."
    if top1 - top2 < min_margin:
        return False, "Top two diseases are too close to call. Retake the photo or consult an expert."
    return True, "OK"


def parse_plantvillage_label(label: str) -> Tuple[str, str]:
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')."""
    if "___" in label:
        crop, disease = label.split("___", 1)
    else:
        crop, disease = "Unknown", label
    crop = crop.replace("_", " ").replace(",", "").strip()
    disease = disease.replace("_", " ").strip()
    return crop, disease


# ─────────────────────────────────────────────
# PyTorch + Grad-CAM
# ─────────────────────────────────────────────
class GradCAMHook:
    """Stores the target layer's activation from the latest forward pass."""

    def __init__(self, target_layer):
        self.activations = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output  # keep graph: gradients are taken with autograd.grad

    def remove(self):
        self._handle.remove()


def _clean_state_dict(state) -> Dict:
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def _build_head_and_classes(state: Dict) -> Tuple[str, int]:
    """Detect classifier head layout and class count from checkpoint keys."""
    if "classifier.3.1.weight" in state:  # Dropout + Linear (this project's layout)
        return "dropout_linear", int(state["classifier.3.1.weight"].shape[0])
    if "classifier.3.weight" in state:  # plain torchvision Linear
        return "linear", int(state["classifier.3.weight"].shape[0])
    raise RuntimeError(
        "Checkpoint has no recognisable MobileNetV3-Small classifier head "
        "(expected 'classifier.3.weight' or 'classifier.3.1.weight')."
    )


class FasalGradCAM:
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: Optional[int] = None,
        device: str = "cpu",
        allow_untrained: bool = False,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for FasalGradCAM. Use ONNXInference for torch-free deployment.")
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self.model = self._load_model(model_path, num_classes, allow_untrained)
        self.model.eval()
        self.hook = GradCAMHook(self.model.features[-1])  # last conv block, taken AFTER its activation
        logger.info(f"Grad-CAM ready on {device} ({self.num_classes} classes)")

    # -- loading ------------------------------------------------------
    def _load_model(self, model_path, num_classes, allow_untrained):
        have_ckpt = bool(model_path) and os.path.exists(model_path)
        if not have_ckpt:
            if not allow_untrained:
                raise FileNotFoundError(
                    f"Model weights not found at {model_path!r}. Refusing to run with random weights "
                    "(pass allow_untrained=True only for UI demos)."
                )
            logger.warning("RUNNING WITH UNTRAINED WEIGHTS - predictions are meaningless.")
            self.num_classes = num_classes or 38
            head = "dropout_linear"
            state = None
        else:
            try:
                raw = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception as exc:
                raise RuntimeError(f"Could not read checkpoint {model_path}: {exc}") from exc
            state = _clean_state_dict(raw)
            head, inferred = _build_head_and_classes(state)
            if num_classes is not None and num_classes != inferred:
                raise RuntimeError(f"num_classes={num_classes} but checkpoint has {inferred} classes.")
            self.num_classes = inferred

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        if head == "dropout_linear":
            model.classifier[-1] = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features, self.num_classes))
        else:
            model.classifier[-1] = nn.Linear(in_features, self.num_classes)

        if state is not None:
            model.load_state_dict(state, strict=True)  # raises on ANY mismatch
            logger.info(f"Loaded weights from {model_path}")
        return model.to(self.device)

    # -- inference ----------------------------------------------------
    def _tensor(self, image_bgr: np.ndarray):
        return torch.from_numpy(preprocess_bgr(image_bgr)).to(self.device)

    def predict(self, image_bgr: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Returns (class_idx, confidence_percent, probabilities)."""
        with self._lock, torch.no_grad():
            probs = F.softmax(self.model(self._tensor(image_bgr)), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]) * 100.0, probs

    def generate_heatmap(
        self,
        image_bgr: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """Returns (class_idx, confidence_percent, cam_array[H,W] in 0..1, overlay_bgr)."""
        with self._lock:
            with torch.enable_grad():
                x = self._tensor(image_bgr)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                if class_idx is None:
                    class_idx = int(probs.argmax(dim=1).item())
                confidence = float(probs[0, class_idx].item()) * 100.0

                acts = self.hook.activations  # (1, C, h, w), part of the graph
                grads = torch.autograd.grad(logits[0, class_idx], acts)[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * acts).sum(dim=1)).squeeze(0).detach().cpu().numpy()

        peak = float(cam.max())
        cam = cam / peak if (np.isfinite(peak) and peak > 0.0) else np.zeros_like(cam)

        h, w = image_bgr.shape[:2]
        cam = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        cam = np.clip(cam, 0.0, 1.0).astype(np.float32)

        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        glow = cv2.GaussianBlur(heatmap, (21, 21), 0)
        glowing = cv2.addWeighted(heatmap, 0.7, glow, 0.3, 0)
        # blend only where the model attends, so quiet areas keep the original photo
        alpha = (cam * 0.6)[:, :, None]
        overlay = (image_bgr.astype(np.float32) * (1 - alpha) + glowing.astype(np.float32) * alpha).astype(np.uint8)
        return class_idx, confidence, cam, overlay

    def close(self):
        self.hook.remove()


# ─────────────────────────────────────────────
# ONNX path (no torch needed)
# ─────────────────────────────────────────────
class ONNXInference:
    def __init__(self, onnx_path: str):
        if ort is None:
            raise ImportError("onnxruntime is required for ONNXInference: pip install onnxruntime")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_bgr: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Returns (class_idx, confidence_percent, probabilities)."""
        out = self.session.run(None, {self.input_name: preprocess_bgr(image_bgr)})[0][0].astype(np.float64)
        already_probs = bool(out.min() >= 0.0 and out.max() <= 1.0 and abs(out.sum() - 1.0) < 1e-3)
        probs = out if already_probs else _stable_softmax(out)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]) * 100.0, probs.astype(np.float32)
