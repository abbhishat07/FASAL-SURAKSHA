"""
====================================================================
FASAL SURAKSHA - utils/gradcam.py
Gradient-weighted Class Activation Mapping (Grad-CAM) engine.
Produces glowing heatmap overlays on the final convolutional layer
of MobileNetV3-Small for explainable disease localization.
====================================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # onnxruntime not installed; ONNXInference will raise a clear error at init time
from torchvision import models, transforms
from typing import Optional, Tuple
from loguru import logger

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class GradCAMHook:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute_cam(self, input_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        if self.gradients is None or self.activations is None:
            return np.zeros(input_size[::-1], dtype=np.float32)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, input_size, interpolation=cv2.INTER_LANCZOS4)
        return cam.astype(np.float32)


class FasalGradCAM:
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 38,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model = self._load_model(model_path)
        self.model.eval()

        target_layer = self.model.features[-1][0]
        self.hook = GradCAMHook(self.model, target_layer)
        logger.info(f"Grad-CAM initialized on {device}")

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, self.num_classes)
        )
        model = model.to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                model.load_state_dict(state, strict=False)
                logger.info(f"Loaded weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights ({e}); using unweighted baseline.")
        return model

    def predict(self, image_bgr: np.ndarray) -> Tuple[int, float, np.ndarray]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = PREPROCESS(rgb).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)
        pred_class = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item()) * 100.0
        return pred_class, confidence, logits.detach().cpu().numpy()[0]

    def generate_heatmap(
        self,
        image_bgr: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> Tuple[int, float, np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = PREPROCESS(rgb).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, class_idx].item()) * 100.0

        score = logits[0, class_idx]
        score.backward()

        h, w = image_bgr.shape[:2]
        cam_array = self.hook.compute_cam(input_size=(w, h))

        # Glowing heatmap overlay
        cam_uint8 = (cam_array * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        glow = cv2.GaussianBlur(heatmap, (21, 21), 0)
        glowing_heatmap = cv2.addWeighted(heatmap, 0.7, glow, 0.3, 0)
        overlay = cv2.addWeighted(image_bgr, 0.45, glowing_heatmap, 0.55, 0)

        return class_idx, confidence, cam_array, overlay


class ONNXInference:
    def __init__(self, onnx_path: str):
        if ort is None:
            raise ImportError(
                "onnxruntime is required for ONNXInference. "
                "Install it with: pip install onnxruntime"
            )
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_bgr: np.ndarray) -> Tuple[int, float, np.ndarray]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        img  = (img - mean) / std
        img  = img.transpose(2, 0, 1)[np.newaxis, :]

        outputs = self.session.run(None, {self.input_name: img})
        logits  = outputs[0][0]
        probs   = np.exp(logits) / np.exp(logits).sum()
        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100.0
        return class_idx, confidence, probs
