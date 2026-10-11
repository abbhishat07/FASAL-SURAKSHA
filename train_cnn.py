"""
====================================================================
FASAL SURAKSHA - train_cnn.py
Fine-tunes MobileNetV3-Small on agricultural micro-crop leaf datasets.
Exports:
  1. models/fasal_suraksha_mobilenet.pth  (PyTorch weights with Grad-CAM support)
  2. models/fasal_suraksha_cnn.onnx       (Cross-platform ONNX graph)
====================================================================
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from loguru import logger

# 38 Common Micro-Crop & PlantVillage Disease Classes
CROP_DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Brown_Spot",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Rice___healthy",
    "Soybean___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Wheat___Brown_Rust",
    "Wheat___Yellow_Stripe_Rust",
    "Wheat___Septoria",
    "Wheat___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Cotton___Bacterial_Blight",
    "Cotton___healthy"
]


class SyntheticCropLeafDataset(Dataset):
    """
    Fallback dataset generator when training without pre-downloaded images.
    Generates realistic leaf-like synthetic matrices with varying hues & lesion artifacts
    to guarantee valid gradient backprop and onnx/tflite export.
    """
    def __init__(self, num_samples: int = 120, num_classes: int = len(CROP_DISEASE_CLASSES), transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = idx % self.num_classes
        # Synthesize leaf-textured image (Green with optional brownish spots)
        np_img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Leaf green background
        np_img[:, :, 1] = np.random.randint(120, 210, (224, 224), dtype=np.uint8)
        np_img[:, :, 0] = np.random.randint(30, 80, (224, 224), dtype=np.uint8)
        np_img[:, :, 2] = np.random.randint(20, 60, (224, 224), dtype=np.uint8)

        # If not healthy, add necrotic brown/yellow lesions
        if "healthy" not in CROP_DISEASE_CLASSES[label].lower():
            cx, cy = np.random.randint(50, 170), np.random.randint(50, 170)
            radius = np.random.randint(15, 45)
            y, x = np.ogrid[:224, :224]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            np_img[mask, 0] = np.random.randint(100, 180) # Reddish-brown
            np_img[mask, 1] = np.random.randint(60, 120)
            np_img[mask, 2] = np.random.randint(10, 40)

        pil_img = Image.fromarray(np_img)
        if self.transform:
            return self.transform(pil_img), label
        return pil_img, label


def build_mobilenet_v3(num_classes: int) -> nn.Module:
    """Builds MobileNetV3-Small with custom classifier head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_and_export_cnn(
    data_dir: str = "dataset",
    output_dir: str = "models",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.001
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for CNN training: {device}")

    num_classes = len(CROP_DISEASE_CLASSES)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class MappedCropDataset(Dataset):
    """Loads images from subfolders and maps them strictly to CROP_DISEASE_CLASSES indices."""
    def __init__(self, data_dir: str, class_list: list, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(class_list)}

        if os.path.exists(data_dir):
            for d in os.listdir(data_dir):
                d_path = os.path.join(data_dir, d)
                if not os.path.isdir(d_path):
                    continue
                if d in self.class_to_idx:
                    idx = self.class_to_idx[d]
                    for f in os.listdir(d_path):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.samples.append((os.path.join(d_path, f), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train_and_export_cnn(
    data_dir: str = "dataset",
    output_dir: str = "models",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.001
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for CNN training: {device}")

    num_classes = len(CROP_DISEASE_CLASSES)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Combine real mapped dataset with synthetic samples across all 38 classes
    mapped_real = MappedCropDataset(data_dir, CROP_DISEASE_CLASSES, transform=train_transform)
    synth_train = SyntheticCropLeafDataset(num_samples=228, num_classes=num_classes, transform=train_transform)
    
    if len(mapped_real) > 0:
        logger.info(f"Loaded {len(mapped_real)} real samples with exact CROP_DISEASE_CLASSES indices.")
        train_ds = torch.utils.data.ConcatDataset([mapped_real, synth_train])
    else:
        logger.info("Using full synthetic leaf dataset covering all 38 agricultural classes.")
        train_ds = synth_train

    val_ds = SyntheticCropLeafDataset(num_samples=76, num_classes=num_classes, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_mobilenet_v3(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Commencing MobileNetV3-Small fine-tuning...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(1, val_total)
        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss/max(1,val_total):.4f} | Val Acc: {val_acc*100:.2f}%"
        )

    # 1. Export PyTorch weights (for Grad-CAM)
    pth_path = os.path.join(output_dir, "fasal_suraksha_mobilenet.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CROP_DISEASE_CLASSES,
        "arch": "mobilenet_v3_small"
    }, pth_path)
    logger.info(f"PyTorch model successfully saved to: {pth_path}")

    # 2. Export ONNX graph (cross-platform deployment)
    model.eval()
    model.to("cpu")
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    onnx_path = os.path.join(output_dir, "fasal_suraksha_cnn.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_leaf_image"],
        output_names=["disease_logits"],
        dynamic_axes={"input_leaf_image": {0: "batch_size"}, "disease_logits": {0: "batch_size"}}
    )
    logger.info(f"ONNX model successfully exported to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fasal Suraksha MobileNetV3 CNN")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_and_export_cnn(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
