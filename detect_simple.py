"""
Simple sliding-window detection for dental radiography.
Finds all tissue regions (dentin, enamel, pulp) across the full image.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'slm/resolution_best_densenet_model.pth'
PATCH_SIZE = 224
STRIDE = 56
CONFIDENCE_THRESHOLD = 0.90
NMS_IOU_THRESHOLD = 0.35

CLASS_NAMES = ['dentin', 'enamel', 'pulp']
CLASS_COLORS = {
    'dentin': (255, 0, 0),      # Red
    'enamel': (0, 255, 0),      # Green
    'pulp': (0, 0, 255),        # Blue
}

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

_model = None

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    """Load DenseNet121 checkpoint for 3-class tissue detection."""
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 3)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def get_model():
    """Return a cached model instance, loading on first use."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _slide_positions(length, patch_size, stride):
    """Window start positions that cover the full axis, including the far edge."""
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area)


def non_max_suppression(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """Keep spatially distinct detections; suppress overlapping lower-confidence boxes."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d[5], reverse=True)
    kept = []
    for det in detections:
        if all(_box_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def _get_font(size=14):
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def detect(image_path, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Scan the full image with a sliding window and return all tissue detections.
    Each detection: (x1, y1, x2, y2, class_name, confidence)
    """
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    print(f"Image size: {w}x{h}")

    # Upscale very small images so the sliding window can run
    work_image = image
    scale = 1.0
    if w < PATCH_SIZE or h < PATCH_SIZE:
        scale = max(PATCH_SIZE / w, PATCH_SIZE / h)
        work_image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        w, h = work_image.size
        print(f"Upscaled to: {w}x{h}")

    detections = []
    print("Scanning image...")
    model = get_model()

    for y in _slide_positions(h, PATCH_SIZE, STRIDE):
        for x in _slide_positions(w, PATCH_SIZE, STRIDE):
            patch = work_image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            input_tensor = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, pred = torch.max(probs, 0)

            if conf.item() >= confidence_threshold:
                class_name = CLASS_NAMES[pred.item()]
                if scale != 1.0:
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + PATCH_SIZE) / scale)
                    y2 = int((y + PATCH_SIZE) / scale)
                else:
                    x1, y1, x2, y2 = x, y, x + PATCH_SIZE, y + PATCH_SIZE
                detections.append((x1, y1, x2, y2, class_name, conf.item()))

    print(f"Raw detections: {len(detections)}")
    detections = non_max_suppression(detections)

    if not detections:
        print(f"❌ No detections found with confidence >= {confidence_threshold*100:.0f}%")
        return image, []

    print(f"\n✅ Found {len(detections)} region(s) (>= {confidence_threshold*100:.0f}% confidence):")
    for i, (x1, y1, x2, y2, class_name, conf) in enumerate(detections, 1):
        print(
            f"   [{i}] {class_name.upper():6s}  {conf*100:5.1f}%  "
            f"box=({x1}, {y1})-({x2}, {y2})"
        )

    return draw_detections(image, detections), detections


def draw_detections(image, detections):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = _get_font()

    for x1, y1, x2, y2, class_name, conf in detections:
        color = CLASS_COLORS[class_name]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label_text = f"{class_name.upper()} {conf*100:.0f}%"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle([x1, label_y, x1 + text_width + 8, y1], fill=color)
        draw.text((x1 + 4, label_y), label_text, fill=(255, 255, 255), font=font)

    return result


if __name__ == '__main__':
    print(f"Using device: {device}")
    print("Loading model...")
    get_model()
    print("✅ Model loaded!")

    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        raise SystemExit(1)

    result, detections = detect(image_path)

    plt.figure(figsize=(14, 8))
    plt.imshow(result)
    plt.axis('off')
    if detections:
        summary = ", ".join(f"{d[4]}({d[5]*100:.0f}%)" for d in detections)
        plt.title(f"Detections ({len(detections)}): {summary}", fontsize=12, fontweight='bold')
    else:
        plt.title("No detections found", fontsize=14)
    plt.tight_layout()
    plt.show()
    print("\n✅ Image displayed!")
