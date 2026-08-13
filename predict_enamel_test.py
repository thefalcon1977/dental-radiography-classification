"""
Classify enamel test images and write predictions CSV.

Output format mirrors external_test_predictions.csv:
  file,class_name,target,target_label,pred_idx,pred_label,prob_positive

For this 3-class model (dentin/enamel/pulp), prob_positive is P(enamel).
Results are written to test_predictions/.
"""

from __future__ import annotations

import csv
import os
from typing import TypeAlias

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "slm/resolution_best_densenet_model.pth"
INPUT_DIR = "image-testing/enamel_test"
OUTPUT_DIR = "test_predictions"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "enamel_test_predictions.csv")

CLASS_NAMES: list[str] = ["dentin", "enamel", "pulp"]
TARGET_CLASS = "enamel"
TARGET_IDX = CLASS_NAMES.index(TARGET_CLASS)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

ProbList: TypeAlias = list[float]
RowDict: TypeAlias = dict[str, str | int | float]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model(model_path: str) -> nn.Module:
    """Load DenseNet121 checkpoint for 3-class tissue classification.

    Args:
        model_path: Path to model checkpoint (.pth).

    Returns:
        Model in eval mode on the selected device.
    """
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(CLASS_NAMES))

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def list_images(input_dir: str) -> list[str]:
    """List image filenames in a directory (sorted).

    Args:
        input_dir: Directory containing test images.

    Returns:
        Sorted list of image filenames.
    """
    files: list[str] = []
    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(name)
    return sorted(files, key=str.lower)


def predict_image(model: nn.Module, image_path: str) -> tuple[int, ProbList]:
    """Run classification on a single image.

    Args:
        model: Trained DenseNet model.
        image_path: Path to the image file.

    Returns:
        Tuple of (predicted class index, class probability list).
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    pred_idx = int(torch.argmax(probs).item())
    prob_list: ProbList = [float(p) for p in probs.cpu().tolist()]
    return pred_idx, prob_list


def build_row(
    relative_path: str,
    pred_idx: int,
    probs: ProbList,
) -> RowDict:
    """Build one CSV row matching external_test_predictions.csv columns.

    Args:
        relative_path: Relative file path written to the CSV.
        pred_idx: Predicted class index.
        probs: Softmax probabilities for all classes.

    Returns:
        Row dictionary ready for csv.DictWriter.
    """
    return {
        "file": relative_path,
        "class_name": TARGET_CLASS,
        "target": TARGET_IDX,
        "target_label": TARGET_CLASS,
        "pred_idx": pred_idx,
        "pred_label": CLASS_NAMES[pred_idx],
        "prob_positive": probs[TARGET_IDX],
    }


def main() -> None:
    """Classify enamel test images and write predictions CSV."""
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(INPUT_DIR):
        raise SystemExit(f"Input directory not found: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    image_names = list_images(INPUT_DIR)
    if not image_names:
        raise SystemExit(f"No images found in {INPUT_DIR}")

    print(f"Found {len(image_names)} images in {INPUT_DIR}")
    rows: list[RowDict] = []
    correct = 0

    for name in image_names:
        image_path = os.path.join(INPUT_DIR, name)
        relative_path = os.path.join(INPUT_DIR, name).replace("\\", "/")
        pred_idx, probs = predict_image(model, image_path)
        row = build_row(relative_path, pred_idx, probs)
        rows.append(row)

        is_correct = pred_idx == TARGET_IDX
        correct += int(is_correct)
        mark = "OK" if is_correct else "MISS"
        print(
            f"[{mark}] {name:20s}  "
            f"pred={CLASS_NAMES[pred_idx]:6s}  "
            f"P(enamel)={probs[TARGET_IDX]:.4f}"
        )

    fieldnames = [
        "file",
        "class_name",
        "target",
        "target_label",
        "pred_idx",
        "pred_label",
        "prob_positive",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accuracy = 100.0 * correct / len(rows)
    print(f"\nWrote {len(rows)} rows -> {OUTPUT_CSV}")
    print(f"Accuracy on enamel_test: {correct}/{len(rows)} ({accuracy:.1f}%)")
    print(
        "Note: prob_positive = P(enamel) for this 3-class model "
        "(dentin / enamel / pulp)."
    )


if __name__ == "__main__":
    main()
