"""Project-wide constants. Class index order must not change."""

from __future__ import annotations

CLASS_NAMES: list[str] = ["dentin", "enamel", "pulp"]  # 0, 1, 2

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "dentin": (255, 0, 0),  # red
    "enamel": (0, 255, 0),  # green
    "pulp": (0, 0, 255),  # blue
}

DEFAULT_MODEL_PATH = "slm/resolution_best_densenet_model.pth"

IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

PREDICTIONS_DIR = "test_predictions"
