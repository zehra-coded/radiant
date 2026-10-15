"""
Image preprocessing and validation for the vision module.

Raises typed errors on invalid input. Never silently accepts a file
that isn't a readable image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from backend.core.config import VISION_INPUT_SIZE


class VisionInputError(Exception):
    """Base class for vision input errors."""


class ImageNotFoundError(VisionInputError):
    """The image path does not exist or is not a regular file."""


class UnsupportedImageError(VisionInputError):
    """The file is not a supported / readable image format."""


class EmptyImageError(VisionInputError):
    """The file is empty or has zero pixels."""


def validate_image_path(path: Path | str) -> Path:
    """Validate that *path* points at a readable, non-empty image file."""
    p = Path(path)
    if not p.exists():
        raise ImageNotFoundError(f"Image not found: {p}")
    if not p.is_file():
        raise ImageNotFoundError(f"Not a regular file: {p}")
    if p.stat().st_size == 0:
        raise EmptyImageError(f"Image file is empty: {p}")
    return p


def load_image(path: Path | str) -> Image.Image:
    """Open an image and convert to single-channel grayscale.

    Raises ``UnsupportedImageError`` on unreadable / corrupt files.
    Raises ``EmptyImageError`` if the image has zero pixels.
    """
    p = validate_image_path(path)
    try:
        img = Image.open(p)
        img.load()
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"Unreadable or unsupported image: {p}") from exc
    except OSError as exc:
        raise UnsupportedImageError(f"Failed to read image: {p}") from exc

    if img.size[0] == 0 or img.size[1] == 0:
        raise EmptyImageError(f"Image has zero pixels: {p}")

    return img.convert("L")


def to_model_tensor(img: Image.Image, size: int = VISION_INPUT_SIZE) -> torch.Tensor:
    """Convert a PIL image to the tensor torchxrayvision expects.

    The ``densenet121-res224-nih`` checkpoint was trained on images
    normalized to the range [-1024, 1024] (torchxrayvision's
    ``xrv.datasets.normalize`` with maxval=255). Feeding it values in
    [0, 1] produces meaningless scores near the 0.50 decision boundary.

    Output shape: (1, 1, size, size), dtype float32, values in [-1024, 1024].
    """
    resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)

    # torchxrayvision normalization: (2 * (pixel / 255) - 1) * 1024
    # Maps [0, 255] -> [-1024, 1024], matching the training pipeline.
    arr = (2.0 * (arr / 255.0) - 1.0) * 1024.0

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def preprocess(path: Path | str) -> torch.Tensor:
    """Full preprocessing pipeline for a single image path."""
    return to_model_tensor(load_image(path))