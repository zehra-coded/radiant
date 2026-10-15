"""
Unit tests for the vision module.

These tests NEVER load the real DenseNet model. They use mocks and
tiny fixtures. The real model is tested only in
tests/integration/test_vision_real.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from backend.core.config import VISION_INPUT_SIZE
from backend.vision.preprocess import (
    EmptyImageError,
    ImageNotFoundError,
    UnsupportedImageError,
    load_image,
    preprocess,
    validate_image_path,
)
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult
from backend.vision.thresholds import classify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_png(tmp_path: Path) -> Path:
    p = tmp_path / "valid.png"
    Image.new("L", (224, 224), color=128).save(p, format="PNG")
    return p


@pytest.fixture()
def wrong_type_file(tmp_path: Path) -> Path:
    p = tmp_path / "not_an_image.txt"
    p.write_text("this is not an image", encoding="utf-8")
    return p


@pytest.fixture()
def corrupt_png(tmp_path: Path) -> Path:
    p = tmp_path / "corrupt.png"
    # A .png extension but garbage bytes.
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage-not-a-real-png")
    return p


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


def test_validate_missing_file_raises() -> None:
    with pytest.raises(ImageNotFoundError):
        validate_image_path("this/path/does/not/exist.png")


def test_validate_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(EmptyImageError):
        validate_image_path(p)


def test_load_valid_image_returns_pil(valid_png: Path) -> None:
    img = load_image(valid_png)
    assert img.mode == "L"
    assert img.size == (224, 224)


def test_load_wrong_type_raises(wrong_type_file: Path) -> None:
    with pytest.raises(UnsupportedImageError):
        load_image(wrong_type_file)


def test_load_corrupt_png_raises(corrupt_png: Path) -> None:
    with pytest.raises(UnsupportedImageError):
        load_image(corrupt_png)


def test_preprocess_returns_correct_tensor_shape(valid_png: Path) -> None:
    tensor = preprocess(valid_png)
    assert tensor.shape == (1, 1, VISION_INPUT_SIZE, VISION_INPUT_SIZE)
    assert tensor.dtype.is_floating_point


def test_preprocess_values_in_model_range(valid_png: Path) -> None:
    """Model input must be in [-1024, 1024] to match torchxrayvision's
    training pipeline. See docs/models.md for details.
    """
    tensor = preprocess(valid_png)
    assert float(tensor.min()) >= -1024.0
    assert float(tensor.max()) <= 1024.0


def test_preprocess_mid_gray_maps_to_zero(tmp_path: Path) -> None:
    """Pixel value 128 (mid-gray) should map to approximately 0.0:
    (2 * (128/255) - 1) * 1024 ~= 4.0.
    """
    p = tmp_path / "midgray.png"
    Image.new("L", (224, 224), color=128).save(p, format="PNG")
    tensor = preprocess(p)
    mid_value = float(tensor.mean())
    assert -10.0 <= mid_value <= 10.0


def test_preprocess_black_maps_to_minus_1024(tmp_path: Path) -> None:
    """Pixel value 0 should map to exactly -1024.0."""
    p = tmp_path / "black.png"
    Image.new("L", (224, 224), color=0).save(p, format="PNG")
    tensor = preprocess(p)
    assert float(tensor.min()) == -1024.0
    assert float(tensor.max()) == -1024.0


def test_preprocess_white_maps_to_plus_1024(tmp_path: Path) -> None:
    """Pixel value 255 should map to exactly +1024.0."""
    p = tmp_path / "white.png"
    Image.new("L", (224, 224), color=255).save(p, format="PNG")
    tensor = preprocess(p)
    assert float(tensor.min()) == 1024.0
    assert float(tensor.max()) == 1024.0


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.95, "present"),
        (0.50, "present"),
        (0.49, "uncertain"),
        (0.30, "uncertain"),
        (0.29, "absent"),
        (0.00, "absent"),
    ],
)
def test_classify_thresholds(score: float, expected: str) -> None:
    assert classify(score) == expected


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_nih_labels_has_14_entries() -> None:
    assert len(NIH_LABELS) == 14
    assert len(set(NIH_LABELS)) == 14


def test_vision_finding_accepts_valid_data() -> None:
    f = VisionFinding(name="Effusion", model_score=0.42, status="uncertain")
    assert f.name == "Effusion"
    assert f.model_score == 0.42
    assert f.status == "uncertain"


def test_vision_finding_rejects_score_out_of_range() -> None:
    with pytest.raises(Exception):
        VisionFinding(name="Effusion", model_score=1.5, status="present")


def test_vision_result_contains_14_findings_in_order() -> None:
    findings = [
        VisionFinding(name=n, model_score=0.1, status="absent") for n in NIH_LABELS
    ]
    result = VisionResult(findings=findings, checkpoint="densenet121-res224-nih")
    assert len(result.findings) == 14
    assert [f.name for f in result.findings] == list(NIH_LABELS)
    assert result.calibration_status.startswith("NOT CALIBRATED")