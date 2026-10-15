"""
Integration test: loads the real DenseNet-121 and runs inference on
ONE real NIH ChestX-ray14 image.

This test is heavy. It is skipped automatically if the NIH dataset is
not present on disk.

Banner meanings:
- "REAL CHEST-X-RAY INFERENCE": a real NIH image was used. Model scores
  are still NOT clinical probabilities.
- "PIPELINE SMOKE TEST — NOT CLINICALLY MEANINGFUL": the model ran, but
  on a non-clinical placeholder image.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from backend.core.config import NIH_IMAGES_DIR, NIH_TRAIN_VAL_LIST


def _first_nih_image() -> Path | None:
    """Return the first image listed in train_val_list.txt, if present."""
    if not NIH_TRAIN_VAL_LIST.exists() or not NIH_IMAGES_DIR.exists():
        return None
    for line in NIH_TRAIN_VAL_LIST.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        candidate = NIH_IMAGES_DIR / name
        if candidate.exists():
            return candidate
    return None


@pytest.mark.integration
def test_vision_real_inference_on_one_nih_image() -> None:
    image = _first_nih_image()
    if image is None:
        pytest.skip(
            "REAL MODEL TEST: NOT RUN — image/data unavailable "
            f"(checked {NIH_TRAIN_VAL_LIST} and {NIH_IMAGES_DIR})"
        )

    # Import here so unit tests don't pay the cost of importing torchxrayvision.
    from backend.vision.model import VisionModel

    t0 = time.perf_counter()
    model = VisionModel()
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = model.predict(image)
    t_infer = time.perf_counter() - t1

    print()
    print("=" * 68)
    print("REAL CHEST-X-RAY INFERENCE — model scores only, NOT clinical")
    print("=" * 68)
    print(f"Image:               {image}")
    print(f"Checkpoint:          {result.checkpoint}")
    print(f"Model load time:     {t_load:.2f} s")
    print(f"Inference time:      {t_infer:.2f} s")
    print(f"Calibration status:  {result.calibration_status}")
    print()
    print("14 NIH finding scores (raw sigmoid; NOT probabilities):")
    for f in result.findings:
        print(f"  {f.name:<20s}  {f.model_score:0.4f}   [{f.status}]")
    print("=" * 68)

    assert len(result.findings) == 14
    assert all(0.0 <= f.model_score <= 1.0 for f in result.findings)