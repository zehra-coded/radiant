"""
Threshold classification for vision model scores.

All thresholds are UNVALIDATED PROTOTYPE THRESHOLDS. They are not
clinical cutoffs. Changing them changes prototype behavior, not
diagnostic accuracy.
"""
from __future__ import annotations

from backend.core.config import (
    VISION_PRESENT_THRESHOLD,
    VISION_UNCERTAIN_HIGH,
    VISION_UNCERTAIN_LOW,
)
from backend.vision.schema import VisionStatus


def classify(score: float) -> VisionStatus:
    """Classify a model score as present / uncertain / absent.

    UNVALIDATED PROTOTYPE HEURISTIC.

    - score >= VISION_PRESENT_THRESHOLD            -> present
    - VISION_UNCERTAIN_LOW <= score < VISION_PRESENT_THRESHOLD -> uncertain
    - score < VISION_UNCERTAIN_LOW                 -> absent
    """
    if score >= VISION_PRESENT_THRESHOLD:
        return "present"
    if score >= VISION_UNCERTAIN_LOW:
        return "uncertain"
    return "absent"


def describe_thresholds() -> dict[str, float]:
    """Return the currently configured thresholds. For docs and tests."""
    return {
        "present": VISION_PRESENT_THRESHOLD,
        "uncertain_low": VISION_UNCERTAIN_LOW,
        "uncertain_high": VISION_UNCERTAIN_HIGH,
        "note": "UNVALIDATED PROTOTYPE THRESHOLDS",
    }