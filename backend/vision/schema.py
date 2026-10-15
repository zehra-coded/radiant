"""
Vision module schema.

VisionResult carries the 14 NIH finding labels and their corresponding
model scores. Field names use ``model_score`` — never ``probability``
or ``confidence`` — per project rule 3.14 (raw sigmoid outputs are NOT
calibrated probabilities).
"""
from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, Field

# The 14 official NIH ChestX-ray14 finding labels, in a stable order.
NIH_LABELS: Final[tuple[str, ...]] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

VisionStatus = Literal["present", "uncertain", "absent"]


class VisionFinding(BaseModel):
    """One finding from the vision model."""

    name: str = Field(..., description="One of the 14 NIH labels.")
    model_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw sigmoid output. NOT a calibrated probability.",
    )
    status: VisionStatus = Field(
        ...,
        description="Prototype threshold classification. UNVALIDATED PROTOTYPE THRESHOLD.",
    )


class VisionResult(BaseModel):
    """Complete vision module output for a single image."""

    findings: list[VisionFinding] = Field(
        ..., description="Exactly 14 findings, one per NIH label, in NIH_LABELS order."
    )
    checkpoint: str = Field(..., description="Model checkpoint name used.")
    calibration_status: str = Field(
        default="NOT CALIBRATED — raw sigmoid output",
        description="Fixed string; never claim calibration.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Non-fatal notes (e.g. label mapping warnings).",
    )