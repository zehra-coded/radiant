"""
Fusion module schema.

FusionResult is the output of the cross-modal verifier. It compares
the vision model's findings against the report's findings for each of
the 14 NIH labels and classifies the overall consistency.

A conflict is a disagreement between two structured inputs. It is NOT
a medical error. Neither modality is treated as ground truth.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OverallConsistency = Literal["high", "partial", "conflicting", "insufficient"]


class FusionConflict(BaseModel):
    """A single-label disagreement between vision and report."""

    name: str = Field(..., description="One of the 14 NIH labels.")
    vision_status: str = Field(
        ..., description="Vision status for this label (present/uncertain/absent)."
    )
    report_status: str = Field(
        ...,
        description=(
            "Report status for this label. Either an explicit status "
            "(present/negated/uncertain) or 'not_mentioned' if the "
            "report does not reference this label."
        ),
    )
    reason: str = Field(
        ...,
        description="Human-readable description of why this is a conflict.",
    )


class FusionResult(BaseModel):
    """Cross-modal comparison of vision and report findings."""

    agreed: list[str] = Field(
        default_factory=list,
        description="Labels where vision and report agree (present+present or absent+negated).",
    )
    vision_only: list[str] = Field(
        default_factory=list,
        description="Labels the vision model marks present but the report does not mention.",
    )
    report_only: list[str] = Field(
        default_factory=list,
        description="Labels the report marks present but the vision model does not.",
    )
    conflicts: list[FusionConflict] = Field(
        default_factory=list,
        description="Labels where vision and report disagree (e.g. vision present + report negated).",
    )
    overall_consistency: OverallConsistency = Field(
        ...,
        description=(
            "high / partial / conflicting / insufficient. "
            "Derived from the ratio of agreed to total labels, "
            "with overrides for critical-label conflicts and "
            "empty / missing modalities."
        ),
    )
    agreement_ratio: float | None = Field(
        default=None,
        description=(
            "Computed ratio = agreed / (agreed + vision_only + report_only + conflicts). "
            "None if there were no labels to compare (both empty)."
        ),
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Non-fatal notes (e.g. missing modality, critical conflict override).",
    )