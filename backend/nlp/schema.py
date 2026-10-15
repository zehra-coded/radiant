"""
Report NLP schema.

ReportFinding represents one finding extracted from a free-text report.
ReportExtraction is the complete output for one report.

The ``uncertainty`` field is a deterministic heuristic score (0.0-1.0),
NOT a calibrated probability. See rule 3.14.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FindingStatus = Literal["present", "negated", "uncertain"]

Anatomy = Literal[
    "left",
    "right",
    "bilateral",
    "upper",
    "lower",
    "basilar",
    "apical",
    "diffuse",
    "focal",
    "unspecified",
]


class ReportFinding(BaseModel):
    """One finding extracted from a report."""

    name: str = Field(..., description="One of the 14 NIH labels.")
    anatomy: Anatomy = Field(
        default="unspecified",
        description="Anatomical modifier, if any. 'unspecified' if not stated.",
    )
    status: FindingStatus = Field(
        ...,
        description="present / negated / uncertain, per deterministic cues.",
    )
    uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic heuristic score, NOT a calibrated probability. "
            "1.0 = explicit uncertainty cue ('possible', 'cannot exclude'). "
            "0.5 = borderline or ambiguous phrasing. "
            "0.0 = explicit present or negated statement."
        ),
    )
    evidence_span: str = Field(
        ...,
        description="The substring of the report the finding was extracted from.",
    )
    spatial_info: str | None = Field(
        default=None,
        description="Any spatial term captured (e.g. 'left lower', 'bilateral').",
    )


class ReportExtraction(BaseModel):
    """Complete NLP extraction for a single report."""

    findings: list[ReportFinding] = Field(
        default_factory=list,
        description="Extracted findings, deduplicated by (name, anatomy, status).",
    )
    raw_text: str = Field(..., description="The original report text.")
    notes: list[str] = Field(
        default_factory=list,
        description="Non-fatal notes (e.g. unparsed sentences, warnings).",
    )