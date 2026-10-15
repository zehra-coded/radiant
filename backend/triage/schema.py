"""
Triage schema.

ReviewPriority is a routing suggestion within a research prototype.
It is NOT a clinical triage category.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReviewPriority = Literal["ROUTINE", "MONITOR", "URGENT_REVIEW", "EMERGENCY_REVIEW"]


class TriageResult(BaseModel):
    """Output of the triage engine."""

    priority: ReviewPriority = Field(
        ...,
        description="Prototype review-priority. NOT a clinical triage category.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons for the chosen priority, in order.",
    )
    source_signals: list[str] = Field(
        default_factory=list,
        description="The uncertainty signals that contributed to this priority.",
    )
    decision_trace: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of rule evaluations, for traceability. "
            "Each entry names the rule and whether it fired."
        ),
    )
    notes: list[str] = Field(default_factory=list)