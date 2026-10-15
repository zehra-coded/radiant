"""
Safety gate schema.

Reason codes are strings, not enums, so they can be logged, displayed,
and serialized without ambiguity. The canonical set is defined in
REASON_CODES below.
"""
from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

# Canonical reason codes (per spec §Stage 9).
REASON_URGENT_OR_EMERGENCY: Final[str] = "urgent_or_emergency"
REASON_VISION_REPORT_CONFLICT: Final[str] = "vision_report_conflict"
REASON_LOW_CONFIDENCE_FINDING: Final[str] = "low_confidence_finding"
REASON_INSUFFICIENT_EVIDENCE: Final[str] = "insufficient_evidence"
REASON_DIAGNOSIS_REQUEST_OUT_OF_SCOPE: Final[str] = "diagnosis_request_out_of_scope"
REASON_TREATMENT_REQUEST_OUT_OF_SCOPE: Final[str] = "treatment_request_out_of_scope"
REASON_MALFORMED_INPUT: Final[str] = "malformed_input"
REASON_MODEL_FAILURE: Final[str] = "model_failure"
REASON_RETRIEVAL_FAILURE: Final[str] = "retrieval_failure"
REASON_GROUNDING_VALIDATION_FAILURE: Final[str] = "grounding_validation_failure"
REASON_CITATION_VALIDATION_FAILURE: Final[str] = "citation_validation_failure"
REASON_PRIVACY_VIOLATION: Final[str] = "privacy_violation_hard_fail"

REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        REASON_URGENT_OR_EMERGENCY,
        REASON_VISION_REPORT_CONFLICT,
        REASON_LOW_CONFIDENCE_FINDING,
        REASON_INSUFFICIENT_EVIDENCE,
        REASON_DIAGNOSIS_REQUEST_OUT_OF_SCOPE,
        REASON_TREATMENT_REQUEST_OUT_OF_SCOPE,
        REASON_MALFORMED_INPUT,
        REASON_MODEL_FAILURE,
        REASON_RETRIEVAL_FAILURE,
        REASON_GROUNDING_VALIDATION_FAILURE,
        REASON_CITATION_VALIDATION_FAILURE,
        REASON_PRIVACY_VIOLATION,
    }
)

DISCLAIMER: Final[str] = (
    "This system is a research prototype for clinical decision support "
    "only. It is not a diagnostic tool. All outputs must be reviewed by "
    "a licensed physician."
)


class SafetyResult(BaseModel):
    """Output of a safety gate check."""

    allowed: bool = Field(
        ..., description="Whether generation (or release) may proceed."
    )
    requires_human_review: bool = Field(
        ..., description="Whether the case must be routed to a human reviewer."
    )
    blocked_or_redirected: bool = Field(
        default=False,
        description="True if the request was blocked or redirected to a safe message.",
    )
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Reason codes (see REASON_CODES) explaining the decision.",
    )
    user_safe_message: str = Field(
        ...,
        description=(
            "Text safe to display to a user. Always includes the required "
            "disclaimer."
        ),
    )