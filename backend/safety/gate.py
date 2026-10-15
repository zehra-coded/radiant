"""
Safety gate.

Two functions:

- pre_generation_check: runs before any LLM or template generation.
- post_generation_check: runs after generation to validate the output.

Fail closed. When in doubt, block, require human review, and emit the
required disclaimer.
"""
from __future__ import annotations

import re

from backend.core.logging import get_logger
from backend.fusion.schema import FusionResult
from backend.rag.schema import Chunk
from backend.safety.schema import (
    DISCLAIMER,
    REASON_CITATION_VALIDATION_FAILURE,
    REASON_DIAGNOSIS_REQUEST_OUT_OF_SCOPE,
    REASON_GROUNDING_VALIDATION_FAILURE,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_LOW_CONFIDENCE_FINDING,
    REASON_MALFORMED_INPUT,
    REASON_MODEL_FAILURE,
    REASON_PRIVACY_VIOLATION,
    REASON_RETRIEVAL_FAILURE,
    REASON_TREATMENT_REQUEST_OUT_OF_SCOPE,
    REASON_URGENT_OR_EMERGENCY,
    REASON_VISION_REPORT_CONFLICT,
    SafetyResult,
)
from backend.safety.schema import SafetyResult  # noqa: F811
from backend.triage.schema import TriageResult

logger = get_logger(__name__)

# Simple out-of-scope detectors. These are heuristics, not classifiers.
_DIAGNOSIS_PATTERNS = re.compile(
    r"\b(diagnose|diagnosis|what disease|do i have|cancer|malignan|tumor)\b",
    re.IGNORECASE,
)
_TREATMENT_PATTERNS = re.compile(
    r"\b(treat|treatment|therapy|medication|prescribe|dose|dosage|drug)\b",
    re.IGNORECASE,
)

# Citation format: [source:doc_id:chunk_id]
_CITATION_RE = re.compile(r"\[source:([^\]:]+):([^\]]+)\]")


def _disclaimer_present(text: str) -> bool:
    return "research prototype" in text.lower() and "not a diagnostic" in text.lower()


def _ensure_disclaimer(text: str) -> str:
    if _disclaimer_present(text):
        return text
    return f"{text.rstrip()}\n\n{DISCLAIMER}"


def pre_generation_check(
    user_query: str,
    triage: TriageResult | None,
    fusion: FusionResult | None,
    evidence: list[Chunk],
    *,
    vision_failed: bool = False,
    report_failed: bool = False,
    retrieval_failed: bool = False,
) -> SafetyResult:
    """Runs BEFORE generation.

    Returns a SafetyResult. If `allowed` is False, generation must not
    proceed; use `user_safe_message` instead.
    """
    reasons: list[str] = []

    if not user_query or not user_query.strip():
        reasons.append(REASON_MALFORMED_INPUT)

    if vision_failed or report_failed:
        reasons.append(REASON_MODEL_FAILURE)

    if retrieval_failed:
        reasons.append(REASON_RETRIEVAL_FAILURE)

    if fusion is not None and fusion.conflicts:
        reasons.append(REASON_VISION_REPORT_CONFLICT)

    if triage is not None and triage.priority in ("URGENT_REVIEW", "EMERGENCY_REVIEW"):
        reasons.append(REASON_URGENT_OR_EMERGENCY)

    if triage is not None and "low_confidence" in triage.source_signals:
        reasons.append(REASON_LOW_CONFIDENCE_FINDING)

    if _DIAGNOSIS_PATTERNS.search(user_query or ""):
        reasons.append(REASON_DIAGNOSIS_REQUEST_OUT_OF_SCOPE)

    if _TREATMENT_PATTERNS.search(user_query or ""):
        reasons.append(REASON_TREATMENT_REQUEST_OUT_OF_SCOPE)

    if not evidence:
        reasons.append(REASON_INSUFFICIENT_EVIDENCE)

    # Hard-block conditions. Everything else can still be generated but
    # will require human review.
    hard_blocks = {
        REASON_DIAGNOSIS_REQUEST_OUT_OF_SCOPE,
        REASON_TREATMENT_REQUEST_OUT_OF_SCOPE,
        REASON_MALFORMED_INPUT,
        REASON_PRIVACY_VIOLATION,
    }
    blocked = bool(set(reasons) & hard_blocks)

    requires_review = bool(reasons)

    if blocked:
        message = (
            "This request is outside the scope of this research prototype. "
            "RADIANT does not provide diagnoses, treatment recommendations, "
            "or patient-specific medical advice. "
        )
    elif requires_review:
        message = (
            "Generation may proceed, but the output will be marked for "
            "human review because one or more safety signals are present."
        )
    else:
        message = "Generation may proceed."

    return SafetyResult(
        allowed=not blocked,
        requires_human_review=requires_review,
        blocked_or_redirected=blocked,
        reason_codes=sorted(set(reasons)),
        user_safe_message=_ensure_disclaimer(message),
    )


def post_generation_check(
    generated_text: str,
    triage: TriageResult | None,
    fusion: FusionResult | None,
    evidence: list[Chunk],
    generator: str,
) -> SafetyResult:
    """Runs AFTER generation.

    Validates:
    - every [source:doc_id:chunk_id] citation resolves to a retrieved chunk
    - no diagnosis or treatment language is present
    - the disclaimer is present (or appended)
    """
    reasons: list[str] = []

    if generator not in ("template", "llm", "amd_token_factory"):
        reasons.append(REASON_GROUNDING_VALIDATION_FAILURE)

    # 1. Citation validation.
    valid_ids = {(c.document_id, c.chunk_id) for c in evidence}
    found = _CITATION_RE.findall(generated_text)
    bad: list[str] = []
    for doc_id, chunk_id in found:
        if (doc_id.strip(), chunk_id.strip()) not in valid_ids:
            bad.append(f"{doc_id}:{chunk_id}")
    if bad:
        reasons.append(REASON_CITATION_VALIDATION_FAILURE)
        logger.warning("fabricated citations: %s", bad)

    # 2. Out-of-scope language check.
    if _DIAGNOSIS_PATTERNS.search(generated_text):
        reasons.append(REASON_GROUNDING_VALIDATION_FAILURE)
    if _TREATMENT_PATTERNS.search(generated_text):
        reasons.append(REASON_GROUNDING_VALIDATION_FAILURE)

    # 3. Evidence support check: if generation claims findings but evidence
    # is empty, flag as unsupported.
    if not evidence and _has_finding_claims(generated_text):
        reasons.append(REASON_INSUFFICIENT_EVIDENCE)

    requires_review = bool(reasons)
    blocked = bool(
        set(reasons)
        & {
            REASON_CITATION_VALIDATION_FAILURE,
            REASON_GROUNDING_VALIDATION_FAILURE,
        }
    )

    if blocked:
        message = (
            "The generated explanation did not pass validation and has been "
            "withheld. A human reviewer must inspect the structured findings "
            "and retrieved evidence directly."
        )
    elif requires_review:
        message = (
            "The generated explanation passed basic validation but is marked "
            "for human review."
        )
    else:
        message = "The generated explanation passed all validations."

    return SafetyResult(
        allowed=not blocked,
        requires_human_review=requires_review or _triage_requires_review(triage),
        blocked_or_redirected=blocked,
        reason_codes=sorted(set(reasons)),
        user_safe_message=_ensure_disclaimer(message),
    )


def _has_finding_claims(text: str) -> bool:
    keywords = ("finding", "consistent with", "suggest", "present", "identified")
    lower = text.lower()
    return any(k in lower for k in keywords)


def _triage_requires_review(triage: TriageResult | None) -> bool:
    if triage is None:
        return False
    return triage.priority in ("URGENT_REVIEW", "EMERGENCY_REVIEW")