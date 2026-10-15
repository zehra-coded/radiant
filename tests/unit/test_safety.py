"""Unit tests for the safety gate and privacy enforcement. No network."""
from __future__ import annotations

import pytest

from backend.fusion.schema import FusionConflict, FusionResult
from backend.rag.schema import Chunk
from backend.safety.gate import post_generation_check, pre_generation_check
from backend.safety.privacy import PrivacyViolation, enforce
from backend.safety.schema import (
    REASON_CITATION_VALIDATION_FAILURE,
    REASON_DIAGNOSIS_REQUEST_OUT_OF_SCOPE,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_MALFORMED_INPUT,
    REASON_MODEL_FAILURE,
    REASON_PRIVACY_VIOLATION,
    REASON_RETRIEVAL_FAILURE,
    REASON_TREATMENT_REQUEST_OUT_OF_SCOPE,
    REASON_URGENT_OR_EMERGENCY,
    REASON_VISION_REPORT_CONFLICT,
)
from backend.triage.schema import TriageResult


def _chunk(doc: str = "fixture_pneumothorax", chunk: str = "000") -> Chunk:
    return Chunk(
        document_id=doc,
        chunk_id=chunk,
        text="sample",
        title="t",
        source="src",
        url=None,
        license="CC0",
        provenance="DEVELOPMENT_FIXTURE",
    )


def _triage(priority: str, signals: list[str] | None = None) -> TriageResult:
    return TriageResult(
        priority=priority,  # type: ignore[arg-type]
        reasons=["test"],
        source_signals=signals or [],
        decision_trace=["test"],
        notes=[],
    )


def _fusion_with_conflict() -> FusionResult:
    return FusionResult(
        agreed=[],
        vision_only=[],
        report_only=[],
        conflicts=[
            FusionConflict(
                name="Effusion",
                vision_status="present",
                report_status="negated",
                reason="test",
            )
        ],
        overall_consistency="conflicting",
        agreement_ratio=0.0,
    )


# ----- pre_generation_check -----


def test_pre_generation_blocks_diagnosis_request() -> None:
    r = pre_generation_check(
        "Can you diagnose my patient's chest X-ray?",
        _triage("ROUTINE"),
        None,
        [_chunk()],
    )
    assert r.blocked_or_redirected
    assert REASON_DIAGNOSIS_REQUEST_OUT_OF_SCOPE in r.reason_codes
    assert "research prototype" in r.user_safe_message.lower()


def test_pre_generation_blocks_treatment_request() -> None:
    r = pre_generation_check(
        "What medication should I prescribe?",
        _triage("ROUTINE"),
        None,
        [_chunk()],
    )
    assert r.blocked_or_redirected
    assert REASON_TREATMENT_REQUEST_OUT_OF_SCOPE in r.reason_codes


def test_pre_generation_blocks_malformed_input() -> None:
    r = pre_generation_check("", _triage("ROUTINE"), None, [_chunk()])
    assert r.blocked_or_redirected
    assert REASON_MALFORMED_INPUT in r.reason_codes


def test_pre_generation_flags_urgent_triage() -> None:
    r = pre_generation_check(
        "summarize this case",
        _triage("URGENT_REVIEW"),
        None,
        [_chunk()],
    )
    assert r.requires_human_review
    assert REASON_URGENT_OR_EMERGENCY in r.reason_codes
    assert not r.blocked_or_redirected


def test_pre_generation_flags_conflict() -> None:
    r = pre_generation_check(
        "summarize",
        _triage("ROUTINE"),
        _fusion_with_conflict(),
        [_chunk()],
    )
    assert REASON_VISION_REPORT_CONFLICT in r.reason_codes


def test_pre_generation_flags_insufficient_evidence() -> None:
    r = pre_generation_check("summarize", _triage("ROUTINE"), None, [])
    assert REASON_INSUFFICIENT_EVIDENCE in r.reason_codes


def test_pre_generation_model_failure() -> None:
    r = pre_generation_check(
        "summarize", _triage("ROUTINE"), None, [_chunk()], vision_failed=True
    )
    assert REASON_MODEL_FAILURE in r.reason_codes
    assert r.requires_human_review


def test_pre_generation_retrieval_failure() -> None:
    r = pre_generation_check(
        "summarize",
        _triage("ROUTINE"),
        None,
        [],
        retrieval_failed=True,
    )
    assert REASON_RETRIEVAL_FAILURE in r.reason_codes


def test_pre_generation_disclaimer_always_present() -> None:
    r = pre_generation_check("summarize", _triage("ROUTINE"), None, [_chunk()])
    assert "not a diagnostic" in r.user_safe_message.lower()
    assert "licensed physician" in r.user_safe_message.lower()


# ----- post_generation_check -----


def test_post_generation_valid_citation() -> None:
    text = "See [source:fixture_pneumothorax:000] for detail."
    r = post_generation_check(
        text, _triage("ROUTINE"), None, [_chunk()], generator="template"
    )
    assert r.allowed
    assert REASON_CITATION_VALIDATION_FAILURE not in r.reason_codes


def test_post_generation_fabricated_citation() -> None:
    text = "See [source:fake_doc:999] for detail."
    r = post_generation_check(
        text, _triage("ROUTINE"), None, [_chunk()], generator="llm"
    )
    assert not r.allowed
    assert r.blocked_or_redirected
    assert REASON_CITATION_VALIDATION_FAILURE in r.reason_codes


def test_post_generation_diagnosis_language_flagged() -> None:
    text = "The diagnosis is pneumonia."
    r = post_generation_check(
        text, _triage("ROUTINE"), None, [_chunk()], generator="llm"
    )
    assert not r.allowed


def test_post_generation_unsupported_claim_flagged() -> None:
    text = "Finding: consolidation is present."
    r = post_generation_check(
        text, _triage("ROUTINE"), None, [], generator="llm"
    )
    assert REASON_INSUFFICIENT_EVIDENCE in r.reason_codes or not r.allowed


def test_post_generation_requires_review_for_urgent() -> None:
    text = "Case summary with no citations."
    r = post_generation_check(
        text, _triage("URGENT_REVIEW"), None, [_chunk()], generator="llm"
    )
    assert r.requires_human_review


# ----- privacy -----


def test_privacy_drops_non_allowlisted_field() -> None:
    out = enforce({"vision_scores": {"a": 1}, "notes_internal": "hi"})
    assert "vision_scores" in out
    assert "notes_internal" not in out


def test_privacy_blocks_denylisted_field() -> None:
    with pytest.raises(PrivacyViolation):
        enforce({"vision_scores": {}, "patient_name": "X"})


def test_privacy_blocks_nested_denylisted_field() -> None:
    with pytest.raises(PrivacyViolation):
        enforce({"vision_scores": {"nested": {"mrn": "12345"}}})