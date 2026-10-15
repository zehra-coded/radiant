"""Unit tests for the LLM reporter. Mocked client, no network."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.fusion.schema import FusionConflict, FusionResult
from backend.llm import amd_client, payload as payload_builder
from backend.llm.reporter import generate_explanation
from backend.nlp.schema import ReportExtraction
from backend.rag.schema import Chunk
from backend.safety.schema import DISCLAIMER
from backend.triage.schema import TriageResult
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult


def _chunk() -> Chunk:
    return Chunk(
        document_id="fixture_effusion",
        chunk_id="fixture_effusion::000",
        text="pleural effusion fixture text",
        title="Effusion",
        source="RADIANT",
        url=None,
        license="CC0",
        provenance="DEVELOPMENT_FIXTURE",
    )


def _vision() -> VisionResult:
    return VisionResult(
        findings=[
            VisionFinding(name="Effusion", model_score=0.75, status="present"),
            *[
                VisionFinding(name=n, model_score=0.05, status="absent")
                for n in NIH_LABELS
                if n != "Effusion"
            ],
        ],
        checkpoint="test",
    )


def _report() -> ReportExtraction:
    return ReportExtraction(raw_text="No effusion.", findings=[])


def _fusion() -> FusionResult:
    return FusionResult(
        agreed=[],
        vision_only=[],
        report_only=[],
        conflicts=[
            FusionConflict(
                name="Effusion",
                vision_status="present",
                report_status="negated",
                reason="disagreement",
            )
        ],
        overall_consistency="conflicting",
        agreement_ratio=0.0,
    )


def _triage() -> TriageResult:
    return TriageResult(
        priority="URGENT_REVIEW",
        reasons=["conflict"],
        source_signals=["vision_report_conflict"],
        decision_trace=["rule_vision_report_conflict: FIRED"],
        notes=[],
    )


def test_payload_contains_no_image_fields() -> None:
    p = payload_builder.build(_vision(), _report(), _fusion(), _triage(), [_chunk()])
    # Absolutely no image bytes/path/embedding keys anywhere.
    for forbidden in ("image", "image_path", "image_bytes", "embedding", "pixels"):
        assert forbidden not in p
        # Also ensure no nested key matches.
        def _walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    assert forbidden not in k.lower(), f"nested key {k}"
                    _walk(v)
            elif isinstance(o, list):
                for x in o:
                    _walk(x)
        _walk(p)


def test_payload_strips_non_allowlisted() -> None:
    p = payload_builder.build(_vision(), _report(), _fusion(), _triage(), [_chunk()])
    assert "disclaimer" in p
    assert "vision_scores" in p
    # Something not on the allowlist must not appear.
    assert "internal_debug" not in p


@patch("backend.llm.amd_client.is_configured", return_value=False)
def test_template_path_when_unconfigured(mock_cfg) -> None:
    out = generate_explanation(
        "summarize",
        _vision(), _report(), _fusion(), _triage(), [_chunk()],
    )
    assert out.generator == "template"
    assert out.fallback_reason and "not configured" in out.fallback_reason.lower()
    assert DISCLAIMER in out.text


@patch("backend.llm.amd_client.is_configured", return_value=True)
@patch("backend.llm.amd_client.call", side_effect=amd_client.LlmCallFailed("boom"))
def test_template_fallback_on_llm_failure(mock_call, mock_cfg) -> None:
    out = generate_explanation(
        "summarize",
        _vision(), _report(), _fusion(), _triage(), [_chunk()],
    )
    assert out.generator == "template"
    assert "failed" in (out.fallback_reason or "").lower()


@patch("backend.llm.amd_client.is_configured", return_value=True)
@patch("backend.llm.amd_client.call", return_value="See [source:fake:999].")
def test_invalid_citation_falls_back_to_template(mock_call, mock_cfg) -> None:
    out = generate_explanation(
        "summarize",
        _vision(), _report(), _fusion(), _triage(), [_chunk()],
    )
    assert out.generator == "template"
    assert "post_generation_check" in (out.fallback_reason or "")


@patch("backend.llm.amd_client.is_configured", return_value=True)
@patch(
    "backend.llm.amd_client.call",
    return_value="Findings: see [source:fixture_effusion:fixture_effusion::000].",
)
def test_valid_citation_passes(mock_call, mock_cfg) -> None:
    out = generate_explanation(
        "summarize",
        _vision(), _report(), _fusion(), _triage(), [_chunk()],
    )
    assert out.generator == "amd_token_factory"


@patch("backend.llm.amd_client.is_configured", return_value=True)
@patch(
    "backend.llm.amd_client.call",
    return_value="Here is my medical diagnosis for this patient.",
)
def test_diagnosis_language_in_llm_output_blocks(mock_call, mock_cfg) -> None:
    out = generate_explanation(
        "summarize",
        _vision(), _report(), _fusion(), _triage(), [_chunk()],
    )
    assert out.generator == "template"
    assert out.fallback_reason