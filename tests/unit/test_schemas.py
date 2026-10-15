"""Schema composition and JSON round-trip tests."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from backend.schemas.result import (
    EvidenceItem,
    ExplanationBlock,
    HumanReviewBlock,
    RadiantResult,
    UncertaintySignalsModel,
)
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult


def _minimal_vision() -> VisionResult:
    return VisionResult(
        findings=[
            VisionFinding(name=n, model_score=0.5, status="uncertain")
            for n in NIH_LABELS
        ],
        checkpoint="test",
    )


def test_minimal_result_serializes() -> None:
    r = RadiantResult(
        request_id="req-test",
        vision=_minimal_vision(),
        generator="none",
    )
    j = r.model_dump_json()
    back = RadiantResult.model_validate_json(j)
    assert back.request_id == "req-test"
    assert back.vision is not None
    assert len(back.vision.findings) == 14


def test_evidence_item_from_chunk() -> None:
    from backend.rag.schema import Chunk

    c = Chunk(
        document_id="d1",
        chunk_id="d1::000",
        text="sample text",
        title="Sample",
        source="RADIANT",
        url=None,
        license="CC0",
        provenance="DEVELOPMENT_FIXTURE",
    )
    e = EvidenceItem.from_chunk(c)
    assert e.document_id == "d1"
    assert e.provenance == "DEVELOPMENT_FIXTURE"


def test_uncertainty_signals_from_dict() -> None:
    m = UncertaintySignalsModel.from_dict(
        {"low_confidence": ["a", "b"], "vision_report_conflict": ["c"]}
    )
    names = [s.signal for s in m.signals]
    assert names.count("low_confidence") == 2
    assert names.count("vision_report_conflict") == 1


def test_radiant_result_full_round_trip() -> None:
    r = RadiantResult(
        request_id="req-1",
        vision=_minimal_vision(),
        generator="template",
        explanation=ExplanationBlock(
            text="Some explanation text.",
            generator="template",
            fallback_reason=None,
        ),
        human_review=HumanReviewBlock(required=True, reasons=["test"]),
    )
    j = r.model_dump_json()
    back = RadiantResult.model_validate_json(j)
    assert back.explanation is not None
    assert back.explanation.text == "Some explanation text."
    assert back.human_review.required is True


def test_no_module_schema_redefined() -> None:
    """Every schema that a module defines should be importable from
    that module and be the SAME object as what schemas.result re-exports.
    """
    from backend.fusion import schema as fusion_schema
    from backend.nlp import schema as nlp_schema
    from backend.rag import schema as rag_schema
    from backend.safety import schema as safety_schema
    from backend.triage import schema as triage_schema
    from backend.vision import schema as vision_schema

    from backend.schemas import result as schemas_result

    # Sample of identity checks: the re-exported class must be the same
    # object as the one defined in the owning module.
    assert schemas_result.VisionResult is vision_schema.VisionResult
    assert schemas_result.VisionFinding is vision_schema.VisionFinding
    assert schemas_result.ReportExtraction is nlp_schema.ReportExtraction
    assert schemas_result.FusionResult is fusion_schema.FusionResult
    assert schemas_result.TriageResult is triage_schema.TriageResult
    assert schemas_result.Chunk is rag_schema.Chunk
    assert schemas_result.SafetyResult is safety_schema.SafetyResult


def test_radiant_result_explanation_cannot_alter_structured_fields() -> None:
    """Changing the explanation text does not affect any other field."""
    r1 = RadiantResult(
        request_id="req-1",
        vision=_minimal_vision(),
        generator="amd_token_factory",
        explanation=ExplanationBlock(text="A", generator="amd_token_factory"),
    )
    r2 = RadiantResult(
        request_id="req-1",
        vision=_minimal_vision(),
        generator="amd_token_factory",
        explanation=ExplanationBlock(text="B", generator="amd_token_factory"),
    )
    # Only the explanation.text differs.
    d1 = r1.model_dump()
    d2 = r2.model_dump()
    d1["explanation"].pop("text")
    d2["explanation"].pop("text")
    assert d1 == d2


def test_vision_finding_out_of_range_rejected() -> None:
    with pytest.raises(Exception):
        VisionFinding(name="Atelectasis", model_score=2.0, status="present")