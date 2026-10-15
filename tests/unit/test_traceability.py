"""Traceability tests.

Confirms that the explanation text is not load-bearing: no structured
field's value can depend on it.
"""
from __future__ import annotations

from backend.schemas.result import (
    ExplanationBlock,
    HumanReviewBlock,
    RadiantResult,
)
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult


def _vision() -> VisionResult:
    return VisionResult(
        findings=[
            VisionFinding(name=n, model_score=0.42, status="uncertain")
            for n in NIH_LABELS
        ],
        checkpoint="test",
    )


def test_explanation_text_is_isolated() -> None:
    """Mutating the text field does not touch any other field."""
    r = RadiantResult(
        request_id="req",
        vision=_vision(),
        human_review=HumanReviewBlock(required=True, reasons=["conflict"]),
        generator="template",
        explanation=ExplanationBlock(text="initial", generator="template"),
    )
    original = r.model_dump()
    r.explanation.text = "changed to something entirely different"
    after = r.model_dump()
    original["explanation"]["text"] = after["explanation"]["text"]
    assert original == after


def test_top_level_generator_matches_explanation_generator() -> None:
    """If explanation is present, the top-level generator must agree."""
    r = RadiantResult(
        request_id="req",
        vision=_vision(),
        generator="amd_token_factory",
        explanation=ExplanationBlock(text="x", generator="amd_token_factory"),
    )
    assert r.generator == r.explanation.generator


def test_human_review_block_is_independent() -> None:
    r = RadiantResult(
        request_id="req",
        vision=_vision(),
        human_review=HumanReviewBlock(required=True, reasons=["a", "b"]),
        generator="template",
        explanation=ExplanationBlock(text="does not matter", generator="template"),
    )
    r.explanation.text = "changed"
    assert r.human_review.required is True
    assert r.human_review.reasons == ["a", "b"]