"""
Consolidated RADIANT output schema.

This module composes the schemas that each module owns. It does NOT
redefine any of them.

- VisionResult, VisionFinding       — backend/vision/schema.py
- ReportExtraction, ReportFinding   — backend/nlp/schema.py
- FusionResult, FusionConflict      — backend/fusion/schema.py
- TriageResult                      — backend/triage/schema.py
- Chunk, RetrievalResult            — backend/rag/schema.py
- SafetyResult                      — backend/safety/schema.py
- ReporterOutput (dataclass)        — backend/llm/reporter.py

The models defined HERE are only the ones that do not already exist:

- UncertaintySignal
- UncertaintySignalsModel
- EvidenceItem
- FinalFinding
- RadiantResult

Traceability rule: every field in RadiantResult except `explanation.text`
is produced by deterministic pipeline code. The explanation text is a
separate string field that cannot alter any other field — there is no
mechanism for it to do so.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Re-exports so downstream code can import everything from one place.
from backend.fusion.schema import FusionConflict, FusionResult  # noqa: F401
from backend.nlp.schema import ReportExtraction, ReportFinding  # noqa: F401
from backend.rag.schema import Chunk  # noqa: F401
from backend.safety.schema import DISCLAIMER, SafetyResult  # noqa: F401
from backend.triage.schema import ReviewPriority, TriageResult  # noqa: F401
from backend.vision.schema import VisionFinding, VisionResult  # noqa: F401


class UncertaintySignal(BaseModel):
    """One uncertainty signal with its reason."""

    signal: str = Field(..., description="Signal name, e.g. low_confidence.")
    reason: str = Field(..., description="Human-readable reason.")


class UncertaintySignalsModel(BaseModel):
    """Serializable snapshot of the uncertainty signals for a request."""

    signals: list[UncertaintySignal] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, mapping: dict[str, list[str]]) -> "UncertaintySignalsModel":
        items: list[UncertaintySignal] = []
        for name in sorted(mapping.keys()):
            for reason in mapping[name]:
                items.append(UncertaintySignal(signal=name, reason=reason))
        return cls(signals=items)


class EvidenceItem(BaseModel):
    """A retrieved chunk in the API output shape.

    Mirrors Chunk but excludes the embedder-side fields and includes the
    retrieval score. Chunk itself is not modified.
    """

    document_id: str
    chunk_id: str
    title: str
    text: str
    source: str
    url: str | None = None
    license: str
    provenance: Literal["DEVELOPMENT_FIXTURE", "LICENSED_SOURCE", "UNVERIFIED_SOURCE"]
    score: float | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "EvidenceItem":
        return cls(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            text=chunk.text,
            source=chunk.source,
            url=chunk.url,
            license=chunk.license,
            provenance=chunk.provenance,  # type: ignore[arg-type]
            score=chunk.score,
        )


class FinalFinding(BaseModel):
    """One finding presented in the final output.

    Combines the vision status (if any) and the report status (if any)
    for a single NIH label. This is a derived view; it does not replace
    the upstream models.
    """

    name: str
    vision_status: str | None = None
    vision_model_score: float | None = None
    report_status: str | None = None
    report_anatomy: str | None = None
    in_conflict: bool = False


class ExplanationBlock(BaseModel):
    """The generated explanation text plus provenance."""

    text: str = Field(..., description="The explanation string. Opaque to all other fields.")
    generator: Literal["amd_token_factory", "template"] = Field(
        ..., description="Which generator produced the text."
    )
    fallback_reason: str | None = Field(
        default=None,
        description="If template fallback was used, why. None if the AMD LLM was used.",
    )


class HumanReviewBlock(BaseModel):
    """Human-review state for this request."""

    required: bool
    reasons: list[str] = Field(default_factory=list)


class RadiantResult(BaseModel):
    """Complete RADIANT response for one analysis request.

    Every field except ExplanationBlock.text is derived from deterministic
    pipeline code. ExplanationBlock.text cannot modify any other field.
    """

    request_id: str = Field(..., description="Opaque identifier for this request.")
    vision: VisionResult | None = Field(default=None)
    report_analysis: ReportExtraction | None = Field(default=None)
    cross_modal_analysis: FusionResult | None = Field(default=None)
    uncertainty: UncertaintySignalsModel = Field(default_factory=UncertaintySignalsModel)
    triage: TriageResult | None = Field(default=None)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    explanation: ExplanationBlock | None = Field(default=None)
    safety: SafetyResult | None = Field(default=None)
    human_review: HumanReviewBlock = Field(
        default_factory=lambda: HumanReviewBlock(required=False)
    )
    generator: Literal["amd_token_factory", "template", "none"] = Field(
        default="none",
        description=(
            "The generator used for the explanation. Duplicated at the top level "
            "for convenience; must equal explanation.generator if explanation "
            "is present."
        ),
    )
    disclaimer: str = Field(default=DISCLAIMER)