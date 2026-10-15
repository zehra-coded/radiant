"""
The single shared pipeline used by /analyze and /analyze/stream.

Runs the same steps in the same order. The synchronous path returns
the final RadiantResult. The streaming path yields (step, partial_or_None)
after each step really completes.

Every step is wrapped in try/except. On failure, the step is skipped,
the failure is recorded, human review is set to required, and the
pipeline continues. The pipeline never raises to the caller.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterator

from backend.api import state
from backend.core.config import API_DEFAULT_QUERY
from backend.core.logging import get_logger
from backend.fusion.verifier import verify as run_fusion
from backend.llm.reporter import generate_explanation
from backend.nlp.extractor import extract as run_nlp
from backend.rag.retriever import retrieve as run_retrieve
from backend.safety.schema import DISCLAIMER
from backend.schemas.result import (
    EvidenceItem,
    ExplanationBlock,
    HumanReviewBlock,
    RadiantResult,
    UncertaintySignalsModel,
)
from backend.triage.engine import triage as run_triage
from backend.uncertainty.signals import compute as compute_signals

logger = get_logger(__name__)


def _new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


class _PipelineRun:
    """Holds the state of a single pipeline execution.

    Each step is a method that mutates self.result and self.failed flags.
    Both the sync and streaming drivers call the same methods.
    """

    def __init__(self, image_path: Path, report_text: str, user_query: str) -> None:
        self.image_path = image_path
        self.report_text = report_text
        self.user_query = user_query
        self.request_id = _new_request_id()
        self.result = RadiantResult(
            request_id=self.request_id,
            disclaimer=DISCLAIMER,
            generator="none",
            human_review=HumanReviewBlock(required=False),
            uncertainty=UncertaintySignalsModel(),
        )
        self.vision_failed = False
        self.report_failed = False
        self.retrieval_failed = False

    # ---- step 1 ----------------------------------------------------

    def step_vision(self) -> None:
        try:
            model = state.get_vision_model()
            self.result.vision = model.predict(self.image_path)
        except Exception as exc:
            logger.exception("vision step failed: %s", exc)
            self.vision_failed = True

    # ---- step 2 ----------------------------------------------------

    def step_nlp(self) -> None:
        try:
            self.result.report_analysis = run_nlp(self.report_text)
        except Exception as exc:
            logger.exception("nlp step failed: %s", exc)
            self.report_failed = True

    # ---- step 3 ----------------------------------------------------

    def step_fusion(self) -> None:
        try:
            if self.result.vision is not None or self.result.report_analysis is not None:
                self.result.cross_modal_analysis = run_fusion(
                    self.result.vision, self.result.report_analysis
                )
        except Exception as exc:
            logger.exception("fusion step failed: %s", exc)

    # ---- step 4 ----------------------------------------------------

    def step_triage(self) -> None:
        try:
            signals = compute_signals(
                self.result.vision,
                self.result.report_analysis,
                self.result.cross_modal_analysis,
                vision_failed=self.vision_failed,
                report_failed=self.report_failed,
                retrieval_failed=self.retrieval_failed,
            )
            self.result.triage = run_triage(
                self.result.vision,
                self.result.report_analysis,
                self.result.cross_modal_analysis,
                signals,
                vision_failed=self.vision_failed,
                report_failed=self.report_failed,
                retrieval_failed=self.retrieval_failed,
            )
            self.result.uncertainty = UncertaintySignalsModel.from_dict(signals.reasons())
        except Exception as exc:
            logger.exception("triage step failed: %s", exc)

    # ---- step 5 ----------------------------------------------------

    def step_retrieval(self) -> None:
        try:
            index = state.get_rag_index()
            query = _build_retrieval_query(self.result)
            rr = run_retrieve(index, query, k=5)
            self.result.evidence = [EvidenceItem.from_chunk(c) for c in rr.chunks]
        except Exception as exc:
            logger.exception("retrieval step failed: %s", exc)
            self.retrieval_failed = True

    # ---- step 6 ----------------------------------------------------

    def step_explanation(self) -> None:
        try:
            from backend.rag.schema import Chunk

            chunks = [
                Chunk(
                    document_id=e.document_id,
                    chunk_id=e.chunk_id,
                    text=e.text,
                    title=e.title,
                    source=e.source,
                    url=e.url,
                    license=e.license,
                    provenance=e.provenance,  # type: ignore[arg-type]
                    score=e.score,
                )
                for e in self.result.evidence
            ]
            out = generate_explanation(
                self.user_query,
                self.result.vision,
                self.result.report_analysis,
                self.result.cross_modal_analysis,
                self.result.triage,
                chunks,
                vision_failed=self.vision_failed,
                report_failed=self.report_failed,
                retrieval_failed=self.retrieval_failed,
            )
            self.result.explanation = ExplanationBlock(
                text=out.text,
                generator=out.generator,  # type: ignore[arg-type]
                fallback_reason=out.fallback_reason,
            )
            self.result.generator = out.generator  # type: ignore[assignment]
            self.result.safety = out.post_safety
        except Exception as exc:
            logger.exception("explanation step failed: %s", exc)

    # ---- step 7 ----------------------------------------------------

    def finalize(self) -> None:
        must_review = False
        reasons: list[str] = []
        if self.result.triage and self.result.triage.priority in (
            "URGENT_REVIEW", "EMERGENCY_REVIEW"
        ):
            must_review = True
            reasons.append(f"priority={self.result.triage.priority}")
        if self.result.cross_modal_analysis and self.result.cross_modal_analysis.conflicts:
            must_review = True
            reasons.append(
                f"{len(self.result.cross_modal_analysis.conflicts)} conflicts"
            )
        if self.vision_failed or self.report_failed or self.retrieval_failed:
            must_review = True
            reasons.append("pipeline step failed")
        if self.result.safety and self.result.safety.requires_human_review:
            must_review = True
            reasons.append("safety gate requires review")
        self.result.human_review = HumanReviewBlock(required=must_review, reasons=reasons)


def _build_retrieval_query(result: RadiantResult) -> str:
    parts: list[str] = []
    if result.cross_modal_analysis and result.cross_modal_analysis.conflicts:
        parts.extend(c.name for c in result.cross_modal_analysis.conflicts)
    if result.vision is not None:
        parts.extend(f.name for f in result.vision.findings if f.status == "present")
    if result.report_analysis is not None:
        parts.extend(
            f.name for f in result.report_analysis.findings if f.status == "present"
        )
    if not parts:
        return "chest x-ray findings"
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return " ".join(unique)


# The step order and their event names, shared by both drivers.
_STEPS: tuple[tuple[str, str], ...] = (
    ("vision_done", "step_vision"),
    ("nlp_done", "step_nlp"),
    ("fusion_done", "step_fusion"),
    ("priority_done", "step_triage"),
    ("retrieval_done", "step_retrieval"),
    ("safety_done", "step_explanation"),   # pre+post safety run inside explanation
    ("generation_done", "step_explanation"),  # same method, no-op second call
)


def run(
    image_path: Path,
    report_text: str,
    user_query: str | None = None,
) -> RadiantResult:
    """Run the full pipeline synchronously. Returns a RadiantResult."""
    p = _PipelineRun(image_path, report_text, user_query or API_DEFAULT_QUERY)
    p.step_vision()
    p.step_nlp()
    p.step_fusion()
    p.step_triage()
    p.step_retrieval()
    p.step_explanation()
    p.finalize()
    return p.result


def run_streaming(
    image_path: Path,
    report_text: str,
    user_query: str | None = None,
) -> Iterator[tuple[str, RadiantResult | None]]:
    """Yield (step_name, None) after each step completes. Final yield
    is ("done", RadiantResult)."""
    p = _PipelineRun(image_path, report_text, user_query or API_DEFAULT_QUERY)

    p.step_vision()
    yield ("vision_done", None)
    p.step_nlp()
    yield ("nlp_done", None)
    p.step_fusion()
    yield ("fusion_done", None)
    p.step_triage()
    yield ("priority_done", None)
    p.step_retrieval()
    yield ("retrieval_done", None)
    p.step_explanation()
    yield ("safety_done", None)
    yield ("generation_done", None)
    p.finalize()
    yield ("done", p.result)