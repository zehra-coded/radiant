"""
Uncertainty signals computed from the upstream pipeline.

Signals are strings, not booleans, so the downstream triage engine can
trace *why* a priority was chosen.

Signals implemented (per spec):

- low_confidence         : a vision finding in the uncertain band
                           appears as vision_only
- ambiguous_predictions  : too many vision findings in the uncertain band
- vision_report_conflict : the fusion result contains any conflicts
- insufficient_report    : the report produced no findings
- retrieval_failure      : the RAG retrieval failed (added in Stage 8)
- model_failure          : the vision or NLP module failed
"""
from __future__ import annotations

from typing import Final

from backend.core.config import (
    TRIAGE_AMBIGUOUS_FRACTION,
    TRIAGE_LOW_CONFIDENCE_MAX_SCORE,
)
from backend.fusion.schema import FusionResult
from backend.nlp.schema import ReportExtraction
from backend.vision.schema import VisionResult

# Signal names as constants, so downstream code can reference them safely.
SIGNAL_LOW_CONFIDENCE: Final[str] = "low_confidence"
SIGNAL_AMBIGUOUS_PREDICTIONS: Final[str] = "ambiguous_predictions"
SIGNAL_VISION_REPORT_CONFLICT: Final[str] = "vision_report_conflict"
SIGNAL_INSUFFICIENT_REPORT: Final[str] = "insufficient_report"
SIGNAL_RETRIEVAL_FAILURE: Final[str] = "retrieval_failure"
SIGNAL_MODEL_FAILURE: Final[str] = "model_failure"


class UncertaintySignals:
    """Collected signals with traceable reasons."""

    def __init__(self) -> None:
        self._signals: dict[str, list[str]] = {}

    def add(self, signal: str, reason: str) -> None:
        self._signals.setdefault(signal, []).append(reason)

    def has(self, signal: str) -> bool:
        return signal in self._signals

    def signals(self) -> list[str]:
        return sorted(self._signals.keys())

    def reasons(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._signals.items()}

    def human_readable(self) -> list[str]:
        out: list[str] = []
        for signal in sorted(self._signals.keys()):
            for reason in self._signals[signal]:
                out.append(f"{signal}: {reason}")
        return out


def compute(
    vision: VisionResult | None,
    report: ReportExtraction | None,
    fusion: FusionResult | None,
    *,
    vision_failed: bool = False,
    report_failed: bool = False,
    retrieval_failed: bool = False,
) -> UncertaintySignals:
    """Compute all uncertainty signals from upstream outputs."""
    signals = UncertaintySignals()

    if vision_failed or vision is None:
        signals.add(SIGNAL_MODEL_FAILURE, "vision module failed or returned no result")

    if report_failed or report is None:
        signals.add(SIGNAL_MODEL_FAILURE, "report NLP module failed or returned no result")

    if retrieval_failed:
        signals.add(SIGNAL_RETRIEVAL_FAILURE, "evidence retrieval failed")

    if fusion is None:
        return signals

    if fusion.conflicts:
        for c in fusion.conflicts:
            signals.add(
                SIGNAL_VISION_REPORT_CONFLICT,
                f"{c.name}: {c.reason}",
            )

    if report is not None and not report.findings:
        signals.add(SIGNAL_INSUFFICIENT_REPORT, "report contained no findings")

    if vision is not None:
        uncertain_findings = [f for f in vision.findings if f.status == "uncertain"]
        total = len(vision.findings)
        if total > 0 and (len(uncertain_findings) / total) > TRIAGE_AMBIGUOUS_FRACTION:
            signals.add(
                SIGNAL_AMBIGUOUS_PREDICTIONS,
                f"{len(uncertain_findings)}/{total} vision findings in uncertain band",
            )

        # Low confidence: a vision finding with a model score below the
        # confidence threshold that also appears as vision_only.
        vision_only_set = set(fusion.vision_only) if fusion else set()
        for f in vision.findings:
            if f.name in vision_only_set and f.model_score <= TRIAGE_LOW_CONFIDENCE_MAX_SCORE:
                signals.add(
                    SIGNAL_LOW_CONFIDENCE,
                    f"{f.name}: model_score={f.model_score:.3f} (vision_only, low score)",
                )

    return signals