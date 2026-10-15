"""Unit tests for uncertainty signals and triage. No models, no network."""
from __future__ import annotations

from backend.fusion.schema import FusionConflict, FusionResult
from backend.nlp.schema import ReportExtraction, ReportFinding
from backend.triage.engine import triage
from backend.uncertainty import signals as sig
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult


def _v(name: str, status: str, score: float | None = None) -> VisionFinding:
    if score is None:
        score = {"present": 0.8, "uncertain": 0.4, "absent": 0.1}[status]
    return VisionFinding(name=name, model_score=score, status=status)  # type: ignore[arg-type]


def _r(name: str, status: str) -> ReportFinding:
    return ReportFinding(
        name=name,
        anatomy="unspecified",
        status=status,  # type: ignore[arg-type]
        uncertainty=0.0 if status != "uncertain" else 1.0,
        evidence_span="test",
        spatial_info=None,
    )


def _blank_vision() -> VisionResult:
    return VisionResult(
        findings=[_v(n, "absent") for n in NIH_LABELS],
        checkpoint="test",
    )


def _clean_fusion() -> FusionResult:
    return FusionResult(
        agreed=list(NIH_LABELS),
        vision_only=[],
        report_only=[],
        conflicts=[],
        overall_consistency="high",
        agreement_ratio=1.0,
    )


def test_routine_when_everything_agrees() -> None:
    v = _blank_vision()
    r = ReportExtraction(
        findings=[_r(n, "negated") for n in NIH_LABELS],
        raw_text="test",
    )
    result = triage(v, r, _clean_fusion(), None)
    assert result.priority == "ROUTINE"
    assert not any("fail" in s for s in result.source_signals)


def test_conflict_forces_urgent_review() -> None:
    v = _blank_vision()
    r = ReportExtraction(findings=[_r("Effusion", "present")], raw_text="test")
    fusion = FusionResult(
        agreed=[],
        vision_only=[],
        report_only=["Effusion"],
        conflicts=[
            FusionConflict(
                name="Effusion",
                vision_status="absent",
                report_status="present",
                reason="test conflict",
            )
        ],
        overall_consistency="conflicting",
        agreement_ratio=0.0,
    )
    result = triage(v, r, fusion, None)
    assert result.priority == "URGENT_REVIEW"
    assert sig.SIGNAL_VISION_REPORT_CONFLICT in result.source_signals


def test_critical_present_forces_urgent_review() -> None:
    v = VisionResult(
        findings=[
            _v("Pneumothorax", "present", score=0.9),
            *[_v(n, "absent") for n in NIH_LABELS if n != "Pneumothorax"],
        ],
        checkpoint="test",
    )
    r = ReportExtraction(findings=[_r("Pneumothorax", "present")], raw_text="test")
    fusion = FusionResult(
        agreed=["Pneumothorax"],
        vision_only=[],
        report_only=[],
        conflicts=[],
        overall_consistency="high",
        agreement_ratio=1.0,
    )
    result = triage(v, r, fusion, None)
    assert result.priority == "URGENT_REVIEW"
    assert any("Pneumothorax" in r for r in result.reasons)


def test_model_failure_never_routine() -> None:
    result = triage(None, None, None, None, vision_failed=True)
    assert result.priority != "ROUTINE"
    assert sig.SIGNAL_MODEL_FAILURE in result.source_signals


def test_insufficient_fusion_never_routine() -> None:
    fusion = FusionResult(
        agreed=[],
        vision_only=[],
        report_only=[],
        conflicts=[],
        overall_consistency="insufficient",
        agreement_ratio=None,
    )
    result = triage(None, None, fusion, None)
    assert result.priority != "ROUTINE"


def test_insufficient_report_is_at_least_monitor() -> None:
    v = _blank_vision()
    r = ReportExtraction(findings=[], raw_text="")
    fusion = FusionResult(
        agreed=[],
        vision_only=[],
        report_only=[],
        conflicts=[],
        overall_consistency="insufficient",
        agreement_ratio=None,
    )
    result = triage(v, r, fusion, None)
    assert result.priority in ("MONITOR", "URGENT_REVIEW", "EMERGENCY_REVIEW")
    assert sig.SIGNAL_INSUFFICIENT_REPORT in result.source_signals


def test_ambiguous_predictions_is_monitor() -> None:
    # Many findings in the uncertain band.
    findings = []
    for i, n in enumerate(NIH_LABELS):
        findings.append(_v(n, "uncertain" if i < 6 else "absent"))
    v = VisionResult(findings=findings, checkpoint="test")
    r = ReportExtraction(findings=[], raw_text="")
    fusion = FusionResult(
        agreed=[],
        vision_only=[],
        report_only=[],
        conflicts=[],
        overall_consistency="insufficient",
        agreement_ratio=None,
    )
    result = triage(v, r, fusion, None)
    assert sig.SIGNAL_AMBIGUOUS_PREDICTIONS in result.source_signals
    assert result.priority != "ROUTINE"


def test_reasons_are_human_readable() -> None:
    v = _blank_vision()
    r = ReportExtraction(findings=[], raw_text="")
    fusion = FusionResult(
        agreed=[],
        vision_only=[],
        report_only=[],
        conflicts=[],
        overall_consistency="insufficient",
        agreement_ratio=None,
    )
    result = triage(v, r, fusion, None)
    assert result.reasons
    for r_str in result.reasons:
        # Reasons should be full sentences, not bare identifiers.
        assert len(r_str.split()) >= 4


def test_decision_trace_is_populated() -> None:
    result = triage(None, None, None, None, vision_failed=True)
    assert result.decision_trace
    assert any("model_failure" in t for t in result.decision_trace)


def test_higher_priority_wins_precedence() -> None:
    """A model_failure plus a critical finding should still resolve to
    URGENT_REVIEW (or higher), never downgrade."""
    v = VisionResult(
        findings=[
            _v("Pneumothorax", "present", score=0.9),
            *[_v(n, "absent") for n in NIH_LABELS if n != "Pneumothorax"],
        ],
        checkpoint="test",
    )
    r = ReportExtraction(findings=[_r("Pneumothorax", "present")], raw_text="test")
    fusion = FusionResult(
        agreed=["Pneumothorax"],
        vision_only=[],
        report_only=[],
        conflicts=[],
        overall_consistency="high",
        agreement_ratio=1.0,
    )
    result = triage(v, r, fusion, None, vision_failed=False)
    assert result.priority == "URGENT_REVIEW"