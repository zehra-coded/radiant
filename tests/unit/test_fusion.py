"""Unit tests for the cross-modal verifier. No models, no network."""
from __future__ import annotations

from backend.fusion.verifier import verify
from backend.nlp.schema import ReportExtraction, ReportFinding
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult


def _v(name: str, status: str) -> VisionFinding:
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


def _vision_with(**overrides: str) -> VisionResult:
    findings = [
        _v(name, overrides.get(name, "absent"))
        for name in NIH_LABELS
    ]
    return VisionResult(findings=findings, checkpoint="test")


def test_full_agreement() -> None:
    v = _vision_with(Cardiomegaly="present", Effusion="present")
    r = ReportExtraction(
        findings=[_r("Cardiomegaly", "present"), _r("Effusion", "present")],
        raw_text="test",
    )
    result = verify(v, r)
    assert "Cardiomegaly" in result.agreed
    assert "Effusion" in result.agreed
    assert result.conflicts == []
    assert result.overall_consistency == "high"


def test_simple_conflict() -> None:
    v = _vision_with(Pneumothorax="present")
    r = ReportExtraction(findings=[_r("Pneumothorax", "negated")], raw_text="test")
    result = verify(v, r)
    assert len(result.conflicts) == 1
    assert result.conflicts[0].name == "Pneumothorax"
    assert result.overall_consistency == "conflicting"


def test_critical_label_conflict_override() -> None:
    """Even with a high agreement ratio, a conflict on a critical label
    forces overall_consistency = conflicting."""
    # 10 agreed labels + 1 critical conflict.
    agreed_names = [n for n in NIH_LABELS if n != "Pneumothorax"][:10]
    overrides = {n: "present" for n in agreed_names}
    overrides["Pneumothorax"] = "present"
    v = _vision_with(**overrides)
    r = ReportExtraction(
        findings=[_r(n, "present") for n in agreed_names] + [_r("Pneumothorax", "negated")],
        raw_text="test",
    )
    result = verify(v, r)
    assert result.overall_consistency == "conflicting"
    assert any("critical-label conflict" in n for n in result.notes)


def test_both_modalities_empty_is_insufficient() -> None:
    v = _blank_vision()
    r = ReportExtraction(findings=[], raw_text="")
    result = verify(v, r)
    assert result.overall_consistency == "insufficient"
    assert result.agreement_ratio is None


def test_vision_missing_is_insufficient() -> None:
    r = ReportExtraction(findings=[_r("Effusion", "present")], raw_text="test")
    result = verify(None, r)
    assert result.overall_consistency == "insufficient"
    assert any("vision" in n.lower() for n in result.notes)


def test_report_missing_is_insufficient() -> None:
    v = _vision_with(Effusion="present")
    result = verify(v, None)
    assert result.overall_consistency == "insufficient"
    assert any("report" in n.lower() for n in result.notes)


def test_uncertain_vision_never_conflicts() -> None:
    v = _vision_with(Effusion="uncertain")
    r = ReportExtraction(findings=[_r("Effusion", "negated")], raw_text="test")
    result = verify(v, r)
    assert result.conflicts == []
    # Effusion should be in skipped, not agreed.


def test_uncertain_report_never_conflicts() -> None:
    v = _vision_with(Effusion="present")
    r = ReportExtraction(findings=[_r("Effusion", "uncertain")], raw_text="test")
    result = verify(v, r)
    assert result.conflicts == []


def test_vision_only_when_report_silent() -> None:
    v = _vision_with(Effusion="present")
    r = ReportExtraction(findings=[], raw_text="")
    # report has zero findings — this triggers report_missing=insufficient.
    result = verify(v, r)
    assert result.overall_consistency == "insufficient"


def test_report_only_when_vision_absent() -> None:
    v = _blank_vision()
    r = ReportExtraction(findings=[_r("Effusion", "present")], raw_text="test")
    result = verify(v, r)
    assert "Effusion" in result.report_only


def test_ratio_boundary_high() -> None:
    """A ratio of exactly 0.70 should be 'high'."""
    # 7 agreed + 3 vision_only = ratio 0.70
    agreed_names = NIH_LABELS[:7]
    vision_only_names = NIH_LABELS[7:10]
    overrides = {n: "present" for n in agreed_names + vision_only_names}
    v = _vision_with(**overrides)
    r = ReportExtraction(
        findings=[_r(n, "present") for n in agreed_names],
        raw_text="test",
    )
    result = verify(v, r)
    assert result.overall_consistency == "high"


def test_ratio_boundary_partial() -> None:
    """A ratio of exactly 0.40 should be 'partial'."""
    # 4 agreed + 6 vision_only = 0.40
    agreed_names = NIH_LABELS[:4]
    vision_only_names = NIH_LABELS[4:10]
    overrides = {n: "present" for n in agreed_names + vision_only_names}
    v = _vision_with(**overrides)
    r = ReportExtraction(
        findings=[_r(n, "present") for n in agreed_names],
        raw_text="test",
    )
    result = verify(v, r)
    assert result.overall_consistency == "partial"


def test_ratio_below_partial_is_conflicting() -> None:
    # 3 agreed + 7 vision_only = 0.30
    agreed_names = NIH_LABELS[:3]
    vision_only_names = NIH_LABELS[3:10]
    overrides = {n: "present" for n in agreed_names + vision_only_names}
    v = _vision_with(**overrides)
    r = ReportExtraction(
        findings=[_r(n, "present") for n in agreed_names],
        raw_text="test",
    )
    result = verify(v, r)
    assert result.overall_consistency == "conflicting"