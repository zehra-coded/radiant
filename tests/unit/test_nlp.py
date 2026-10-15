"""Unit tests for the report NLP extractor. Deterministic, no network."""
from __future__ import annotations

from backend.nlp.extractor import extract


def _by_name(extraction, name: str) -> list:
    return [f for f in extraction.findings if f.name == name]


def test_positive_single_finding() -> None:
    r = extract("There is a small left pleural effusion.")
    effusions = _by_name(r, "Effusion")
    assert len(effusions) == 1
    assert effusions[0].status == "present"
    assert effusions[0].anatomy == "left"
    assert effusions[0].uncertainty == 0.0


def test_negated_finding() -> None:
    r = extract("No pneumothorax.")
    pts = _by_name(r, "Pneumothorax")
    assert len(pts) == 1
    assert pts[0].status == "negated"
    assert pts[0].uncertainty == 0.0


def test_uncertain_finding() -> None:
    r = extract("Possible small right pleural effusion.")
    effusions = _by_name(r, "Effusion")
    assert len(effusions) == 1
    assert effusions[0].status == "uncertain"
    assert effusions[0].uncertainty >= 0.8


def test_bilateral_finding() -> None:
    r = extract("Bilateral lower lobe consolidations are present.")
    cons = _by_name(r, "Consolidation")
    assert len(cons) >= 1
    assert cons[0].anatomy == "bilateral"
    assert cons[0].status == "present"


def test_multiple_findings_in_one_sentence() -> None:
    r = extract("Mild cardiomegaly and a small left pleural effusion.")
    assert len(_by_name(r, "Cardiomegaly")) >= 1
    assert len(_by_name(r, "Effusion")) >= 1


def test_mixed_negation_and_presence_in_one_sentence() -> None:
    r = extract("No pneumothorax, but there is a small effusion.")
    pts = _by_name(r, "Pneumothorax")
    effusions = _by_name(r, "Effusion")
    assert len(pts) == 1 and pts[0].status == "negated"
    assert len(effusions) == 1 and effusions[0].status == "present"


def test_double_negation_is_uncertain() -> None:
    r = extract("Cannot exclude a small pneumothorax.")
    pts = _by_name(r, "Pneumothorax")
    assert len(pts) == 1
    assert pts[0].status == "uncertain"
    assert pts[0].uncertainty >= 0.8


def test_empty_report() -> None:
    r = extract("")
    assert r.findings == []
    assert any("empty" in n.lower() or "no parseable" in n.lower() for n in r.notes)


def test_no_findings_report() -> None:
    r = extract("Normal chest radiograph. No acute cardiopulmonary abnormality.")
    # This is a tricky case: "no acute" is a cue but "abnormality" is not a finding.
    assert r.findings == [] or all(f.status == "negated" for f in r.findings)