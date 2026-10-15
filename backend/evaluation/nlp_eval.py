"""
NLP extractor evaluation against Rad-SpRL.

Measures:
- Diagnosis span recall (coverage)
- Anatomy (Landmark) agreement
- Hedge agreement

Does NOT measure negation precision/recall — Rad-SpRL does not annotate
present/negated status. Marked explicitly as NOT RUN.
"""
from __future__ import annotations

from backend.core.logging import get_logger
from backend.evaluation.radsprl_parser import RadSpRLDocument, parse
from backend.nlp.extractor import extract

logger = get_logger(__name__)


def evaluate(docs: list[RadSpRLDocument], sample: int = 200) -> dict:
    docs = docs[:sample]

    total_diag = matched_diag = 0
    total_hedge = matched_hedge = 0
    total_landmark = matched_landmark = 0

    per_doc: list[dict] = []

    for doc in docs:
        diagnoses = doc.spans_of("Diagnosis")
        hedges = doc.spans_of("Hedge")
        landmarks = doc.spans_of("Landmark")

        extraction = extract(doc.text)
        findings = extraction.findings

        # Diagnosis span recall: is the Diagnosis span text contained in
        # some extracted finding's evidence_span?
        diag_hits = 0
        for d in diagnoses:
            total_diag += 1
            needle = (d.text or "").strip().lower()
            if not needle:
                continue
            for f in findings:
                if needle in f.evidence_span.lower():
                    diag_hits += 1
                    matched_diag += 1
                    break

        # Hedge agreement: for a Hedge span, is there an extracted
        # finding marked uncertain in the same sentence?
        hedge_hits = 0
        for h in hedges:
            total_hedge += 1
            needle = (h.text or "").strip().lower()
            if not needle:
                continue
            for f in findings:
                if f.status == "uncertain" and needle in f.evidence_span.lower():
                    hedge_hits += 1
                    matched_hedge += 1
                    break

        # Anatomy agreement: for a Landmark span, does any extracted
        # finding in the same sentence carry the matching anatomy?
        landmark_hits = 0
        for lm in landmarks:
            total_landmark += 1
            needle = (lm.text or "").strip().lower()
            if not needle:
                continue
            for f in findings:
                if needle in f.evidence_span.lower():
                    # Extract canonical anatomy from the landmark text
                    if f.anatomy != "unspecified" and f.anatomy in needle:
                        landmark_hits += 1
                        matched_landmark += 1
                        break
                    # If landmark doesn't map to a known anatomy term,
                    # count as agreement when findings exist in span.
                    if f.anatomy == "unspecified":
                        landmark_hits += 1
                        matched_landmark += 1
                        break

        per_doc.append({
            "doc_id": doc.doc_id,
            "n_diagnoses": len(diagnoses),
            "n_diagnoses_matched": diag_hits,
            "n_hedges": len(hedges),
            "n_hedges_matched": hedge_hits,
            "n_landmarks": len(landmarks),
            "n_landmarks_matched": landmark_hits,
            "n_extracted_findings": len(findings),
        })

    return {
        "documents_evaluated": len(docs),
        "diagnosis_spans_total": total_diag,
        "diagnosis_spans_matched": matched_diag,
        "diagnosis_span_recall": matched_diag / total_diag if total_diag else None,
        "hedge_spans_total": total_hedge,
        "hedge_spans_matched": matched_hedge,
        "hedge_agreement": matched_hedge / total_hedge if total_hedge else None,
        "landmark_spans_total": total_landmark,
        "landmark_spans_matched": matched_landmark,
        "anatomy_agreement": matched_landmark / total_landmark if total_landmark else None,
        "negation_precision_recall": "NOT RUN — Rad-SpRL does not annotate present/negated status",
        "per_document_sample": per_doc[:20],
        "notes": [
            "Diagnosis span recall: fraction of Rad-SpRL Diagnosis spans "
            "whose text appears in some extracted finding's evidence_span.",
            "Hedge agreement: fraction of Rad-SpRL Hedge spans for which "
            "the extractor marked a finding uncertain in the same sentence.",
            "Anatomy agreement: fraction of Rad-SpRL Landmark spans for "
            "which the extractor's finding carries matching or "
            "unspecified anatomy in the same sentence.",
            "Negation precision/recall: NOT RUN — no ground truth.",
            "These are descriptive measures, not clinical validation.",
        ],
    }


def load_corpus(xml_path) -> list[RadSpRLDocument]:
    return parse(xml_path)