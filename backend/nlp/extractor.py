"""
Deterministic report NLP extraction.

V1 uses regex and lexicons only. No LLM. No model. Pure Python.

The extractor works in three passes:

1. Sentence segmentation.
2. Finding detection per sentence, with a preceding-cue window to detect
   negation and uncertainty.
3. Deduplication: the same finding appearing twice with the same status
   is collapsed. Conflicting statuses in the same report are preserved
   as separate findings.
"""
from __future__ import annotations
from typing import Final
import re

from backend.core.logging import get_logger
from backend.nlp.lexicons import (
    ANATOMY_TERMS,
    FINDING_SYNONYMS,
    NEGATION_CUES,
    SPATIAL_TERMS,
    UNCERTAINTY_CUES,
)
from backend.nlp.schema import ReportExtraction, ReportFinding

logger = get_logger(__name__)

# Characters that end a "sentence" for our purposes. Radiology reports
# use periods, semicolons, and newlines as boundaries.
_SENTENCE_SPLIT = re.compile(r"[.\n;]+")

# How many characters to look back from a finding match for a cue.
_CUE_WINDOW = 60


def _segment_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


# Clause-boundary words: if one of these appears between a cue and a
# finding, the cue does not apply to the finding. This is the core fix
# for the negation-scope problem in mixed sentences like
# "No pneumothorax, but there is a small effusion."
_CLAUSE_BOUNDARIES: Final[tuple[str, ...]] = (
    "but",
    "however",
    "although",
    "though",
    "yet",
    "whereas",
    "while",
)


def _find_cue_before(
    sentence: str,
    match_start: int,
    cues: dict[str, float] | tuple[str, ...],
) -> tuple[str, float] | None:
    """Look backwards from ``match_start`` within the cue window and
    return the closest matching cue and its score.

    A cue is ignored if a clause boundary (but, however, ...) appears
    between the cue and the finding. This prevents negation from
    leaking across clauses.

    For a tuple of cues (negation), the score is 0.0.
    For a dict of cues (uncertainty), the score is the dict value.
    Returns None if no cue is found.
    """
    window_start = max(0, match_start - _CUE_WINDOW)
    window = sentence[window_start:match_start].lower()

    # Find the position of the LAST clause boundary in the window.
    # Any cue that starts BEFORE this boundary is out of scope.
    boundary_pos = -1
    for word in _CLAUSE_BOUNDARIES:
        idx = window.rfind(f" {word} ")
        if idx == -1:
            # Also try at the very start of the window.
            if window.startswith(word + " "):
                idx = 0
        if idx > boundary_pos:
            boundary_pos = idx

    best: tuple[str, float] | None = None
    best_end = -1

    items = cues.items() if isinstance(cues, dict) else [(c, 0.0) for c in cues]
    for cue, value in items:
        idx = window.rfind(cue.lower())
        if idx == -1:
            continue
        # If the cue starts before a clause boundary, it's out of scope.
        if boundary_pos >= 0 and idx < boundary_pos:
            continue
        end = idx + len(cue)
        # "no" must be a standalone word, not part of "nodule" etc.
        if cue.lower() == "no":
            before = window[idx - 1] if idx > 0 else " "
            after = window[end] if end < len(window) else " "
            if before.isalpha() or after.isalpha():
                continue
        if end > best_end:
            best_end = end
            best = (cue, value)
    return best


def _find_anatomy(sentence: str, match_start: int, match_end: int) -> tuple[str, str | None]:
    """Look for an anatomy term near the finding.

    Returns (canonical_anatomy, spatial_phrase_or_None).
    """
    # First check for multi-word spatial phrases anywhere near the finding.
    window_start = max(0, match_start - 40)
    window_end = min(len(sentence), match_end + 40)
    window = sentence[window_start:window_end].lower()

    spatial: str | None = None
    for term in SPATIAL_TERMS:
        if term in window:
            spatial = term
            break

    if spatial is not None:
        # Derive canonical anatomy from the first word of the spatial term.
        first = spatial.split()[0]
        return ANATOMY_TERMS.get(first, "unspecified"), spatial

    # Fall back to single-word anatomy tokens.
    for token, canonical in ANATOMY_TERMS.items():
        if re.search(rf"\b{re.escape(token)}\b", window):
            return canonical, None
    return "unspecified", None


def _deduplicate(findings: list[ReportFinding]) -> list[ReportFinding]:
    """Collapse exact (name, anatomy, status) duplicates.

    Preserve the first occurrence. Do NOT collapse same-name-different-status.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[ReportFinding] = []
    for f in findings:
        key = (f.name, f.anatomy, f.status)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def extract(text: str) -> ReportExtraction:
    """Extract findings from a radiology report."""
    findings: list[ReportFinding] = []
    notes: list[str] = []

    sentences = _segment_sentences(text)
    if not sentences:
        return ReportExtraction(
            findings=[],
            raw_text=text,
            notes=["Report was empty or contained no parseable sentences."],
        )

    for sentence in sentences:
        lower = sentence.lower()
        for canonical, synonyms in FINDING_SYNONYMS.items():
            # Sort synonyms by length descending so longer phrases match first.
            for syn in sorted(synonyms, key=len, reverse=True):
                pattern = re.compile(rf"\b{re.escape(syn)}\b")
                for m in pattern.finditer(lower):
                    start, end = m.start(), m.end()

                    # Check uncertainty first — it takes precedence over negation
                    # only if the uncertainty cue is CLOSER to the finding.
                    unc = _find_cue_before(sentence, start, UNCERTAINTY_CUES)
                    neg = _find_cue_before(sentence, start, NEGATION_CUES)

                    unc_dist = (start - (start - _CUE_WINDOW)) if unc else 10**6
                    if unc:
                        # Distance from finding to cue end
                        window = sentence[max(0, start - _CUE_WINDOW):start].lower()
                        idx = window.rfind(unc[0].lower())
                        unc_dist = start - (max(0, start - _CUE_WINDOW) + idx + len(unc[0]))

                    neg_dist = 10**6
                    if neg:
                        window = sentence[max(0, start - _CUE_WINDOW):start].lower()
                        idx = window.rfind(neg[0].lower())
                        neg_dist = start - (max(0, start - _CUE_WINDOW) + idx + len(neg[0]))

                    if unc and unc_dist < neg_dist:
                        status = "uncertain"
                        uncertainty = unc[1]
                    elif neg:
                        status = "negated"
                        uncertainty = 0.0
                    else:
                        status = "present"
                        uncertainty = 0.0

                    anatomy, spatial = _find_anatomy(sentence, start, end)

                    findings.append(
                        ReportFinding(
                            name=canonical,
                            anatomy=anatomy,
                            status=status,
                            uncertainty=uncertainty,
                            evidence_span=sentence.strip(),
                            spatial_info=spatial,
                        )
                    )
                    # Only record the first match per synonym per sentence
                    # to avoid piling up duplicates.
                    break

    findings = _deduplicate(findings)

    if not findings:
        notes.append("No findings were extracted from this report.")

    return ReportExtraction(findings=findings, raw_text=text, notes=notes)