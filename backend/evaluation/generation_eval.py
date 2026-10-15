"""Generation groundedness evaluation."""
from __future__ import annotations

import re

from backend.rag.schema import Chunk

_CITATION_RE = re.compile(r"\[source:([^\]:]+):([^\]]+)\]")

_DISCLAIMER_MARKERS = (
    "research prototype",
    "not a diagnostic",
    "limitations",
    "must be reviewed by",
)


def _citation_validity(text: str, evidence: list[Chunk]) -> tuple[int, int]:
    valid_ids = {(c.document_id, c.chunk_id) for c in evidence}
    total = valid = 0
    for doc_id, chunk_id in _CITATION_RE.findall(text):
        total += 1
        if (doc_id.strip(), chunk_id.strip()) in valid_ids:
            valid += 1
    return valid, total


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _is_grounded(sentence: str) -> bool:
    lower = sentence.lower()
    if any(m in lower for m in _DISCLAIMER_MARKERS):
        return True
    return bool(_CITATION_RE.search(sentence))


def evaluate_one(text: str, evidence: list[Chunk]) -> dict:
    valid, total = _citation_validity(text, evidence)
    sentences = _sentences(text)
    grounded = sum(1 for s in sentences if _is_grounded(s))
    return {
        "citations_total": total,
        "citations_valid": valid,
        "citation_validity_rate": (valid / total) if total else None,
        "sentences_total": len(sentences),
        "sentences_grounded": grounded,
        "groundedness_ratio": (grounded / len(sentences)) if sentences else None,
    }