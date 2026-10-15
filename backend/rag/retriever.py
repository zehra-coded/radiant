"""
Retriever over the RAG index.

Returns [] when nothing meets the minimum similarity threshold. No
hidden fallback, no hard-coded results.
"""
from __future__ import annotations

import numpy as np

from backend.core.config import RAG_MIN_SIMILARITY, RAG_TOP_K_DEFAULT
from backend.core.logging import get_logger
from backend.rag.indexer import RagIndex
from backend.rag.schema import Chunk, RetrievalResult

logger = get_logger(__name__)


def retrieve(
    index: RagIndex,
    query: str,
    k: int = RAG_TOP_K_DEFAULT,
    min_similarity: float = RAG_MIN_SIMILARITY,
) -> RetrievalResult:
    """Retrieve up to k chunks matching query above min_similarity."""
    notes: list[str] = []
    if not query or not query.strip():
        return RetrievalResult(
            query=query,
            chunks=[],
            min_similarity=min_similarity,
            notes=["empty query"],
        )

    q = index.embedder.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Retrieve more than k so we can filter by threshold.
    k_fetch = min(max(k * 3, k), len(index.chunks))
    scores, indices = index.index.search(q, k_fetch)

    out: list[Chunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < min_similarity:
            continue
        chunk = index.chunks[int(idx)].model_copy()
        chunk.score = float(score)
        out.append(chunk)
        if len(out) >= k:
            break

    if not out:
        notes.append(
            f"no chunks met min_similarity={min_similarity:.2f} for query: {query!r}"
        )

    return RetrievalResult(
        query=query,
        chunks=out,
        min_similarity=min_similarity,
        notes=notes,
    )