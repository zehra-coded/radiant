"""
FAISS index builder and loader for the RADIANT knowledge base.

Chunking respects the embedding model's max sequence length. Chunks
are sized in tokens measured with the model's own tokenizer.

Provenance rule: every chunk carries one of DEVELOPMENT_FIXTURE,
LICENSED_SOURCE, UNVERIFIED_SOURCE. The retriever and safety gate use
this to decide whether the chunk may be shown to a user.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import (
    RAG_CHUNK_OVERLAP_TOKENS,
    RAG_CHUNK_TOKENS,
    RAG_EMBEDDING_MODEL,
)
from backend.core.logging import get_logger
from backend.rag.schema import Chunk

logger = get_logger(__name__)

_INDEX_FILENAME = "index.faiss"
_CHUNKS_FILENAME = "chunks.pkl"
_META_FILENAME = "meta.json"

_FIXTURE_MARKER = "DEVELOPMENT FIXTURE — NOT AUTHORITATIVE CLINICAL GUIDANCE"


def _extract_field(text: str, field: str) -> str | None:
    m = re.search(rf"^{field}:\s*(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def _title_from_text(path: Path, text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else path.stem


def _provenance_from_text(text: str) -> str:
    if text.lstrip().startswith(_FIXTURE_MARKER):
        return "DEVELOPMENT_FIXTURE"
    if _extract_field(text, "license") and _extract_field(text, "source"):
        return "LICENSED_SOURCE"
    return "UNVERIFIED_SOURCE"


def _chunk_text(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text into chunks of at most max_tokens tokens with overlap.

    Uses the embedding model's own tokenizer so the count matches what
    the model will see.
    """
    # Split on paragraph boundaries first, then tokenize each paragraph.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current_ids: list[int] = []

    def flush() -> None:
        nonlocal current_ids
        if not current_ids:
            return
        chunk_text = tokenizer.decode(current_ids, skip_special_tokens=True)
        chunks.append(chunk_text.strip())
        # Keep the last overlap_tokens for the next chunk.
        current_ids = current_ids[-overlap_tokens:] if overlap_tokens else []

    for para in paragraphs:
        ids = tokenizer.encode(para, add_special_tokens=False)
        if len(ids) > max_tokens:
            # Split this paragraph by token windows.
            for start in range(0, len(ids), max_tokens - overlap_tokens):
                window = ids[start:start + max_tokens]
                if not window:
                    break
                chunks.append(
                    tokenizer.decode(window, skip_special_tokens=True).strip()
                )
            current_ids = []
            continue
        if len(current_ids) + len(ids) > max_tokens:
            flush()
        current_ids.extend(ids)

    flush()
    return [c for c in chunks if c]


class RagIndex:
    """A loaded FAISS index plus its chunk metadata."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[Chunk],
        embedder: SentenceTransformer,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _INDEX_FILENAME))
        with open(directory / _CHUNKS_FILENAME, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(directory / _META_FILENAME, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_model": RAG_EMBEDDING_MODEL,
                    "chunk_count": len(self.chunks),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory: Path) -> "RagIndex":
        index = faiss.read_index(str(directory / _INDEX_FILENAME))
        with open(directory / _CHUNKS_FILENAME, "rb") as f:
            chunks = pickle.load(f)
        embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)
        return cls(index=index, chunks=chunks, embedder=embedder)


def build_index(
    knowledge_dir: Path,
    embedder: SentenceTransformer | None = None,
) -> RagIndex:
    """Build a FAISS index from all .md / .txt files in *knowledge_dir*."""
    if embedder is None:
        logger.info("loading embedding model: %s", RAG_EMBEDDING_MODEL)
        embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)

    max_tokens = embedder.max_seq_length
    logger.info("embedding max_seq_length = %d tokens", max_tokens)

    docs = sorted(
        [p for p in knowledge_dir.iterdir() if p.suffix in (".md", ".txt")]
    )
    if not docs:
        raise RuntimeError(f"No .md or .txt files found in {knowledge_dir}")

    all_chunks: list[Chunk] = []
    for doc_path in docs:
        text = doc_path.read_text(encoding="utf-8")
        title = _title_from_text(doc_path, text)
        source = _extract_field(text, "source") or "unknown"
        license_ = _extract_field(text, "license") or "unknown"
        provenance = _provenance_from_text(text)
        doc_id = doc_path.stem

        pieces = _chunk_text(text, embedder.tokenizer, max_tokens, RAG_CHUNK_OVERLAP_TOKENS)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    document_id=doc_id,
                    chunk_id=f"{doc_id}::{i:03d}",
                    text=piece,
                    title=title,
                    source=source,
                    url=None,
                    license=license_,
                    provenance=provenance,  # type: ignore[arg-type]
                    score=None,
                )
            )

    logger.info("built %d chunks from %d documents", len(all_chunks), len(docs))

    embeddings = embedder.encode(
        [c.text for c in all_chunks],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return RagIndex(index=index, chunks=all_chunks, embedder=embedder)