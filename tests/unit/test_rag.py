"""Unit tests for the RAG indexer and retriever.

Uses a tiny fixture knowledge base and the real embedding model — but
the tests are small (5 documents) and fast. No network calls during
the test itself; the embedding model may already be cached from
earlier runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.rag.indexer import build_index
from backend.rag.retriever import retrieve
from backend.rag.schema import Chunk

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


@pytest.fixture(scope="module")
def index():
    if not FIXTURE_DIR.exists() or not list(FIXTURE_DIR.glob("*.md")):
        pytest.skip("No knowledge_base fixtures present — RAG tests skipped.")
    return build_index(FIXTURE_DIR)


def test_index_builds_with_expected_chunk_count(index) -> None:
    assert len(index.chunks) > 0
    for c in index.chunks:
        assert isinstance(c, Chunk)
        assert c.document_id
        assert c.chunk_id
        assert c.text


def test_relevant_query_returns_chunks(index) -> None:
    r = retrieve(index, "pneumothorax", k=3)
    assert r.chunks, "expected at least one chunk"
    # At least one of the top chunks should mention pneumothorax.
    joined = " ".join(c.text.lower() for c in r.chunks)
    assert "pneumothorax" in joined


def test_irrelevant_query_returns_empty(index) -> None:
    r = retrieve(index, "xyzzy plugh qwerty nonsense", k=3, min_similarity=0.99)
    assert r.chunks == []
    assert any("no chunks" in n for n in r.notes)


def test_empty_query_returns_empty(index) -> None:
    r = retrieve(index, "", k=3)
    assert r.chunks == []
    assert any("empty" in n for n in r.notes)


def test_metadata_preserved(index) -> None:
    r = retrieve(index, "pleural effusion", k=1)
    assert r.chunks
    c = r.chunks[0]
    assert c.document_id
    assert c.chunk_id
    assert c.license
    assert c.provenance in ("DEVELOPMENT_FIXTURE", "LICENSED_SOURCE", "UNVERIFIED_SOURCE")


def test_no_chunk_exceeds_token_limit(index) -> None:
    limit = index.embedder.max_seq_length
    for c in index.chunks:
        ids = index.embedder.tokenizer.encode(c.text, add_special_tokens=False)
        assert len(ids) <= limit, (
            f"chunk {c.chunk_id} has {len(ids)} tokens, limit {limit}"
        )


def test_index_save_load_round_trip(index, tmp_path: Path) -> None:
    from backend.rag.indexer import RagIndex

    save_dir = tmp_path / "rag"
    index.save(save_dir)
    loaded = RagIndex.load(save_dir)
    assert len(loaded.chunks) == len(index.chunks)
    r = retrieve(loaded, "pneumothorax", k=1)
    assert r.chunks