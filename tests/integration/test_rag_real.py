"""Integration test: build a real index from data/knowledge_base/ and
run real queries."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.rag.indexer import build_index
from backend.rag.retriever import retrieve

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


@pytest.mark.integration
def test_real_index_and_queries() -> None:
    index = build_index(FIXTURE_DIR)
    print()
    print("=" * 68)
    print(f"Index built: {len(index.chunks)} chunks "
          f"from {len(set(c.document_id for c in index.chunks))} docs")
    print(f"Embedding max_seq_length: {index.embedder.max_seq_length}")
    print("=" * 68)

    queries = [
        "pneumothorax",
        "possible pleural effusion",
        "enlarged heart",
        "consolidation in the lung",
        "completely unrelated query about weddings and food",
    ]
    for q in queries:
        r = retrieve(index, q, k=2)
        print()
        print(f"Query: {q}")
        if not r.chunks:
            print("  (no chunks met threshold)")
        for c in r.chunks:
            print(f"  [{c.score:.3f}] {c.chunk_id} ({c.provenance})")
            snippet = c.text[:100].replace("\n", " ")
            print(f"          {snippet}...")