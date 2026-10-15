"""
RAG schema.

Chunk is a single retrievable unit of text with full provenance metadata.
RetrievalResult wraps a list of Chunks with the query that produced them.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Provenance = Literal["DEVELOPMENT_FIXTURE", "LICENSED_SOURCE", "UNVERIFIED_SOURCE"]


class Chunk(BaseModel):
    """One retrievable chunk of text with metadata."""

    document_id: str = Field(..., description="Stable identifier for the source document.")
    chunk_id: str = Field(..., description="Stable identifier for this chunk within the document.")
    text: str = Field(..., description="The chunk text.")
    title: str = Field(..., description="Human-readable title of the source document.")
    source: str = Field(..., description="Source line from the fixture or document.")
    url: str | None = Field(default=None, description="URL if any.")
    license: str = Field(..., description="License line from the fixture or document.")
    provenance: Provenance = Field(
        ...,
        description=(
            "DEVELOPMENT_FIXTURE for project-authored test data. "
            "LICENSED_SOURCE for verified external sources. "
            "UNVERIFIED_SOURCE for anything else — these are blocked from generation."
        ),
    )
    score: float | None = Field(
        default=None,
        description="Similarity score from retrieval. None if the chunk has not been retrieved.",
    )


class RetrievalResult(BaseModel):
    """Complete output of a retrieve() call."""

    query: str = Field(..., description="The query text.")
    chunks: list[Chunk] = Field(default_factory=list)
    min_similarity: float = Field(
        ..., description="The similarity threshold applied (prototype)."
    )
    notes: list[str] = Field(default_factory=list)