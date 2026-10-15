"""
Shared application state: the vision model and the RAG index.

Both are expensive to construct. They are created once and cached for
the process lifetime. Lazy so tests that never touch them pay no cost.
"""
from __future__ import annotations

from backend.core.config import DATA_ROOT
from backend.core.logging import get_logger
from backend.rag.indexer import RagIndex, build_index
from backend.vision.model import VisionModel

logger = get_logger(__name__)

_vision_model: VisionModel | None = None
_rag_index: RagIndex | None = None


def get_vision_model() -> VisionModel:
    global _vision_model
    if _vision_model is None:
        logger.info("constructing vision model (one-time)")
        _vision_model = VisionModel()
    return _vision_model


def get_rag_index() -> RagIndex:
    global _rag_index
    if _rag_index is None:
        logger.info("building RAG index (one-time)")
        _rag_index = build_index(DATA_ROOT / "knowledge_base")
    return _rag_index


def reset() -> None:
    """For tests: clear the cached singletons."""
    global _vision_model, _rag_index
    _vision_model = None
    _rag_index = None