"""
RADIANT FastAPI application.

Endpoints:
    GET  /health                liveness
    POST /analyze               multipart image + report -> RadiantResult
    POST /analyze/stream        same pipeline, Server-Sent Events
    POST /retrieve              {query, k} -> list[EvidenceItem]
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api import pipeline, state
from backend.core.config import (
    API_CORS_ORIGINS,
    API_MAX_UPLOAD_BYTES,
    API_TITLE,
    API_VERSION,
)
from backend.core.logging import get_logger
from backend.rag.retriever import retrieve as run_retrieve
from backend.schemas.result import EvidenceItem, RadiantResult

logger = get_logger(__name__)

app = FastAPI(title=API_TITLE, version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(API_CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _read_upload(upload: UploadFile) -> Path:
    """Read the uploaded image to a temp file. Enforce max size."""
    data = await upload.read()
    if len(data) > API_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {API_MAX_UPLOAD_BYTES} bytes",
        )
    suffix = Path(upload.filename or "upload.png").suffix or ".png"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    Path(tmp_name).write_bytes(data)
    import os
    os.close(fd)
    return Path(tmp_name)


@app.post("/analyze", response_model=RadiantResult)
async def analyze(
    image: UploadFile = File(...),
    report: str = Form(""),
    query: str | None = Form(None),
) -> RadiantResult:
    """Run the full pipeline. Returns a RadiantResult.

    Never raises on pipeline failure — a valid result with failure flags
    is returned instead.
    """
    image_path = await _read_upload(image)
    try:
        return pipeline.run(image_path, report, query)
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("temp image cleanup failed: %s", image_path)


def _sse_format(step: str, payload: dict | None) -> bytes:
    data = json.dumps({"step": step, "payload": payload}, default=str)
    return f"data: {data}\n\n".encode("utf-8")


@app.post("/analyze/stream")
async def analyze_stream(
    image: UploadFile = File(...),
    report: str = Form(""),
    query: str | None = Form(None),
) -> StreamingResponse:
    """Same pipeline, streaming Server-Sent Events after each step
    actually completes. The final event carries the full RadiantResult."""
    image_path = await _read_upload(image)

    def event_source():
        try:
            for step, partial in pipeline.run_streaming(image_path, report, query):
                payload = partial.model_dump() if partial is not None else None
                yield _sse_format(step, payload)
        finally:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("temp image cleanup failed: %s", image_path)

    return StreamingResponse(event_source(), media_type="text/event-stream")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


@app.post("/retrieve", response_model=list[EvidenceItem])
def retrieve_endpoint(body: RetrieveRequest) -> list[EvidenceItem]:
    """Direct evidence retrieval, independent of any image or report."""
    index = state.get_rag_index()
    rr = run_retrieve(index, body.query, k=body.k)
    return [EvidenceItem.from_chunk(c) for c in rr.chunks]