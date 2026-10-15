"""
Health endpoint test.

If FastAPI/httpx are not installed yet, the test is skipped with a clear
reason — matching the Stage 1 "Health verification rule".
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed yet")
httpx = pytest.importorskip("httpx", reason="httpx not installed yet")

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}