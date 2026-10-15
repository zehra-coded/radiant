"""
Quota reader for the AMD Token Factory usage endpoint.

Never fabricates a value. Returns None when the endpoint is not
available or not authorized.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from backend.core.config import LLM_BASE_URL, RADEON_API_KEY
from backend.core.logging import get_logger

logger = get_logger(__name__)

_CACHE_TTL_S = 60


@dataclass
class QuotaSnapshot:
    remaining_pts: float | None
    used_pts: float | None
    fetched_at: float  # unix timestamp


_cache: QuotaSnapshot | None = None


def get_quota(force: bool = False) -> QuotaSnapshot | None:
    """Return cached quota info, refreshing if stale.

    Returns None if the endpoint is unavailable or unauthorized.
    """
    global _cache
    now = time.time()
    if not force and _cache and (now - _cache.fetched_at) < _CACHE_TTL_S:
        return _cache

    if not RADEON_API_KEY or not LLM_BASE_URL:
        return None

    import httpx  # local import to avoid hard dependency at module load

    base = LLM_BASE_URL.rstrip("/")
    url = f"{base}/usage?include_recent=true"
    try:
        r = httpx.get(
            url,
            headers={"Authorization": f"Bearer {RADEON_API_KEY}"},
            timeout=10.0,
        )
        if r.status_code != 200:
            logger.warning("quota endpoint returned %d", r.status_code)
            return None
        data = r.json()
    except Exception as exc:
        logger.warning("quota fetch failed: %s", exc)
        return None

    remaining = _first_present(data, ("remaining", "remaining_pts", "remainingPoints"))
    used = _first_present(data, ("used", "used_pts", "usedPoints"))

    snap = QuotaSnapshot(
        remaining_pts=_to_float(remaining),
        used_pts=_to_float(used),
        fetched_at=now,
    )
    _cache = snap
    return snap


def _first_present(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None