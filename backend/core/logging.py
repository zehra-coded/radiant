"""
Structured logging for RADIANT.

Rules:
- INFO is the default level.
- Prompt bodies, evidence text and report text are NEVER logged at INFO.
- DEBUG is off unless explicitly enabled via LOG_LEVEL.
"""
from __future__ import annotations

import logging
import sys
from typing import Final

from backend.core.config import LOG_LEVEL

_CONFIGURED: bool = False

_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging() -> None:
    """Idempotently configure the root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL.upper())

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with RADIANT configuration applied."""
    configure_logging()
    return logging.getLogger(name)