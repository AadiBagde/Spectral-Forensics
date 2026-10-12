"""Lightweight logging helpers."""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI scripts."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; ensure basic config exists."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
