"""Logging setup for the rag package (stdlib only)."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_NOISY = ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "faiss")
_HANDLER_NAME = "rag-default"


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stderr`` is at emit time (survives stream swaps)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore[override]
    def stream(self) -> object:
        return sys.stderr

    @stream.setter
    def stream(self, value: object) -> None:
        pass


def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """Attach one stderr handler to the ``rag`` logger; safe to call repeatedly."""
    logger = logging.getLogger("rag")
    logger.setLevel(level)
    if not any(h.get_name() == _HANDLER_NAME for h in logger.handlers):
        handler = _StderrHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``rag`` logger (``rag.<name>``)."""
    if name == "rag" or name.startswith("rag."):
        return logging.getLogger(name)
    return logging.getLogger(f"rag.{name}")
