import logging

import pytest

from rag.core import MissingDependencyError, Registry, RegistryError, require
from rag.core.logging import configure_logging, get_logger


def test_registry_register_and_get():
    reg: Registry = Registry("splitter")

    @reg.register("Recursive")
    def build():
        return "built"

    assert reg.get("recursive") is build
    assert reg.get("RECURSIVE")() == "built"
    assert "recursive" in reg
    assert reg.available() == ["recursive"]
    assert list(reg) == ["recursive"] and len(reg) == 1


def test_registry_unknown_lists_available():
    reg: Registry = Registry("store")
    reg.add("chroma", object())
    reg.add("faiss", object())
    with pytest.raises(RegistryError, match="unknown store 'qdrant'. Available: chroma, faiss"):
        reg.get("qdrant")


def test_registry_duplicate_and_override():
    reg: Registry = Registry("loader")
    reg.add("pdf", 1)
    with pytest.raises(RegistryError, match="already registered"):
        reg.add("PDF", 2)
    reg.add("pdf", 3, override=True)
    assert reg.get("pdf") == 3


def test_require_missing_names_extra():
    with pytest.raises(MissingDependencyError, match=r"pip install 'ragchatbot\[qdrant\]'"):
        require("definitely_not_installed_module_xyz", "qdrant")
    # also catchable as ImportError
    with pytest.raises(ImportError):
        require("definitely_not_installed_module_xyz", "qdrant")


def test_require_present():
    assert require("json").dumps({}) == "{}"


def test_configure_logging_idempotent():
    logger = configure_logging("DEBUG")
    configure_logging("INFO")
    assert logger.level == logging.INFO
    assert sum(h.get_name() == "rag-default" for h in logger.handlers) == 1
    assert get_logger("ingestion").name == "rag.ingestion"
    assert get_logger("rag.stores").name == "rag.stores"


def test_require_core_dependency_message():
    with pytest.raises(MissingDependencyError, match="pip install -r requirements.txt"):
        require("definitely_not_installed_module_xyz")


def test_setup_tracing(monkeypatch):
    from rag.config.schema import ObservabilityConfig
    from rag.observability.tracing import setup_tracing

    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    setup_tracing(ObservabilityConfig(langsmith=False))
    import os

    assert "LANGSMITH_TRACING" not in os.environ
    setup_tracing(ObservabilityConfig(langsmith=True, project="p1"))
    assert os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_PROJECT"] == "p1"


def test_logging_follows_stderr_swaps(capsys):
    import io
    import sys

    logger = configure_logging("INFO")
    swapped = io.StringIO()
    old, sys.stderr = sys.stderr, swapped
    try:
        get_logger("t").info("hello")
    finally:
        sys.stderr = old
    assert "hello" in swapped.getvalue()
    assert logger.handlers
