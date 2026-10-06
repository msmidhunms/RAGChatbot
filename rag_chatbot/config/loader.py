"""Build a validated ``RAGConfig`` from layered sources.

Precedence, lowest to highest:

1. built-in defaults (``schema.py``)
2. YAML file (``--config``; or the ``RAG_CONFIG`` env var)
3. env vars ``RAG__SECTION__KEY=value``  e.g. ``RAG__RETRIEVAL__K=10``
4. explicit overrides ``section.key=value``  e.g. ``--set retrieval.k=10``

Values in layers 3 and 4 are parsed as YAML scalars, so ``10`` is an int,
``true`` a bool and ``[0.5, 0.5]`` a list. Relative paths in the config are
resolved against the current working directory.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from rag_chatbot.config.schema import RAGConfig
from rag_chatbot.core.exceptions import ConfigError, MissingCredentialsError

ENV_PREFIX = "RAG__"
CONFIG_PATH_ENV = "RAG_CONFIG"


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    use_dotenv: bool = True,
) -> RAGConfig:
    """Load, merge and validate the configuration.

    ``env`` defaults to ``os.environ`` (after loading ``.env`` when
    ``use_dotenv`` is true); pass a dict in tests to isolate them.
    """
    if env is None:
        if use_dotenv:
            load_dotenv(override=False)
        env = os.environ

    if path is None:
        path = env.get(CONFIG_PATH_ENV) or None

    data: dict[str, Any] = read_yaml(path) if path else {}
    deep_merge(data, env_overrides(env))
    deep_merge(data, parse_overrides(overrides))

    try:
        return RAGConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, path)) from None


def read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level")
    return data


def env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in env.items():
        if not key.upper().startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        if parts:
            set_nested(out, parts, parse_value(raw), source=key)
    return out


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"override must look like section.key=value, got {item!r}")
        parts = [p for p in key.split(".") if p]
        set_nested(out, parts, parse_value(raw.strip()), source=item)
    return out


def parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw) if raw != "" else ""
    except yaml.YAMLError:
        return raw


def set_nested(target: dict[str, Any], parts: list[str], value: Any, *, source: str) -> None:
    node = target
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set {'.'.join(parts)} from {source}: '{part}' is not a section")
        node = child
    node[parts[-1]] = value


def deep_merge(base: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``other`` into ``base`` in place; ``other`` wins."""
    for key, value in other.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def format_validation_error(exc: ValidationError, path: str | Path | None) -> str:
    where = f" ({path})" if path else ""
    lines = [f"invalid configuration{where}:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def dump_config(cfg: RAGConfig) -> dict[str, Any]:
    """JSON-safe dict using the same keys as the YAML files."""
    return cfg.model_dump(mode="json", by_alias=True)


# ----------------------------------------------------------------- credentials
LLM_KEY_ENV = {"google_genai": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
EMBEDDING_KEY_ENV = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}


def required_env_vars(cfg: RAGConfig) -> dict[str, str]:
    """Env vars the selected components need, mapped to the setting that needs them."""
    needed: dict[str, str] = {}

    def need(var: str | None, why: str) -> None:
        if var:
            needed.setdefault(var, why)

    need(LLM_KEY_ENV.get(cfg.llm.provider), f"llm.provider={cfg.llm.provider}")
    need(EMBEDDING_KEY_ENV.get(cfg.embeddings.provider), f"embeddings.provider={cfg.embeddings.provider}")
    judge = cfg.evaluation.judge_llm
    if judge is not None:
        need(LLM_KEY_ENV.get(judge.provider), f"evaluation.judge_llm.provider={judge.provider}")
    if cfg.reranker.type == "cohere":
        need("COHERE_API_KEY", "reranker.type=cohere")
    if cfg.vector_store.type == "pgvector":
        need(cfg.vector_store.pgvector.connection_env, "vector_store.type=pgvector")
    if cfg.observability.langsmith:
        need("LANGSMITH_API_KEY", "observability.langsmith=true")
    return needed


def missing_env_vars(cfg: RAGConfig, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Subset of ``required_env_vars`` that is unset or empty."""
    env = os.environ if env is None else env
    return {var: why for var, why in required_env_vars(cfg).items() if not env.get(var)}


def ensure_env(var: str | None, why: str, env: Mapping[str, str] | None = None) -> None:
    """Raise ``MissingCredentialsError`` if ``var`` is unset or empty."""
    env = os.environ if env is None else env
    if var and not env.get(var):
        raise MissingCredentialsError(
            f"{var} is not set but required by {why}. Add it to .env or the environment."
        )
