"""Chat model factory: one config section -> a LangChain chat model.

All providers go through ``init_chat_model``; the per-provider factories only
translate the shared ``LLMConfig`` fields into that provider's kwargs and
check the credential and integration package up front, so failures name
the fix instead of surfacing deep inside a request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel

from rag_chatbot.config.loader import LLM_KEY_ENV, ensure_env
from rag_chatbot.config.schema import LLMConfig
from rag_chatbot.core.registry import Registry, require

LLMFactory = Callable[[LLMConfig], BaseChatModel]
LLM_PROVIDERS: Registry[LLMFactory] = Registry("llm provider")


def build_llm(cfg: LLMConfig, *, env: Mapping[str, str] | None = None) -> BaseChatModel:
    ensure_env(LLM_KEY_ENV.get(cfg.provider), f"llm.provider={cfg.provider}", env)
    return LLM_PROVIDERS.get(cfg.provider)(cfg)


def _init(cfg: LLMConfig, **kwargs: Any) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return init_chat_model(
        model=cfg.model, model_provider=cfg.provider, temperature=cfg.temperature, **kwargs
    )


def _hosted(cfg: LLMConfig) -> BaseChatModel:
    return _init(
        cfg,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        base_url=cfg.base_url,
    )


@LLM_PROVIDERS.register("google_genai")
def _google(cfg: LLMConfig) -> BaseChatModel:
    require("langchain_google_genai")
    return _hosted(cfg)


@LLM_PROVIDERS.register("openai")
def _openai(cfg: LLMConfig) -> BaseChatModel:
    require("langchain_openai", "openai")
    return _hosted(cfg)


@LLM_PROVIDERS.register("anthropic")
def _anthropic(cfg: LLMConfig) -> BaseChatModel:
    require("langchain_anthropic", "anthropic")
    return _hosted(cfg)


@LLM_PROVIDERS.register("ollama")
def _ollama(cfg: LLMConfig) -> BaseChatModel:
    require("langchain_ollama", "local")
    # Ollama names the output cap num_predict and takes the timeout via the HTTP client.
    return _init(
        cfg,
        num_predict=cfg.max_tokens,
        base_url=cfg.base_url,
        client_kwargs={"timeout": cfg.timeout} if cfg.timeout else None,
    )
