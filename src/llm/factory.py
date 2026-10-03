"""Build a LangChain chat model from settings + (env-based or keyring) key.

Resolution order:
    1. ``MINIMAX_API_KEY`` env var (or whatever the deployment injected)
    2. OS keyring (set via legacy UI / CLI)
    3. raise

``MINIMAX_BASE_URL`` (optional) lets the deployment point at an
Anthropic-compatible proxy without code changes.

``MINIMAX_MODEL`` is a *deployment default* — once the user picks a model
in the UI, their choice wins. Only when nothing else is set do we fall
back to the env-injected model name.
"""
from __future__ import annotations

import asyncio
import atexit
from collections import OrderedDict
from functools import lru_cache
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.constants import DEFAULT_LLM_MODELS, DEFAULT_LLM_PROVIDER
from config.settings import get_settings
from src.core.logging import logger
from src.security.keyring_store import KeyringStore


DEFAULT_MODEL_FALLBACK = DEFAULT_LLM_MODELS[DEFAULT_LLM_PROVIDER]


def get_api_key(provider: str, keyring_service: str) -> Optional[str]:
    """Read the API key from the OS keyring (no in-memory cache by default)."""
    account = f"{provider}_api_key"
    store = KeyringStore(service=keyring_service)
    return store.get(account)


def _resolve_api_key(provider: str) -> Optional[str]:
    """Env var wins; fall back to keyring for the legacy path."""
    s = get_settings()
    if s.llm_api_key:
        return s.llm_api_key
    return get_api_key(provider, s.keyring_service)


def _resolve_model(model: Optional[str]) -> str:
    """Resolution: explicit call > user-config > env default.

    ``MINIMAX_MODEL`` is a *deployment default* — once the user picks a
    model in the UI (saved to ``UserConfig.llm_model``), their choice
    wins. Only when the user hasn't picked anything do we fall back to
    the env-injected default.
    """
    s = get_settings()
    return model or s.llm_model or s.llm_model_override or DEFAULT_MODEL_FALLBACK


def _base_url_kwargs(provider: str) -> dict:
    """Map ``MINIMAX_BASE_URL`` to the per-provider kwarg.

    Different LangChain clients use different param names:
        ChatOpenAI     → ``openai_api_base``
        ChatAnthropic  → ``anthropic_api_url``
    """
    s = get_settings()
    if not s.llm_base_url:
        return {}
    return {
        "openai": {"openai_api_base": s.llm_base_url},
        "anthropic": {"anthropic_api_url": s.llm_base_url},
    }.get(provider, {})


# Process-wide client cache. Five LLM nodes (route / rewrite / grade /
# hallucination / generate) each used to construct a fresh
# ChatAnthropic/ChatOpenAI per call: Pydantic validation + httpx
# client init + provider-config parsing, every time. Memoizing on the
# tuple that fully determines the client (provider, model name,
# temperature, thinking flag, base URL, API key) makes the warm path a
# single dict lookup. API key rotation naturally invalidates the cache
# because the key is part of the key — when the user rotates keys via
# settings, the next call sees a different key and constructs a fresh
# client. Tests call ``reset_llm_cache()`` to drop stale instances
# after monkey-patching settings.
#
# v2.0.28.4 P1-4 — eviction hygiene: the underlying httpx.AsyncClient
# in each ChatAnthropic/ChatOpenAI holds a connection pool. Pre-PR-4
# when ``cache_clear()`` ran (or the LRU evicted a stale entry on key
# rotation), the old client was garbage-collected eventually but its
# ``aclose()`` was never called — so the connection pool + FD stayed
# open until Python's cyclic GC caught up. After 50 key rotations
# (e.g. an operator cycling through credentials during incident
# response) the process would run out of file descriptors.
#
# Fix: on cache eviction (``OrderedDict.popitem(last=False)``) or
# ``reset_llm_cache()`` / ``close_all_llm_clients()``, we explicitly
# call ``aclose()`` on the evicted client. The OrderedDict itself
# holds the only strong references — once popped, the GC can collect
# the client normally.


def _try_close_client(client: BaseChatModel) -> None:
    """Best-effort async-close of an evicted LLM client.

    Some LangChain client versions expose ``aclose()`` (async close of
    the underlying httpx.AsyncClient); some don't. We try both shapes
    and log at DEBUG if neither exists — never raise, since this
    runs on eviction and we don't want a bad close to crash the
    caller that triggered the cache mutation.
    """
    try:
        if hasattr(client, "aclose"):
            aclose = client.aclose
            # ``aclose`` may be sync or async depending on LangChain
            # version. Try the async path first (newer versions).
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(aclose())
                else:
                    loop.run_until_complete(aclose())
            except RuntimeError:
                # No running loop (e.g. process shutdown) — fall back
                # to fire-and-forget via asyncio.run.
                try:
                    asyncio.run(aclose())
                except Exception:
                    pass
        elif hasattr(client, "close"):
            client.close()
    except Exception as exc:
        logger.debug(
            f"op=llm_cache_eviction close_failed type={type(exc).__name__}: {exc}"
        )


def _on_lru_cache_evict(evicted: BaseChatModel) -> None:
    """Hook called from ``_cached_chat_model`` when the LRU evicts
    or we call ``reset_llm_cache()``. Best-effort close + log so a
    future FD-exhaustion operator can correlate.
    """
    _try_close_client(evicted)


# v2.0.28.4 P1-4 — manual cache so we can hook eviction. ``@lru_cache``
# doesn't expose an eviction callback, so we wrap an ``OrderedDict``
# (insertion-order preserved, ``move_to_end`` for LRU touch, and
# ``popitem(last=False)`` for evict-oldest). Maxsize matches the
# previous ``@lru_cache(maxsize=32)`` — keeping it identical avoids
# a behavior change for warm-path callers.
_LLM_CACHE_MAXSIZE = 32
_LLM_CACHE: "OrderedDict[tuple, BaseChatModel]" = OrderedDict()


def _cached_chat_model(
    provider: str,
    resolved_model: str,
    temperature: float,
    enable_thinking: bool,
    base_url: str,
    api_key: str,
) -> BaseChatModel:
    """Memoize the LangChain chat model on the tuple of fields that
    fully determine the client (provider, model name, temperature,
    thinking flag, base URL, API key).

    v2.0.28.4 P1-4 — replaced the previous ``@lru_cache`` so we can
    hook eviction and close the underlying httpx client. See
    ``_on_lru_cache_evict`` above for the FD-leak rationale.
    """
    cache_key = (
        provider,
        resolved_model,
        temperature,
        enable_thinking,
        base_url,
        api_key,
    )
    cached = _LLM_CACHE.get(cache_key)
    if cached is not None:
        # Touch for LRU semantics.
        _LLM_CACHE.move_to_end(cache_key)
        return cached

    # Construct a fresh client (same logic as pre-PR-4).
    thinking: Optional[dict] = None
    if enable_thinking and provider == "anthropic":
        # Budget tuned down from 8000 → 1500 because every thinking token
        # is generated serially before the first visible token, so a big
        # budget directly inflates TTFB. 1500 covers the rewrite /
        # planning pass for retrieval-grounded answers; anything more just
        # spends latency on marginal reasoning quality.
        thinking = {"type": "enabled", "budget_tokens": 1500}
        # Anthropic requires temperature=1.0 when extended thinking is on;
        # fold it into the cache key (caller passes 1.0 explicitly when
        # enable_thinking=True so the same cache slot is hit).
    if provider == "openai":
        kwargs: dict = {
            "model": resolved_model,
            "api_key": api_key,
            "temperature": temperature,
            "streaming": True,
        }
        if base_url:
            kwargs["openai_api_base"] = base_url
        client: BaseChatModel = ChatOpenAI(**kwargs)
    elif provider == "anthropic":
        kwargs = {
            "model": resolved_model,
            "api_key": api_key,
            "temperature": temperature,
            "streaming": True,
            "thinking": thinking,
        }
        if base_url:
            kwargs["anthropic_api_url"] = base_url
        client = ChatAnthropic(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    # Evict oldest if at capacity, closing the evicted client.
    while len(_LLM_CACHE) >= _LLM_CACHE_MAXSIZE:
        try:
            oldest_key, oldest_client = _LLM_CACHE.popitem(last=False)
        except KeyError:
            break
        _on_lru_cache_evict(oldest_client)

    _LLM_CACHE[cache_key] = client
    return client


def reset_llm_cache() -> None:
    """Drop all cached LangChain clients. Tests only.

    v2.0.28.4 P1-4 — also closes each evicted client's underlying
    httpx pool so tests don't leak FDs across test runs.
    """
    while _LLM_CACHE:
        try:
            _, evicted = _LLM_CACHE.popitem()
        except KeyError:
            break
        _on_lru_cache_evict(evicted)


def close_all_llm_clients() -> None:
    """Synchronously close every currently cached LLM client.

    Called from FastAPI lifespan shutdown (PR-4 of this Item) and from
    ``atexit`` as a last-resort fallback for non-FastAPI entry points
    (e.g. ``main.py`` if it ever imports the cache). Pre-PR-4 the
    process just exited and let Python's GC close the httpx clients
    eventually — usually fine, but during rapid restart loops
    (development with ``--reload``) the FD count would creep up.
    """
    while _LLM_CACHE:
        try:
            _, evicted = _LLM_CACHE.popitem()
        except KeyError:
            break
        _try_close_client(evicted)


# Best-effort cleanup at interpreter shutdown. This handles the case
# where FastAPI's lifespan teardown didn't run (e.g. SIGKILL, ``atexit``
# chain never reached). Cheap to register; ``atexit`` handlers run
# after main exits and before Python's final GC.
atexit.register(close_all_llm_clients)


def build_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    enable_thinking: bool = False,
) -> BaseChatModel:
    """Build (or fetch from cache) a LangChain chat model.

    ``enable_thinking=True`` opts the model into extended reasoning and
    surfaces it via streaming ``thinking`` blocks. Today only Anthropic
    honors it: ``thinking={"type": "enabled", "budget_tokens": 1500}`` is
    passed to ``ChatAnthropic`` and ``temperature`` is forced to ``1.0``
    (Anthropic's requirement when extended thinking is on). Other
    providers ignore the flag — OpenAI's chat.completions path drops
    reasoning text silently in ``langchain_openai==1.6.0``, so opting
    in here does nothing for them.

    Caching: the underlying ``ChatAnthropic``/``ChatOpenAI`` is shared
    across calls with the same effective config — see
    ``_cached_chat_model``. Every node used to build a fresh client
    per call (5 nodes × per-turn = 5 wasted constructions), paying
    Pydantic validation + httpx client init each time. The cache key
    includes the API key so rotation through the settings dialog
    naturally invalidates stale entries.
    """
    s = get_settings()
    provider = provider or s.llm_provider
    resolved_model = _resolve_model(model)
    api_key = _resolve_api_key(provider)
    if not api_key:
        raise RuntimeError(
            f"No API key found for provider {provider!r}. "
            f"Set MINIMAX_API_KEY in your .env (preferred) or save one "
            f"via the OS keyring."
        )

    # Anthropic requires temperature=1.0 when extended thinking is on;
    # rewrite before computing the cache key so the thinking and
    # non-thinking variants of the same model don't collide.
    if enable_thinking and provider == "anthropic":
        temperature = 1.0

    base_url = s.llm_base_url or ""
    return _cached_chat_model(
        provider,
        resolved_model,
        temperature,
        enable_thinking,
        base_url,
        api_key,
    )


def build_cheap_model(temperature: float = 0.0) -> BaseChatModel:
    """A small, fast model for grading/routing/classification.

    P1-2-fix: previously hardcoded to ``claude-3-5-haiku-latest``
    regardless of provider, which silently routed every
    route / rewrite / grade / hallucination call through a model the
    deployment didn't actually configure — the cascade of
    "fail-open" / "smart-quote repair" / "truncation recovery" code
    paths were workarounds for the resulting garbage JSON. The cheap
    model is now configurable via :attr:`Settings.llm_cheap_model`
    (persisted in ``settings.json``) with a per-provider default that
    matches what the deployment actually serves.
    """
    s = get_settings()
    # Per-provider default: matches the deployment's main model when
    # possible (we know that works), falls back to a small OpenAI
    # model when the user is on the openai provider. For an
    # Anthropic-direct deployment without the MiniMax proxy, haiku is
    # the right answer; for a MiniMax-proxy deployment the user should
    # override via Settings.
    cheap_map = {
        "openai": "gpt-4o-mini",
        "anthropic": "MiniMax-M3",  # same as the deployment default
    }
    model = (
        s.llm_cheap_model
        or cheap_map.get(s.llm_provider)
        or s.llm_model
    )
    return build_chat_model(
        model=model,
        temperature=temperature,
    )


__all__ = ["build_chat_model", "build_cheap_model", "get_api_key", "reset_llm_cache"]
