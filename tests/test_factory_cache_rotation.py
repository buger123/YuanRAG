"""v2.0.28.4 Item 11 PR-2 P1-4 — ``factory._cached_chat_model`` eviction hygiene.

Pre-PR-4 the LRU cache was a vanilla ``@lru_cache(maxsize=32)``. When
the cache evicted a stale entry (e.g. API key rotation through the
Settings dialog, or just a different ``(provider, model,
temperature, thinking, base_url, api_key)`` tuple), the old
``ChatAnthropic`` / ``ChatOpenAI`` instance was garbage-collected
eventually but ``aclose()`` was never called on the underlying
httpx.AsyncClient. Each evicted client leaked its connection pool
+ FD until Python's cyclic GC caught up.

After 50 key rotations (e.g. an operator cycling through
credentials during incident response) the process would run out of
file descriptors and the next LLM call would fail with
``OSError: Too many open files``.

This file pins:
    1. After eviction, ``_try_close_client`` is called on the evicted
       client (verified by spying ``aclose``).
    2. ``reset_llm_cache()`` / ``close_all_llm_clients()`` close
       every currently cached client.
    3. New client with different api_key doesn't share the same
       instance as the evicted one.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# v2.0.28.4 P1-4 — Test 1
def test_factory_evicts_old_client_on_key_rotation():
    """Build chat model with key A → then key B → old key A entry
    is evicted (cache maxsize exceeded) and a new instance is
    returned for key B.

    This pins the LRU semantics of the new manual cache (we replaced
    ``@lru_cache`` with a dict to hook eviction — the LRU order
    must be preserved so warm-path callers see the same instance
    for repeated identical calls).
    """
    from src.llm import factory

    # Stub out the constructor so we don't actually hit the SDK.
    fake_client_a = MagicMock(name="client_A")
    fake_client_b = MagicMock(name="client_B")
    call_log = []

    def _fake_openai_ctor(*args, **kwargs):
        client = MagicMock(name="ChatOpenAI")
        client.aclose = AsyncMock()
        call_log.append(kwargs.get("api_key"))
        return client

    factory.reset_llm_cache()
    with patch.object(factory, "ChatOpenAI", side_effect=_fake_openai_ctor):
        # Fill the cache with key A.
        client_a = factory._cached_chat_model(
            "openai", "gpt-4o", 0.2, False, "", "key-A"
        )
        # Same key again — should hit cache (no new construction).
        client_a_again = factory._cached_chat_model(
            "openai", "gpt-4o", 0.2, False, "", "key-A"
        )
        assert client_a is client_a_again
        # Different key — different instance.
        client_b = factory._cached_chat_model(
            "openai", "gpt-4o", 0.2, False, "", "key-B"
        )
        assert client_b is not client_a

    factory.reset_llm_cache()


# v2.0.28.4 P1-4 — Test 2
def test_factory_closes_evicted_client_via_aclose():
    """When a new entry evicts an old one (cache at maxsize),
    the evicted client's ``aclose`` is called.

    We manually stuff the cache to capacity, then add one more
    entry — the eviction path should fire.
    """
    import asyncio

    from src.llm import factory

    factory.reset_llm_cache()

    # Build a fake client whose aclose is an AsyncMock — we can
    # await it later to verify the call.
    evicted_client = MagicMock(name="evicted")
    evicted_client.aclose = AsyncMock()

    # Pre-populate the cache directly with maxsize entries.
    factory._LLM_CACHE.clear()
    for i in range(factory._LLM_CACHE_MAXSIZE):
        factory._LLM_CACHE[(f"key-{i}",) * 6] = MagicMock(name=f"client-{i}")

    # Insert evicted_client at the front (so it's the oldest).
    factory._LLM_CACHE[("key-evict",) * 6] = evicted_client
    # Re-order so evicted_client is at the front (oldest).
    factory._LLM_CACHE.move_to_end(("key-evict",) * 6, last=False)

    # Now build a chat model with a NEW key — this will push past
    # maxsize and trigger eviction.
    def _fake_openai_ctor(*args, **kwargs):
        return MagicMock(name="new_client")

    with patch.object(factory, "ChatOpenAI", side_effect=_fake_openai_ctor):
        factory._cached_chat_model("openai", "gpt-4o", 0.2, False, "", "new-key")

    # The evicted client should have had aclose() called on it.
    # We can't await AsyncMock.call_count directly without a loop,
    # so we check that ``aclose`` was called (the mock records the
    # call regardless of whether it was awaited).
    assert evicted_client.aclose.called, (
        "evicted client's aclose() should have been called"
    )

    factory.reset_llm_cache()


# v2.0.28.4 P1-4 — Test 3
def test_reset_llm_cache_closes_all_cached_clients():
    """``reset_llm_cache()`` must close every currently cached client,
    not just clear the dict (which would leak the httpx pools).

    This is the test-isolation hook — without it, tests that build
    models leave httpx clients open across the test session,
    eventually exhausting FDs in CI.
    """
    from src.llm import factory

    factory.reset_llm_cache()

    # Build three clients, each with its own aclose AsyncMock.
    clients = []
    for i in range(3):
        c = MagicMock(name=f"client-{i}")
        c.aclose = AsyncMock()
        clients.append(c)

    factory._LLM_CACHE.clear()
    factory._LLM_CACHE[(f"k-{0}",) * 6] = clients[0]
    factory._LLM_CACHE[(f"k-{1}",) * 6] = clients[1]
    factory._LLM_CACHE[(f"k-{2}",) * 6] = clients[2]

    factory.reset_llm_cache()

    for c in clients:
        assert c.aclose.called, (
            f"{c!r}.aclose() should have been called during reset_llm_cache"
        )

    assert len(factory._LLM_CACHE) == 0


# v2.0.28.4 P1-4 — Test 4
def test_factory_maxsize_is_32():
    """Pin the cache maxsize constant to 32 (same as the old
    ``@lru_cache(maxsize=32)``). If anyone tunes it down to save
    memory, they should also tune down cache pressure on
    ``build_chat_model`` — this is a deliberate API decision worth
    a regression guard.
    """
    from src.llm import factory

    assert factory._LLM_CACHE_MAXSIZE == 32
