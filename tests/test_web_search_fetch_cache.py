"""Tests for the v2.0.18 TTL+LRU bounded ``_FETCH_CACHE``.

What we lock here
-----------------
1. TTL eviction — an entry past ``_CACHE_TTL_SECONDS`` is no longer
   returned (re-fetched on next request).
2. LRU cap — adding more than ``_CACHE_MAX_ENTRIES`` evicts the
   least-recently-used entry.
3. URL fragment isolation — ``#a`` and ``#b`` cache separately
   (RFC 3986 §3.5: fragments are part of the URL).
4. Cancellation — a fetch cancelled mid-flight does not pollute the
   cache (the entry was never ``set()``).
5. Cache size assertion — N distinct URLs fetched → cache size == N
   (no accidental deduplication on URL strings).
6. Negative cache — a failed fetch (``_fetch_one`` returning ``None``)
   is NOT cached — a follow-up re-tries rather than reusing a stale
   failure marker for the full TTL window.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Direct unit tests for the cache primitive (no I/O)
# ---------------------------------------------------------------------------


def test_ttl_evicts_after_window(monkeypatch):
    """Insert an entry, advance the clock past the TTL, expect miss."""
    from src.web_search.fetch import _TTLCache

    cache = _TTLCache(max_entries=10, ttl_seconds=10)
    fake_now = [1000.0]

    def now() -> float:
        return fake_now[0]

    monkeypatch.setattr("time.monotonic", now)

    cache.set("https://x.com/a", "body-a")
    # Inside TTL — hit.
    assert cache.get("https://x.com/a") == "body-a"

    # Advance 11 s — past the 10-s TTL.
    fake_now[0] += 11.0
    assert cache.get("https://x.com/a") is None


def test_lru_cap_evicts_least_recently_used(monkeypatch):
    """Beyond the cap, the oldest entry is evicted first."""
    from src.web_search.fetch import _TTLCache

    cache = _TTLCache(max_entries=3, ttl_seconds=600)
    fake_now = [1000.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_now[0])

    cache.set("a", "1")
    fake_now[0] += 1
    cache.set("b", "2")
    fake_now[0] += 1
    cache.set("c", "3")
    # Touch "a" so "b" becomes LRU.
    assert cache.get("a") == "1"
    fake_now[0] += 1

    # Insert a 4th entry — "b" should be evicted.
    cache.set("d", "4")
    assert len(cache) == 3
    assert cache.get("a") == "1"
    assert cache.get("c") == "3"
    assert cache.get("d") == "4"
    assert cache.get("b") is None  # evicted


def test_url_fragments_are_distinct_keys():
    """Per RFC 3986 §3.5, ``#a`` and ``#b`` are distinct URLs."""
    from src.web_search.fetch import _TTLCache

    cache = _TTLCache(max_entries=10, ttl_seconds=600)
    cache.set("https://x.com/page#s1", "body-s1")
    cache.set("https://x.com/page#s2", "body-s2")
    assert cache.get("https://x.com/page#s1") == "body-s1"
    assert cache.get("https://x.com/page#s2") == "body-s2"
    assert len(cache) == 2


def test_get_returns_none_for_missing_key():
    from src.web_search.fetch import _TTLCache

    cache = _TTLCache()
    assert cache.get("nope") is None
    assert "nope" not in cache


def test_set_overwrites_existing_key():
    """Same URL twice → second value wins; cap not exceeded."""
    from src.web_search.fetch import _TTLCache

    cache = _TTLCache(max_entries=10, ttl_seconds=600)
    cache.set("u", "v1")
    cache.set("u", "v2")
    assert cache.get("u") == "v2"
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# Integration with ``_fetch_one`` (cancellation, negative cache, size)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_fetch_does_not_pollute_cache():
    """A cancelled ``_fetch_one`` must not insert a partial entry.

    Without this guard, a cancellation at the network boundary could
    cache whatever was in-flight (which is either empty bytes or a
    truncated body) and then return that for the next 10 min."""
    from src.web_search import fetch as fetch_mod

    # Reset to a known-empty state.
    fetch_mod._FETCH_CACHE.clear()

    def raise_cancelled(*args, **kwargs):
        """Sync raise — ``_fetch_url_sync`` is a sync function called
        from inside ``asyncio.to_thread`` so it must raise the exception
        synchronously. ``asyncio.to_thread`` re-raises it at the await
        site as ``asyncio.CancelledError``, which the ``_fetch_one``
        outer try/except propagates back to the caller."""
        raise asyncio.CancelledError()

    with patch.object(fetch_mod, "_fetch_url_sync", side_effect=raise_cancelled):
        with pytest.raises(asyncio.CancelledError):
            await fetch_mod._fetch_one("https://example.com/cancel", 4000)

    # Nothing was cached.
    assert "https://example.com/cancel" not in fetch_mod._FETCH_CACHE


@pytest.mark.asyncio
async def test_failed_fetch_is_not_cached():
    """A ``_fetch_one`` returning ``None`` (anti-bot / 5xx / DNS fail)
    must not be cached as ``None`` — a follow-up re-tries instead of
    reusing the cached failure marker for 10 min."""
    from src.web_search import fetch as fetch_mod

    fetch_mod._FETCH_CACHE.clear()

    def fail_sync(url, timeout):
        return ("", False)  # failure path in _fetch_url_sync

    with patch.object(fetch_mod, "_fetch_url_sync", side_effect=fail_sync):
        result = await fetch_mod._fetch_one("https://example.com/down", 4000)

    assert result is None
    assert "https://example.com/down" not in fetch_mod._FETCH_CACHE


@pytest.mark.asyncio
async def test_successful_fetch_caches_result(monkeypatch):
    """A successful extraction lands in the cache."""
    from src.web_search import fetch as fetch_mod

    fetch_mod._FETCH_CACHE.clear()

    def ok_sync(url, timeout):
        # Body > 200 chars, with a tiny article container so the regex
        # extraction picks it up.
        body = (
            "<html><body>"
            + "<article>" + ("news text. " * 50) + "</article>"
            + "</body></html>"
        )
        return (body, True)

    with patch.object(fetch_mod, "_fetch_url_sync", side_effect=ok_sync):
        first = await fetch_mod._fetch_one("https://example.com/article", 4000)

    assert first is not None
    assert len(first) > 100
    # Second call hits the cache (no second sync invocation).
    with patch.object(
        fetch_mod,
        "_fetch_url_sync",
        side_effect=AssertionError("must hit cache on second call"),
    ):
        second = await fetch_mod._fetch_one("https://example.com/article", 4000)
    assert second == first


# ---------------------------------------------------------------------------
# Shared SSL context (lock the Bing+fetch dedup)
# ---------------------------------------------------------------------------


def test_make_ssl_context_disables_hostname_check():
    """The shared context must disable cert checks — Bing's CDN edge +
    fresh Windows installs both need this tolerance."""
    import ssl

    from src.web_search.fetch import _make_ssl_context

    ctx = _make_ssl_context()
    assert ctx.check_hostname is False
    assert ctx.verify_mode == ssl.CERT_NONE