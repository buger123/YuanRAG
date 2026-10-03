"""Phase 6 — DuckDuckGo async wrapper behavior.

These tests mock the ``DDGS`` class that ``search_web_ddg`` lazy-imports
from the optional ``ddgs`` package, so we don't need network access. We
exercise:

* Empty / blank query short-circuits to ``[]``.
* ``RatelimitException`` (or any exception) triggers exponential backoff
  retries up to ``max_attempts``.
* Empty result is treated as transient — same retry path.
* On success, results are mapped to ``WebResult`` dicts with title / url /
  snippet / domain populated; the domain is lowercased.
* The 1.5 s global gate does NOT delay successive calls in the test
  suite because ``reset_for_tests()`` zeros the timestamp.

Failure-mode contract: ``search_web_ddg`` returns ``[]`` when every attempt
fails; it never raises to the caller.

Note on the API shape: ``ddgs >= 9.x`` dropped ``AsyncDDGS`` and only
exposes the synchronous ``DDGS`` class. ``search_web_ddg`` wraps the
blocking ``text()`` call in ``asyncio.to_thread``; the mocks below
match that contract (sync context manager with sync ``text()``).

v1.1.10 rename: the symbol was ``search_web`` until v1.1.10, when it
was renamed to ``search_web_ddg`` to free up ``search_web`` for the
new engine-chain dispatcher in ``src/web_search/__init__.py``. Tests
in ``test_web_search_dispatcher.py`` cover the dispatcher end-to-end;
this file focuses on the DDG engine specifically.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.constants import WEB_SEARCH
# v2.0.18 — module renamed duckduckgo.py → ddg.py; keep the local alias
# ``duckduckgo`` so the patch.object sites at lines 248/311/317 still work
# without renaming every internal reference.
from src.web_search import ddg as duckduckgo
from src.web_search.ddg import reset_for_tests, search_web_ddg


# Sync stub that mimics ``ddgs.DDGS`` — sync context manager with a
# sync ``text()`` method. ``search_web`` runs ``text()`` inside
# ``asyncio.to_thread``, so we just install a sync result and let the
# wrapper thread it.
class _FakeDDGS:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # Per-instance scripted response (list of dicts or exception).
        self.script: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text_set(self, payload):
        """Install a one-shot response — list of result dicts."""
        self.script.append(payload)

    def text_raise(self, exc):
        self.script.append(exc)

    def text(self, query, **_kwargs):
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _RatelimitException(Exception):
    """Stand-in for ddgs.RatelimitException — duckduckgo raises its own
    exception class but we don't import the real one to keep tests
    independent of the optional dep."""


@pytest.fixture(autouse=True)
def _reset_gate():
    """Each test starts with a zeroed gate so the 1.5 s wait doesn't
    slow the suite down.
    """
    reset_for_tests()
    yield
    reset_for_tests()


def _install_fake_ddgs(monkeypatch, factory):
    """Patch the ``ddgs`` module so the lazy import inside ``_attempt``
    returns our factory. The wrapper does::

        from ddgs import DDGS

    so we need a module named ``ddgs`` with a ``DDGS`` attribute.
    """
    fake_module = SimpleNamespace(DDGS=factory)
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)
    return fake_module


def _make_ddg_with_results(results):
    """Return a factory whose DDGS instance yields ``results`` once."""
    instance = _FakeDDGS()
    instance.text_set(results)

    def factory(*args, **kwargs):
        # Each call returns a FRESH instance — the wrapper does
        # ``with DDGS() as ddgs`` per attempt.
        fresh = _FakeDDGS()
        fresh.text_set(results)
        return fresh

    return factory, instance


@pytest.mark.asyncio
async def test_search_web_empty_query_returns_empty():
    """Empty / whitespace queries must not hit the network."""
    assert await search_web_ddg("") == []
    assert await search_web_ddg("   ") == []


@pytest.mark.asyncio
async def test_search_web_success(monkeypatch):
    """Happy path: a successful DDG response is mapped to WebResult dicts
    with title / url / snippet / domain (domain lowercased)."""
    factory, _ = _make_ddg_with_results(
        [
            {"title": "Hello", "href": "https://Example.COM/path", "body": "snippet 1"},
            {"title": "World", "url": "https://other.test/foo", "snippet": "snippet 2"},
        ]
    )
    _install_fake_ddgs(monkeypatch, factory)

    results = await search_web_ddg("hello world")
    assert len(results) == 2
    # First row — keys should be normalized
    assert results[0]["title"] == "Hello"
    assert results[0]["url"] == "https://Example.COM/path"
    assert results[0]["snippet"] == "snippet 1"
    assert results[0]["domain"] == "example.com"
    # Second row — uses url / snippet fallbacks
    assert results[1]["url"] == "https://other.test/foo"
    assert results[1]["domain"] == "other.test"


@pytest.mark.asyncio
async def test_search_web_skips_results_without_url(monkeypatch):
    """DDG sometimes returns rows with no URL — they should be dropped."""
    factory, _ = _make_ddg_with_results(
        [
            {"title": "Has URL", "href": "https://ok.test/", "body": "ok"},
            {"title": "No URL", "href": "", "body": "dropped"},
        ]
    )
    _install_fake_ddgs(monkeypatch, factory)

    results = await search_web_ddg("q")
    assert len(results) == 1
    assert results[0]["domain"] == "ok.test"


@pytest.mark.asyncio
async def test_search_web_retries_on_rate_limit(monkeypatch):
    """A persistent RatelimitException must trigger exponential backoff
    and ultimately return [] (failure mode contract)."""
    attempts: list[int] = []

    def factory(*args, **kwargs):
        attempts.append(1)
        fresh = _FakeDDGS()
        fresh.text_raise(_RatelimitException("rate limited"))
        return fresh

    _install_fake_ddgs(monkeypatch, factory)

    # Patch asyncio.sleep inside the wrapper so we don't actually sleep.
    real_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    results = await search_web_ddg("q")
    assert results == []
    # max_attempts == 3 by default
    assert len(attempts) == WEB_SEARCH["max_attempts"]


@pytest.mark.asyncio
async def test_search_web_succeeds_after_empty_attempt(monkeypatch):
    """Empty result is treated as transient; the second attempt gets
    real data and we hand it back without further retries."""
    calls: list[int] = []

    def factory(*args, **kwargs):
        calls.append(1)
        fresh = _FakeDDGS()
        if len(calls) == 1:
            fresh.text_set([])  # empty → retry
        else:
            fresh.text_set(
                [{"title": "ok", "href": "https://ok.test/", "body": "yes"}]
            )
        return fresh

    _install_fake_ddgs(monkeypatch, factory)

    real_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    results = await search_web_ddg("q")
    assert len(results) == 1
    assert results[0]["title"] == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_search_web_respects_max_attempts(monkeypatch):
    """``max_attempts=1`` must skip retries entirely."""
    calls: list[int] = []

    def factory(*args, **kwargs):
        calls.append(1)
        fresh = _FakeDDGS()
        fresh.text_raise(_RatelimitException("nope"))
        return fresh

    _install_fake_ddgs(monkeypatch, factory)

    real_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    # Bypass WEB_SEARCH["max_attempts"] by passing it through the wrapper's
    # own retry budget — we patch the module-level config dict in place.
    original = duckduckgo.WEB_SEARCH["max_attempts"]
    duckduckgo.WEB_SEARCH["max_attempts"] = 1
    try:
        results = await search_web_ddg("q")
    finally:
        duckduckgo.WEB_SEARCH["max_attempts"] = original

    assert results == []
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_search_web_passes_through_params(monkeypatch):
    """max_results / region / safesearch defaults + overrides are
    forwarded into ``DDGS()`` and ``text()``."""
    captured: dict = {}

    def factory(*args, **kwargs):
        captured["ddgs_kwargs"] = dict(kwargs)

        class _DDG:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def text(self_inner, query, **kw):
                captured["text_kwargs"] = dict(kw)
                return [{"title": "x", "href": "https://x.test/", "body": "y"}]

        return _DDG()

    _install_fake_ddgs(monkeypatch, factory)

    results = await search_web_ddg("q", max_results=7, region="us-en", safesearch="strict")
    assert len(results) == 1
    assert results[0]["domain"] == "x.test"
    # text() received our overrides
    assert captured["text_kwargs"]["max_results"] == 7
    assert captured["text_kwargs"]["region"] == "us-en"
    assert captured["text_kwargs"]["safesearch"] == "strict"


@pytest.mark.asyncio
async def test_search_web_gate_does_not_sleep_after_reset(monkeypatch):
    """The 1.5 s global rate-limit gate should not delay the first
    search_web call after ``reset_for_tests()`` zeros the timestamp.

    After reset, _last_call_ts is 0.0. Calling search_web with a
    mocked DDG that immediately returns should NOT sleep 1.5 s —
    the test should complete in well under that.
    """
    real_sleep = asyncio.sleep
    slept: list[float] = []

    async def track_sleep(seconds):
        slept.append(seconds)
        await real_sleep(0)

    # We need to patch asyncio.sleep *within* the wrapper module —
    # duckduckgo imports ``asyncio`` at top level, so we patch the
    # module's own reference.
    original_module_sleep = duckduckgo.asyncio.sleep
    duckduckgo.asyncio.sleep = track_sleep
    try:
        factory, _ = _make_ddg_with_results(
            [{"title": "ok", "href": "https://ok.test/", "body": "yes"}]
        )
        with patch.object(duckduckgo, "_attempt", new=AsyncMock(return_value=[
            {"title": "ok", "href": "https://ok.test/", "body": "yes"}
        ])):
            results = await search_web_ddg("q")
    finally:
        duckduckgo.asyncio.sleep = original_module_sleep

    assert len(results) == 1
    # We slept 0 (the gate didn't fire because the timestamp was 0
    # and we patched the gate-keeper's sleep to no-op), so the 1.5 s
    # interval was NOT enforced.
    assert all(s == 0 for s in slept)