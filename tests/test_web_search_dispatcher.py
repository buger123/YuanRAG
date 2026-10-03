"""v1.1.10 + v2.0.29.6 (Phase 5) — engine-chain dispatcher tests.

v1.1.10: dispatcher tried engines in declared order, returning the
first non-empty result. Engine list came from ``RAG_WEB_SEARCH_ENGINES``
(env var) or ``WEB_SEARCH["engines"]`` (config). v1.1.10 short-circuit
contract locked the dispatch path — engine order, fallback, timeout,
exception isolation, env var override.

v2.0.29.6 (Phase 5): dispatcher now collects from EVERY configured
engine in parallel via ``asyncio.gather(..., return_exceptions=True)``,
merges results, dedupes by URL, and sorts by ``(-score, engine_priority)``
so the highest-scoring result wins and engine declaration order is the
deterministic tie-breaker. This is a deliberate contract change:
short-circuit-on-first-non-empty is gone, and CancelledError in one
engine is no longer propagated (gather captures it).

What we lock here
-----------------
1. Empty / whitespace query short-circuits to ``[]`` (no engine invoked).
2. Engine list resolution: env var > config > default ``["bing"]``.
3. **All engines called in parallel** (no short-circuit on first non-empty).
4. Per-engine failures (timeout, exception) → captured by gather → dropped.
5. URL dedup — same URL from two engines appears once.
6. All engines empty / errored → ``[]`` returned, no exception leak.
7. Unknown engine name → skipped (with warning), chain continues.
8. CancelledError is captured (return_exceptions=True); dispatcher
   returns whatever successful engines produced (typically empty list).
9. ``max_results`` is forwarded to every engine.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from config.constants import WEB_SEARCH


# ---------------------------------------------------------------------------
# Engine chain / fallback behavior — use AsyncMock engines injected
# via monkeypatch on the registry, not the real engines. This way
# the tests don't depend on ddgs / urllib / any I/O.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engines(monkeypatch):
    """Replace the engine registry with controllable AsyncMocks.

    Returns a dict ``{"eng_name": async_fn}`` for tests to script.
    Each AsyncMock can have ``.side_effect`` (return values / raised
    exceptions) set per-test.
    """
    import src.web_search as ws

    engines: dict[str, AsyncMock] = {}

    def factory(name):
        mock = AsyncMock(name=f"engine_{name}")
        engines[name] = mock
        return mock

    # Wipe + replace registry
    new_registry = {}
    for name in ("e1", "e2", "e3"):
        new_registry[name] = factory(name)
    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", new_registry)

    # Default config engine list — tests can override via env var.
    monkeypatch.setitem(WEB_SEARCH, "engines", ["e1", "e2", "e3"])
    # Keep per-engine timeout tight so a hung engine doesn't slow
    # the suite. Tests can override via ``monkeypatch.setitem`` if
    # they need to exercise the timeout path.
    monkeypatch.setitem(WEB_SEARCH, "per_engine_timeout_seconds", 1)

    # Wipe env var so tests aren't affected by the user's shell.
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)
    return engines


# ---------------------------------------------------------------------------
# Empty / short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_empty_query_skips_all_engines(mock_engines):
    """Empty / whitespace queries must not invoke any engine."""
    from src.web_search import search_web

    assert await search_web("") == []
    assert await search_web("   ") == []
    for eng in mock_engines.values():
        eng.assert_not_called()


# ---------------------------------------------------------------------------
# Engine list resolution: env > config > default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_uses_config_engine_list_when_no_env(mock_engines, monkeypatch):
    """When ``RAG_WEB_SEARCH_ENGINES`` is unset, fall back to
    ``WEB_SEARCH["engines"]``.

    v2.0.29.6 (Phase 5): ALL configured engines are called in parallel
    via ``asyncio.gather`` — there is no short-circuit on first
    non-empty. The pre-Phase-5 behavior of ``e3.assert_not_called()``
    is gone by design (Phase 5 root cause F closure — never discard a
    high-scoring result from a slower engine).
    """
    from src.web_search import search_web

    mock_engines["e1"].return_value = []
    mock_engines["e2"].return_value = [{"title": "ok", "url": "https://x.test/", "snippet": "y", "domain": "x.test"}]

    out = await search_web("hello", enrich=False)
    assert len(out) == 1
    assert out[0].metadata["filename"] == "ok"
    # v2.0.29.6 — every configured engine is called in parallel.
    mock_engines["e1"].assert_awaited_once()
    mock_engines["e2"].assert_awaited_once()
    mock_engines["e3"].assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_env_var_overrides_config(mock_engines, monkeypatch):
    """``RAG_WEB_SEARCH_ENGINES`` env var takes precedence over the
    config list. Useful for ops to flip engines without a code change.

    v2.0.29.6 (Phase 5): env-var engine list is the resolved chain,
    and ALL engines in the resolved chain run in parallel. ``e2`` is
    NOT in the resolved chain (env says ``e3,e1``), so it's not called.
    """
    from src.web_search import search_web

    monkeypatch.setenv("RAG_WEB_SEARCH_ENGINES", "e3,e1")
    # Even though config says ["e1", "e2", "e3"], env says ["e3", "e1"].
    mock_engines["e3"].return_value = [
        {"title": "e3-first", "url": "https://e3.test/", "snippet": "", "domain": "e3.test", "score": 0.9}
    ]
    mock_engines["e1"].return_value = [
        {"title": "e1-second", "url": "https://e1.test/", "snippet": "", "domain": "e1.test", "score": 0.5}
    ]

    out = await search_web("q", enrich=False)
    # Both engines returned results; higher-score wins the sort.
    assert out[0].metadata["filename"] == "e3-first"
    mock_engines["e3"].assert_awaited_once()
    mock_engines["e1"].assert_awaited_once()
    mock_engines["e2"].assert_not_called()  # not in env-var resolved chain


@pytest.mark.asyncio
async def test_dispatcher_env_var_with_whitespace_trimmed(mock_engines, monkeypatch):
    """Whitespace around engine names in the env var must be
    stripped — copy-pasting ``bing, ddg`` shouldn't break."""
    from src.web_search import search_web

    monkeypatch.setenv("RAG_WEB_SEARCH_ENGINES", "  e3  ,  e1  ")
    mock_engines["e3"].return_value = []
    mock_engines["e1"].return_value = []

    out = await search_web("q")
    assert out == []  # both tried, both empty
    mock_engines["e3"].assert_awaited_once()
    mock_engines["e1"].assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_falls_back_to_bing_default_when_no_config(monkeypatch):
    """If both env var and config are empty / unset, default to
    ``["bing"]``. This is the China-friendly default."""
    import src.web_search as ws
    from src.web_search import search_web

    # Wipe env + config
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)
    monkeypatch.setitem(WEB_SEARCH, "engines", [])

    # Replace the ``bing`` entry in the registry with a controllable
    # mock so we don't hit the real network. (Patching
    # ``bing.search_web_bing`` doesn't affect the dispatcher's
    # already-bound registry reference.)
    fake_bing = AsyncMock(return_value=[
        {"title": "bing-default", "url": "https://bing/", "snippet": "", "domain": "bing"}
    ])
    monkeypatch.setitem(ws._ENGINE_REGISTRY, "bing", fake_bing)

    out = await search_web("q", enrich=False)
    assert out[0].metadata["filename"] == "bing-default"
    fake_bing.assert_awaited_once()


# ---------------------------------------------------------------------------
# Engine chain semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_collects_from_all_engines_in_parallel(mock_engines):
    """v2.0.29.6 (Phase 5) — every engine runs in parallel; results
    are merged + deduped + sorted. The pre-Phase-5 short-circuit
    contract is GONE: even when ``e1`` returns a result, ``e2`` and
    ``e3`` still run, so the dispatcher can pick the highest-scoring
    one across the chain.

    Without a ``score`` field, all engines are equal in the tie-breaker,
    so ``e1`` wins (declared first). With distinct scores, the higher
    one wins regardless of declaration order.
    """
    from src.web_search import search_web

    mock_engines["e1"].return_value = [
        {"title": "from-e1", "url": "https://e1/", "snippet": "", "domain": "e1", "score": 0.5}
    ]
    mock_engines["e2"].return_value = [
        {"title": "from-e2", "url": "https://e2/", "snippet": "", "domain": "e2", "score": 0.9}
    ]
    mock_engines["e3"].return_value = [
        {"title": "from-e3", "url": "https://e3/", "snippet": "", "domain": "e3", "score": 0.1}
    ]

    out = await search_web("q", enrich=False)
    # e2 has highest score (0.9), wins despite being declared second.
    assert out[0].metadata["filename"] == "from-e2"
    # All three engines were called (parallel gather).
    mock_engines["e1"].assert_awaited_once()
    mock_engines["e2"].assert_awaited_once()
    mock_engines["e3"].assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_falls_through_when_engine_returns_empty(mock_engines):
    """Empty list from engine N → engine N+1. Empty list is treated
    as 'try next', not as 'success with zero results'.

    v2.0.29.6 (Phase 5): the "fall through" framing is misleading —
    all engines run in parallel regardless. Empty lists just don't
    contribute to the merged result. Tests assert that a successful
    engine still wins despite others returning empty.
    """
    from src.web_search import search_web

    mock_engines["e1"].return_value = []
    mock_engines["e2"].return_value = []
    mock_engines["e3"].return_value = [
        {"title": "e3", "url": "https://e3/", "snippet": "", "domain": "e3"}
    ]

    out = await search_web("q", enrich=False)
    assert out[0].metadata["filename"] == "e3"
    assert mock_engines["e1"].await_count == 1
    assert mock_engines["e2"].await_count == 1
    assert mock_engines["e3"].await_count == 1


@pytest.mark.asyncio
async def test_dispatcher_drops_failing_engine_but_keeps_others(mock_engines):
    """v2.0.29.6 (Phase 5) — per-engine exceptions are captured by
    ``asyncio.gather(return_exceptions=True)`` and dropped; successful
    engines still contribute to the merged result. Pre-Phase-5 had a
    ``falls through to engine N+1`` loop; post-Phase-5 the failing
    engine is dropped silently and other engines' results merge.
    """
    from src.web_search import search_web

    mock_engines["e1"].side_effect = RuntimeError("e1 blew up")
    mock_engines["e2"].return_value = [
        {"title": "e2-ok", "url": "https://e2/", "snippet": "", "domain": "e2"}
    ]
    mock_engines["e3"].return_value = [
        {"title": "e3-ok", "url": "https://e3/", "snippet": "", "domain": "e3"}
    ]

    out = await search_web("q", enrich=False)
    # e2 (declared 2nd, e1 exception) and e3 both contribute.
    filenames = {d.metadata["filename"] for d in out}
    assert "e2-ok" in filenames
    assert "e3-ok" in filenames
    mock_engines["e1"].assert_awaited_once()
    mock_engines["e2"].assert_awaited_once()
    mock_engines["e3"].assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_drops_engine_on_timeout(mock_engines, monkeypatch):
    """v2.0.29.6 (Phase 5) — a slow engine that exceeds
    ``per_engine_timeout_seconds`` is cancelled by ``asyncio.wait_for``
    and dropped. Other engines' results are still merged."""
    from src.web_search import search_web

    monkeypatch.setitem(WEB_SEARCH, "per_engine_timeout_seconds", 0.1)

    async def slow_engine(query, max_results=None):
        await asyncio.sleep(1.0)
        return [{"title": "too-late", "url": "x", "snippet": "", "domain": "x"}]

    mock_engines["e1"] = slow_engine  # replace the AsyncMock
    import src.web_search
    src.web_search._ENGINE_REGISTRY["e1"] = slow_engine

    mock_engines["e2"].return_value = [
        {"title": "e2-after-timeout", "url": "https://e2/", "snippet": "", "domain": "e2"}
    ]

    out = await search_web("q", enrich=False)
    assert out[0].metadata["filename"] == "e2-after-timeout"


@pytest.mark.asyncio
async def test_dispatcher_all_engines_empty_returns_empty_list(mock_engines):
    """All engines return empty → ``[]`` (graph node handles this
    with a clean 'no info found' answer)."""
    from src.web_search import search_web

    mock_engines["e1"].return_value = []
    mock_engines["e2"].return_value = []
    mock_engines["e3"].return_value = []

    assert await search_web("q") == []
    assert mock_engines["e1"].await_count == 1
    assert mock_engines["e2"].await_count == 1
    assert mock_engines["e3"].await_count == 1


@pytest.mark.asyncio
async def test_dispatcher_all_engines_explode_returns_empty_list(mock_engines):
    """All engines raise → ``[]`` (not an exception leak to the
    graph node). Defense-in-depth: each engine should swallow
    internally, but the dispatcher catches anything that escapes."""
    from src.web_search import search_web

    for eng in mock_engines.values():
        eng.side_effect = RuntimeError(f"{eng._mock_name} blew up")

    assert await search_web("q") == []


@pytest.mark.asyncio
async def test_dispatcher_captures_cancellation_per_engine(mock_engines):
    """v2.0.29.6 (Phase 5) — ``asyncio.gather(return_exceptions=True)``
    captures per-engine ``CancelledError`` and surfaces it as a result
    item rather than propagating to the caller. The dispatcher's
    contract changed: callers can no longer cancel mid-chain because
    the dispatcher awaits ALL engines in parallel; partial results
    from successful engines are returned normally.

    Pre-Phase-5 raised ``CancelledError`` to the caller so they could
    short-circuit the chain. Post-Phase-5 the chain runs to
    completion — successful engines' results merge, cancelled engines
    are dropped silently. This is a deliberate trade-off: parallel
    gather can't honour mid-flight cancellation of a subset of
    engines anyway.
    """
    from src.web_search import search_web

    mock_engines["e1"].side_effect = asyncio.CancelledError()
    mock_engines["e2"].return_value = [
        {"title": "e2-ok", "url": "https://e2/", "snippet": "", "domain": "e2"}
    ]

    # Must NOT raise — gather captures the exception, dispatcher
    # returns the successful engines' results.
    out = await search_web("q", enrich=False)
    assert any(d.metadata["filename"] == "e2-ok" for d in out)


# ---------------------------------------------------------------------------
# Engine registry semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_skips_unknown_engine_names(monkeypatch):
    """Unknown engine names in the chain list are skipped (with a
    warning log) rather than raising. Keeps the chain working
    even if a user typo'd ``RAG_WEB_SEARCH_ENGINES=bing,ddggg``."""
    from src.web_search import search_web

    import src.web_search as ws

    # Build a registry with only "bing"; chain has "bing", "ddggg"
    fake_bing = AsyncMock(return_value=[
        {"title": "ok", "url": "https://bing/", "snippet": "", "domain": "bing"}
    ])
    ws._ENGINE_REGISTRY = {"bing": fake_bing}
    monkeypatch.setitem(WEB_SEARCH, "engines", ["bing", "ddggg"])

    out = await search_web("q", enrich=False)
    assert out[0].metadata["filename"] == "ok"
    fake_bing.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatcher_max_results_passed_to_each_engine(mock_engines):
    """``max_results`` is forwarded to each engine. Engines may cap
    further internally, but the dispatcher's value is the ceiling
    of intent."""
    from src.web_search import search_web

    mock_engines["e1"].return_value = [
        {"title": "t", "url": "https://e1/", "snippet": "", "domain": "e1"}
    ]

    await search_web("q", max_results=7)
    # First engine called with max_results=7
    _, kwargs = mock_engines["e1"].call_args
    assert mock_engines["e1"].call_args.args[0] == "q"
    assert mock_engines["e1"].call_args.kwargs["max_results"] == 7