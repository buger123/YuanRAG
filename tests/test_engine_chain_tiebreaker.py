"""v2.0.29.6 (Phase 5) — web_search engine chain tie-breaker.

Pre-Phase-5 short-circuited on the first non-empty engine — if Bing
returned 5 mediocre results and DDG was about to return 5 excellent
ones, the user only saw Bing's. Phase 5 collects from every
configured engine in parallel via ``asyncio.gather``, dedupes by URL,
then sorts by ``(-score, engine_priority)`` so the highest-scoring
result wins and engine declaration order is the deterministic
tie-breaker.

These tests pin:
  1. The dispatcher runs all engines in parallel (not short-circuit).
  2. URL dedup — same URL from two engines appears once.
  3. Tie-breaker — equal-score results from different engines:
     declared-order engine wins.
  4. Score ordering — higher-score result comes first.
  5. Empty/all-failed engines → empty docs list (no crash).
  6. Internal ``_engine`` tag does not leak into Document metadata.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest


# ============================================================
# 1. Dispatcher shape: parallel gather (not short-circuit)
# ============================================================


def test_dispatcher_uses_asyncio_gather():
    """The dispatcher collects from all engines via ``asyncio.gather``.

    Pre-Phase-5 used a ``for name in engines: ... return docs`` loop
    that exited on first non-empty result. Post-Phase-5 must use
    ``asyncio.gather`` so all engines run in parallel.
    """
    import src.web_search as web_search_mod

    src = inspect.getsource(web_search_mod.search_web)
    assert "asyncio.gather" in src, (
        "search_web dispatcher should use asyncio.gather for parallel engine execution"
    )
    # And must NOT short-circuit on first non-empty result.
    assert "if results:" not in src or src.count("if results:") >= 2, (
        "search_web should not short-circuit on first non-empty engine result"
    )


def test_dispatcher_dedupes_by_url():
    """The dispatcher dedupes merged results by URL."""
    import src.web_search as web_search_mod

    src = inspect.getsource(web_search_mod.search_web)
    assert "seen_urls" in src, (
        "search_web should track seen URLs for cross-engine dedup"
    )


def test_dispatcher_sorts_by_neg_score_then_engine_priority():
    """Tie-breaker uses ``(-score, engine_priority)`` key."""
    import src.web_search as web_search_mod

    src = inspect.getsource(web_search_mod.search_web)
    assert "engine_priority" in src
    assert "-float" in src or "-r.get(\"score\"" in src or "-r.get('score'" in src, (
        "Tie-breaker should sort by negative score first"
    )


# ============================================================
# 2. Behavioural: dispatcher with stubbed engines
# ============================================================


@pytest.mark.asyncio
async def test_dispatcher_collects_from_all_engines(monkeypatch):
    """Both engines run; merged list contains results from each."""

    async def fake_bing(query, max_results=None, preferred_domains=None):
        return [
            {"url": "https://bing-result.com/a", "title": "Bing A", "snippet": "x", "score": 0.9},
        ]

    async def fake_ddg(query, max_results=None, preferred_domains=None):
        return [
            {"url": "https://ddg-result.com/b", "title": "DDG B", "snippet": "y", "score": 0.8},
        ]

    import src.web_search as web_search_mod
    monkeypatch.setattr(web_search_mod, "_ENGINE_REGISTRY", {"bing": fake_bing, "ddg": fake_ddg})
    monkeypatch.setattr(web_search_mod, "_resolve_engine_list", lambda: ["bing", "ddg"])
    # Stub enrich to skip network page-fetch.
    async def no_enrich(docs, max_chars=None):
        return docs
    monkeypatch.setattr(web_search_mod, "_enrich_with_pages", no_enrich)

    docs = await web_search_mod.search_web("test query", enrich=False)
    urls = {d.metadata["url"] for d in docs}
    assert "https://bing-result.com/a" in urls
    assert "https://ddg-result.com/b" in urls, (
        "DDG result should NOT be dropped just because Bing returned first"
    )


@pytest.mark.asyncio
async def test_dispatcher_dedupes_same_url_across_engines(monkeypatch):
    """Same URL from two engines appears once."""

    async def fake_bing(query, max_results=None, preferred_domains=None):
        return [{"url": "https://shared.com/x", "title": "Bing", "snippet": "x", "score": 0.9}]

    async def fake_ddg(query, max_results=None, preferred_domains=None):
        return [{"url": "https://shared.com/x", "title": "DDG", "snippet": "x", "score": 0.7}]

    import src.web_search as web_search_mod
    monkeypatch.setattr(web_search_mod, "_ENGINE_REGISTRY", {"bing": fake_bing, "ddg": fake_ddg})
    monkeypatch.setattr(web_search_mod, "_resolve_engine_list", lambda: ["bing", "ddg"])
    async def no_enrich(docs, max_chars=None):
        return docs
    monkeypatch.setattr(web_search_mod, "_enrich_with_pages", no_enrich)

    docs = await web_search_mod.search_web("q", enrich=False)
    shared = [d for d in docs if d.metadata["url"] == "https://shared.com/x"]
    assert len(shared) == 1, (
        f"Expected dedup to 1 entry for shared URL, got {len(shared)}"
    )


@pytest.mark.asyncio
async def test_tie_breaker_equal_score_declared_order_wins(monkeypatch):
    """Two engines return same score → declared order (bing=0, ddg=1) wins."""

    async def fake_bing(query, max_results=None, preferred_domains=None):
        return [{"url": "https://bing.com/x", "title": "Bing", "snippet": "x", "score": 0.5}]

    async def fake_ddg(query, max_results=None, preferred_domains=None):
        return [{"url": "https://ddg.com/x", "title": "DDG", "snippet": "x", "score": 0.5}]

    import src.web_search as web_search_mod
    monkeypatch.setattr(web_search_mod, "_ENGINE_REGISTRY", {"bing": fake_bing, "ddg": fake_ddg})
    monkeypatch.setattr(web_search_mod, "_resolve_engine_list", lambda: ["bing", "ddg"])
    async def no_enrich(docs, max_chars=None):
        return docs
    monkeypatch.setattr(web_search_mod, "_enrich_with_pages", no_enrich)

    docs = await web_search_mod.search_web("q", enrich=False)
    # Bing declared first → its result wins the tie.
    assert docs[0].metadata["url"] == "https://bing.com/x", (
        f"Tied-score result should be ordered by engine declaration; "
        f"got {docs[0].metadata['url']!r}"
    )
    assert docs[1].metadata["url"] == "https://ddg.com/x"


@pytest.mark.asyncio
async def test_higher_score_wins_regardless_of_engine_order(monkeypatch):
    """DDG (declared 2nd) with higher score beats Bing (declared 1st)."""

    async def fake_bing(query, max_results=None, preferred_domains=None):
        return [{"url": "https://bing.com/x", "title": "Bing", "snippet": "x", "score": 0.3}]

    async def fake_ddg(query, max_results=None, preferred_domains=None):
        return [{"url": "https://ddg.com/x", "title": "DDG", "snippet": "x", "score": 0.9}]

    import src.web_search as web_search_mod
    monkeypatch.setattr(web_search_mod, "_ENGINE_REGISTRY", {"bing": fake_bing, "ddg": fake_ddg})
    monkeypatch.setattr(web_search_mod, "_resolve_engine_list", lambda: ["bing", "ddg"])
    async def no_enrich(docs, max_chars=None):
        return docs
    monkeypatch.setattr(web_search_mod, "_enrich_with_pages", no_enrich)

    docs = await web_search_mod.search_web("q", enrich=False)
    assert docs[0].metadata["url"] == "https://ddg.com/x", (
        f"Higher-score result should win; got {docs[0].metadata['url']!r}"
    )


@pytest.mark.asyncio
async def test_dispatcher_handles_all_engines_failing(monkeypatch):
    """All engines raise → empty list, no crash."""

    async def fake_failing_engine(query, max_results=None, preferred_domains=None):
        raise RuntimeError("simulated engine outage")

    import src.web_search as web_search_mod
    monkeypatch.setattr(web_search_mod, "_ENGINE_REGISTRY", {"bing": fake_failing_engine})
    monkeypatch.setattr(web_search_mod, "_resolve_engine_list", lambda: ["bing"])

    docs = await web_search_mod.search_web("q", enrich=False)
    assert docs == [], "All-failing chain should return empty list"


@pytest.mark.asyncio
async def test_internal_engine_tag_does_not_leak_into_metadata(monkeypatch):
    """The internal ``_engine`` tag is stripped before Document conversion."""

    async def fake_bing(query, max_results=None, preferred_domains=None):
        return [{"url": "https://bing.com/x", "title": "B", "snippet": "x", "score": 0.9}]

    import src.web_search as web_search_mod
    monkeypatch.setattr(web_search_mod, "_ENGINE_REGISTRY", {"bing": fake_bing})
    monkeypatch.setattr(web_search_mod, "_resolve_engine_list", lambda: ["bing"])
    async def no_enrich(docs, max_chars=None):
        return docs
    monkeypatch.setattr(web_search_mod, "_enrich_with_pages", no_enrich)

    docs = await web_search_mod.search_web("q", enrich=False)
    for d in docs:
        assert "_engine" not in d.metadata, (
            f"_engine internal tag leaked into metadata: {d.metadata}"
        )