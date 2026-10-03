"""v2.0.18 — ``search_web`` public surface + new kwargs + TypedDict migration.

The web-search simplification moved ``search_web`` from
``src/agent/tools/web_search.py`` into ``src/web_search/__init__.py``
and changed three things callers can observe:

1. Return type: ``list[WebResult]`` → ``list[Document]``.
2. New kwarg: ``enrich: bool = True`` — opt-out of page-fetch
   enrichment (useful for tests / offline mode).
3. New kwarg: ``preferred_domains: list[str] | None = None`` —
   caller can override the query-keyword-extracted domain hint.

Plus a smaller refactor:

4. ``WebResult`` is now a ``TypedDict``, not a ``class WebResult(dict)``.
5. ``_domain_of(url)`` moved from ``ddg.py`` into ``_domain_hints.py``
   alongside ``WebResult``.

These tests pin the new public surface so future refactors
don't silently break the Phase 3 pure-async state machine's
contract with ``search_web``.

What we lock here
-----------------
1. Return type is ``list[Document]`` with ``source_kind="web"``.
2. ``enrich=False`` skips the ``_fetch_one`` call entirely.
3. ``enrich=True`` (default) invokes ``_fetch_one`` once per doc.
4. ``preferred_domains=...`` overrides the query-keyword hint.
5. ``WebResult`` is a ``TypedDict``, not a dict subclass.
6. ``_domain_of`` is importable from ``src.web_search._domain_hints``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Return type is list[Document]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_web_returns_documents_with_source_kind_web(monkeypatch):
    """``search_web()`` returns ``list[Document]`` (not ``list[WebResult]``).

    Phase 3 readiness: the return type uses ``langchain_core.Document``
    only, no LangGraph types. The pure-async state machine will call
    this directly and feed the docs straight into its own accumulator.
    """
    from langchain_core.documents import Document
    import src.web_search as ws
    from config.constants import WEB_SEARCH

    fake_engine = AsyncMock(return_value=[
        {"title": "A", "url": "https://a.test/", "snippet": "snip A", "domain": "a.test"},
        {"title": "B", "url": "https://b.test/", "snippet": "snip B", "domain": "b.test"},
    ])
    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"fake": fake_engine})
    monkeypatch.setitem(WEB_SEARCH, "engines", ["fake"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    out = await ws.search_web("q", enrich=False)

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(d, Document) for d in out)
    # v2.0.18 contract — every returned doc carries source_kind="web"
    # so the runner can route it to the correct citation chip.
    assert all(d.metadata.get("source_kind") == "web" for d in out)
    # The engine's title becomes ``filename`` (for the <WebSource> chip)
    # and the body is ``title + "\n\n" + snippet``.
    assert out[0].metadata["filename"] == "A"
    assert out[0].page_content == "A\n\nsnip A"
    assert out[1].metadata["filename"] == "B"


# ---------------------------------------------------------------------------
# 2. enrich=False skips fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_false_does_not_call_fetch_one(monkeypatch):
    """``enrich=False`` short-circuits before any HTTP I/O.

    Tests use this to avoid the noise of real urllib attempts; ops
    can use it for offline / smoke runs without network.
    """
    import src.web_search as ws
    from config.constants import WEB_SEARCH

    fake_engine = AsyncMock(return_value=[
        {"title": "A", "url": "https://a.test/", "snippet": "snip", "domain": "a.test"},
    ])
    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"fake": fake_engine})
    monkeypatch.setitem(WEB_SEARCH, "engines", ["fake"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    with patch("src.web_search._fetch_one", new=AsyncMock(return_value="never-called body")) as fetch_mock:
        out = await ws.search_web("q", enrich=False)

    assert len(out) == 1
    # page_content is the snippet (no fetch happened).
    assert "snip" in out[0].page_content
    fetch_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 3. enrich=True (default) calls _fetch_one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enrich_true_default_invokes_fetch_one_per_doc(monkeypatch):
    """``enrich=True`` (the default) calls ``_fetch_one`` once per doc.

    Verifies the default matches the documented behavior in the
    ``search_web`` docstring — callers who don't pass ``enrich``
    still get enrichment.
    """
    import src.web_search as ws
    from config.constants import WEB_SEARCH

    fake_engine = AsyncMock(return_value=[
        {"title": "A", "url": "https://a.test/", "snippet": "snip A", "domain": "a.test"},
        {"title": "B", "url": "https://b.test/", "snippet": "snip B", "domain": "b.test"},
    ])
    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"fake": fake_engine})
    monkeypatch.setitem(WEB_SEARCH, "engines", ["fake"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    async def fake_fetch(url, max_chars):
        calls.append(url)
        return f"BODY({url})"

    calls: list[str] = []
    with patch("src.web_search._fetch_one", new=fake_fetch):
        out = await ws.search_web("q")  # no enrich kwarg → default True

    # ``_fetch_one`` was called for each of the 2 docs.
    assert len(calls) == 2
    assert calls == ["https://a.test/", "https://b.test/"]
    # page_content now has the fetched body, not the snippet.
    assert out[0].page_content == "A\n\nBODY(https://a.test/)"
    assert out[1].page_content == "B\n\nBODY(https://b.test/)"


# ---------------------------------------------------------------------------
# 4. preferred_domains overrides query-keyword hint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preferred_domains_kwarg_overrides_query_keyword_hint(monkeypatch):
    """``preferred_domains=...`` short-circuits ``extract_domain_hints``.

    Even if the query would otherwise produce hints (e.g. ``腾讯新闻``),
    the explicit ``preferred_domains`` wins.
    """
    import src.web_search as ws
    from config.constants import WEB_SEARCH

    received_kwargs: list[dict] = []

    async def fake_engine(query, max_results=None, **kwargs):
        received_kwargs.append(kwargs)
        return [
            {"title": "A", "url": "https://a.test/", "snippet": "", "domain": "a.test"},
        ]

    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"fake": fake_engine})
    monkeypatch.setitem(WEB_SEARCH, "engines", ["fake"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    # A query that would normally yield ["inews.qq.com", ...] — pass
    # an entirely different preferred list and verify it's what the
    # engine sees.
    await ws.search_web(
        "腾讯新闻 2026.5.6",
        enrich=False,
        preferred_domains=["nytimes.com", "bbc.com"],
    )

    assert received_kwargs, "engine never called"
    assert received_kwargs[0]["preferred_domains"] == ["nytimes.com", "bbc.com"]


@pytest.mark.asyncio
async def test_preferred_domains_none_falls_back_to_keyword_hint(monkeypatch):
    """``preferred_domains=None`` (the default) → engine sees the
    query-keyword hint (not an empty list).
    """
    import src.web_search as ws
    from config.constants import WEB_SEARCH

    received_kwargs: list[dict] = []

    async def fake_engine(query, max_results=None, **kwargs):
        received_kwargs.append(kwargs)
        return []

    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"fake": fake_engine})
    monkeypatch.setitem(WEB_SEARCH, "engines", ["fake"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    await ws.search_web("腾讯新闻 2026.5.6", enrich=False)
    assert received_kwargs
    # The dispatcher's hint computation ran (not bypassed by an
    # empty override). The exact list comes from _DOMAIN_HINTS.
    assert received_kwargs[0]["preferred_domains"] == [
        "inews.qq.com", "new.qq.com", "qq.com",
    ]


# ---------------------------------------------------------------------------
# 5. WebResult is a TypedDict (not a dict subclass)
# ---------------------------------------------------------------------------


def test_web_result_is_typed_dict_not_dict_subclass():
    """``WebResult`` is now a ``TypedDict`` — runtime is plain dict.

    Pre-v2.0.18 used ``class WebResult(dict)`` (a v1 type theater:
    subclass without behavior). TypedDict has zero runtime overhead,
    supports both ``r["title"]`` and the kwarg constructor, and is
    friendlier to mypy strict-mode.
    """
    from typing import TypedDict
    from src.web_search._domain_hints import WebResult

    assert issubclass(WebResult, dict), "TypedDict must be a dict subclass"
    # TypedDict sets ``__total__``; the old ``class WebResult(dict)``
    # doesn't have this attribute.
    assert hasattr(WebResult, "__total__"), (
        "WebResult must be a TypedDict, not a plain dict subclass"
    )
    # Kwarg constructor still works (no caller-visible breakage).
    r = WebResult(title="x", url="https://x/", snippet="", domain="x")
    assert r["title"] == "x"
    assert r["url"] == "https://x/"


# ---------------------------------------------------------------------------
# 6. _domain_of lives in _domain_hints
# ---------------------------------------------------------------------------


def test_domain_of_lives_in_domain_hints_module():
    """``_domain_of(url)`` is importable from
    ``src.web_search._domain_hints`` (not from ``ddg.py``).

    v2.0.18 move — co-located with ``WebResult`` because they're
    always used together (``WebResult(domain=_domain_of(url), ...)`).
    """
    from src.web_search._domain_hints import _domain_of

    assert _domain_of("https://inews.qq.com/x") == "inews.qq.com"
    # ``netloc`` preserves port (per RFC 3986) — fine for our
    # purposes; the domain matcher treats the full netloc as one
    # string, so we don't need to strip the port.
    assert _domain_of("https://NEWS.QQ.COM:8080/y?z=1") == "news.qq.com:8080"
    assert _domain_of("not a url") == ""
    assert _domain_of("") == ""


__all__ = [
    "test_search_web_returns_documents_with_source_kind_web",
    "test_enrich_false_does_not_call_fetch_one",
    "test_enrich_true_default_invokes_fetch_one_per_doc",
    "test_preferred_domains_kwarg_overrides_query_keyword_hint",
    "test_preferred_domains_none_falls_back_to_keyword_hint",
    "test_web_result_is_typed_dict_not_dict_subclass",
    "test_domain_of_lives_in_domain_hints_module",
]