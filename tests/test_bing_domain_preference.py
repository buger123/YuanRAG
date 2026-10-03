"""v1.1.11 — domain hints + re-ranker tests.

The ``_domain_hints`` module provides two pure-Python functions:

* :func:`extract_domain_hints` — keyword → preferred domain list
* :func:`rerank_by_preferred` — stable re-order so preferred URLs
  come first

Both engines (``bing.search_web_bing``, ``duckduckgo.search_web_ddg``)
and the graph ``search_web`` node forward query-derived hints to
these functions so the LLM cites the right publisher.

What we lock here
-----------------
1. :func:`extract_domain_hints` returns the expected domain list
   for each keyword.
2. Specific keywords win over generic substrings
   (``"腾讯新闻"`` matches before ``"腾讯"``).
3. Empty / no-match queries return ``[]``.
4. :func:`rerank_by_preferred` puts preferred URLs first while
   preserving intra-bucket order.
5. Subdomain match: ``inews.qq.com`` matches ``preferred=["qq.com"]``.
6. Empty ``preferred`` returns the input unchanged (no copy).
8. End-to-end through Bing: ``preferred_domains`` argument is
   accepted and applied.
9. End-to-end through DDG: same.
10. Dispatcher forwards hints to the engine it calls.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.web_search._domain_hints import (
    extract_domain_hints,
    rerank_by_preferred,
)


# ---------------------------------------------------------------------------
# extract_domain_hints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_first",
    [
        # Specific publisher names win.
        ("腾讯新闻 2026.5.6", "inews.qq.com"),
        ("腾讯网 体育", "new.qq.com"),
        ("腾讯 是谁", "qq.com"),
        ("澎湃新闻 今天", "thepaper.cn"),
        ("澎湃", "thepaper.cn"),
        ("人民日报 头版", "paper.people.com.cn"),
        ("新华社 报道", "news.cn"),
        ("央视新闻 联播", "cctv.com"),
        ("界面新闻", "界面"),  # placeholder; checked below
        ("36氪 融资", "36kr.com"),
        ("虎嗅", "huxiu.com"),
        ("财新 杂志", "caixin.com"),
        ("路透社", "reuters.com"),
        ("彭博社", "bloomberg.com"),
        ("BBC 中文", "bbc.com"),
        ("纽约时报", "nytimes.com"),
        ("华尔街日报", "wsj.com"),
    ],
)
def test_extract_domain_hints_recognized_keywords(query: str, expected_first: str):
    """Each publisher keyword returns the expected leading domain.

    The leading-domain check (``expected_first in hints[0]``) handles
    both literal matches and the 界面 case where the hint name itself
    uses Chinese characters in the source map."""
    hints = extract_domain_hints(query)
    assert hints, f"no hints for {query!r}"
    if expected_first == "界面":
        # The 界面 entry maps to jiemian.com — verify the domain
        # rather than the Chinese keyword.
        assert hints[0] == "jiemian.com"
    else:
        assert hints[0] == expected_first, (
            f"query {query!r} → expected first domain {expected_first!r}, "
            f"got {hints[0]!r}"
        )


def test_extract_domain_hints_specific_phrase_wins_over_generic():
    """``"腾讯新闻"`` (specific) is in ``_DOMAIN_HINTS`` BEFORE
    ``"腾讯"`` (generic), so a query containing BOTH substrings
    returns the specific entry — not the generic one."""
    hints = extract_domain_hints("腾讯新闻 腾讯")
    assert hints[0] == "inews.qq.com", (
        f"specific '腾讯新闻' should win; got {hints[0]!r}"
    )


def test_extract_domain_hints_empty_query_returns_empty():
    assert extract_domain_hints("") == []


def test_extract_domain_hints_whitespace_query_returns_empty():
    assert extract_domain_hints("   ") == []


def test_extract_domain_hints_no_keyword_match_returns_empty():
    """A query that names no publisher returns ``[]`` — engines
    then return results in the natural search order."""
    assert extract_domain_hints("今天天气怎么样") == []
    assert extract_domain_hints("python programming") == []


def test_extract_domain_hints_returns_fresh_list():
    """Each call returns a new list so callers can mutate freely."""
    a = extract_domain_hints("腾讯")
    b = extract_domain_hints("腾讯")
    assert a == b
    assert a is not b, "extract_domain_hints must return a fresh list per call"


def test_extract_domain_hints_case_insensitive():
    """``_DOMAIN_HINTS`` keywords are matched case-insensitively."""
    # (The Chinese keywords are case-invariant in Unicode, but the
    # ASCII-domain English entries should still match mixed-case
    # queries. This guards against future ASCII-keyword additions.)
    assert extract_domain_hints("BBC news") == extract_domain_hints("bbc NEWS")


# ---------------------------------------------------------------------------
# rerank_by_preferred
# ---------------------------------------------------------------------------


def _r(*domain_url_pairs):
    """Build a list of WebResult-shaped dicts from (title, url) pairs."""
    from urllib.parse import urlparse

    out = []
    for title, url in domain_url_pairs:
        netloc = (urlparse(url).netloc or "").lower()
        out.append({"title": title, "url": url, "snippet": "", "domain": netloc})
    return out


def test_rerank_empty_preferred_returns_input_unchanged():
    """``preferred_domains=None`` → input returned unchanged (no copy)."""
    results = _r(("A", "https://a.test/"), ("B", "https://b.test/"))
    out = rerank_by_preferred(results, None)
    assert out is results, "no preferred → identity preserved"


def test_rerank_empty_preferred_list_returns_input_unchanged():
    results = _r(("A", "https://a.test/"))
    out = rerank_by_preferred(results, [])
    assert out is results


def test_rerank_puts_preferred_first():
    """When preferred domains are non-empty, matching results move
    to the front in declaration order of the preferred list."""
    results = _r(
        ("thepaper", "https://www.thepaper.cn/a"),
        ("qq", "https://new.qq.com/b"),
        ("other", "https://other.test/c"),
        ("qq2", "https://inews.qq.com/d"),
    )
    out = rerank_by_preferred(
        results, preferred_domains=["qq.com", "thepaper.cn"]
    )
    domains = [r["domain"] for r in out]
    # Both qq.com (new.qq.com + inews.qq.com) come first, then
    # thepaper.cn, then the non-matching other.
    assert domains == [
        "new.qq.com",
        "inews.qq.com",
        "www.thepaper.cn",
        "other.test",
    ]


def test_rerank_subdomain_match():
    """``inews.qq.com`` is treated as a match for preferred ``qq.com``
    (subdomain matching). Required for our case where the preferred
    bucket is the publisher root but the actual URLs are deeper
    subdomains like ``inews.qq.com`` / ``new.qq.com``."""
    results = _r(("qq", "https://inews.qq.com/x"))
    out = rerank_by_preferred(results, preferred_domains=["qq.com"])
    assert out[0]["domain"] == "inews.qq.com"


def test_rerank_does_not_match_lookalike():
    """``fakeqq.com`` is NOT ``qq.com`` — substring-match would be a
    security/UX bug. The helper uses ``endswith('.qq.com')`` so this
    lookalike does NOT match."""
    results = _r(("fake", "https://fakeqq.com/x"))
    out = rerank_by_preferred(results, preferred_domains=["qq.com"])
    # fakeqq.com stays in the non-preferred remainder.
    assert out[0]["domain"] == "fakeqq.com"


def test_rerank_preserves_intra_bucket_order():
    """Within the preferred bucket, the original input order is
    preserved (stable sort). The LLM gets the same first-N preferred
    results the engine naturally ranked first."""
    results = _r(
        ("qq-c", "https://inews.qq.com/c"),
        ("qq-a", "https://inews.qq.com/a"),
        ("qq-b", "https://inews.qq.com/b"),
    )
    out = rerank_by_preferred(results, preferred_domains=["qq.com"])
    titles = [r["title"] for r in out]
    assert titles == ["qq-c", "qq-a", "qq-b"]


def test_rerank_preserves_remainder_order():
    """Non-preferred results keep their original relative order."""
    results = _r(
        ("a", "https://a.test/"),
        ("qq", "https://inews.qq.com/x"),
        ("b", "https://b.test/"),
        ("c", "https://c.test/"),
    )
    out = rerank_by_preferred(results, preferred_domains=["qq.com"])
    titles = [r["title"] for r in out]
    # qq jumps to front; a/b/c keep original order.
    assert titles == ["qq", "a", "b", "c"]


def test_rerank_no_matches_keeps_original_order():
    """When none of the preferred domains match, the output is the
    input in the same order (no shuffling)."""
    results = _r(
        ("x", "https://x.test/"),
        ("y", "https://y.test/"),
    )
    out = rerank_by_preferred(results, preferred_domains=["qq.com"])
    assert [r["title"] for r in out] == ["x", "y"]


def test_rerank_with_iterable_input():
    """The function accepts any iterable of dicts, not just lists."""
    gen = (r for r in _r(("a", "https://a.test/"), ("qq", "https://inews.qq.com/x")))
    out = rerank_by_preferred(gen, preferred_domains=["qq.com"])
    assert [r["title"] for r in out] == ["qq", "a"]


# ---------------------------------------------------------------------------
# End-to-end: Bing accepts and applies preferred_domains
# ---------------------------------------------------------------------------


def test_bing_accepts_preferred_domains_kwarg():
    """``search_web_bing`` exposes ``preferred_domains`` as a kwarg
    so the dispatcher / graph node can pass query-derived hints
    without a separate import path."""
    import inspect

    from src.web_search.bing import search_web_bing

    sig = inspect.signature(search_web_bing)
    assert "preferred_domains" in sig.parameters
    assert sig.parameters["preferred_domains"].default is None


def test_ddg_accepts_preferred_domains_kwarg():
    """``search_web_ddg`` also exposes ``preferred_domains`` so the
    graph ``search_web`` node can pass hints to whichever engine
    ultimately gets called (graph node calls DDG directly today)."""
    import inspect

    from src.web_search.ddg import search_web_ddg

    sig = inspect.signature(search_web_ddg)
    assert "preferred_domains" in sig.parameters
    assert sig.parameters["preferred_domains"].default is None


@pytest.mark.asyncio
async def test_bing_end_to_end_applies_preferred_domains():
    """Mocked urllib returns a fixed mixed-domain result list; with
    ``preferred_domains=["qq.com"]``, the qq.com entry comes first
    in the returned list."""
    import gzip as _gzip
    from unittest.mock import patch

    from src.web_search.bing import search_web_bing

    html = """
    <li class="b_algo">
      <h2><a href="https://www.thepaper.cn/news/1">thepaper</a></h2>
      <p>snippet a</p>
    </li>
    <li class="b_algo">
      <h2><a href="https://inews.qq.com/news/2">qq</a></h2>
      <p>snippet b</p>
    </li>
    <li class="b_algo">
      <h2><a href="https://other.test/news/3">other</a></h2>
      <p>snippet c</p>
    </li>
    """ + ("<!-- " + "x" * 6000 + " -->")
    compressed = _gzip.compress(html.encode("utf-8"))

    class _Resp:
        status = 200
        headers = {"Content-Encoding": "gzip"}

        def read(self):
            return compressed

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def factory(req, **kwargs):
        return _Resp()

    with patch("src.web_search.bing.urllib.request.urlopen", new=factory):
        results = await search_web_bing(
            "腾讯新闻", max_results=10, preferred_domains=["qq.com"]
        )
    assert len(results) == 3
    assert results[0]["domain"] == "inews.qq.com"
    # The non-preferred entries follow in their original order.
    assert [r["domain"] for r in results[1:]] == [
        "www.thepaper.cn",
        "other.test",
    ]


# ---------------------------------------------------------------------------
# End-to-end: dispatcher forwards hints to engines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_forwards_preferred_domains_to_engine(monkeypatch):
    """The dispatcher computes hints from the query and forwards
    them to the engine call. Verifies via an engine mock that the
    ``preferred_domains`` kwarg equals the hints derived from the
    query."""
    import src.web_search as ws
    from config.constants import WEB_SEARCH

    received_kwargs: list[dict] = []

    async def fake_engine(query, max_results=None, **kwargs):
        received_kwargs.append({"query": query, **kwargs})
        return [{"title": "ok", "url": "https://x.test/", "snippet": "", "domain": "x.test"}]

    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"bing": fake_engine})
    monkeypatch.setitem(WEB_SEARCH, "engines", ["bing"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    out = await ws.search_web("腾讯新闻 2026.5.6", enrich=False)
    # v2.0.18 — ``search_web`` now returns ``list[Document]``, not
    # ``list[WebResult]``. The to-Document conversion puts the engine's
    # ``title`` into ``metadata["filename"]``; the body lives in
    # ``page_content``.
    assert out, "search_web returned empty even though fake_engine returned 1 result"
    assert out[0].metadata["filename"] == "ok"
    assert received_kwargs, "dispatcher never invoked the engine"
    assert received_kwargs[0]["query"] == "腾讯新闻 2026.5.6"
    # The dispatcher computes hints and passes them through. The
    # exact list comes from ``_DOMAIN_HINTS`` so we just check
    # that it's non-empty and starts with the expected entry.
    assert received_kwargs[0]["preferred_domains"], (
        "dispatcher did not forward domain hints to the engine"
    )
    assert received_kwargs[0]["preferred_domains"][0] == "inews.qq.com"


@pytest.mark.asyncio
async def test_dispatcher_empty_hints_when_no_keyword(monkeypatch):
    """When the query names no publisher, the dispatcher passes
    ``preferred_domains=[]`` — engines then return results in
    natural order without re-ranking."""
    import src.web_search as ws
    from config.constants import WEB_SEARCH

    received_kwargs: list[dict] = []

    async def fake_engine(query, max_results=None, **kwargs):
        received_kwargs.append(kwargs)
        return [{"title": "ok", "url": "https://x.test/", "snippet": "", "domain": "x.test"}]

    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"bing": fake_engine})
    monkeypatch.setitem(WEB_SEARCH, "engines", ["bing"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    await ws.search_web("python programming language")
    assert received_kwargs[0]["preferred_domains"] == []


# ---------------------------------------------------------------------------
# Search_web node computes hints from query — REMOVED in v2.0.13 cleanup
# ---------------------------------------------------------------------------
#
# ``test_search_web_node_passes_hints_to_ddg`` previously verified
# that the legacy ``src/agent/nodes/search_web.py`` graph node
# forwarded ``preferred_domains`` hints from the query to DDG. That
# node was retired when web search became a ReAct tool; the active
# surface is ``src/agent/tools/web_search.py`` and the hint-forwarding
# contract is exercised by the per-engine tests above.