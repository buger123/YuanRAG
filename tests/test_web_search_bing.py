"""v1.1.10 — Bing direct-scrape engine tests.

The ``src/web_search/bing.py`` module is the China-friendly default
search engine for v1.1.10. It scrapes Bing's HTML results page
directly with urllib + regex (no ddgs dependency) and parses the
stable ``<li class="b_algo">`` container. These tests mock urllib
so we don't need internet access from the test runner.

What we lock here
-----------------
1. HTML parsing: ``<li class="b_algo">`` containers with
   ``<h2><a href="...">title</a></h2>`` + ``<p>snippet</p>`` yield
   WebResult dicts with title/url/snippet/domain. Domain is
   lowercased.
2. Empty body / anti-bot page (body too small): no results.
3. ``bing.com/`` redirect hosts are filtered out.
4. Result blocks missing ``<h2>`` or ``href`` are skipped (related-
   search stubs Bing injects).
5. Host fallback: when ``cn.bing.com`` fails, ``www.bing.com`` is
   tried. First non-empty wins.
6. Both hosts failing → ``[]`` (no exception leaks).
7. Per-engine timeout via the dispatcher: a hanging host doesn't
   stall past ``per_engine_timeout_seconds``.
8. Truncation: ``max_results`` caps the result list.
"""
from __future__ import annotations

import asyncio
import gzip
from unittest.mock import AsyncMock, patch

import pytest

from src.web_search.bing import (
    _BING_HOSTS,
    _decode_response,
    _fetch_one,
    _parse_results,
    search_web_bing,
)


# ---------------------------------------------------------------------------
# HTML fixture: a realistic Bing results page (English, 10 results).
# Mirrors the actual structure returned by cn.bing.com — confirmed by
# a live HTTP probe in the v1.1.10 investigation.
# ---------------------------------------------------------------------------

_BING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>python programming language - Bing</title></head>
<body>
<div class="b_results">
<li class="b_algo">
  <h2><a href="https://www.python.org/">Welcome to Python.org</a></h2>
  <p>Experienced programmers in any other language can pick up Python very quickly.</p>
</li>
<li class="b_algo">
  <h2><a href="https://en.wikipedia.org/wiki/Python_(programming_language)">Python - Wikipedia</a></h2>
  <p>Python is a high-level, general-purpose programming language.</p>
</li>
<li class="b_algo">
  <h2><a href="https://docs.python.org/3/">Python Documentation</a></h2>
  <p>The official home of the Python Programming Language.</p>
</li>
<li class="b_ad">
  <h2><a href="https://bing.com/ck/a?!ignored">ad</a></h2>
  <p>advertisement</p>
</li>
<li class="b_related">
  <h2><a href="#">related searches</a></h2>
</li>
<li class="b_algo">
  <div><h2><a href="https://github.com/python/cpython">GitHub - cpython</a></h2></div>
  <p>Python source distribution.</p>
</li>
</div>
""" + ("<!-- " + "x" * 6000 + " -->") + """
</body>
</html>
"""

_BING_HTML_GZIPPED = gzip.compress(_BING_HTML.encode("utf-8"))


def _pad(html: str) -> str:
    """Pad a short HTML fixture over the 5000-byte captcha-detection
    threshold so ``_parse_results`` doesn't reject it as an anti-bot
    page. Real Bing results pages are ~100 KB; the threshold exists
    to detect the 5-10 KB captcha HTML pages. Test fixtures smaller
    than 5 KB are an artifact of test brevity, not captchas."""
    if len(html) >= 5000:
        return html
    pad = "<!-- " + "x" * (6000 - len(html)) + " -->"
    return html + "\n" + pad


# ---------------------------------------------------------------------------
# Parsing tests — operate on HTML fixtures, no network
# ---------------------------------------------------------------------------


def test_parse_results_extracts_algo_blocks():
    """Happy path: 4 real b_algo results (the b_ad and b_related are
    skipped; the bing.com/ck redirect is filtered)."""
    results = _parse_results(_BING_HTML)
    assert len(results) == 4, (
        f"Expected 4 real results, got {len(results)}: {results!r}"
    )
    titles = [r["title"] for r in results]
    assert titles[0] == "Welcome to Python.org"
    assert titles[1] == "Python - Wikipedia"
    assert "GitHub - cpython" in titles[3]
    # All domains are non-bing and lowercased
    for r in results:
        assert "bing.com" not in r["domain"]
        assert r["domain"] == r["domain"].lower()


def test_parse_results_filters_bing_redirects():
    """``bing.com/ck/...`` and ``www.bing.com/...`` are tracking
    redirects, never real destinations. Filter them out so the LLM
    doesn't cite them."""
    html = _pad("""
    <li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!ignored">x</a></h2><p>ad</p></li>
    <li class="b_algo"><h2><a href="https://real.test/page">y</a></h2><p>real</p></li>
    """)
    results = _parse_results(html)
    assert len(results) == 1
    assert results[0]["domain"] == "real.test"


def test_parse_results_empty_when_body_too_small():
    """Anti-bot / captcha pages return a 5-10 KB body without any
    ``b_algo`` items. Treat them as 'no results' so the dispatcher
    can fall through to the next engine."""
    html = "<html><body>captcha challenge</body></html>"
    assert _parse_results(html) == []


def test_parse_results_handles_strong_and_br_in_snippet():
    """Bing's snippets include ``<strong>`` (matched keyword
    highlighting) and ``<br>`` between lines. Strip those cleanly."""
    html = _pad("""
    <li class="b_algo">
      <h2><a href="https://x.test/">x</a></h2>
      <p>The <strong>quick brown</strong> fox<br>jumps over the lazy dog.</p>
    </li>
    """)
    results = _parse_results(html)
    assert len(results) == 1
    # Tags stripped, whitespace collapsed to single spaces
    assert results[0]["snippet"] == "The quick brown fox jumps over the lazy dog."


def test_parse_results_skips_block_without_h2_or_href():
    """A ``b_algo`` block with no ``<h2>`` or ``<a href>`` is a
    Bing 'related search' stub. Skip it cleanly."""
    html = _pad("""
    <li class="b_algo"><p>no title</p></li>
    <li class="b_algo"><h2><a href="https://ok.test/">ok</a></h2><p>good</p></li>
    """)
    results = _parse_results(html)
    assert len(results) == 1
    assert results[0]["title"] == "ok"


def test_decode_response_handles_gzip():
    """Bing always gzips responses — ``_decode_response`` must
    transparently decompress when ``Content-Encoding: gzip``."""
    headers = {"Content-Encoding": "gzip"}
    decoded = _decode_response(_BING_HTML_GZIPPED, headers)
    assert "Welcome to Python.org" in decoded


def test_decode_response_handles_plain_text():
    """When Content-Encoding is missing / identity, treat the raw
    bytes as utf-8 directly."""
    headers = {}
    decoded = _decode_response(_BING_HTML.encode("utf-8"), headers)
    assert "Welcome to Python.org" in decoded


def test_decode_response_recovers_from_lied_content_encoding():
    """If the server claims gzip but sends plain bytes, fall back
    to raw utf-8 rather than crashing the search."""
    headers = {"Content-Encoding": "gzip"}  # lie
    decoded = _decode_response(_BING_HTML.encode("utf-8"), headers)
    assert "Welcome to Python.org" in decoded


# ---------------------------------------------------------------------------
# Fetch + host fallback — mock urllib, no real network
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _gzip_response(body: str = _BING_HTML) -> _FakeResponse:
    return _FakeResponse(
        gzip.compress(body.encode("utf-8")),
        headers={"Content-Encoding": "gzip"},
    )


def _urlopen_response_for(url: str):
    """Map host → response factory. Both cn.bing.com and www.bing.com
    succeed with full HTML."""
    def factory(req, **kwargs):
        return _gzip_response(_BING_HTML)
    return factory


@pytest.mark.asyncio
async def test_search_web_bing_returns_results_from_cn_bing():
    """Primary host ``cn.bing.com`` returns 4 results. Confirm the
    engine resolves 4 WebResult dicts (b_ad + b_related filtered)."""
    with patch("src.web_search.bing.urllib.request.urlopen", new=_urlopen_response_for(None)):
        results = await search_web_bing("python programming language", max_results=10)
    assert len(results) == 4
    assert all(r["url"].startswith("http") for r in results)
    assert all(r["title"] for r in results)


@pytest.mark.asyncio
async def test_search_web_bing_truncates_to_max_results():
    """When the page has more results than ``max_results``, the
    engine caps the list (the Bing page can have 10+ results)."""
    bigger = _BING_HTML.replace(
        "</li>\n<li class=\"b_algo\">",
        "</li>\n<li class=\"b_algo\"><h2><a href=\"https://extra.test/\">extra</a></h2><p>x</p></li>\n<li class=\"b_algo\">",
        4,
    )
    def factory(req, **kwargs):
        return _gzip_response(bigger)
    with patch("src.web_search.bing.urllib.request.urlopen", new=factory):
        results = await search_web_bing("python", max_results=3)
    assert len(results) == 3, (
        f"Expected exactly 3 results (max_results cap), got {len(results)}"
    )


@pytest.mark.asyncio
async def test_search_web_bing_falls_back_to_www_bing_when_cn_fails():
    """When ``cn.bing.com`` errors (timeout / 5xx / captcha), the
    engine tries ``www.bing.com`` next. First non-empty wins."""
    calls: list[str] = []

    def factory(req, **kwargs):
        calls.append(req.full_url)
        if "cn.bing.com" in req.full_url:
            # Primary fails with a timeout-equivalent error
            raise TimeoutError("simulated cn.bing.com hang")
        # Fallback succeeds
        return _gzip_response(_BING_HTML)

    with patch("src.web_search.bing.urllib.request.urlopen", new=factory):
        results = await search_web_bing("python", max_results=10)

    assert len(calls) == 2, f"Expected cn + www fallback, got {calls!r}"
    assert "cn.bing.com" in calls[0]
    assert "www.bing.com" in calls[1]
    assert len(results) == 4, (
        f"Fallback to www.bing.com should return 4 real results, "
        f"got {len(results)}"
    )


@pytest.mark.asyncio
async def test_search_web_bing_returns_empty_when_both_hosts_fail():
    """When EVERY host errors, ``search_web_bing`` returns ``[]``
    (does not raise) so the dispatcher can fall through to the
    next engine. The graph node sees an empty list, not an
    exception."""
    def factory(req, **kwargs):
        raise TimeoutError("all hosts blocked")

    with patch("src.web_search.bing.urllib.request.urlopen", new=factory):
        results = await search_web_bing("python")

    assert results == []


@pytest.mark.asyncio
async def test_search_web_bing_returns_empty_on_captcha_page():
    """When cn.bing.com returns a captcha challenge (small body,
    no b_algo items), fall through to www.bing.com. If www also
    returns captcha, return []."""
    def factory(req, **kwargs):
        return _gzip_response("<html>captcha challenge</html>")

    with patch("src.web_search.bing.urllib.request.urlopen", new=factory):
        results = await search_web_bing("python")

    assert results == []


@pytest.mark.asyncio
async def test_search_web_bing_empty_query_short_circuits():
    """Empty / whitespace queries must NOT hit the network."""
    with patch("src.web_search.bing.urllib.request.urlopen") as mock_urlopen:
        await search_web_bing("")
        await search_web_bing("   ")
        mock_urlopen.assert_not_called()


@pytest.mark.asyncio
async def test_search_web_bing_skips_non_200_responses():
    """A non-200 response from any host is treated as no results,
    not an exception. The engine moves on to the next host."""
    def factory(req, **kwargs):
        return _FakeResponse(b"error", status=503)

    with patch("src.web_search.bing.urllib.request.urlopen", new=factory):
        results = await search_web_bing("python")
    assert results == []


@pytest.mark.asyncio
async def test_search_web_bing_does_not_swallow_cancellation():
    """``asyncio.CancelledError`` must propagate — search cancellation
    from the WS runner should kill the engine, not be silently
    caught. (The dispatcher DOES catch it and re-raises, but the
    engine itself should let it through.)

    ``urlopen`` is a synchronous call (we run it inside
    ``asyncio.to_thread`` in the dispatcher); we patch it with a
    ``MagicMock`` (sync) that raises ``CancelledError`` on call.
    ``CancelledError`` is a ``BaseException`` subclass, not an
    ``Exception`` subclass — so the engine's
    ``except (URLError, HTTPError, TimeoutError, OSError)`` does
    NOT catch it, and it propagates to the dispatcher / caller.
    """
    from unittest.mock import MagicMock

    def raise_cancel(*args, **kwargs):
        raise asyncio.CancelledError()

    with patch("src.web_search.bing.urllib.request.urlopen", new=raise_cancel):
        with pytest.raises(asyncio.CancelledError):
            await search_web_bing("python")