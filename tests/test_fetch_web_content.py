"""Tests for ``src.web_search.fetch`` + ``src.web_search.readability``.

The ReAct rewrite (v2.0) killed the ``fetch_web_content`` graph node
itself (the linear ``search_web → fetch_web_content → generate``
topology no longer exists). What survived is the URL fetcher
(``_fetch_one`` + the readability-style extraction helpers +
``_FETCH_CACHE``) — ``src/web_search/__init__.py::search_web`` uses
it to enrich snippet-only search results with real article content.

What we lock here
-----------------
1. ``_extract_main_text`` prefers ``<article>`` over ``<main>`` over
   ``<body>`` and strips script / style / nav / footer / header /
   aside.
2. ``<br>`` becomes a space (no word-boundary fusion).
3. ``_fetch_url_sync`` returns ``("", False)`` on timeout / 4xx /
   5xx / HTTPError.
4. ``_fetch_one`` returns ``None`` for anti-bot-sized bodies (< 200
   chars after extraction).
5. ``_FETCH_CACHE`` dedupes URLs within a process.
6. Long content is truncated to ``_MAX_CHARS_PER_DOC``.

The orchestration logic that used to live in
``nodes.fetch_web_content`` (calling ``_fetch_one`` in parallel via
``asyncio.gather``, building the state delta, propagating
``CancelledError``) is gone with the node. The
``src/web_search/__init__.py::search_web`` call site exercises it
end-to-end.
"""
from __future__ import annotations

import asyncio
import gzip
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

# v2.0.18 — urllib machinery + readability extraction both live in
# the ``src/web_search/`` package now. ``page_fetch.py`` was the
# pre-v2.0.18 home of these symbols; that file was deleted in
# Phase 2 Step 7.
from src.web_search.fetch import (
    _FETCH_CACHE,
    _MAX_CHARS_PER_DOC,
    _decode_response,
    _fetch_one,
    _fetch_url_sync,
)
from src.web_search.readability import _extract_main_text, _strip_html


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------


_ARTICLE_HTML = """
<!DOCTYPE html>
<html><head><title>News</title></head>
<body>
<nav>navigation</nav>
<article>
<h1>Apple launches M4 MacBook Pro</h1>
<p>The new MacBook Pro features the M4 chip and ships next month.
Industry analysts expect strong demand from creative professionals.</p>
<p>Pricing starts at $1,599 for the base 14-inch model.</p>
</article>
<footer>copyright</footer>
</body>
</html>
""" + ("<!-- " + "x" * 6000 + " -->")


_MAIN_HTML = """
<!DOCTYPE html>
<html><head><title>Sports</title></head>
<body>
<header>site header</header>
<main>
<h1>Champions League Final</h1>
<p>Real Madrid defeated Manchester City 3-1 in a thrilling final.</p>
<p>Bellingham scored the winning goal in stoppage time.</p>
</main>
<aside>sidebar ads</aside>
</body>
</html>
""" + ("<!-- " + "x" * 6000 + " -->")


_BODY_HTML = """
<!DOCTYPE html>
<html><head><title>Tech</title></head>
<body>
<div class="content">
<p>This article covers the latest developments in AI safety research.</p>
<p>Multiple labs have published new findings on alignment techniques.</p>
</div>
</body>
</html>
""" + ("<!-- " + "x" * 6000 + " -->")


_LONG_HTML = """
<!DOCTYPE html>
<html><head><title>Long</title></head>
<body>
<article>
<p>""" + ("word " * 2000) + """</p>
</article>
</body>
</html>
""" + ("<!-- " + "x" * 6000 + " -->")


# ---------------------------------------------------------------------------
# Mock urllib infrastructure
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


def _gzip_html_response(html: str) -> _FakeResponse:
    return _FakeResponse(
        gzip.compress(html.encode("utf-8")),
        headers={"Content-Encoding": "gzip"},
    )


def _reset_cache():
    """Per-test cache wipe so cross-test state doesn't leak."""
    _FETCH_CACHE.clear()


@pytest.fixture(autouse=True)
def _clear_cache_each_test():
    _reset_cache()
    yield
    _reset_cache()


# ---------------------------------------------------------------------------
# Pure-Python extraction tests (no network)
# ---------------------------------------------------------------------------


def test_extract_main_text_prefers_article():
    """``<article>`` wins over ``<main>`` and ``<body>``."""
    html = _ARTICLE_HTML + _MAIN_HTML + _BODY_HTML
    text = _extract_main_text(html)
    assert "Apple launches M4 MacBook Pro" in text
    assert "Champions League Final" not in text
    assert "navigation" not in text
    assert "copyright" not in text


def test_extract_main_text_falls_back_to_main():
    """No ``<article>`` → use ``<main>``."""
    text = _extract_main_text(_MAIN_HTML)
    assert "Champions League Final" in text
    assert "Real Madrid" in text
    assert "site header" not in text
    assert "sidebar ads" not in text


def test_extract_main_text_falls_back_to_body():
    """No ``<article>`` or ``<main>`` → use ``<body>`` after stripping
    script / style / nav / footer / header / aside."""
    text = _extract_main_text(_BODY_HTML)
    assert "AI safety research" in text
    assert "alignment techniques" in text


def test_extract_main_text_strips_script_and_style():
    """``<script>`` and ``<style>`` blocks must NOT bleed into the
    extracted content."""
    html = """
    <html><body>
    <article>
      <script>alert('xss');</script>
      <style>.foo { color: red; }</style>
      <p>Real article text.</p>
    </article>
    </body></html>
    """ + ("<!-- " + "x" * 6000 + " -->")
    text = _extract_main_text(html)
    assert "Real article text" in text
    assert "alert" not in text
    assert "color: red" not in text


def test_extract_main_text_strips_br_to_space():
    """``<br>`` must become a space, not vanish — so word boundaries
    don't fuse (``"fox<br>jumps"`` → ``"fox jumps"``, not ``"foxjumps"``)."""
    html = """
    <html><body>
    <article>
      <p>The <strong>quick brown</strong> fox<br>jumps over the lazy dog.</p>
    </article>
    </body></html>
    """ + ("<!-- " + "x" * 6000 + " -->")
    text = _extract_main_text(html)
    assert "fox jumps" in text
    assert "foxjumps" not in text


def test_extract_main_text_returns_empty_for_empty_html():
    assert _extract_main_text("") == ""


def test_strip_html_preserves_text_drops_tags():
    """``_strip_html`` is the inner helper that converts HTML to text.
    Tags vanish, text remains, whitespace collapses."""
    out = _strip_html("<p>hello <b>world</b></p>  <br>foo")
    assert "hello world" in out
    assert "foo" in out
    assert "<" not in out


def test_decode_response_handles_gzip():
    """``_decode_response`` transparently gunzips when header is set."""
    raw = gzip.compress(b"hello world")
    headers = {"Content-Encoding": "gzip"}
    assert _decode_response(raw, headers) == "hello world"


def test_decode_response_passthrough_on_plain():
    """No gzip header → raw ``bytes.decode`` with ``errors='replace'``."""
    raw = b"plain text"
    headers = {}
    assert _decode_response(raw, headers) == "plain text"


# ---------------------------------------------------------------------------
# Sync _fetch_url_sync tests (mock urllib at the src.web_search.fetch surface)
# ---------------------------------------------------------------------------


def test_fetch_url_sync_returns_html_on_200():
    with patch(
        "src.web_search.fetch.urllib.request.urlopen",
        return_value=_gzip_html_response(_ARTICLE_HTML),
    ):
        html, ok = _fetch_url_sync("https://x.test/article", 5)
    assert ok is True
    assert "Apple launches M4 MacBook Pro" in html


def test_fetch_url_sync_returns_empty_on_timeout():
    with patch(
        "src.web_search.fetch.urllib.request.urlopen",
        side_effect=TimeoutError("simulated hang"),
    ):
        html, ok = _fetch_url_sync("https://x.test/timeout", 1)
    assert ok is False
    assert html == ""


def test_fetch_url_sync_returns_empty_on_404():
    def factory(req, **kwargs):
        raise __import__("urllib.error").error.HTTPError(
            "https://x.test/404", 404, "Not Found", {}, None
        )

    with patch(
        "src.web_search.fetch.urllib.request.urlopen",
        new=factory,
    ):
        html, ok = _fetch_url_sync("https://x.test/404", 5)
    assert ok is False
    assert html == ""


def test_fetch_url_sync_returns_empty_on_503():
    def factory(req, **kwargs):
        return _FakeResponse(b"server error", status=503)

    with patch(
        "src.web_search.fetch.urllib.request.urlopen",
        new=factory,
    ):
        html, ok = _fetch_url_sync("https://x.test/503", 5)
    assert ok is False
    assert html == ""


# ---------------------------------------------------------------------------
# Async _fetch_one tests (anti-bot rejection, cache, truncation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_one_returns_none_on_anti_bot_small_body():
    """A short body without real article content is rejected as
    anti-bot / captcha by ``_fetch_one`` (which calls
    ``_extract_main_text`` and returns ``None`` when the body is too
    small to be a real article)."""
    def factory(req, **kwargs):
        return _gzip_html_response("<html>captcha</html>")

    with patch(
        "src.web_search.fetch.urllib.request.urlopen",
        new=factory,
    ):
        result = await _fetch_one(
            "https://x.test/captcha", max_chars=4000, timeout=5
        )
    assert result is None, (
        f"Anti-bot body should be rejected; got {result!r}"
    )


@pytest.mark.asyncio
async def test_fetch_one_caches_second_call():
    """Within a process, the second call for the same URL must hit
    the in-process cache, not ``urllib``. This is what makes the
    "同一个 URL 两次 fetch 返回不同内容" complaint impossible."""
    call_count = {"n": 0}

    def factory(req, **kwargs):
        call_count["n"] += 1
        return _gzip_html_response(_ARTICLE_HTML)

    with patch(
        "src.web_search.fetch.urllib.request.urlopen",
        new=factory,
    ):
        a = await _fetch_one("https://x.test/cache", max_chars=4000, timeout=5)
        b = await _fetch_one("https://x.test/cache", max_chars=4000, timeout=5)

    assert call_count["n"] == 1, (
        f"Expected cache to dedup; urllib was called {call_count['n']} times"
    )
    assert a == b
    assert a is not None
    assert "Apple launches M4 MacBook Pro" in a


@pytest.mark.asyncio
async def test_fetch_one_truncates_long_content():
    """Article body longer than ``max_chars`` is truncated to fit so
    a single 50k-word article can't blow the context window."""
    def factory(req, **kwargs):
        return _gzip_html_response(_LONG_HTML)

    with patch(
        "src.web_search.fetch.urllib.request.urlopen",
        new=factory,
    ):
        text = await _fetch_one(
            "https://x.test/long", max_chars=1000, timeout=5
        )

    assert text is not None
    # Truncated to roughly max_chars + tiny ellipsis.
    assert len(text) <= 1005


def test_max_chars_per_doc_constant():
    """Pin the constant value — web_search.py uses 4000 to compute
    per-doc budget. Drift here would silently change context window
    pressure."""
    assert _MAX_CHARS_PER_DOC == 4000


def test_decode_response_handles_broken_gzip():
    """If the server claims gzip but the body isn't valid gzip, fall
    through to plain decode rather than raising."""
    raw = b"not actually gzip"
    headers = {"Content-Encoding": "gzip"}
    # Must not raise; returns the raw text via the plain-decode path.
    out = _decode_response(raw, headers)
    assert "not actually gzip" in out