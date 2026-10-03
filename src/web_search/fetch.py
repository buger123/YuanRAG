"""URL fetcher — single source of truth for "snippets aren't enough,
fetch the real page".

v2.0.18 — extracted from the pre-Phase-2 page-fetch shim (now deleted).
(now a re-export shim — to be deleted in Phase 2 Step 7). Owns:

* the urllib machinery (``_fetch_url_sync``, ``_fetch_one``)
* the shared response decoder (``_decode_response``, gzip-aware)
* the shared permissive SSL context factory (``_make_ssl_context``,
  used by both this fetcher and ``bing.py``)
* the bounded cache (``_TTLCache`` + module-level ``_FETCH_CACHE``)

Why stdlib-only (no BeautifulSoup / readability-lxml)
-----------------------------------------------------
Keeps the fetcher's import cost zero and the failure surface small.
News articles have predictable structure (``<article>`` / ``<main>`` /
``<body>`` + script/style stripping) that's good enough for the
grounding task; we don't need a full readability implementation.

Failure policy
--------------
This fetcher is **failure-tolerant by design**:

* Per-URL fetch error (timeout, 5xx, anti-bot challenge, non-HTML
  body) → ``None``, caller falls back to the original ``page_content``
  (the snippet) — never drops the document.
* Per-URL timeout = ``_PER_URL_TIMEOUT`` (8 s).
* Empty body (< 200 chars after decode) → ``None``, treated as anti-bot
  page or captcha.

Cache policy
------------
v2.0.18 — ``_FETCH_CACHE`` was an unbounded ``Dict[str, str]`` that
would grow forever in a long-lived server (a 4 GB landmine at ~1M
distinct URLs). Now a bounded TTL+LRU cache:

* TTL = ``_CACHE_TTL_SECONDS`` (10 min) — news bodies don't change
  inside 10 min for the same query, but CN publishers sometimes rotate
  ad slots. 10 min is the sweet spot: long enough that a user's
  multi-turn follow-up ("tell me more about X" 30s after the original
  "腾讯新闻 about X") hits cache, short enough that a re-ask after
  30 min gets fresh content.
* LRU cap = ``_CACHE_MAX_ENTRIES`` (512) — 5 web results × 102
  distinct topics per user session ≈ 500 entry headroom; long-lived
  servers evict naturally without growing.
* Cache key = full URL verbatim (per RFC 3986 §3.5 fragments are
  part of the URL — ``https://x.com/a#s1`` and ``https://x.com/a#s2``
  are distinct URLs and cache separately).

Sole purpose of this cap is the long-lived-server landmine — short
test runs and ad-hoc single-shot processes will never hit it.
"""
from __future__ import annotations

import asyncio
import gzip
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Optional, Tuple

from src.core.logging import logger
from src.web_search.readability import _extract_main_text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-URL timeout. Mirrors src/web_search/bing.py:_fetch_one so the two
# fetches (search vs. follow-up page fetch) feel similar to a real user.
_PER_URL_TIMEOUT = 8

# Maximum chars per document after extraction. News articles run 2k-10k
# chars; capping at 4000 keeps LLM context manageable (5 URLs × 4k = 20k
# chars ≈ 5k tokens, well under the 200k context window but enough for a
# genuine summary).
_MAX_CHARS_PER_DOC = 4000

# Cache bounds. See module docstring for the "why".
_CACHE_TTL_SECONDS = 600  # 10 minutes
_CACHE_MAX_ENTRIES = 512

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/json;q=0.8,*/*;q=0.5"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------
# Cache: TTL + LRU bounded
# ---------------------------------------------------------------------------


class _TTLCache:
    """Bounded LRU with per-entry TTL.

    Used for ``_FETCH_CACHE``: keeps recent page bodies for fast
    re-fetch within a TTL window, while bounding memory in long-lived
    processes.

    Public surface intentionally minimal: ``get(key)`` returns the
    cached value (or ``None`` on miss / TTL expiry); ``set(key, value)``
    inserts and evicts the least-recently-used entry past the cap.
    Negative caching (None values for known-bad URLs) is NOT used —
    see ``_fetch_one`` for why a failed fetch is treated as
    uncached.

    v2.0.28.5 P1-R2 — added ``self._lock`` (threading.Lock) so the
    compound get + move_to_end / set + evict sequence is atomic.
    Pre-PR-3, CPython 3.12's GIL accidentally made these safe (one
    bytecode op at a time, no concurrent mutation of the same dict).
    CPython 3.13+ free-threaded mode breaks that guarantee; even on
    3.12, two threads on a cold key both miss, both fetch the URL,
    both populate (last write wins). The lock is held only across
    the dict ops (~µs) — never across network I/O.
    """

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES, ttl_seconds: int = _CACHE_TTL_SECONDS):
        self._data: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                # Expired — drop and report miss.
                self._data.pop(key, None)
                return None
            # Touch: mark as recently used.
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (time.monotonic(), value)
            # Evict LRU entries past the cap.
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def clear(self) -> None:
        """Test-only hook to clear the cache between test cases."""
        with self._lock:
            self._data.clear()


# Process-wide cache: URL → extracted main-text. Survives across turns
# in the same process. Bounded (see module docstring).
_FETCH_CACHE = _TTLCache()


# ---------------------------------------------------------------------------
# Shared response decoder
# ---------------------------------------------------------------------------


def _decode_response(raw: bytes, headers) -> str:
    """Decode a urllib response body, transparently handling gzip.

    Shared by the page fetcher (this module) and the Bing engine
    (``bing.py``). When the response is gzip-encoded, decompress first
    then decode utf-8 with ``errors="replace"`` (so a stray surrogate
    becomes ``�`` rather than crashing — the LLM handles replacement
    chars without complaint).
    """
    if headers.get("Content-Encoding") == "gzip":
        try:
            return gzip.decompress(raw).decode("utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            pass  # fall through to plain-text decode
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Shared permissive SSL context
# ---------------------------------------------------------------------------

# v2.0.28.5 P1-R3 — module-level cached SSL context. Pre-PR-3 every
# ``_fetch_url_sync`` and Bing scrape built a fresh
# ``ssl.SSLContext`` (``ssl.create_default_context`` + 2 attribute
# sets), which is ~0.5-2 ms per call. A typical 5-URL page enrichment
# paid that cost 5×. ``SSLContext`` is immutable after construction
# (no ``set_*`` calls mutate it post-build), so sharing one instance
# across threads is safe per the stdlib contract. ``_make_ssl_context``
# stays for back-compat callers (e.g. ``bing.py``) that import it by
# name; they transparently get the same shared instance.
_SHARED_SSL_CONTEXT: "ssl.SSLContext | None" = None


def _make_ssl_context() -> ssl.SSLContext:
    """Return the process-wide permissive SSL context, building it
    lazily on first call.

    Both the Bing CDN edge and various Chinese publishers have been
    known to serve with chains that don't validate against the system
    trust store (fresh Windows installs without intermediate cert
    refresh; corporate MITM proxies that don't re-issue certs).
    Tracking the HTTP-level failure (timeout, status code) is enough
    — we don't need cert validation for *content* correctness on a
    public web search.

    v2.0.28.5 P1-R3 — cache the context at module scope instead of
    rebuilding on every call. ``SSLContext`` is documented as
    thread-safe to share once configured; we only ever read
    ``check_hostname`` / ``verify_mode`` (which are fixed at build
    time), never mutate them post-build.
    """
    global _SHARED_SSL_CONTEXT
    if _SHARED_SSL_CONTEXT is None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _SHARED_SSL_CONTEXT = ctx
    return _SHARED_SSL_CONTEXT


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _fetch_url_sync(url: str, timeout: int) -> Tuple[str, bool]:
    """Synchronous URL fetch. Returns ``(html_or_empty, success)``.

    Failure modes that resolve to ``("", False)``:
      * any ``URLError`` / ``HTTPError`` (DNS, connection refused, 4xx,
        5xx)
      * ``TimeoutError``
      * any other ``OSError``
      * non-200 response
    """
    req = urllib.request.Request(url, headers=_HEADERS)
    ctx = _make_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status != 200:
                # v2.0.28.5 F-R9 partial — promote timeouts/HTTP errors
                # to WARNING + add ``op=`` label so an operator chasing
                # a flaky page-fetch sees the signal at default INFO.
                # Non-200 non-429 responses stay at DEBUG (captcha /
                # 5xx noise is high volume).
                if resp.status == 429:
                    logger.warning(
                        f"op=web_search.fetch rate_limited url={url[:80]!r} status=429"
                    )
                else:
                    logger.debug(
                        f"page_fetch: {url} returned HTTP {resp.status}"
                    )
                return "", False
            raw = resp.read()
            return _decode_response(raw, resp.headers), True
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
    ) as exc:
        # v2.0.28.5 F-R9 partial — promote timeouts to WARNING.
        # Pre-PR-3 was DEBUG, invisible at default INFO level when
        # the operator is chasing "why is enrichment slow?" TimeoutError
        # is the highest-signal failure (network actually down / host
        # unreachable) and warrants visibility.
        exc_type = type(exc).__name__
        if isinstance(exc, TimeoutError):
            logger.warning(
                f"op=web_search.fetch timeout url={url[:80]!r} "
                f"timeout_sec={timeout}"
            )
        elif isinstance(exc, urllib.error.HTTPError) and getattr(exc, "code", None) == 429:
            logger.warning(
                f"op=web_search.fetch rate_limited url={url[:80]!r} status=429"
            )
        else:
            logger.debug(
                f"page_fetch: {url} failed: {exc_type}: {exc}"
            )
        return "", False


async def _fetch_one(
    url: str,
    max_chars: int,
    *,
    timeout: int = _PER_URL_TIMEOUT,
) -> Optional[str]:
    """Async wrapper: fetch ``url``, extract main text, return truncated.

    Returns ``None`` on any failure (caller falls back to snippet).
    Honors ``timeout`` (per-URL) by passing it to ``urllib.request.urlopen``;
    the synchronous body runs on a worker thread (``asyncio.to_thread``)
    so the event loop stays responsive while ``asyncio.gather`` fans
    out across URLs in parallel.

    Cache contract: a successful extraction is cached under the full
    URL; a failed fetch (``None``) is NOT cached — the snippet fallback
    path is the only "negative cache" and it lives in the caller, not
    here. (Otherwise a transient 503 would be pinned for 10 min.)
    """
    cached = _FETCH_CACHE.get(url)
    if cached is not None:
        return cached

    def _blocking() -> Tuple[str, bool]:
        return _fetch_url_sync(url, timeout)

    try:
        html, ok = await asyncio.to_thread(_blocking)
    except asyncio.CancelledError:
        raise  # propagate cancellation
    except Exception as exc:
        logger.debug(
            f"page_fetch: {url} raised: {type(exc).__name__}: {exc}"
        )
        return None

    if not ok or len(html) < 200:
        # Anti-bot pages / captcha / empty body. Treat as "no real
        # content" — caller falls back to the snippet so the document
        # survives.
        return None

    text = _extract_main_text(html)
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    _FETCH_CACHE.set(url, text)
    return text


__all__ = [
    "_fetch_one",
    "_decode_response",
    "_fetch_url_sync",
    "_make_ssl_context",
    "_TTLCache",
    "_MAX_CHARS_PER_DOC",
    "_PER_URL_TIMEOUT",
    "_FETCH_CACHE",
    "_HEADERS",
    "_CACHE_TTL_SECONDS",
    "_CACHE_MAX_ENTRIES",
]