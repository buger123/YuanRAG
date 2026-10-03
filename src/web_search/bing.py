"""Direct Bing HTML scrape — China-friendly default web search engine.

Why this exists
---------------
``ddgs.DDGS().text(query)`` fans out across engines (bing, brave, duckduckgo,
google, mojeek, yahoo, yandex, wikipedia). The default ``backend="auto"``
runs them in parallel and any failure to reach a blocked engine costs
the full per-engine timeout. From networks where most engines are
unreachable (China behind GFW, corporate firewalls, isolated lab
networks), the entire search degrades to "exhausted retries: timed
out" with zero results.

The v1.1.10 fix sidesteps ddgs for Bing specifically:
* ``ddgs.Bing`` has ``disabled = True`` (line 34 of the ddgs source),
  so even passing ``backend="bing"`` to ddgs falls through to another
  engine. ddgs gives us no working Bing path.
* ``https://cn.bing.com/search?q=...`` resolves in ~0.3 s from
  networks behind the GFW and returns clean HTML with stable
  selectors (``<li class="b_algo">``, ``<h2>``, ``<p>``) — none of
  the random-class-name obfuscation Baidu/Sogou use.
* ``https://www.bing.com/search?q=...`` works too but takes ~10 s
  from behind GFW (some kind of challenge page first); keep it as
  the fallback host.

This module implements the scrape with stdlib only (``urllib``,
``gzip``, ``re``) — no extra dependency, no headless browser, no
API key. Defends against the three things that go wrong in
practice: gzip-encoded response (always), anti-bot hint pages
(returned by some networks as ~5 KB HTML), and 5xx errors.

The shape we return is the same ``WebResult`` dict the rest of
``src.web_search`` uses — title/url/snippet/domain. ``_domain_of``
is imported from ``duckduckgo`` to keep domain extraction in one
place; if Bing returns a CN domain or tracking redirect, it goes
through the same parser.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from typing import Optional

from src.core.logging import logger
# v2.0.18 — ``WebResult`` (TypedDict) + ``_domain_of`` moved here from
# ``src/web_search/ddg.py`` so the type and its constructor helper live
# in one place.
from src.web_search._domain_hints import WebResult, _domain_of, rerank_by_preferred
# v2.0.18 — shared gzip decoder + permissive SSL context (was
# duplicated between Bing's own urllib loop and page_fetch's loop).
from src.web_search.fetch import _decode_response, _make_ssl_context


# Public entry points in priority order. ``cn.bing.com`` resolves
# fastest from networks behind the GFW; ``www.bing.com`` is the
# fallback when the CN edge node is rate-limiting or returning a
# captcha challenge page (rare — Bing treats repeated scraping from
# the same IP the same way Google does, but 5s timeout caps the
# worst case).
_BING_HOSTS = ("https://cn.bing.com/search", "https://www.bing.com/search")

# Static regexes for the only structural HTML elements Bing's search
# results pages use that have been stable since the 2019 redesign.
# Bing obfuscates the surrounding chrome (header, ads, related
# searches) with random CSS class names, but the ``b_algo`` result
# containers themselves haven't changed.
_RE_RESULT_ITEM = re.compile(
    r'<li class="b_algo"[^>]*>(.*?)</li>',
    re.DOTALL,
)
_RE_TITLE = re.compile(r"<h2[^>]*>(.*?)</h2>", re.DOTALL)
_RE_HREF = re.compile(r'href="(https?://[^"]+)"')
_RE_BODY = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL)
_RE_TAG_STRIP = re.compile(r"<[^>]+>")

# Anti-bot / captcha pages return a very small HTML body with no
# ``b_algo`` items. Detect by minimum body length rather than
# substring (Bing's anti-bot HTML changes weekly; substring
# detection rots in days).
_MIN_BODY_FOR_VALID_RESULTS = 5000

# Headers mimic a vanilla desktop Chrome. Bing serves a smaller /
# less-bot-resistant HTML to UAs that look like scripts or
# curl/python-urllib default. ``Accept-Language`` is set
# dynamically per call (see ``_headers_for_locale``) so Chinese
# queries get a CN index and English queries get the international
# one — the default ``en-US,en;q=0.9`` previously shipped here was
# sending every CN query through the en-US fallback path.
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _headers_for_locale(locale: str) -> dict[str, str]:
    """Build request headers for the requested locale.

    ``locale`` is one of ``"zh-CN"``, ``"en-US"``, or ``"auto"``.
    ``"auto"`` sniffs the query for CJK characters and picks the
    matching locale. The locale controls both ``Accept-Language``
    (which CN users absolutely need — Bing routes based on it) and
    ``X-Forwarded-For``-style hints (which help with CN edge nodes
    that do geographic throttling).

    v2.0.4 — closing the "Bing returns en-US results for a CN
    query" root cause that fed P0-1 (LLM had to invent CN-local
    facts because Bing returned the wrong index).
    """
    if locale not in ("zh-CN", "en-US"):
        locale = "en-US"
    return {
        "User-Agent": _DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": (
            "zh-CN,zh;q=0.9,en;q=0.5" if locale == "zh-CN"
            else "en-US,en;q=0.9"
        ),
        "Accept-Encoding": "gzip, deflate",
    }


def _detect_locale(query: str) -> str:
    """Return ``"zh-CN"`` if ``query`` contains CJK characters, else
    ``"en-US"``. Used by ``search_web_bing`` when the caller passes
    ``locale="auto"`` (the default)."""
    if not query:
        return "en-US"
    for ch in query:
        # CJK Unified Ideographs + Hiragana/Katakana + Hangul
        if (
            "一" <= ch <= "鿿"
            or "぀" <= ch <= "ヿ"
            or "가" <= ch <= "힯"
        ):
            return "zh-CN"
    return "en-US"


def _decode_response(raw: bytes, headers) -> str:
    """Back-compat re-export — actual implementation is
    ``src/web_search.fetch._decode_response``. Bing-specific callers
    that import this name keep working; new code should import from
    ``src.web_search.fetch`` directly."""
    from src.web_search.fetch import _decode_response as _impl
    return _impl(raw, headers)


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Bing's snippets include inline ``<strong>`` (matched keyword
    highlighting) and ``<br>`` between lines; we want clean text.
    ``errors="replace"`` above means a stray surrogate won't crash
    the decode but might leave a ``�`` in the snippet —
    acceptable; the LLM handles replacement chars without complaint.

    ``<br>`` becomes a literal space BEFORE tag stripping so the
    word that follows doesn't fuse into the previous one (e.g.
    ``"fox<br>jumps"`` → ``"fox jumps"``, not ``"foxjumps"``).
    """
    # Normalize <br> / <br/> / <br /> to a space. Case-insensitive
    # so ``<BR>`` / ``<Br>`` are also handled.
    spaced = re.sub(r"<br\s*/?>", " ", html, flags=re.IGNORECASE)
    text = _RE_TAG_STRIP.sub("", spaced)
    # Collapse runs of whitespace. Newlines inside snippets come
    # from <br> tags; ``\\s+`` covers those plus tab/space runs.
    return re.sub(r"\s+", " ", text).strip()


def _parse_results(html: str) -> list[WebResult]:
    """Extract WebResult dicts from a Bing results HTML page.

    Returns ``[]`` for:
    * Anti-bot / captcha pages (body too small)
    * Bing's "no results found" empty result page
    * Any parse failure (log debug, never raise)
    """
    if len(html) < _MIN_BODY_FOR_VALID_RESULTS:
        logger.debug(
            f"Bing response body too small ({len(html)}B) — "
            "likely anti-bot / captcha page"
        )
        return []

    out: list[WebResult] = []
    for item in _RE_RESULT_ITEM.findall(html):
        title_m = _RE_TITLE.search(item)
        href_m = _RE_HREF.search(item)
        body_m = _RE_BODY.search(item)
        # All three are required for a usable result. A ``b_algo``
        # block with no <h2> or no href is a "related search" stub
        # Bing sometimes injects — skip those.
        if not (title_m and href_m):
            continue
        title = _strip_html(title_m.group(1))
        url = href_m.group(1).strip()
        # Bing sometimes prepends ``bing.com/ck/...`` redirector
        # URLs to outbound links; the actual URL is in the href
        # attribute on the title <a> tag. ``_RE_HREF`` already
        # matched the first ``href="https://..."`` which is the
        # real one — but as defense in depth, filter out any
        # ``bing.com/`` hostnames since they're always tracking
        # redirects, not real destinations.
        if "bing.com/" in url:
            continue
        snippet = _strip_html(body_m.group(1)) if body_m else ""
        if not title or not url:
            continue
        out.append(
            WebResult(
                title=title,
                url=url,
                snippet=snippet,
                domain=_domain_of(url),
            )
        )
    return out


def _fetch_one(host: str, query: str, max_results: int, timeout: int, locale: str = "en-US", *, rate_limited: list[bool] | None = None) -> list[WebResult]:
    """One Bing host attempt. Returns ``[]`` on any error or empty page.

    Catches every exception type urllib raises plus HTTPError so a
    single blocked host doesn't crash the fallback chain. The
    fallback chain upstream (``search_web``) decides what to do with
    an empty list.

    v2.0.4 — adds ``mkt=<locale>`` and ``setlang=<locale>`` to the
    Bing URL so the search engine returns language-matched results
    instead of the default en-US index. ``setlang`` controls UI
    chrome (which we don't see); ``mkt`` controls result content
    ranking and is the load-bearing one for our use case.

    v2.0.28.5 F-R6 — 429 short-circuit. ``rate_limited`` is a
    single-element list used as a mutable flag shared across the
    host loop; when a host returns HTTP 429, we set
    ``rate_limited[0] = True`` and return ``[]`` so the caller can
    short-circuit the rest of the Bing chain (since rate-limit
    typically affects both hosts in the same ASN). HTTPError with
    status 429 is the canonical "Retry-After" response shape, but
    Bing's CDN often returns 429 as a plain status without the
    Retry-After header — so we treat the status code as the
    load-bearing signal.
    """
    from urllib.parse import quote

    qs = (
        f"q={quote(query)}"
        f"&mkt={quote(locale)}"
        f"&setlang={quote(locale)}"
        # Force Bing to honour our locale even if its CDN returns a
        # cached version: ``cc=`` and ``first=1`` are v1-stable
        # params Bing has supported since the 2018 redesign.
        f"&cc={quote(locale.split('-')[-1])}"
        f"&first=1"
    )
    url = f"{host}?{qs}"
    req = urllib.request.Request(url, headers=_headers_for_locale(locale))
    # v2.0.18 — shared permissive SSL context (was inlined here +
    # duplicated in page_fetch.py). See src/web_search/fetch.py.
    ctx = _make_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status == 429:
                # v2.0.28.5 F-R6 — rate-limited; signal the caller.
                logger.warning(
                    f"op=web_search.bing rate_limited host={host!r} "
                    f"query={query[:40]!r} status=429"
                )
                if rate_limited is not None:
                    rate_limited[0] = True
                return []
            if resp.status != 200:
                logger.debug(f"Bing {host} returned HTTP {resp.status}")
                return []
            raw = resp.read()
            html = _decode_response(raw, resp.headers)
    except urllib.error.HTTPError as exc:
        # 429 raised as HTTPError by urllib (CDN doesn't always
        # return 429 as a plain status). Promote to WARNING so an
        # operator sees rate-limit pressure in logs.
        if getattr(exc, "code", None) == 429:
            logger.warning(
                f"op=web_search.bing rate_limited host={host!r} "
                f"query={query[:40]!r} status=429 (HTTPError)"
            )
            if rate_limited is not None:
                rate_limited[0] = True
            return []
        logger.debug(f"Bing {host} returned HTTP {exc.code}")
        return []
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ) as exc:
        logger.debug(f"Bing {host} fetch failed: {type(exc).__name__}: {exc}")
        return []

    results = _parse_results(html)
    # Truncate to ``max_results`` here so callers don't need to.
    if len(results) > max_results:
        results = results[:max_results]
    return results


async def search_web_bing(
    query: str,
    max_results: Optional[int] = None,
    timeout: int = 8,
    preferred_domains: Optional[list[str]] = None,
    locale: str = "auto",
) -> list[WebResult]:
    """Search Bing for ``query`` and return up to ``max_results`` results.

    Tries each Bing host in priority order (``cn.bing.com`` first,
    ``www.bing.com`` second). First host that returns at least one
    result wins. If every host returns empty (anti-bot on both, or
    transient 5xx), returns ``[]`` so the fallback chain can try
    the next engine.

    ``timeout`` is per-host — the caller should pick something like
    ``max_results // 2 + 4`` so a 5-result search waits at most
    ~6-7s across both hosts. We don't share a single timeout across
    the chain because cn.bing.com usually returns in 0.3s and the
    ``www.bing.com`` fallback is rare.

    ``preferred_domains`` (v1.1.11) — if non-empty, results whose
    domain matches one of the preferred entries are reordered to
    the front (preserving declaration order and intra-bucket order).
    The total result count is still capped by ``max_results`` AFTER
    rerank, so a query with many inews.qq.com hits plus a few
    unrelated ones returns the top-N inews.qq.com first. Used by
    the dispatcher to nudge "腾讯新闻" queries toward qq.com results
    so the LLM cites the right source.

    ``locale`` (v2.0.4) — ``"auto"`` (default) sniffs the query for
    CJK characters and picks ``"zh-CN"`` / ``"en-US"``. The locale
    is passed as Bing URL params ``mkt=`` / ``setlang=`` /
    ``cc=`` so Bing returns language-matched results. Without this,
    a CN query to ``www.bing.com`` would silently return the en-US
    index, which is the root cause of P0-1's "fabricated CN-local
    facts" bug.

    v2.0.28.5 F-R6 — 429 short-circuit. If the first host returns
    429 (rate-limited), the second host almost certainly will too
    (same ASN, same client IP). Stop walking the chain and return
    ``[]`` immediately so the dispatcher falls through to the next
    engine (e.g. DDG) instead of paying another timeout.
    """
    if not query or not query.strip():
        return []
    if locale == "auto":
        locale = _detect_locale(query)
    max_results = max_results or 5
    # Mutable single-element list shared across the host loop — Python
    # idiom for "nonlocal bool" without ``global`` / ``nonlocal`` in
    # nested function scope.
    rate_limited: list[bool] = [False]
    last_results: list[WebResult] = []
    for host in _BING_HOSTS:
        results = _fetch_one(host, query, max_results, timeout, locale=locale, rate_limited=rate_limited)
        if rate_limited[0]:
            # F-R6 — first host signalled 429; skip remaining hosts.
            logger.debug(
                f"Bing: chain short-circuited due to 429 from {host!r}"
            )
            return []
        if results:
            return rerank_by_preferred(results, preferred_domains)
        # Keep the most-recent empty-page result to log if even the
        # fallback returned nothing useful — but only as a debug
        # breadcrumb, not as a warning (legit "no results" queries
        # legitimately return empty).
        last_results = results
    if not last_results:
        logger.debug("Bing: all hosts returned no usable results")
    return []


__all__ = ["search_web_bing"]