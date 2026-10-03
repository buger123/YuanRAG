"""DuckDuckGo async wrapper with rate-limit safety.

v2.0.18 — file renamed from ``duckduckgo.py`` to ``ddg.py``. The
package ``src/web_search/`` (this directory) and the engine module
have confusingly similar names; the rename eliminates that ambiguity
and removes the backreference ``src.web_search.duckduckgo.search_web``
that confused grep users. Function names + behavior are unchanged.

The ``search_web_ddg`` coroutine returns a list of ``WebResult``
dicts (title, url, snippet, domain). DuckDuckGo's HTML endpoints
aggressively throttle (~30 req/min/IP triggers HTTP 202). We defend
with three layers:

1. ``asyncio.Semaphore(max_concurrency)`` — caps in-flight DDG calls.
2. Global wall-clock gate (``min_interval_seconds``) — every successful
   ``search_web_ddg`` bumps a module-level timestamp; subsequent calls
   wait until that interval has elapsed before issuing the HTTP request.
3. Exponential backoff retries on ``RatelimitException`` or empty result,
   up to ``max_attempts`` (1.5s / 3.0s / 6.0s by default).

Failure mode: if every attempt fails (transient DDG outage, persistent
rate-limit, network down), ``search_web_ddg`` returns ``[]``. The graph
falls through to ``generate_answer`` with empty docs; the existing
``GENERATE_SYSTEM`` prompt produces a clean "no information found"
answer.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

from config.constants import WEB_SEARCH
from src.core.logging import logger
# v2.0.18 — ``WebResult`` (TypedDict) and ``_domain_of`` moved out of
# this module into ``src/web_search/_domain_hints.py`` because they're
# always used together (``WebResult(title=..., domain=_domain_of(url),
# ...)``). Co-locating the type and the helper keeps the construction
# site readable.
from src.web_search._domain_hints import WebResult, _domain_of, rerank_by_preferred


# Module-level state. Tests call ``reset_for_tests()`` to skip the gate
# (so a hundred-test suite doesn't take 1.5 s per call).
_semaphore: asyncio.Semaphore = asyncio.Semaphore(WEB_SEARCH["max_concurrency"])
_last_call_ts: float = 0.0
_lock: asyncio.Lock = asyncio.Lock()
_CLOCK = time.monotonic


def _now() -> float:
    return _CLOCK()


async def _gate() -> None:
    """Wait until at least ``min_interval_seconds`` has elapsed since the
    last successful ``search_web_ddg`` call. Cancellable; safe under
    concurrent callers (lock).
    """
    global _last_call_ts
    interval = WEB_SEARCH["min_interval_seconds"]
    async with _lock:
        elapsed = _now() - _last_call_ts
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)
        _last_call_ts = _now()


async def _attempt(query: str, max_results: int, region: str, safesearch: str) -> list[dict]:
    """One DuckDuckGo round-trip. Imports ddgs lazily so the module is
    importable in environments without the optional dep (tests).

    ddgs >= 9.x dropped ``AsyncDDGS`` and now only exposes the synchronous
    ``DDGS`` class. We wrap the blocking ``text()`` call in
    ``asyncio.to_thread`` so the event loop stays responsive — the
    semaphore above still caps in-flight DDG calls.
    """
    # Lazy import — keeps import-time side-effect free for tests that
    # never call search_web.
    from ddgs import DDGS  # type: ignore

    def _blocking() -> list[dict]:
        # DDGS is a context manager that cleans up its internal HTTP
        # client on exit; we always use it that way so connection
        # resources don't leak across calls.
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
            )
            return list(results or [])

    return await asyncio.to_thread(_blocking)


async def search_web_ddg(
    query: str,
    max_results: Optional[int] = None,
    region: Optional[str] = None,
    safesearch: Optional[str] = None,
    preferred_domains: Optional[list[str]] = None,
) -> list[WebResult]:
    """Search DuckDuckGo for ``query`` and return up to ``max_results``
    results. Retries with exponential backoff on rate-limit / empty result.

    Renamed from ``search_web`` to ``search_web_ddg`` in v1.1.10 — the
    dispatcher in ``src/web_search/__init__.py`` calls this as one
    engine in a chain. Tests that previously patched
    ``src.web_search.duckduckgo.search_web`` need to patch
    ``search_web_ddg`` instead. The ``src.agent.nodes.search_web``
    node continues to import ``search_web`` (the dispatcher) so its
    call site doesn't change.

    v2.0.18 — module moved from ``src/web_search/duckduckgo.py`` to
    ``src/web_search/ddg.py``; function names unchanged.

    ``preferred_domains`` (v1.1.11) — accepted for parity with the
    Bing engine. If non-empty, results whose domain matches one of
    the preferred entries are reordered to the front before being
    returned. The ``search_web`` graph node computes hints from the
    query (e.g. "腾讯新闻" → ``["inews.qq.com", "new.qq.com",
    "qq.com"]``) and forwards them here so the LLM cites the
    publisher the user actually asked about rather than whatever
    third-party mirror showed up first in the DDG results.
    """
    if not query or not query.strip():
        return []

    max_results = max_results or WEB_SEARCH["max_results"]
    region = region or WEB_SEARCH["region"]
    safesearch = safesearch or WEB_SEARCH["safesearch"]
    backoff = WEB_SEARCH["backoff_seconds"]
    max_attempts = WEB_SEARCH["max_attempts"]

    # ddgs may raise RatelimitException or return []. We treat both as
    # transient — retry with backoff up to max_attempts. Last error wins.
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        async with _semaphore:
            await _gate()
            try:
                raw = await _attempt(query, max_results, region, safesearch)
                if raw:
                    out: list[WebResult] = []
                    for r in raw:
                        url = (r.get("href") or r.get("url") or "").strip()
                        if not url:
                            continue
                        out.append(
                            WebResult(
                                title=(r.get("title") or "").strip(),
                                url=url,
                                snippet=(r.get("body") or r.get("snippet") or "").strip(),
                                domain=_domain_of(url),
                            )
                        )
                    if out:
                        return rerank_by_preferred(out, preferred_domains)
                # Empty response — treat as transient.
                logger.debug(f"DDG returned no results on attempt {attempt + 1}")
            except Exception as exc:
                last_exc = exc
                logger.debug(f"DDG attempt {attempt + 1} failed: {exc}")
        # Backoff between attempts (skip after last attempt).
        if attempt + 1 < max_attempts:
            try:
                await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
            except IndexError:
                await asyncio.sleep(backoff[-1])

    if last_exc is not None:
        logger.warning(f"search_web_ddg exhausted retries: {last_exc}")
    return []


def reset_for_tests() -> None:
    """Reset the rate-limit gate so tests don't sleep 1.5 s between calls.
    Also replaces the semaphore (so a fresh test event loop gets a fresh
    un-acquired semaphore — otherwise a semaphore created on a different
    loop would deadlock here).
    """
    global _semaphore, _last_call_ts, _lock
    try:
        # Best-effort release of in-flight permits.
        for _ in range(WEB_SEARCH["max_concurrency"]):
            _semaphore.release()
    except ValueError:
        pass
    _semaphore = asyncio.Semaphore(WEB_SEARCH["max_concurrency"])
    _last_call_ts = 0.0
    _lock = asyncio.Lock()


__all__ = ["WebResult", "search_web_ddg", "reset_for_tests"]