"""Web search fallback package — engine-chain dispatcher + page-fetch enrichment.

v2.0.18 — public API change: ``search_web`` now returns
``list[Document]`` (not ``list[WebResult]``) and accepts an
``enrich`` kwarg plus an explicit ``preferred_domains`` override.
The ``@tool`` wrapper lives in ``tool.py`` (thin edge code that
imports LangChain's ``tool`` decorator). The internal to-Document
conversion + parallel page-fetch enrichment moved here from
``tools/web_search.py`` so the dispatcher and its post-processing
are in one file.

Exports
-------
* ``search_web`` — top-level entry point. Engine-chain dispatcher
  + Document conversion + parallel page-fetch enrichment. Returns
  ``list[Document]``. **No LangGraph types in this surface** — the
  only LangChain dependency is ``langchain_core.documents.Document``,
  which Phase 3's pure-async state machine will call directly.
* ``search_web_ddg`` / ``search_web_bing`` — direct per-engine entry
  points (advanced use; the dispatcher is what callers normally use).
* ``reset_for_tests`` — clears DDG rate-limit state.
* ``WebResult`` — TypedDict for engine-internal result dict shape.

What ``search_web`` returns
---------------------------
``list[Document]`` with ``source_kind="web"`` metadata. Empty list =
no engine in the chain returned anything AND no enrichment salvaged
the snippet. Callers fall through to the "no information found"
prompt branch.

Failure modes
-------------
* Engine returns ``[]`` → dispatcher falls through to next engine;
  if all empty, returns ``[]``.
* Engine raises (URLError, timeout, etc.) → logged at DEBUG, falls
  through.
* ``asyncio.CancelledError`` → propagated (cancellation must work).
* Page-fetch enrichment failure → snippet survives (the doc isn't
  dropped); per-URL timeout = 8s.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
from typing import Optional
from urllib.parse import urlparse

from config.constants import WEB_SEARCH
from langchain_core.documents import Document
from src.agent.nodes._web_sources import _is_web_doc, _position_score
from src.core.logging import logger

from src.web_search._domain_hints import WebResult, extract_domain_hints
from src.web_search.ddg import (
    reset_for_tests,
    search_web_ddg,
)
from src.web_search.bing import search_web_bing
from src.web_search.fetch import _fetch_one


# Env var name that overrides ``WEB_SEARCH["engines"]``. Comma-
# separated; whitespace stripped. Example:
#   RAG_WEB_SEARCH_ENGINES=bing,ddg
_ENGINES_ENV = "RAG_WEB_SEARCH_ENGINES"


# Engine registry. Each entry maps the chain-name (what the user
# puts in the env var or config) to a coroutine. Adding a new
# engine = drop a new line here, no other wiring needed.
_ENGINE_REGISTRY: dict[str, "callable"] = {
    "bing": search_web_bing,
    "ddg": search_web_ddg,
}


def _resolve_engine_list() -> list[str]:
    """Pick the engine chain. Env var wins over config.

    Falls back to ``["bing"]`` if both are unset / empty. Why Bing
    and not DDG: Bing direct scrape works behind the GFW; the ddgs
    default backend fan-out (which Bing cannot be — see the module
    docstring of ``bing.py``) does not. Users outside China can
    set ``RAG_WEB_SEARCH_ENGINES=ddg`` (or ``=bing,ddg`` to use
    Bing first and DDG as fallback).
    """
    env_value = os.environ.get(_ENGINES_ENV, "").strip()
    if env_value:
        engines = [e.strip() for e in env_value.split(",") if e.strip()]
        if engines:
            return engines
    cfg = WEB_SEARCH.get("engines") or ["bing"]
    return [str(e).strip() for e in cfg if str(e).strip()] or ["bing"]


def _doc_id_for_url(url: str) -> str:
    """Stable, short, content-derived id for a web result.

    v2.0.29.6 (Phase 5) — encode the domain segment so cross-engine
    results pointing to the same article on the same host collapse
    to a single ``doc_id``. Pre-Phase-5 used ``web-{sha1[:12]}`` of
    the URL alone; two URLs that hashed to the same digest (rare but
    possible) collided with no domain disambiguation, and operators
    couldn't tell from a ``doc_id`` which host the result came from.

    Format: ``web-{domain}-{digest}`` where ``domain`` is the URL's
    netloc with leading ``www.`` stripped and non-``[a-z0-9-]``
    characters replaced by ``-`` (truncated to 32 chars). ``digest``
    is the first 8 hex chars of ``sha1(url)`` — short enough to be
    readable in logs, long enough to be effectively unique within a
    single host (16^8 ≈ 4.3 × 10^9).
    """
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    safe_domain = re.sub(r"[^a-z0-9-]", "-", domain)[:32] or "unknown"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"web-{safe_domain}-{digest}"


def _to_doc(result: WebResult, position: int) -> Document:
    """Convert a WebResult dict into a Document with web metadata.

    ``position`` is the 0-based index of ``result`` in the engine's
    ranked list — used to compute a position-decay ``score`` so the
    frontend ``<WebSource>`` chip renders a non-zero number that
    reflects the engine's own ranking. v2.0.5 — previously hardcoded
    to ``0.0`` which made every web source indistinguishable.

    v2.0.18 — moved from ``tools/web_search.py`` into the dispatcher
    so the to-Document conversion lives next to the engine dispatch
    that produces the WebResult. Internal helper, not exported.
    """
    url = result.get("url") or ""
    return Document(
        page_content=(result.get("title") or "") + "\n\n" + (result.get("snippet") or ""),
        metadata={
            "source_kind": "web",
            "url": url,
            "domain": result.get("domain") or "",
            "filename": result.get("title") or url,
            "doc_id": _doc_id_for_url(url),
            # v2.0.29.6 (Phase 5) — deterministic per-result chunk_id.
            # Pre-Phase-5 was empty string, which made the Phase 4
            # PR-1 dedup key ``(doc_id, chunk_id, content_lead)``
            # collapse distinct web results from the same URL on the
            # same domain into a single dedup bucket. With chunk_id
            # populated, each web result keeps its own identity.
            "chunk_id": _doc_id_for_url(url),
            "score": _position_score(position),
        },
    )


async def _enrich_with_pages(
    docs: list[Document],
    max_chars: int,
) -> list[Document]:
    """Run page-fetch enrichment on each doc in parallel; return
    docs whose ``page_content`` was successfully replaced (or kept
    if fetch failed).

    Failure-tolerant: a per-URL fetch error leaves the original
    snippet in place. The doc is dropped only if its
    ``page_content`` ends up empty (no URL, fetch failed, snippet
    was empty). See ``_is_web_doc`` for the filtering criterion.

    v2.0.28.5 F-R7 — ``return_exceptions=True``. Pre-PR-3, the first
    fetch that raised (e.g. ``MemoryError`` from a malformed HTML
    parser path, or any unexpected urllib exception that escapes
    ``_fetch_one``'s internal try/except) would propagate via
    ``asyncio.gather`` and cancel the gather — the remaining tasks
    would be cancelled mid-flight, dropping ALL subsequent docs
    from the result. Now ``return_exceptions=True`` lets each task
    raise independently; we filter ``BaseException`` results to
    ``None`` (treated as "fetch failed; keep snippet") and let the
    surviving fetches complete.
    """
    try:
        tasks = [
            _fetch_one(d.metadata.get("url") or "", max_chars)
            for d in docs
            if d.metadata.get("url")
        ]
        if tasks:
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for doc, body in zip(docs, fetched):
                # F-R7 — an exception (BaseException subclass) from
                # one task must not poison the doc; treat it as a
                # fetch failure and keep the snippet.
                if isinstance(body, BaseException):
                    logger.warning(
                        f"op=web_search.enrich fetch_exception "
                        f"url={(doc.metadata or {}).get('url', '')[:80]!r} "
                        f"type={type(body).__name__}: {body}"
                    )
                    continue
                # ``body is not None`` is the success signal: an empty
                # string is a valid fetch result (a page with no
                # content); ``None`` is a failure. Catching the empty
                # case as a failure drops it.
                if body is None:
                    continue
                title = (doc.metadata or {}).get("filename") or ""
                if title:
                    doc.page_content = f"{title}\n\n{body}"
                else:
                    doc.page_content = body
    except Exception as exc:
        # Enrichment failure is non-fatal — snippet path survives.
        logger.warning(
            f"op=web_search.enrich outer_failure type={type(exc).__name__}: {exc}"
        )

    return [d for d in docs if d.page_content and _is_web_doc(d)]


async def search_web(
    query: str,
    max_results: Optional[int] = None,
    *,
    enrich: bool = True,
    preferred_domains: Optional[list[str]] = None,
) -> list[Document]:
    """Try each engine in order; return the first non-empty result,
    enriched to ``list[Document]`` with optional page-fetch.

    Returns ``[]`` when every engine returns empty (network blocked
    on every host, anti-bot on every backend, or query legitimately
    has no web results). The graph falls through to
    ``generate_answer`` with empty docs in that case.

    The dispatcher never raises — each engine implementation is
    responsible for swallowing its own exceptions and returning
    ``[]`` on failure. The dispatcher only catches
    ``asyncio.CancelledError`` (cancellation propagates correctly
    through the chain) and ``Exception`` as a final safety net so
    a buggy engine doesn't take down the graph.

    Args:
        query: User question. Empty / whitespace-only short-circuits
            to ``[]`` (consistent with pre-v1.1.10 behavior).
        max_results: Per-engine max. Each engine may cap this lower
            internally.
        enrich: When True (default), fetch each URL in parallel and
            replace the snippet with the real article body. Set False
            for tests or offline mode that don't want HTTP I/O.
        preferred_domains: Override the query-keyword-extracted
            domain hints. If None, hints are computed from the query
            via :func:`extract_domain_hints` (closes the
            "腾讯新闻 answers cite thepaper.cn" bug class).

    v2.0.18 — signature change: return type is now ``list[Document]``
    (not ``list[WebResult]``); new ``enrich`` + ``preferred_domains``
    kwargs. The ``@tool`` wrapper in ``tool.py`` translates this to
    the JSON wire format that ``react_generate`` consumes.

    v2.0.5 — page enrichment is opt-out via ``enrich=False``.

    Phase 3 readiness: the signature uses ``langchain_core.Document``
    only; no LangGraph types. The pure-async state machine will
    call this directly with the same kwargs.
    """
    if not query or not query.strip():
        return []
    engines = _resolve_engine_list()
    max_results = max_results or WEB_SEARCH.get("max_results") or 5
    per_engine_timeout = WEB_SEARCH.get("per_engine_timeout_seconds") or 8
    # v1.1.11 — compute domain hints from the query so each engine
    # can reorder results toward the source the user actually asked
    # about (e.g. "腾讯新闻" → inews.qq.com / new.qq.com / qq.com).
    # The caller can override via ``preferred_domains`` for testing
    # or when the hint is already known. Engines that don't recognize
    # this kwarg raise TypeError and the dispatcher falls through
    # silently — we wrap in try below so a future engine with a
    # stricter signature doesn't break the chain.
    if preferred_domains is None:
        preferred_domains = extract_domain_hints(query)

    # v2.0.29.6 (Phase 5) — engine chain tie-breaker: collect from
    # ALL configured engines via asyncio.gather (parallel, not
    # sequential), dedupe by URL, then sort by ``(-score, engine_priority)``
    # so the highest-scoring result wins and engine declaration order
    # is the deterministic tie-breaker. Pre-Phase-5 short-circuited
    # on the first non-empty engine — if Bing returned 5 mediocre
    # results and DDG was about to return 5 excellent ones, the user
    # only saw Bing's. The new behaviour runs both engines in parallel
    # and surfaces the best of all of them.
    engine_priority = {name: idx for idx, name in enumerate(_ENGINE_REGISTRY)}

    async def _gather_one(name: str) -> list[dict]:
        engine_fn = _ENGINE_REGISTRY.get(name)
        if engine_fn is None:
            logger.warning(
                f"web_search: unknown engine {name!r} (known: "
                f"{sorted(_ENGINE_REGISTRY.keys())}) — skipping"
            )
            return []
        try:
            results = await asyncio.wait_for(
                engine_fn(
                    query,
                    max_results=max_results,
                    preferred_domains=preferred_domains,
                ),
                timeout=per_engine_timeout,
            )
            # Stash the engine name on each hit so the tie-breaker
            # can use declared-order priority. Stripped before wire
            # emit (``_to_doc`` does not surface ``_engine``).
            for r in results or []:
                r["_engine"] = name
            return results or []
        except asyncio.TimeoutError:
            logger.debug(
                f"web_search engine {name!r} timed out after "
                f"{per_engine_timeout}s, dropping from chain"
            )
            return []
        except asyncio.CancelledError:
            # Cancellation propagates — don't swallow.
            raise
        except Exception as exc:
            # Any other engine failure: log + drop from chain.
            logger.debug(
                f"web_search engine {name!r} raised "
                f"{type(exc).__name__}: {exc} — dropping from chain"
            )
            return []

    last_engine_name = "<none>"
    gathered = await asyncio.gather(
        *[_gather_one(name) for name in engines],
        return_exceptions=True,
    )
    # Flatten, dedupe by URL, sort by (-score, engine_priority).
    merged: list[dict] = []
    seen_urls: set[str] = set()
    for engine_results in gathered:
        if isinstance(engine_results, BaseException):
            # asyncio.gather(return_exceptions=True) surfaces
            # CancelledError or other unexpected exceptions here.
            # We already handle CancelledError inside _gather_one
            # (re-raise), so this branch is for unexpected errors.
            logger.debug(
                f"web_search gather caught {type(engine_results).__name__}: "
                f"{engine_results}"
            )
            continue
        last_engine_name = engine_results[0].get("_engine", "<none>") if engine_results else last_engine_name
        for r in engine_results:
            url = r.get("url") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            merged.append(r)

    if merged:
        merged.sort(
            key=lambda r: (
                -float(r.get("score", 0.0) or 0.0),
                engine_priority.get(r.get("_engine", ""), 999),
            )
        )
        # Strip the internal ``_engine`` tag before Document conversion
        # so it doesn't leak into metadata (wire surface stays clean).
        for r in merged:
            r.pop("_engine", None)
        docs = [_to_doc(r, i) for i, r in enumerate(merged)]
        if enrich:
            docs = await _enrich_with_pages(docs, max_chars=4000)
        return docs

    # All engines returned empty / failed. Log a breadcrumb so the
    # operator can see which chain was tried when they look at
    # ``data/logs/app.log`` — the existing per-engine WARNING
    # already shows up for genuine outages.
    # v2.0.28.5 F-R9 partial — promote to WARNING + add ``op=``
    # label. Pre-PR-3 was DEBUG, invisible at the default INFO level
    # when an operator is chasing "why did this query return nothing?".
    # WARNING is appropriate because every turn hitting this branch
    # means the WHOLE web search chain failed — the user gets no web
    # grounding — and the operator should be able to spot it.
    logger.warning(
        f"op=web_search.dispatcher chain_exhausted engines={engines!r} "
        f"query={query[:40]!r} last_engine={last_engine_name!r}"
    )
    return []


__all__ = [
    "WebResult",
    "search_web",
    "search_web_ddg",
    "search_web_bing",
    "reset_for_tests",
]