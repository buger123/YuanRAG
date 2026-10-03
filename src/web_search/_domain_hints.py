"""Domain hints: query keyword → preferred domain list, plus re-ranker.

v1.1.11 — closing the "cited URLs don't match the source" bug class.

What this module does
---------------------
When a user asks about a specific source (e.g. "腾讯新闻", "澎湃新闻",
"人民日报"), search engines return a mix of URLs from many domains —
some from the named source, some from third-party mirrors, some
entirely unrelated. Without a domain preference signal, the LLM picks
arbitrary URLs to cite and the user ends up with `[1] thepaper.cn`
labeled as 腾讯新闻 — fabrication by misattribution.

This module provides two small functions:

1. :func:`extract_domain_hints` — scans a query for keywords that
   point to a specific publisher / source and returns the ordered
   list of preferred domains. Used by both the dispatcher (CLI /
   debug path) and the ``search_web`` graph node to inject domain
   preference into the search call.

2. :func:`rerank_by_preferred` — re-orders an existing list of
   ``WebResult`` dicts so that URLs matching ``preferred_domains``
   appear first. Order within the preferred bucket is preserved
   (it's a stable sort); order of non-preferred is preserved.

We do NOT filter out non-preferred results — that's too aggressive.
If the user asks about 腾讯新闻 and Bing returns zero qq.com results,
we still want the LLM to see whatever came back (with appropriate
caveats in the answer). Re-ranking only nudges the LLM toward the
right citations when both buckets are non-empty.

Sharing
-------
Both ``src/web_search/bing.py`` and ``src/web_search/ddg.py``
import :func:`rerank_by_preferred` from here, so the same preference
logic applies regardless of which engine actually returned the URLs.
The dispatcher and the ``search_web`` graph node compute hints via
:func:`extract_domain_hints` and forward them to whichever engine
they call.
"""
from __future__ import annotations

import re
from typing import Iterable, List, TypedDict
from urllib.parse import urlparse as _urlparse


# Keyword → ordered domain preference list.
#
# Add new entries by appending; the first matching keyword wins, so
# longer / more specific phrases go BEFORE shorter ones
# ("腾讯新闻" before "腾讯"). Order WITHIN a list also matters:
# earlier domains are preferred over later ones.
#
# Coverage rationale (v1.1.11): the bug we shipped a fix for was
# "腾讯新闻 query returned 0/4 from the qq.com family". Other
# publishers are listed pre-emptively based on the same failure
# shape — a query that names a publisher but gets URLs from many
# unrelated domains.
_DOMAIN_HINTS: list[tuple[str, list[str]]] = [
    # Specific phrases first (longer match wins on ties)
    ("腾讯新闻", ["inews.qq.com", "new.qq.com", "qq.com"]),
    ("腾讯网", ["new.qq.com", "qq.com", "inews.qq.com"]),
    ("腾讯", ["qq.com", "inews.qq.com", "new.qq.com"]),
    ("澎湃新闻", ["thepaper.cn"]),
    ("澎湃", ["thepaper.cn"]),
    ("人民日报", ["paper.people.com.cn", "peopleapp.com", "people.cn"]),
    ("新华社", ["news.cn", "xinhuanet.com"]),
    ("央视新闻", ["cctv.com", "news.cctv.com"]),
    ("央视", ["cctv.com"]),
    ("界面", ["jiemian.com"]),
    ("36氪", ["36kr.com"]),
    ("虎嗅", ["huxiu.com"]),
    ("财新", ["caixin.com"]),
    ("路透", ["reuters.com"]),
    ("彭博", ["bloomberg.com"]),
    ("BBC", ["bbc.com", "bbc.co.uk"]),
    ("纽约时报", ["nytimes.com"]),
    ("华尔街日报", ["wsj.com"]),
]


def extract_domain_hints(query: str) -> List[str]:
    """Return the ordered preferred-domain list for ``query``.

    Empty list when no keyword matches. The query is matched as a
    substring (case-insensitive). The first matching entry wins, so
    "腾讯新闻" matches before "腾讯" would — even though both
    substrings are present, the more-specific entry comes first in
    ``_DOMAIN_HINTS``.

    Returned list is a fresh copy — callers may mutate freely.
    """
    if not query:
        return []
    q_lower = query.lower()
    for keyword, domains in _DOMAIN_HINTS:
        if keyword.lower() in q_lower:
            return list(domains)
    return []


def _domain_matches(domain: str, preferred: str) -> bool:
    """True if ``domain`` matches ``preferred`` exactly or as a subdomain.

    e.g. ``inews.qq.com`` matches ``qq.com`` (preferred) but not
    ``fakeqq.com``. ``qq.com`` matches ``qq.com``. ``QQ.COM`` matches
    ``qq.com`` (case-insensitive).
    """
    if not domain or not preferred:
        return False
    d = domain.lower()
    p = preferred.lower()
    return d == p or d.endswith("." + p)


def rerank_by_preferred(
    results: Iterable[dict],
    preferred_domains: Iterable[str] | None,
) -> list[dict]:
    """Stable re-order so that URLs in ``preferred_domains`` come first.

    Each preferred domain gets its own bucket in declaration order
    (so ``["qq.com", "thepaper.cn"]`` puts qq.com matches ahead of
    thepaper.cn matches). Within each bucket, original order is
    preserved. After all preferred buckets, the non-preferred
    remainder keeps its original order.

    Returns a fresh list — does not mutate ``results``.

    Empty / None ``preferred_domains`` → returns ``list(results)``
    unchanged (no copy unless needed for safety; tests assert the
    identity is preserved).
    """
    preferred = list(preferred_domains or [])
    # Fast path: no preferred domains → nothing to re-rank. Return
    # the input as-is when it's already a list (avoids an unnecessary
    # copy on every query that names no publisher).
    if not preferred:
        if isinstance(results, list):
            return results
        return list(results)

    materialized = list(results)
    if not materialized:
        return materialized

    preferred_buckets: list[list[dict]] = [[] for _ in preferred]
    remainder: list[dict] = []
    for r in materialized:
        if not isinstance(r, dict):
            remainder.append(r)
            continue
        domain = r.get("domain", "") or ""
        matched = False
        for i, p in enumerate(preferred):
            if _domain_matches(domain, p):
                preferred_buckets[i].append(r)
                matched = True
                break
        if not matched:
            remainder.append(r)

    out: list[dict] = []
    for bucket in preferred_buckets:
        out.extend(bucket)
    out.extend(remainder)
    return out


# ---------------------------------------------------------------------------
# WebResult + URL → domain helper
# ---------------------------------------------------------------------------


class WebResult(TypedDict):
    """Single web-search result. Plain dict at runtime (TypedDict is a
    type-hint-only class) so ``WebResult(title=..., url=...)`` is a
    regular dict constructor — zero behavioral change vs the
    pre-v2.0.18 ``class WebResult(dict):`` form.

    Lives here (rather than in ``ddg.py``) because ``_domain_of`` and
    ``WebResult`` are always used together: ``WebResult(title=...,
    domain=_domain_of(url), ...)``. Co-locating them keeps the
    construction site readable.

    v2.0.18 — migrated from ``class WebResult(dict):`` (a v1 type
    theater: dict-subclass with type annotations that the runtime
    ignored) to ``TypedDict``. Backward-compatible at zero cost:
    dict-style access (``r["title"]``) and the kwarg constructor
    (``WebResult(title=...)``) both work; attribute-style access
    (``r.title``) does not work anymore but no internal code used it.
    """

    title: str
    url: str
    snippet: str
    domain: str


def _domain_of(url: str) -> str:
    """Extract the lowercase ``netloc`` of ``url`` (e.g. ``"x.com"``),
    or ``""`` if parsing fails. Used by both engines to populate
    ``WebResult.domain`` and by ``_domain_matches`` for preferred-domain
    checks.

    v2.0.18 — moved from ``src/web_search/ddg.py`` here because
    ``WebResult`` and ``_domain_of`` are always used together
    (``WebResult(title=..., domain=_domain_of(url), ...)``). One
    import site per call rather than two.
    """
    try:
        return (_urlparse(url).netloc or "").lower()
    except Exception:
        return ""


__all__ = ["extract_domain_hints", "rerank_by_preferred", "WebResult", "_domain_of"]