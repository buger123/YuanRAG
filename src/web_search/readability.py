"""HTML → plain-text readability extraction, stdlib-only.

v2.0.18 — extracted from the pre-Phase-2 page-fetch shim (now deleted).
(where it sat alongside the urllib machinery and confused the module's
"what does this file do" summary). Now lives at
``src/web_search/readability.py`` because the only caller is the page
fetcher at ``src/web_search/fetch.py`` (post-Step-3) — readability
extraction is the readability-style cleaning of the raw HTML, not the
HTTP fetch itself.

Why stdlib-only (no BeautifulSoup / readability-lxml)
----------------------------------------------------
Keeps the import cost zero and the failure surface small. News
articles have predictable structure (``<article>`` / ``<main>`` /
``<body>`` + script/style stripping) that's good enough for the
grounding task; we don't need a full readability implementation.

The 6 module-level compiled regexes below are deliberately not
parameterized — adding parameters ("article selector", "min chars")
would suggest more configurability than the heuristic deserves. If a
future corpus shows a publisher needs different selectors, add a
*specific* regex + a *specific* guard, not a generic parameter.
"""
from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Compiled patterns (module-level so the compile cost is paid once)
# ---------------------------------------------------------------------------

# Drop entire ``<script>`` / ``<style>`` blocks — their content would
# otherwise pollute the article text. ``re.DOTALL`` so ``.*?`` spans
# newlines (script bodies often do).
_RE_SCRIPT_STYLE = re.compile(
    r"<(?:script|style|noscript)\b[^>]*>.*?</(?:script|style|noscript)>",
    re.DOTALL | re.IGNORECASE,
)

# Drop structural chrome we know is boilerplate.
_RE_STRIP_BLOCKS = re.compile(
    r"<(?:nav|footer|header|aside|form)\b[^>]*>.*?</(?:nav|footer|header|aside|form)>",
    re.DOTALL | re.IGNORECASE,
)

# Article-content blocks (preferred). Try ``<article>`` first, fall
# back to ``<main>``, fall back to whole body.
_RE_ARTICLE = re.compile(
    r"<article\b[^>]*>(.*?)</article>",
    re.DOTALL | re.IGNORECASE,
)
_RE_MAIN = re.compile(
    r"<main\b[^>]*>(.*?)</main>",
    re.DOTALL | re.IGNORECASE,
)
_RE_BODY = re.compile(
    r"<body\b[^>]*>(.*?)</body>",
    re.DOTALL | re.IGNORECASE,
)

# All remaining HTML tags → empty (we keep their text content).
_RE_TAG = re.compile(r"<[^>]+>")

# ``<br>`` → space BEFORE tag stripping so word boundaries don't fuse.
_RE_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)

# Whitespace runs.
_RE_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def _strip_html(html: str) -> str:
    """Strip tags, collapse whitespace, return plain text."""
    spaced = _RE_BR.sub(" ", html)
    return _RE_WS.sub(" ", _RE_TAG.sub("", spaced)).strip()


def _extract_main_text(html: str) -> str:
    """Readability-style extraction: ``<article>`` > ``<main>`` > body.

    Falls back gracefully when none of those tags exist — strips
    scripts / styles / nav / footer / header / aside out of the raw
    HTML first, then returns the cleaned-up body. This is intentionally
    stdlib-only (no BeautifulSoup, no readability-lxml) to keep the
    fetcher's import cost zero and the failure surface small.

    The 200-char threshold is a heuristic — a real article body is
    usually 1000+ chars; a 200-char ``<article>`` tag with only metadata
    inside is treated as "no article" so we fall through to ``<main>``
    or the body fallback.
    """
    if not html:
        return ""
    # Drop blocks that contain only chrome / scripts before searching
    # for the main container.
    cleaned = _RE_SCRIPT_STYLE.sub("", html)
    cleaned = _RE_STRIP_BLOCKS.sub("", cleaned)

    for pattern in (_RE_ARTICLE, _RE_MAIN, _RE_BODY):
        m = pattern.search(cleaned)
        if m:
            text = _strip_html(m.group(1))
            if len(text) >= 200:
                return text

    # Last resort: strip everything from the whole document.
    return _strip_html(cleaned)


__all__ = ["_strip_html", "_extract_main_text"]