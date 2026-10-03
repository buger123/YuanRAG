"""v2.0.29.4 (Phase 4 PR-2) — pre-embed ingestion cleaning.

Why this exists
---------------
Garbage in → garbage out. Parsers (PDF OCR fallback, Docling,
pypdfium2 text layer, xlsx) can emit chunks that are technically
present but worthless to the retriever:

  * Whitespace-only / empty chunks (PDF blank pages, xlsx empty rows)
  * Very short chunks under a configurable noise threshold (OCR noise,
    accidental whitespace)
  * Exact-duplicate chunks (the same content emitted twice in one
    ingestion pass — common in OCR re-runs and chunker boundary drift)
  * Chunks whose text contains bytes that fail UTF-8 round-trip (PDF
    garbled by a non-UTF-8-aware decoder — typically the OCR engine
    drops bytes mid-codepoint)

Each one of these silently inflates the chunk table, pushes
legitimate chunks down the top-K, and wastes LLM context. The cheap
"pre-embed cleaning" stage here applies 4 deterministic rules in
order. No ML, no semantic dedup (SimHash / MinHash are PR-5+
scope); just structural hygiene.

Order of rules
--------------
The order matters because the rules are not commutative. Each step
narrows the corpus:

  1. **Empty / whitespace-only drop** — cheap; eliminates the largest
     class of junk first.
  2. **Short-chunk drop** — sub-noise-floor content (configurable,
     default 10 chars). OCR noise is often 1-5 chars long; legitimate
     chunk headers can be 5-15 chars; 10 is a safe default that
     catches the worst offenders without losing real content.
  3. **Exact-duplicate drop** — SHA-256 of the stripped text. The
     first occurrence wins; later identicals are dropped. This is a
     pre-embed dedup that prevents two LanceDB rows with byte-identical
     content from competing for the same query slot.
  4. **UTF-8 round-trip** — encode then decode; if the decode round-
     trips to the original string the chunk is safe. Otherwise the
     text contained surrogate code points or other invalid UTF-8
     sequences and would corrupt downstream BM25 tokenisation.

All drops are logged at debug level so an operator can audit the
cleaning rate via ``logs/app.log`` (set ``LOG_LEVEL=DEBUG`` for the
ingestion module to see them).

Threading
---------
Pure function on the input list. No I/O, no shared state, no
imports beyond stdlib. Safe to call from any context.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List

from src.core.logging import logger

# Drop chunks whose stripped length is below this many chars. Tuned
# against the OCR-noise distribution seen on scanned PDFs in
# production (typical noise = 1-5 chars after pypdfium2's text-layer
# fallback). 10 chars is conservative; legitimate content can be
# shorter than that but very rarely.
MIN_CHUNK_CHARS = 10


def _normalize_for_dedup(text: str) -> str:
    """Strip whitespace + collapse internal runs of whitespace for the
    dedup key.

    Two chunks that differ only by trailing whitespace or newlines
    count as duplicates for retrieval purposes — the BM25 indexer
    tokenises them identically and they'd compete for the same
    query slot. Normalising here lets the SHA-256 catch them.
    """
    return " ".join(text.split())


def _sha256(text: str) -> str:
    """UTF-8 SHA-256 of the normalised text. ``errors="replace"`` so a
    non-UTF-8 source string doesn't crash the cleaner mid-loop (it'd
    be caught by step 4 below anyway, but defence-in-depth here
    keeps the cleaner robust on hostile input)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_chunks(chunks: Iterable[dict]) -> List[dict]:
    """Apply 4 deterministic cleaning rules in order.

    Each input chunk is expected to have a ``"text"`` key (the chunk
    body). The returned chunks preserve the original shape — this
    function only removes entries, never rewrites fields.

    A "dropped" chunk is logged once at debug level (not warning —
    cleaning junk is the expected outcome, not an error).
    """
    seen_hashes: set[str] = set()
    out: List[dict] = []
    drops_empty = drops_short = drops_dup = drops_utf8 = 0

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        # Rule 1: empty / whitespace-only.
        if not text:
            drops_empty += 1
            continue
        # Rule 2: below the noise floor.
        if len(text) < MIN_CHUNK_CHARS:
            drops_short += 1
            continue
        # Rule 3: exact-duplicate (normalised SHA-256 match).
        norm = _normalize_for_dedup(text)
        h = _sha256(norm)
        if h in seen_hashes:
            drops_dup += 1
            continue
        # Rule 4: UTF-8 round-trip safety.
        try:
            # ``errors="strict"`` so a surrogate or invalid sequence
            # raises. Round-trip check: encode-then-decode must
            # produce the same string. The decode side accepts the
            # encoded bytes back into ``str``.
            if text.encode("utf-8", errors="strict").decode("utf-8", errors="strict") != text:
                drops_utf8 += 1
                continue
        except UnicodeError:
            drops_utf8 += 1
            continue
        # All four rules passed.
        seen_hashes.add(h)
        out.append(chunk)

    total_dropped = drops_empty + drops_short + drops_dup + drops_utf8
    if total_dropped:
        # Aggregate debug log — operators see one line per ingestion
        # pass with the breakdown. The four sub-counts let a
        # operator spot "OCR is producing lots of garbage" vs "the
        # chunker is duplicating" without parsing the per-chunk
        # logs.
        logger.debug(
            f"cleaning: dropped {total_dropped} chunk(s) — "
            f"empty={drops_empty} short={drops_short} "
            f"dup={drops_dup} utf8={drops_utf8}"
        )
    return out


__all__ = ["clean_chunks", "MIN_CHUNK_CHARS"]