"""v2.0.29.4 (Phase 4 PR-2) — pre-embed ingestion cleaning tests.

The ``clean_chunks`` helper applies 4 deterministic rules to strip
garbage before chunks reach the embedder / LanceDB:

  1. Empty / whitespace-only drop
  2. Sub-noise-floor drop (< MIN_CHUNK_CHARS chars)
  3. Exact-duplicate drop (normalised SHA-256)
  4. UTF-8 round-trip safety check

These tests pin each rule independently and a couple of integration
shapes (ordering preserved, no side effects on the input). Pure
function tests — no LanceDB, no loguru capture (debug logs aren't
asserted; the test doesn't depend on the operator's ``LOG_LEVEL``).

Why pure-function tests are enough
----------------------------------
``clean_chunks`` is intentionally pure: no I/O, no shared state,
no imports beyond stdlib. So the tests don't need to mock anything
or run against a real DB — they exercise the function directly.
Integration with ``run_ingestion`` is covered by the Layer 5
verifier (``tests/e2e/verify-v2_0_29_4_p2-layer5.py``).
"""
from __future__ import annotations

from src.ingestion.cleaning import MIN_CHUNK_CHARS, clean_chunks


def _chunk(text: str, idx: int = 0) -> dict:
    """Build a minimal chunk dict mirroring the parser output shape."""
    return {"text": text, "meta": {"chunk_index": idx}}


def test_cleaning_drops_empty_chunks():
    """Rule 1 — empty / whitespace-only chunks are dropped."""
    chunks = [_chunk(""), _chunk("   "), _chunk("\n\n\t"), _chunk("real content here")]
    out = clean_chunks(chunks)
    assert len(out) == 1
    assert out[0]["text"] == "real content here"


def test_cleaning_drops_short_chunks():
    """Rule 2 — chunks under MIN_CHUNK_CHARS are dropped (OCR noise).

    ``MIN_CHUNK_CHARS=10`` is the production default; a 9-char chunk
    is treated as noise even if it's otherwise valid UTF-8.
    """
    below = "x" * (MIN_CHUNK_CHARS - 1)  # 9 chars
    above = "x" * MIN_CHUNK_CHARS  # exactly 10 chars — survives
    chunks = [_chunk(below), _chunk(above), _chunk("legit paragraph")]
    out = clean_chunks(chunks)
    assert [c["text"] for c in out] == [above, "legit paragraph"]


def test_cleaning_drops_exact_duplicates():
    """Rule 3 — same normalised text after SHA-256 dedups to one chunk.

    The first occurrence wins; later identicals drop. Whitespace
    differences collapse so trailing-newline duplicates count.
    """
    base = "the quick brown fox jumps over the lazy dog"
    chunks = [
        _chunk(base, idx=0),
        _chunk(base + "\n", idx=1),  # trailing newline normalises
        _chunk(base + "   ", idx=2),  # trailing whitespace normalises
        _chunk(base + " extra", idx=3),  # different content — survives
    ]
    out = clean_chunks(chunks)
    assert len(out) == 2
    assert out[0]["meta"]["chunk_index"] == 0  # first occurrence wins
    assert out[1]["meta"]["chunk_index"] == 3


def test_cleaning_drops_non_utf8_garbled():
    """Rule 4 — UTF-8 round-trip failure drops the chunk.

    A standalone surrogate (``\\udcff``) is a Python string-level
    construct that the strict UTF-8 encoder rejects. Such chunks
    would corrupt the BM25 indexer downstream.
    """
    bad = "garbled \udcff byte sequence"
    good = "perfectly fine UTF-8 string here"
    chunks = [_chunk(bad), _chunk(good)]
    out = clean_chunks(chunks)
    assert [c["text"] for c in out] == [good]


def test_cleaning_preserves_unique_chunks():
    """Happy path — clean corpus passes through unchanged."""
    chunks = [
        _chunk("alpha alpha alpha"),
        _chunk("beta beta beta"),
        _chunk("gamma gamma gamma"),
    ]
    out = clean_chunks(chunks)
    assert len(out) == 3
    assert [c["text"] for c in out] == [c["text"] for c in chunks]


def test_cleaning_preserves_input_ordering():
    """Survivor order = input order (deterministic; no sorting)."""
    # Each chunk must exceed MIN_CHUNK_CHARS so the noise-floor rule
    # doesn't drop them — the test is about ordering, not about
    # cleaning decisions.
    chunks = [_chunk(f"row {i:03d} content here", idx=i) for i in range(5)]
    out = clean_chunks(chunks)
    assert [c["meta"]["chunk_index"] for c in out] == [0, 1, 2, 3, 4]


def test_cleaning_is_pure_no_side_effects():
    """``clean_chunks`` doesn't mutate its input list."""
    chunks = [_chunk("alpha"), _chunk(""), _chunk("alpha")]
    snapshot = list(chunks)
    clean_chunks(chunks)
    assert chunks == snapshot  # input unchanged


def test_cleaning_handles_empty_input():
    """Empty iterable → empty list, no errors."""
    assert clean_chunks([]) == []


def test_cleaning_aggregates_drop_counts_in_debug_log():
    """All four drop categories combined → one debug-log line.

    We don't pin the exact log message (debug logs are operator-facing
    and may evolve). Instead, we monkeypatch the logger to capture
    and assert that AT LEAST ONE debug call fired when there were
    drops. This is a behaviour pin, not a string pin.
    """
    from src.ingestion import cleaning as cleaning_mod

    captured = []
    original_debug = cleaning_mod.logger.debug

    def spy_debug(msg, *args, **kwargs):
        captured.append(msg)

    cleaning_mod.logger.debug = spy_debug
    try:
        chunks = [
            _chunk(""),  # empty
            _chunk("x"),  # short
            _chunk("real content here"),
            _chunk("real content here"),  # duplicate of the above
            _chunk("bad \udcff surrogate"),  # utf-8 fail
            _chunk("survivor text"),
        ]
        out = clean_chunks(chunks)
    finally:
        cleaning_mod.logger.debug = original_debug

    # Two survivors: "real content here" (first occurrence) + "survivor text"
    assert len(out) == 2
    # Drop was non-empty so a debug log fired
    assert captured, "expected at least one debug log when drops happened"
    assert any("cleaning: dropped" in str(m) for m in captured)