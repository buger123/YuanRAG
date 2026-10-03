"""Tests for the reverse index in ``src.storage.lancedb_store``.

The reverse index collapses the N+1 ``GET /sessions`` query into a
single dict-lookup pass. These tests pin its invariants:

1. ``add_chunks`` extends the doc <-> thread index incrementally.
2. ``delete_by_doc_id`` removes a doc's edges and invalidates ONLY
   the owning thread(s), not the entire cache.
3. ``delete_by_thread_id`` drops the thread's index entries.
4. ``count_documents_by_threads`` returns ``(count, first_filename)``
   per thread from the warm index.
5. After ``reset_for_tests`` the index is empty until rebuilt on
   next access.
"""
from __future__ import annotations

import pytest


def _make_record(doc_id: str, thread_id: str, filename: str = "report.pdf"):
    """Build a minimal ChunkRecord for the lancedb_store tests."""
    from src.storage.schema import ChunkRecord

    return ChunkRecord(
        chunk_id=f"{doc_id}-c1",
        doc_id=doc_id,
        text="hello world",
        vector=[0.0] * 1024,
        sparse={1: 1.0},
        filename=filename,
        source_type="file",
        source_path=f"{doc_id}/{filename}",
        mime_type="application/pdf",
        chunk_index=0,
        chunk_type="text",
        page_number=1,
        section=None,
        sheet=None,
        headers=None,
        row_start=None,
        row_end=None,
        thread_id=thread_id,
    )


# ============================================================
# count_documents_by_threads — the /sessions N+1 fix
# ============================================================


def test_count_documents_by_threads_returns_empty_for_empty_index():
    """With no chunks uploaded, every thread maps to ``(0, "")``."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    out = lancedb_store.count_documents_by_threads(["alpha", "beta"])
    assert out == {"alpha": (0, ""), "beta": (0, "")}


def test_count_documents_by_threads_aggregates_per_thread():
    """Multiple docs on the same thread contribute to one count;
    multiple threads with one doc each contribute independently."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    # Thread "alpha" gets 2 docs, "beta" gets 1 doc
    lancedb_store.add_chunks(
        [
            _make_record("d1", "alpha", filename="alpha-1.pdf"),
            _make_record("d2", "alpha", filename="alpha-2.pdf"),
            _make_record("d3", "beta", filename="beta-1.pdf"),
        ]
    )
    out = lancedb_store.count_documents_by_threads(["alpha", "beta", "gamma"])
    assert out["alpha"] == (2, "alpha-1.pdf")  # first filename wins
    assert out["beta"] == (1, "beta-1.pdf")
    assert out["gamma"] == (0, "")  # never seen → empty


def test_count_documents_by_threads_first_filename_is_stable():
    """The ``first_filename`` is captured on the FIRST ``add_chunks``
    for a thread and doesn't change when later docs are added (matches
    the historical ``list_threads_with_documents`` ordering)."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks([_make_record("d1", "alpha", filename="first.pdf")])
    lancedb_store.add_chunks([_make_record("d2", "alpha", filename="second.pdf")])
    out = lancedb_store.count_documents_by_threads(["alpha"])
    assert out["alpha"] == (2, "first.pdf")


# ============================================================
# Reverse-index maintenance on mutations
# ============================================================


def test_delete_by_doc_id_only_invalidates_owning_threads():
    """``delete_by_doc_id`` must drop the doc's edges AND invalidate
    the cache for the threads that owned it. Threads that didn't own
    this doc must keep their caches (regression: the previous code
    nuked the whole cache on every delete)."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    # Three threads, one doc each
    lancedb_store.add_chunks(
        [
            _make_record("d1", "alpha"),
            _make_record("d2", "beta"),
            _make_record("d3", "gamma"),
        ]
    )
    # Warm the per-thread doc caches so we can verify what stays valid
    lancedb_store.list_documents_cached("alpha")
    lancedb_store.list_documents_cached("beta")
    lancedb_store.list_documents_cached("gamma")

    # Delete only beta's doc
    lancedb_store.delete_by_doc_id("d2")

    # Reverse index: d2's edges gone, d1 + d3 intact
    assert "d2" not in lancedb_store._doc_to_threads
    assert lancedb_store._thread_to_docs.get("alpha") == {"d1"}
    assert "beta" not in lancedb_store._thread_to_docs  # empty → dropped
    assert lancedb_store._thread_to_docs.get("gamma") == {"d3"}


def test_delete_by_thread_id_drops_all_owned_docs():
    """``delete_by_thread_id`` removes the thread AND every doc's edge
    that referenced it."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks(
        [
            _make_record("d1", "alpha"),
            _make_record("d2", "alpha"),
            _make_record("d3", "beta"),
        ]
    )
    lancedb_store.delete_by_thread_id("alpha")

    assert "alpha" not in lancedb_store._thread_to_docs
    assert lancedb_store._thread_to_docs.get("beta") == {"d3"}
    # alpha's docs no longer reference alpha in the reverse index
    assert lancedb_store._doc_to_threads.get("d1", set()) == set()
    assert lancedb_store._doc_to_threads.get("d2", set()) == set()


def test_count_after_delete_reflects_lost_docs():
    """After deleting a thread's docs, ``count_documents_by_threads``
    returns ``(0, "")`` for that thread."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks(
        [
            _make_record("d1", "alpha"),
            _make_record("d2", "beta"),
        ]
    )
    lancedb_store.delete_by_thread_id("alpha")
    out = lancedb_store.count_documents_by_threads(["alpha", "beta"])
    assert out["alpha"] == (0, "")
    assert out["beta"] == (1, "report.pdf")


# ============================================================
# reset_for_tests clears the index
# ============================================================


def test_reset_for_tests_clears_reverse_index():
    """``reset_for_tests`` must clear the reverse index so the next
    test sees a fresh state — path-keyed singletons shouldn't leak."""
    from src.storage import lancedb_store

    lancedb_store.add_chunks([_make_record("d1", "alpha")])
    assert lancedb_store._thread_to_docs  # non-empty
    lancedb_store.reset_for_tests()
    assert lancedb_store._doc_to_threads == {}
    assert lancedb_store._thread_to_docs == {}
    assert lancedb_store._thread_first_filename == {}


# ============================================================
# Single-scan invariant
# ============================================================


def test_count_documents_by_threads_warms_index_in_one_scan():
    """First call builds the index from a single ``table.to_arrow()``
    scan. Subsequent calls don't scan again — verified by counting
    ``get_table`` calls via monkey-patching."""
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()
    lancedb_store.add_chunks([_make_record("d1", "alpha")])
    lancedb_store.add_chunks([_make_record("d2", "beta")])
    lancedb_store.reset_for_tests()

    # Count get_table calls during count_documents_by_threads
    call_count = {"n": 0}
    real_get_table = lancedb_store.get_table

    def counting_get_table(*a, **kw):
        call_count["n"] += 1
        return real_get_table(*a, **kw)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(lancedb_store, "get_table", counting_get_table)
    try:
        # First call after reset: cold index → ONE scan
        lancedb_store.count_documents_by_threads(["alpha", "beta"])
        first_call_count = call_count["n"]
        assert first_call_count == 1, (
            f"first call should trigger one to_arrow scan, got {first_call_count}"
        )
        # Second call: warm index → ZERO scans
        lancedb_store.count_documents_by_threads(["alpha", "beta"])
        second_call_count = call_count["n"]
        assert second_call_count == first_call_count, (
            f"second call should NOT scan; went from {first_call_count} to "
            f"{second_call_count}"
        )
    finally:
        monkey.undo()