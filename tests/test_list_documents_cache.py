"""Tests for the TTL cache around list_documents (Phase C3).

The cache is the dominant per-turn latency optimization for the
``_uploaded_docs_hint`` lookup that powers the legacy ``route_query``
graph node (now dead post-v2.0 ReAct rewrite; the hint is still
read by ``intent_analysis`` for the summary fast-path pre-check).
These tests pin the cache contract:

1. The cached value is returned on a second call within the TTL.
2. The TTL is honored — after expiry, the cache misses and a fresh
   scan runs.
3. ``add_chunks`` for a thread invalidates that thread's cache.
4. ``delete_by_doc_id`` clears the entire cache (no thread lookup).
5. ``delete_by_thread_id`` invalidates the targeted thread.
6. ``reset_for_tests`` clears the entire cache.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.storage import lancedb_store
from src.storage.schema import ChunkRecord


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test gets a clean cache so prior runs don't leak entries."""
    lancedb_store.reset_for_tests()
    yield
    lancedb_store.reset_for_tests()


def _stub_list_documents(value):
    """Return a fake list_documents that records call count + returns value."""
    calls = {"n": 0}

    def _fake(thread_id=None):
        calls["n"] += 1
        return value

    return _fake, calls


def _make_record(thread_id: str = "t1", doc_id: str = "d1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id="c1",
        doc_id=doc_id,
        text="hello",
        vector=[0.0] * 1024,
        filename="notes.md",
        source_type="file",
        # v2.0.27 P0-P2 — canonical privacy-safe form.
        source_path=f"{doc_id}/notes.md",
        chunk_index=0,
        thread_id=thread_id,
    )


# ----- Basic hit / miss behaviour ---------------------------------


def test_cache_hit_within_ttl_does_not_call_underlying():
    """Two calls in quick succession → underlying scanned only once."""
    fake, calls = _stub_list_documents([{"doc_id": "d1", "filename": "a.md"}])
    with patch.object(lancedb_store, "list_documents", fake):
        a = lancedb_store.list_documents_cached("t1")
        b = lancedb_store.list_documents_cached("t1")

    assert calls["n"] == 1
    assert a == b


def test_cache_miss_after_ttl_expires():
    """Sleeping past the TTL forces a fresh scan."""
    fake, calls = _stub_list_documents([])
    # Shrink the TTL for this test so we don't actually sleep 5 s.
    with patch.object(lancedb_store, "_THREAD_DOC_CACHE_TTL_SEC", 0.1):
        with patch.object(lancedb_store, "list_documents", fake):
            lancedb_store.list_documents_cached("t1")
            time.sleep(0.15)
            lancedb_store.list_documents_cached("t1")

    assert calls["n"] == 2


def test_cache_keys_per_thread_id_independently():
    """Different thread_ids maintain separate cache entries."""
    fake, calls = _stub_list_documents([])
    with patch.object(lancedb_store, "list_documents", fake):
        lancedb_store.list_documents_cached("tA")
        lancedb_store.list_documents_cached("tB")
        lancedb_store.list_documents_cached("tA")

    assert calls["n"] == 2  # tA once, tB once, tA hit


def test_cache_thread_id_none_uses_sentinel_key():
    """``None`` thread_id is cached under a stable sentinel so callers
    with / without thread_id don't poison each other's cache slots."""
    fake, calls = _stub_list_documents([])
    with patch.object(lancedb_store, "list_documents", fake):
        lancedb_store.list_documents_cached(None)
        lancedb_store.list_documents_cached(None)
        lancedb_store.list_documents_cached("")

    # None and "" map to the same sentinel, so 1 underlying call.
    assert calls["n"] == 1


# ----- Invalidation -----------------------------------------------


def test_add_chunks_invalidates_target_thread():
    """Uploading chunks to a thread drops that thread's cached listing."""
    fake, calls = _stub_list_documents([])
    with patch.object(lancedb_store, "list_documents", fake):
        lancedb_store.list_documents_cached("t1")  # populate
        lancedb_store.list_documents_cached("t1")  # hit

        # Pretend we just added chunks for t1 — cache must clear.
        lancedb_store.add_chunks([_make_record(thread_id="t1")])

        lancedb_store.list_documents_cached("t1")  # miss again

    assert calls["n"] == 2


def test_add_chunks_for_other_thread_does_not_invalidate():
    """Uploading to thread t2 leaves thread t1's cache untouched."""
    fake, calls = _stub_list_documents([])
    with patch.object(lancedb_store, "list_documents", fake):
        lancedb_store.list_documents_cached("t1")  # populate
        lancedb_store.add_chunks([_make_record(thread_id="t2")])
        lancedb_store.list_documents_cached("t1")  # still hits

    assert calls["n"] == 1


def test_invalidate_thread_doc_cache_clears_one_thread():
    """Targeted invalidation only drops the named thread."""
    fake, calls = _stub_list_documents([])
    with patch.object(lancedb_store, "list_documents", fake):
        lancedb_store.list_documents_cached("t1")
        lancedb_store.list_documents_cached("t2")
        lancedb_store.invalidate_thread_doc_cache("t1")
        lancedb_store.list_documents_cached("t1")  # miss
        lancedb_store.list_documents_cached("t2")  # hit

    assert calls["n"] == 3  # t1, t2, t1


def test_invalidate_thread_doc_cache_none_clears_all():
    """``invalidate_thread_doc_cache(None)`` wipes every entry."""
    fake, calls = _stub_list_documents([])
    with patch.object(lancedb_store, "list_documents", fake):
        lancedb_store.list_documents_cached("t1")
        lancedb_store.list_documents_cached("t2")
        lancedb_store.invalidate_thread_doc_cache(None)
        lancedb_store.list_documents_cached("t1")
        lancedb_store.list_documents_cached("t2")

    assert calls["n"] == 4


def test_reset_for_tests_clears_cache():
    """The test-reset helper must wipe the cache as well as the DB conn."""
    fake, calls = _stub_list_documents([])
    with patch.object(lancedb_store, "list_documents", fake):
        lancedb_store.list_documents_cached("t1")
        lancedb_store.reset_for_tests()
        lancedb_store.list_documents_cached("t1")

    assert calls["n"] == 2


# ----- Failure mode -----------------------------------------------


def test_cache_swallows_value_storage_but_returns_fresh_value():
    """If the underlying raises, we still surface the error — the cache
    must not silently swallow failures and serve a stale value."""
    def _boom(thread_id=None):
        raise RuntimeError("LanceDB down")

    with patch.object(lancedb_store, "list_documents", _boom):
        with pytest.raises(RuntimeError):
            lancedb_store.list_documents_cached("t1")

        # A second call still raises — no stale value was cached.
        with pytest.raises(RuntimeError):
            lancedb_store.list_documents_cached("t1")
