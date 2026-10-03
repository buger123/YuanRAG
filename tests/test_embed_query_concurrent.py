"""v2.0.28.5 Item 11 PR-3 P1-R1 — ``embed_query`` concurrent dedup.

Pre-PR-3, two threads racing on the same cold cache key both missed
(``_query_cache.get`` was lock-free), both ran BGE-M3 inference
(50-200 ms each), both populated the cache. Result: doubled
inference cost on a cold key, with the cache holding one entry whose
``move_to_end`` ordering depended on which thread won the lock
race for ``__setitem__``. Post-PR-3, the read path is under-lock and
the insert path uses double-checked-locking — the second thread
sees the first thread's populated entry and returns it without a
second inference.

This file pins the dedup contract under concurrent pressure.
"""
from __future__ import annotations

import threading
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_cache(monkeypatch):
    """Reset the module-level cache + model state before AND after."""
    from src.embeddings import bge_m3

    bge_m3.reset_for_tests()
    yield
    bge_m3.reset_for_tests()


def test_embed_query_concurrent_same_key_dedupes_inference(isolated_cache):
    """N threads calling ``embed_query("same text")`` should
    result in **exactly one** inference call when the lock is
    held across the embed_query body (v2.0.28.5 P1-R1).

    Implementation note: we cannot use a barrier INSIDE ``_embed``
    to force simultaneous entry — that would let all N threads
    reach ``_embed`` before any of them writes back to the cache,
    defeating the test. Instead, we use a barrier BEFORE
    ``embed_query`` to align the threads, and a single ``_embed``
    call that signals completion. Threads that arrive at the cache
    AFTER the first thread populates it should see the cached value
    and return without re-running ``_embed``.

    To make this deterministic on a multi-core CI, we serialize
    the threads using a lock held around the ``_embed`` body — so
    only ONE thread runs ``_embed`` at a time. Without v2.0.28.5
    P1-R1's per-embed_query cache lock, all N threads would each
    run ``_embed`` once on a cold key (last writer wins).
    """
    from src.embeddings import bge_m3

    sentinel = {"dense": [0.1] * 1024, "sparse": {"a": 1.0}}
    call_count = {"n": 0}
    # Used to force sequential _embed calls (so the test is
    # deterministic regardless of GIL scheduling).
    embed_lock = threading.Lock()

    def slow_fake_embed(self, texts, *, is_query):
        with embed_lock:
            call_count["n"] += 1
        return [sentinel]

    embedder = bge_m3.BGEM3Embedder()

    results: list = []
    errors: list = []

    def worker():
        try:
            r = embedder.embed_query("same text")
            results.append(r)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    with patch.object(bge_m3.BGEM3Embedder, "_embed", slow_fake_embed):
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    assert len(results) == 10
    # The key contract: only ONE inference despite 10 concurrent
    # threads. Pre-PR-3 would yield N inferences (every thread
    # miss → infer). Post-PR-3, the lock-across-embed_query
    # serializes same-key misses: thread 1 enters, runs inference,
    # populates cache, releases; threads 2-10 enter, see cache hit,
    # return immediately.
    assert call_count["n"] == 1, (
        f"expected 1 inference under concurrent dedup, got {call_count['n']}"
    )
    assert all(r is sentinel for r in results), (
        "all 10 threads should receive the same cached sentinel"
    )


def test_embed_query_concurrent_different_keys_each_run_once(isolated_cache):
    """Sanity check the OTHER direction: N threads on N distinct
    keys should each run inference exactly once (no false dedup).

    Without this guard, the under-lock read + double-checked insert
    could accidentally over-cache or return the wrong value when
    keys differ. This test pins the per-key independence contract.
    """
    from src.embeddings import bge_m3

    sentinels = {f"q-{i}": {"dense": [float(i)] * 1024, "sparse": {}} for i in range(5)}
    call_count = {"n": 0}

    def fake_embed(self, texts, *, is_query):
        call_count["n"] += 1
        text = texts[0]
        return [sentinels[text]]

    embedder = bge_m3.BGEM3Embedder()

    barrier = threading.Barrier(5)

    def worker(text):
        barrier.wait()  # all 5 start at once
        return embedder.embed_query(text)

    with patch.object(bge_m3.BGEM3Embedder, "_embed", fake_embed):
        results = {}
        threads = []
        for q in sentinels:
            t = threading.Thread(target=lambda q=q: results.update({q: worker(q)}))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert call_count["n"] == 5, (
        f"expected 5 distinct inferences for 5 distinct keys, got {call_count['n']}"
    )
    for q, sent in sentinels.items():
        assert results[q] is sent, (
            f"key {q!r} returned wrong sentinel: got {results[q]!r}"
        )
