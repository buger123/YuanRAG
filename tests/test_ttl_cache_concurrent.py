"""v2.0.28.5 Item 11 PR-3 P1-R2 — ``_TTLCache`` thread-safe compound ops.

Pre-PR-3, ``_TTLCache.get`` and ``_TTLCache.set`` operated on the
underlying ``OrderedDict`` without any thread sync. CPython 3.12's
GIL accidentally serialized the bytecode, but two threads racing on
a cold key both fetched the URL (last write wins on the OrderedDict
mutation). CPython 3.13+ free-threaded mode breaks that guarantee
entirely. Post-PR-3, every public method takes ``self._lock`` — the
lock is held only across dict ops, never across network I/O.

This file pins the contract under concurrent pressure.
"""
from __future__ import annotations

import threading
import time

import pytest

from src.web_search.fetch import _TTLCache


def test_ttl_cache_concurrent_get_set_no_corruption():
    """5 threads alternating ``get`` + ``set`` × 100 iterations
    must not corrupt the cache invariant (``len <= max``) or raise.

    Without the lock, ``OrderedDict.move_to_end`` + ``__setitem__``
    interleaving across threads can produce ``KeyError`` (pop during
    iteration) or ``len > max`` (race on eviction). With the lock,
    the contract is preserved regardless of GIL behavior.
    """
    cache = _TTLCache(max_entries=64, ttl_seconds=600)
    errors: list = []

    def worker(thread_id: int):
        try:
            for i in range(100):
                key = f"t{thread_id}-k{i}"
                cache.set(key, f"v{thread_id}-{i}")
                _ = cache.get(key)
                _ = cache.get(f"t{(thread_id + 1) % 5}-k{(i + 1) % 100}")
                # Force eviction to race with other threads' sets.
                if i % 10 == 0:
                    for j in range(70):  # over-fill past cap
                        cache.set(f"t{thread_id}-flood-{i}-{j}", f"v{j}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"threads raised: {errors}"
    # Cache invariant must hold: len <= max_entries (64).
    assert len(cache) <= 64, (
        f"cache exceeded cap: len={len(cache)}, max=64"
    )


def test_ttl_cache_concurrent_same_key_only_one_winner():
    """N threads ``set``-ing the SAME key concurrently should leave
    the cache with exactly ONE entry for that key, not N.

    Without the lock, ``__setitem__`` would race on
    ``OrderedDict`` mutation. With the lock, the serialized compound
    op (move_to_end + __setitem__ + evict) preserves the invariant.
    """
    cache = _TTLCache(max_entries=64, ttl_seconds=600)
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        cache.set("shared-key", f"v-{threading.get_ident()}")

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # Exactly one entry for "shared-key" — last writer wins.
    assert "shared-key" in cache
    assert len(cache) <= 64
    # The value should be a string from one of the workers (any of them).
    val = cache.get("shared-key")
    assert val is not None
    assert val.startswith("v-")


def test_ttl_cache_ttl_expiry_still_works_under_concurrent_get():
    """TTL expiry path (``get`` removes expired entry) must remain
    consistent under concurrent reads.

    We pre-populate an entry with a very short TTL, then run 5
    threads calling ``get`` after the TTL has elapsed. Each thread
    should see ``None`` (expired) and the cache should not contain
    the key afterward.
    """
    cache = _TTLCache(max_entries=64, ttl_seconds=1)
    cache.set("expiring", "value")

    # Let the TTL elapse.
    time.sleep(1.1)

    barrier = threading.Barrier(5)
    results: list = []
    errors: list = []

    def worker():
        try:
            barrier.wait()
            results.append(cache.get("expiring"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    # All threads see None — TTL expiry is consistent.
    assert all(r is None for r in results), f"some threads saw cached value: {results}"
    assert "expiring" not in cache
