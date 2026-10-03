"""QW #7: in-process LRU cache for ``embed_query``.

Why a test rather than just code:
The cache is observable only as a perf win on a real BGE-M3 load
(50-200 ms saved per hit). Without a test we couldn't tell whether
``reset_for_tests`` actually flushes it, or whether the cap is
enforced, or whether a model-swap invalidates entries. Each of
those is a regression waiting to happen; locking them in via tests
catches them the moment someone "simplifies" the cache.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_cache(monkeypatch):
    """Reset the module-level cache before AND after the test so no
    shared state leaks across tests. Pairs with the autouse
    ``_isolate_user_data_dir`` in ``conftest.py``.
    """
    from src.embeddings import bge_m3

    bge_m3.reset_for_tests()
    yield
    bge_m3.reset_for_tests()


def test_embed_query_caches_repeated_calls(isolated_cache):
    """Same query → second call returns the SAME dict object.

    We don't actually load BGE-M3 (way too expensive for a unit
    test). Instead we patch ``_embed`` so it returns a sentinel
    dict and counts calls. The cache key is the input string, so
    two calls with the same text should hit the cache; two calls
    with different text should both reach ``_embed``.
    """
    from src.embeddings import bge_m3

    sentinel_a = {"dense": [0.1] * 1024, "sparse": {"a": 1.0}}
    sentinel_b = {"dense": [0.2] * 1024, "sparse": {"b": 2.0}}
    call_count = {"n": 0}

    def fake_embed(self, texts, *, is_query):
        call_count["n"] += 1
        return [sentinel_a if texts == ["hi"] else sentinel_b]

    embedder = bge_m3.BGEM3Embedder()
    with patch.object(bge_m3.BGEM3Embedder, "_embed", fake_embed):
        first = embedder.embed_query("hi")
        second = embedder.embed_query("hi")
        third = embedder.embed_query("hello")

    assert first is second  # cache hit returns the SAME object
    assert first is sentinel_a
    assert third is sentinel_b
    assert call_count["n"] == 2  # "hi" + "hello", "hi" second time cached


def test_embed_query_resets_via_reset_for_tests(isolated_cache):
    """``reset_for_tests`` must clear the cache so the next test
    starts from a cold state — same contract as the model singleton.
    """
    from src.embeddings import bge_m3

    embedder = bge_m3.BGEM3Embedder()
    sentinel = {"dense": [0.0] * 1024, "sparse": {}}

    with patch.object(bge_m3.BGEM3Embedder, "_embed", lambda self, t, *, is_query: [sentinel]):
        embedder.embed_query("warm")
        assert len(bge_m3._query_cache) == 1
        bge_m3.reset_for_tests()
        assert len(bge_m3._query_cache) == 0
        embedder.embed_query("warm")
        assert len(bge_m3._query_cache) == 1


def test_embed_query_evicts_oldest_at_maxsize(isolated_cache):
    """Once the cache exceeds ``_QUERY_CACHE_MAXSIZE``, the oldest
    entry must be evicted (FIFO within LRU). Bounded growth is the
    whole reason we have a maxsize.
    """
    from src.embeddings import bge_m3

    embedder = bge_m3.BGEM3Embedder()
    sentinel = {"dense": [0.0] * 1024, "sparse": {}}

    with patch.object(bge_m3.BGEM3Embedder, "_embed", lambda self, t, *, is_query: [sentinel]):
        # Fill past the cap with unique queries.
        for i in range(bge_m3._QUERY_CACHE_MAXSIZE + 10):
            embedder.embed_query(f"q-{i}")
        # The cache must be bounded, not growing unboundedly.
        assert len(bge_m3._query_cache) <= bge_m3._QUERY_CACHE_MAXSIZE


def test_embed_query_cache_key_scoped_by_model_name(isolated_cache):
    """If ``_model_name`` changes, the cache key for the same text
    must change too — otherwise a model swap could serve stale
    vectors. The simplest check: insert one entry, mutate
    ``_model_name``, insert the same text again, expect TWO
    ``_embed`` calls (no hit).
    """
    from src.embeddings import bge_m3

    embedder = bge_m3.BGEM3Embedder()
    sentinel = {"dense": [0.0] * 1024, "sparse": {}}
    call_count = {"n": 0}

    def fake_embed(self, texts, *, is_query):
        call_count["n"] += 1
        return [sentinel]

    with patch.object(bge_m3.BGEM3Embedder, "_embed", fake_embed):
        bge_m3._model_name = "model-A"
        embedder.embed_query("same text")
        bge_m3._model_name = "model-B"  # simulate a model swap
        embedder.embed_query("same text")

    assert call_count["n"] == 2  # cache must NOT have served model-A's vector under model-B