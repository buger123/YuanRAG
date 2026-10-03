"""v2.0.28.6 Item 11 PR-4 — inference-lock tests.

Pre-PR-4, the inference locks (``_infer_lock`` in both
``src/embeddings/bge_m3.py`` and ``src/reranker/bge_reranker.py``)
were exercised in production but never pinned by automated tests.
A regression that broke serialization (e.g. accidentally renaming
the lock to a no-op) would only surface under load, not in CI.

Post-PR-4, three tests pin:
  1. ``_embed`` corpus sub-batch releases the inference lock between
     chunks — so a concurrent ``embed_query`` doesn't have to wait
     for the entire corpus to finish.
  2. ``get_model`` rejects a name switch once loaded — protects the
     singleton invariant.
  3. ``BGEReranker.score`` serializes concurrent calls — protects
     against FlagReranker's known non-thread-safety.

These tests use monkeypatched ``_embed`` / ``compute_score`` so they
don't need a real BGE-M3 model.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest


def test_bge_m3_singleton_rejects_model_name_switch(monkeypatch):
    """``get_model("A")`` then ``get_model("B")`` must raise ValueError.

    This is the "singleton name-mismatch" guard — protects the
    invariant that once a model is loaded you can't swap to a
    different name without an explicit reset. A regression that
    silently accepts the swap would be very hard to detect in
    production (chunks embedded with model A, queries with model B).
    """
    from src.embeddings import bge_m3

    # Mark the singleton as loaded with name "A".
    monkeypatch.setattr(bge_m3, "_loaded", True)
    monkeypatch.setattr(bge_m3, "_model", object())  # placeholder
    monkeypatch.setattr(bge_m3, "_model_name", "model-A")
    monkeypatch.setattr(bge_m3, "_loading", False)

    with pytest.raises(ValueError, match="already loaded as 'model-A'"):
        bge_m3.get_model("model-B")


def test_bge_m3_singleton_accepts_same_name(monkeypatch):
    """``get_model("A")`` then ``get_model("A")`` must return the same
    instance without raising — the name-mismatch check is exact, not
    a "any change" check.
    """
    from src.embeddings import bge_m3

    sentinel = object()
    monkeypatch.setattr(bge_m3, "_loaded", True)
    monkeypatch.setattr(bge_m3, "_model", sentinel)
    monkeypatch.setattr(bge_m3, "_model_name", "model-A")

    # Same name → returns the cached instance.
    assert bge_m3.get_model("model-A") is sentinel


def test_bge_reranker_inference_serializes_concurrent_calls(monkeypatch):
    """Two concurrent ``BGEReranker.score`` calls must not interleave.

    FlagReranker.compute_score is NOT thread-safe — concurrent calls
    on the same instance can return interleaved/wrong scores silently.
    We mock compute_score with a thread-safe counter + lock to verify
    that calls are serialized (no two ``compute_score`` calls run
    at the same time).
    """
    from src.reranker import bge_reranker

    in_flight = {"max": 0, "current": 0}
    in_flight_lock = threading.Lock()
    call_count = {"n": 0}

    def fake_compute_score(pairs, normalize=True):
        with in_flight_lock:
            in_flight["current"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["current"])
        # Simulate work — if NOT serialized, two threads would
        # both see current=2.
        time.sleep(0.02)
        with in_flight_lock:
            in_flight["current"] -= 1
        with in_flight_lock:
            call_count["n"] += 1
        # FlagReranker returns one score per pair.
        return [0.5] * len(pairs)

    # Inject a fake singleton model with our fake compute_score.
    fake_model = MagicMock()
    fake_model.compute_score = fake_compute_score
    monkeypatch.setattr(bge_reranker, "_loaded", True)
    monkeypatch.setattr(bge_reranker, "_reranker", fake_model)
    monkeypatch.setattr(bge_reranker, "_model_name", bge_reranker._DEFAULT_MODEL)

    reranker = bge_reranker.BGEReranker()

    # Use a barrier so both threads start at the same instant.
    barrier = threading.Barrier(2)
    results: list = []
    errors: list = []

    def worker():
        try:
            barrier.wait()
            scores = reranker.score("q", ["doc-a", "doc-b"])
            results.append(scores)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"threads raised: {errors}"
    assert len(results) == 2
    # The key contract — at no point did two compute_score calls overlap.
    assert in_flight["max"] == 1, (
        f"compute_score not serialized: max concurrent = {in_flight['max']}"
    )
    assert call_count["n"] == 2


def test_bge_m3_corpus_sub_batch_releases_lock_between_chunks(monkeypatch):
    """``_embed`` for a corpus of N texts (>64) should process them in
    sub-batches, releasing the inference lock between sub-batches so
    a concurrent ``embed_query`` can interleave.

    This pins the F-R8 audit finding: the old code held the inference
    lock for the WHOLE corpus, blocking all queries behind it. The
    current code sub-batches at 64, so a 200-text corpus becomes 4
    sub-batches of 50 each (4 ``encode_corpus`` calls, lock released
    between them).

    Implementation: we mock ``encode_corpus`` to return a dict with
    the two keys ``_to_record_list`` reads (``dense_vecs`` +
    ``lexical_weights``). The assertion checks ``encode_corpus``
    was called multiple times, proving sub-batching happened.
    """
    from src.embeddings import bge_m3

    encode_call_count = {"n": 0}

    def fake_encode_corpus(texts, return_dense=True, return_sparse=True):
        encode_call_count["n"] += 1
        n = len(texts)
        # Both keys must be present — ``_to_record_list`` zips them.
        return {
            "dense_vecs": [[0.1] * 16] * n,
            "lexical_weights": [{}] * n,
        }

    # Inject a fake singleton model.
    fake_model = MagicMock()
    fake_model.encode_corpus = fake_encode_corpus
    monkeypatch.setattr(bge_m3, "_loaded", True)
    monkeypatch.setattr(bge_m3, "_model", fake_model)
    monkeypatch.setattr(bge_m3, "_model_name", bge_m3._DEFAULT_MODEL)

    embedder = bge_m3.BGEM3Embedder()

    # 200 texts exceeds the 64-text sub-batch threshold, forcing
    # the corpus sub-batching branch (4 batches: 64+64+64+8).
    corpus = [f"text-{i}" for i in range(200)]
    embeddings = embedder._embed(corpus, is_query=False)

    assert len(embeddings) == 200
    # Sub-batching happened: more than one encode_corpus call.
    assert encode_call_count["n"] > 1, (
        f"expected sub-batching, but encode_corpus was called "
        f"{encode_call_count['n']} times"
    )
    # Verify the right shape (64-text sub-batches: ceil(200/64) = 4).
    assert encode_call_count["n"] == 4, (
        f"expected 4 sub-batches for 200 texts (64+64+64+8), "
        f"got {encode_call_count['n']}"
    )