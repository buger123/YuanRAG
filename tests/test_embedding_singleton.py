"""Regression: the BGE-M3 + reranker singletons must not re-load on every call.

Bug history:
1. A FastEmbed-based implementation called ``TextEmbedding("BAAI/bge-m3")``
   which raises ``ValueError`` in FastEmbed 0.8.0 (BGE-M3 is not in the
   supported model list). The error was swallowed silently by the route
   handler, and the frontend polled forever — logs were flooded with
   "Loading BGE-M3 dense model" lines.
2. ``functools.lru_cache`` doesn't serialize concurrent body execution, so
   N threads racing on a cold cache all ran the heavyweight ONNX/PyTorch
   loader in parallel.

Fix: ``threading.Lock`` + ``_loaded`` flag (manual singleton) with FlagEmbedding.

These tests mock FlagEmbedding's ``BGEM3FlagModel`` and ``FlagReranker`` so
they don't require real models.
"""
from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_flagembed(monkeypatch):
    """Replace FlagEmbedding constructors with mocks that count calls."""
    call_counts = {"embedder": 0, "reranker": 0}
    counter_lock = threading.Lock()

    instances = {"embedder": None, "reranker": None}

    def make_embedder(*args, **kwargs):
        with counter_lock:
            call_counts["embedder"] += 1
        # Simulate init time
        import time
        time.sleep(0.05)
        inst = MagicMock(name="BGEM3FlagModel")
        instances["embedder"] = inst
        return inst

    def make_reranker(*args, **kwargs):
        with counter_lock:
            call_counts["reranker"] += 1
        import time
        time.sleep(0.05)
        inst = MagicMock(name="FlagReranker")
        instances["reranker"] = inst
        return inst

    # Reset singleton state — import with aliases so both module's reset
    # functions are reachable (the second ``from X import Y`` would
    # otherwise shadow the first).
    from src.embeddings.bge_m3 import reset_for_tests as reset_bge
    from src.reranker.bge_reranker import reset_for_tests as reset_rr
    reset_bge()
    reset_rr()

    # Patch the module-level references that the loader uses.
    monkeypatch.setattr("src.embeddings.bge_m3.BGEM3FlagModel", make_embedder)
    monkeypatch.setattr("src.reranker.bge_reranker.FlagReranker", make_reranker)

    yield call_counts

    # Clean up for next test
    reset_bge()
    reset_rr()


def test_bge_m3_singleton_single_call(fake_flagembed):
    from src.embeddings.bge_m3 import get_model, models_loaded

    assert models_loaded() is False
    m = get_model()
    assert fake_flagembed["embedder"] == 1
    assert models_loaded() is True

    # Second call must return the same instance, with zero new constructions.
    m2 = get_model()
    assert m is m2
    assert fake_flagembed["embedder"] == 1


def test_bge_m3_singleton_concurrent_calls(fake_flagembed):
    """Hammer the loader from many threads; constructor must run only once."""
    from src.embeddings.bge_m3 import get_model

    barrier = threading.Barrier(20)
    results = []

    def worker():
        barrier.wait()
        results.append(id(get_model()))

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == 1, f"got {len(set(results))} unique instances"
    assert fake_flagembed["embedder"] == 1, (
        f"expected 1 construction, got {fake_flagembed['embedder']}"
    )


def test_bge_m3_models_loaded_is_cheap(fake_flagembed):
    """models_loaded() must NOT trigger a fresh load on every call."""
    from src.embeddings.bge_m3 import get_model, models_loaded

    assert models_loaded() is False
    get_model()
    assert models_loaded() is True

    before = fake_flagembed["embedder"]
    for _ in range(50):
        assert models_loaded() is True
    assert fake_flagembed["embedder"] == before


def test_reranker_singleton_concurrent(fake_flagembed):
    from src.reranker.bge_reranker import get_model

    barrier = threading.Barrier(20)
    results = []

    def worker():
        barrier.wait()
        results.append(id(get_model()))

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == 1
    assert fake_flagembed["reranker"] == 1


@pytest.mark.asyncio
async def test_models_status_endpoint_does_not_reload(app_under_test, monkeypatch, fake_flagembed):
    """GET /models/status must NOT trigger a fresh model load.

    Regression: the previous _try_load(loader=get_bge_models) handler invoked
    the loader on every poll, spinning up TextEmbedding on every request.
    """
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Pre-load once (simulating a warmup / first chat request).
        from src.embeddings.bge_m3 import get_model as get_bge
        from src.reranker.bge_reranker import get_model as get_rr
        get_bge()
        get_rr()

        before_e = fake_flagembed["embedder"]
        before_r = fake_flagembed["reranker"]

        # Poll 10 times.
        for _ in range(10):
            r = await client.get("/models/status")
            assert r.status_code == 200
            data = r.json()
            assert data["ready"] is True

        assert fake_flagembed["embedder"] == before_e
        assert fake_flagembed["reranker"] == before_r


@pytest.mark.asyncio
async def test_status_before_load_returns_missing(app_under_test, monkeypatch, fake_flagembed):
    """Without any load, /models/status must return ready=false without raising."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/models/status")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is False
        assert data["embedding"]["status"] == "missing"
        assert data["reranker"]["status"] == "missing"


# app_under_test fixture lives in conftest.py (hoisted to share across the
# 6 test files that need a FastAPI app bound to a temp RAG_DATA_DIR).