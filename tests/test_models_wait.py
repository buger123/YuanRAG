"""Tests for ``GET /models/wait`` long-poll endpoint (v2.0.30.0).

Pre-fix the frontend polled ``/models/status`` every 2 s and could
lag up to 2 s behind backend ``ready=True``. The new ``/models/wait``
endpoint blocks the request server-side until both models report
``loaded=True``, so the spinner detaches within ~250 ms of backend
ready in the foregrounded-tab case.

We assert:
  - immediate response when both models are already loaded
  - poll-until-ready behavior when ``_loading=True`` and we flip the
    flag mid-wait
  - 408 on timeout with no state change
  - immediate response on load-error
  - Query validation for ``timeout`` / ``poll_interval``
"""
from __future__ import annotations

import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient


# ----- 1. immediate response when both already loaded --------------------

@pytest.mark.asyncio
async def test_wait_ready_returns_immediately_when_already_loaded(
    app_under_test, monkeypatch
):
    """Pre-set ``_loaded=True`` on both modules — endpoint should
    return in <100 ms with ``ready=True``."""
    from src.embeddings import bge_m3 as bge_mod
    from src.reranker import bge_reranker as rr_mod
    from unittest.mock import MagicMock

    monkeypatch.setattr(bge_mod, "_loaded", True, raising=False)
    monkeypatch.setattr(bge_mod, "_model", MagicMock(name="bge"), raising=False)
    monkeypatch.setattr(bge_mod, "_model_name", bge_mod._DEFAULT_MODEL, raising=False)
    monkeypatch.setattr(rr_mod, "_loaded", True, raising=False)
    monkeypatch.setattr(rr_mod, "_reranker", MagicMock(name="rr"), raising=False)
    monkeypatch.setattr(rr_mod, "_model_name", rr_mod._DEFAULT_MODEL, raising=False)

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        r = await client.get("/models/wait?timeout=5&poll_interval=0.1")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 0.2, f"expected immediate response, took {elapsed:.3f}s"
    data = r.json()
    assert data["ready"] is True
    assert data["embedding"]["status"] == "ready"
    assert data["reranker"]["status"] == "ready"


# ----- 2. polls until ready (loading → loaded) --------------------------

@pytest.mark.asyncio
async def test_wait_ready_polls_until_loaded(app_under_test, monkeypatch):
    """Simulate: ``_loading=True``, after ~1 s flip to ``_loaded=True``.
    Endpoint must wait and surface the transition.
    """
    from src.embeddings import bge_m3 as bge_mod
    from src.reranker import bge_reranker as rr_mod
    from unittest.mock import MagicMock

    # Start: not loaded, but bge is "loading". The endpoint's
    # "not loading and not loaded" guard is therefore False (loading=True).
    monkeypatch.setattr(bge_mod, "_loading", True, raising=False)
    monkeypatch.setattr(bge_mod, "_loaded", False, raising=False)
    monkeypatch.setattr(rr_mod, "_loading", True, raising=False)
    monkeypatch.setattr(rr_mod, "_loaded", False, raising=False)

    async def flip_ready() -> None:
        await asyncio.sleep(0.8)
        bge_mod._loaded = True
        bge_mod._loading = False
        rr_mod._loaded = True
        rr_mod._loading = False
        # Set _model + _model_name to satisfy the loaded-check.
        bge_mod._model = MagicMock(name="bge")
        bge_mod._model_name = bge_mod._DEFAULT_MODEL
        rr_mod._reranker = MagicMock(name="rr")
        rr_mod._model_name = rr_mod._DEFAULT_MODEL

    asyncio.create_task(flip_ready())

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        r = await client.get("/models/wait?timeout=5&poll_interval=0.1")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    assert elapsed >= 0.7, f"expected to wait for flip, returned in {elapsed:.3f}s"
    assert elapsed < 3.0, f"took too long: {elapsed:.3f}s"
    data = r.json()
    assert data["ready"] is True


# ----- 3. 408 on timeout ------------------------------------------------

@pytest.mark.asyncio
async def test_wait_ready_returns_408_on_timeout(app_under_test, monkeypatch):
    """If neither model finishes loading before the deadline, the
    endpoint must respond with HTTP 408 Request Timeout."""
    from src.embeddings import bge_m3 as bge_mod
    from src.reranker import bge_reranker as rr_mod

    # Stay in "loading" forever.
    monkeypatch.setattr(bge_mod, "_loading", True, raising=False)
    monkeypatch.setattr(bge_mod, "_loaded", False, raising=False)
    monkeypatch.setattr(rr_mod, "_loading", True, raising=False)
    monkeypatch.setattr(rr_mod, "_loaded", False, raising=False)

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        r = await client.get("/models/wait?timeout=1&poll_interval=0.2")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 408, f"expected 408, got {r.status_code}: {r.text}"
    assert elapsed >= 0.9, f"expected to wait ~1s, took {elapsed:.3f}s"
    assert elapsed < 2.5, f"took too long: {elapsed:.3f}s"


# ----- 4. immediate response on load error ------------------------------

@pytest.mark.asyncio
async def test_wait_ready_returns_status_on_load_error(
    app_under_test, monkeypatch
):
    """If a load error is set, the endpoint must surface the current
    state immediately (delegating to ``/status`` shape) rather than
    waiting for a happy-path resolution that won't come."""
    from src.embeddings import bge_m3 as bge_mod
    from src.reranker import bge_reranker as rr_mod

    monkeypatch.setattr(bge_mod, "_loading", False, raising=False)
    monkeypatch.setattr(bge_mod, "_loaded", False, raising=False)
    monkeypatch.setattr(bge_mod, "_load_error", RuntimeError("simulated"), raising=False)
    monkeypatch.setattr(rr_mod, "_loading", False, raising=False)
    monkeypatch.setattr(rr_mod, "_loaded", False, raising=False)
    monkeypatch.setattr(rr_mod, "_load_error", None, raising=False)

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        r = await client.get("/models/wait?timeout=5&poll_interval=0.1")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    # Should return immediately — error short-circuits the wait loop.
    assert elapsed < 0.3, f"expected immediate response, took {elapsed:.3f}s"
    data = r.json()
    # Not ready; per-model status reflects the canonical state
    # (delegated to /status which sets _status from loaded/loading only).
    assert data["ready"] is False


# ----- 5. Query validation: timeout -------------------------------------

@pytest.mark.asyncio
async def test_wait_ready_validates_timeout_param(app_under_test):
    """``timeout`` is bounded to ``[1.0, 300.0]``. ``timeout=0`` is invalid."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/models/wait?timeout=0")
    assert r.status_code == 422  # FastAPI validation


# ----- 6. Query validation: poll_interval -------------------------------

@pytest.mark.asyncio
async def test_wait_ready_validates_poll_interval_param(app_under_test):
    """``poll_interval`` is bounded to ``[0.1, 5.0]``."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/models/wait?timeout=2&poll_interval=99")
    assert r.status_code == 422  # FastAPI validation


# ----- 7. surfaces "missing" state immediately (no warmup yet) ---------

@pytest.mark.asyncio
async def test_wait_ready_returns_missing_state_immediately(
    app_under_test, monkeypatch
):
    """When neither model is loading and neither is loaded (the cold
    "no one has clicked Download" state), the endpoint returns the
    canonical missing status instead of waiting for the deadline."""
    from src.embeddings import bge_m3 as bge_mod
    from src.reranker import bge_reranker as rr_mod

    monkeypatch.setattr(bge_mod, "_loading", False, raising=False)
    monkeypatch.setattr(bge_mod, "_loaded", False, raising=False)
    monkeypatch.setattr(bge_mod, "_load_error", None, raising=False)
    monkeypatch.setattr(rr_mod, "_loading", False, raising=False)
    monkeypatch.setattr(rr_mod, "_loaded", False, raising=False)
    monkeypatch.setattr(rr_mod, "_load_error", None, raising=False)

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        r = await client.get("/models/wait?timeout=5&poll_interval=0.1")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 0.3, f"expected immediate response, took {elapsed:.3f}s"
    data = r.json()
    assert data["ready"] is False
    assert data["embedding"]["status"] == "missing"
    assert data["reranker"]["status"] == "missing"