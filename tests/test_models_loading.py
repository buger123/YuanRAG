"""Regression tests for the /models/download event-loop-blocking bug.

Bug history:
    The previous ``POST /models/download`` route used
    ``BackgroundTasks.add_task(_warmup)`` to schedule the synchronous
    warmup function. FastAPI/Starlette runs BackgroundTasks AFTER the
    response is sent but on the SAME asyncio event loop thread. Since
    ``_warmup()`` is a sync function that takes ~30 s for BGE-M3 and
    another ~30 s for the reranker, the entire server (HTTP, WebSocket,
    polling) froze for the duration. Symptom: user clicked Download, the
    WebSocket went silent, the page appeared frozen, the user thought
    the app had crashed and reported "页面黑屏后无响应" (page went black,
    no response).

Fix: three layers
    1. ``asyncio.to_thread(_warmup)`` runs the warmup on a worker thread.
    2. /models/status reports ``status=downloading`` while the load is in
       flight (via a new ``_loading`` flag on each singleton).
    3. ``stream_agent`` checks ``models_loaded()`` at the top and yields
       an error event immediately instead of letting ``retrieve_hybrid``
       synchronously block on ``_load_blocking``.

These tests mock FlagEmbedding's constructors with sleeping stubs so the
warmup actually takes measurable wall time (and the loading flag is
observable), without needing real model weights.
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


# ----- fixtures -----------------------------------------------------------

@pytest.fixture
def slow_flagembed(monkeypatch):
    """Patch FlagEmbedding constructors with stubs that sleep for ~0.3 s.

    Slow enough to keep the ``_loading`` flag observable from a poll
    scheduled right after triggering the download, but fast enough to keep
    the test suite snappy.
    """
    embedder_lock = threading.Lock()
    reranker_lock = threading.Lock()

    def make_embedder(*args, **kwargs):
        time.sleep(0.3)
        inst = MagicMock(name="BGEM3FlagModel")
        inst.encode_queries.return_value = {
            "dense_vecs": [[0.0] * 1024],
            "lexical_weights": [{}],
        }
        inst.encode_corpus.return_value = {
            "dense_vecs": [[0.0] * 1024],
            "lexical_weights": [{}],
        }
        return inst

    def make_reranker(*args, **kwargs):
        time.sleep(0.3)
        return MagicMock(name="FlagReranker")

    from src.embeddings.bge_m3 import reset_for_tests as reset_bge
    from src.reranker.bge_reranker import reset_for_tests as reset_rr
    reset_bge()
    reset_rr()

    monkeypatch.setattr("src.embeddings.bge_m3.BGEM3FlagModel", make_embedder)
    monkeypatch.setattr("src.reranker.bge_reranker.FlagReranker", make_reranker)

    yield

    reset_bge()
    reset_rr()


# ----- 1. /models/status reports "downloading" while in flight -----------

@pytest.mark.asyncio
async def test_status_reports_downloading_during_load(
    app_under_test, monkeypatch, slow_flagembed
):
    """While _loading=True, /models/status must say status=downloading,
    not status=missing. The frontend uses this to render a spinner
    instead of a misleading 'not started' state.
    """
    # Simulate: load just started, not finished yet. Patch the module-level
    # ``_loading`` variable in each singleton's module.
    monkeypatch.setattr("src.embeddings.bge_m3._loading", True)
    monkeypatch.setattr("src.reranker.bge_reranker._loading", True)

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/models/status")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is False
    assert data["embedding"]["status"] == "downloading"
    assert data["reranker"]["status"] == "downloading"


# ----- 2. /models/download returns immediately (doesn't block loop) ------

@pytest.mark.asyncio
async def test_download_endpoint_returns_quickly(
    app_under_test, monkeypatch, slow_flagembed
):
    """Regression: POST /models/download must return WELL BEFORE the
    ~0.6 s wall time of the slow stubs. Previously it blocked the event
    loop for ~30 s.
    """
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        t0 = time.perf_counter()
        r = await client.post("/models/download")
        elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    # Give it generous headroom but still well under the 0.6 s stub time.
    assert elapsed < 0.3, f"download endpoint took {elapsed:.3f}s — likely blocked the event loop"


@pytest.mark.asyncio
async def test_download_endpoint_keeps_status_endpoint_responsive(
    app_under_test, monkeypatch, slow_flagembed
):
    """While the slow warmup is running on the worker thread, /models/status
    must still respond promptly. Previously it blocked along with the warmup.
    """
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Kick off the (slow) download.
        r = await client.post("/models/download")
        assert r.status_code == 200

        # Immediately query status — if the loop were frozen, this would hang.
        t0 = time.perf_counter()
        r = await client.get("/models/status")
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200
        assert elapsed < 0.3, f"status endpoint took {elapsed:.3f}s while download was running"

        # Status should report either "downloading" (if warmup still running)
        # or "ready" (if it somehow finished already — won't here since stubs
        # take 0.6 s).
        data = r.json()
        # We don't assert the exact value — depends on timing — but the
        # endpoint must return a valid schema.
        assert data["embedding"]["status"] in ("missing", "downloading", "ready")
        assert data["reranker"]["status"] in ("missing", "downloading", "ready")


# ----- 3. stream_agent bails fast when models aren't loaded --------------

@pytest.mark.asyncio
async def test_stream_agent_yields_error_when_models_not_loaded(monkeypatch):
    """stream_agent must NOT call the FSM / LLM if either model is not
    loaded — it must yield an error event + done immediately so the WS
    stays responsive.

    v2.0.19 (Phase 3) — the gate is the same shape (the runner still
    checks models_loaded() and bails early), but the symbol it
    guards against is now ``run_fsm`` instead of ``_get_graph``. We
    assert ``run_fsm`` is NOT touched by patching it to a sentinel.
    """
    # Ensure both models are in the "not loaded, not loading" state.
    from src.embeddings.bge_m3 import reset_for_tests as reset_bge
    from src.reranker.bge_reranker import reset_for_tests as reset_rr
    reset_bge()
    reset_rr()

    from src.agent import runner

    # Patch the FSM so we can detect if it gets called — it MUST NOT.
    async def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "run_fsm must not be invoked when models aren't loaded"
        )

    monkeypatch.setattr(runner, "run_fsm", fail_if_called)

    events = []
    agen = runner.stream_agent("thread-x", "你是谁")
    async for ev in agen:
        events.append(ev)

    types = [e["type"] for e in events]
    assert "error" in types, f"expected error event, got {types}"
    assert types[-1] == "done", f"expected final 'done', got {types}"

    # The error message must tell the user what's happening, not leak internals.
    err = next(e for e in events if e["type"] == "error")
    assert "loading" in err["message"].lower() or "wait" in err["message"].lower(), (
        f"error message should mention loading/wait, got: {err['message']!r}"
    )


@pytest.mark.asyncio
async def test_stream_agent_runs_normally_when_models_loaded(monkeypatch):
    """Sanity check: when models are loaded, stream_agent proceeds past
    the gate. We don't actually run the FSM here — just patch
    ``run_fsm`` to an empty async generator and verify the model
    gate doesn't bail early.
    """
    # Force the model-loaded checks to return True.
    # The updated ``models_loaded()`` / ``model_loaded()`` checks _loaded
    # AND _model AND _model_name — all three must be set together.
    import src.embeddings.bge_m3 as bge_mod
    import src.reranker.bge_reranker as rr_mod

    monkeypatch.setattr(bge_mod, "_loaded", True, raising=False)
    monkeypatch.setattr(bge_mod, "_model", MagicMock(name="BGEM3FlagModel"), raising=False)
    monkeypatch.setattr(bge_mod, "_model_name", bge_mod._DEFAULT_MODEL, raising=False)
    monkeypatch.setattr(rr_mod, "_loaded", True, raising=False)
    monkeypatch.setattr(rr_mod, "_reranker", MagicMock(name="FlagReranker"), raising=False)
    monkeypatch.setattr(rr_mod, "_model_name", rr_mod._DEFAULT_MODEL, raising=False)

    from src.agent import runner

    async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
        # Empty FSM stream — model gate passed, FSM produced nothing.
        return
        yield  # make this a generator

    monkeypatch.setattr(runner, "run_fsm", fake_run_fsm)

    events = []
    agen = runner.stream_agent("thread-x", "hello")
    async for ev in agen:
        events.append(ev)

    types = [e["type"] for e in events]
    # No error event — the FSM ran (returned nothing because it's an
    # empty mock) and we got the final done.
    assert "error" not in types
    assert types[-1] == "done"


# ----- 4. Loading flag transitions correctly around get_model -----------

def test_loading_flag_set_during_load(monkeypatch):
    """While ``_load_blocking`` runs, ``model_loading()`` must be True;
    after it finishes (success or failure), ``_loading`` must be False.

    This guarantees /models/status reports ``downloading`` for the
    duration of the actual load.
    """
    import src.embeddings.bge_m3 as mod
    from src.embeddings.bge_m3 import get_model, model_loading, reset_for_tests

    reset_for_tests()

    # The block hasn't run yet — must be False.
    assert mod._loading is False

    observed = {"during": False}

    # Patch the *body* of _load_blocking to record the flag value as it
    # stands inside the lock when the loader runs.
    def fake_load(name):
        # Read the module-level flag, not an import-time snapshot.
        observed["during"] = mod._loading
        return MagicMock(name="BGEM3FlagModel")

    monkeypatch.setattr(mod, "_load_blocking", fake_load)

    get_model()  # triggers the load

    assert observed["during"] is True, "model_loading() was False while load was running"
    assert mod._loading is False
    assert model_loading() is False


def test_loading_flag_cleared_on_load_failure(monkeypatch):
    """If _load_blocking raises, _loading must be cleared (finally)."""
    import src.embeddings.bge_m3 as mod
    from src.embeddings.bge_m3 import get_model, model_loading, reset_for_tests

    reset_for_tests()

    def failing_load(name):
        raise RuntimeError("simulated download failure")

    monkeypatch.setattr(mod, "_load_blocking", failing_load)

    with pytest.raises(RuntimeError):
        get_model()

    # CRITICAL: _loading must be False after failure so subsequent status
    # polls don't lie about "downloading" forever.
    assert mod._loading is False
    assert model_loading() is False


def test_reranker_loading_flag(monkeypatch):
    """Same contract for the reranker."""
    import src.reranker.bge_reranker as mod
    from src.reranker.bge_reranker import get_model, model_loading, reset_for_tests

    reset_for_tests()

    observed = {"during": False}

    def fake_load(name):
        observed["during"] = mod._loading
        return MagicMock(name="FlagReranker")

    monkeypatch.setattr(mod, "_load_blocking", fake_load)

    get_model()
    assert observed["during"] is True
    assert mod._loading is False
    assert model_loading() is False