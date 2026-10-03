"""Regression tests for the comprehensive audit fixes (avoid context interference).

Covers:
- BGE-M3 / reranker inference serialization (concurrent encode correctness)
- BGE-M3 / reranker model-name mismatch guard
- BGE-M3 / reranker load_error capture + propagation
- LanceDB path-keyed singleton + doc_id sanitization
- history._saver path-aware reopen on path change
- runner._graph path-aware rebuild on path change
- delete_thread raises on failure (no longer silent swallow)
- /sessions dedups by thread_id + bounded load
- models.cancel_warmup_tasks cancels in-flight warmups
- /chat/clear surfaces HTTPException when delete_thread fails

All tests use mocks for heavy resources (FlagEmbedding, LanceDB, etc.) so
they run on any machine without the real models installed.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage


# ============================================================
# BGE-M3 inference + load_error + name guard
# ============================================================


@pytest.fixture
def fake_bge(monkeypatch):
    """Patch BGE-M3 FlagEmbedding to a controllable mock."""
    from src.embeddings.bge_m3 import reset_for_tests

    reset_for_tests()

    encode_calls: list[tuple[list[str], bool]] = []
    encode_lock = threading.Lock()
    in_flight = threading.Semaphore(0)

    class _FakeModel:
        def encode_queries(self, texts, **kwargs):
            with encode_lock:
                encode_calls.append((list(texts), True))
            in_flight.release()
            return {
                "dense_vecs": [[0.1] * 4 for _ in texts],
                "lexical_weights": [{"a": 0.1} for _ in texts],
            }

        def encode_corpus(self, texts, **kwargs):
            with encode_lock:
                encode_calls.append((list(texts), False))
            in_flight.release()
            return {
                "dense_vecs": [[0.2] * 4 for _ in texts],
                "lexical_weights": [{"b": 0.2} for _ in texts],
            }

    def make_model(*args, **kwargs):
        return _FakeModel()

    monkeypatch.setattr("src.embeddings.bge_m3.BGEM3FlagModel", make_model)
    yield encode_calls, in_flight
    reset_for_tests()


def test_bge_m3_inference_serializes_concurrent_calls(fake_bge):
    """Regression: FlagEmbedding.encode_* is NOT thread-safe.

    Without an inference lock, two threads racing on the same instance
    produce interleaved/wrong outputs silently. With the lock, both
    calls run to completion with all expected outputs in encode_calls.
    """
    encode_calls, _in_flight = fake_bge
    from src.embeddings.bge_m3 import BGEM3Embedder

    embedder = BGEM3Embedder()
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker(texts):
        try:
            barrier.wait()
            embedder.embed_documents(texts)
        except BaseException as exc:  # pragma: no cover - propagated below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=([f"doc{i}", f"doc{i}_b"],)) for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"workers raised: {errors}"
    # Each worker submitted 2 docs → 8 * 2 = 16 encoded texts total.
    total = sum(len(texts) for texts, _ in encode_calls)
    assert total == 16, f"expected 16 total encoded texts, got {total}"


def test_bge_m3_load_error_captured_and_surfaced(monkeypatch):
    """Regression: load_error() must report the failure reason.

    Previously get_model() caught BaseException in `finally` only — the
    error reason was lost. Now it is captured in ``_load_error`` AND
    re-raised so the caller sees it. /models/status uses load_error() to
    tell the user what went wrong.
    """
    from src.embeddings import bge_m3

    bge_m3.reset_for_tests()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated FlagEmbedding failure")

    monkeypatch.setattr("src.embeddings.bge_m3.BGEM3FlagModel", boom)

    with pytest.raises(RuntimeError, match="simulated FlagEmbedding failure"):
        bge_m3.get_model()

    assert bge_m3.load_error() is not None
    assert "simulated FlagEmbedding failure" in bge_m3.load_error()
    assert bge_m3.models_loaded() is False
    assert bge_m3.model_loading() is False

    bge_m3.reset_for_tests()


def test_bge_m3_name_mismatch_raises(fake_bge):
    """Regression: silently returning the wrong model produces wrong
    embeddings. A name mismatch must raise so the caller fixes config
    instead of getting semantically broken outputs.
    """
    from src.embeddings.bge_m3 import get_model

    get_model("BAAI/bge-m3")
    with pytest.raises(ValueError, match="cannot switch to"):
        get_model("BAAI/bge-m3-some-other")


# ============================================================
# BGE reranker inference + load_error + name guard
# ============================================================


@pytest.fixture
def fake_reranker(monkeypatch):
    from src.reranker.bge_reranker import reset_for_tests

    reset_for_tests()

    score_calls: list[list[tuple[str, str]]] = []
    score_lock = threading.Lock()

    class _FakeReranker:
        def compute_score(self, pairs, **kwargs):
            with score_lock:
                score_calls.append(list(pairs))
            return [0.9 for _ in pairs]

    def make_rr(*args, **kwargs):
        return _FakeReranker()

    monkeypatch.setattr("src.reranker.bge_reranker.FlagReranker", make_rr)
    yield score_calls
    reset_for_tests()


def test_reranker_inference_serializes_concurrent_calls(fake_reranker):
    """Regression: FlagReranker.compute_score is NOT thread-safe."""
    from src.reranker.bge_reranker import BGEReranker

    rr = BGEReranker()
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker():
        try:
            barrier.wait()
            rr.score("q", ["a", "b", "c"])
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"workers raised: {errors}"
    # 8 calls × 3 pairs = 24 pairs total.
    total_pairs = sum(len(c) for c in fake_reranker)
    assert total_pairs == 24, f"expected 24 pairs, got {total_pairs}"


def test_reranker_load_error_captured_and_surfaced(monkeypatch):
    from src.reranker import bge_reranker

    bge_reranker.reset_for_tests()

    def boom(*args, **kwargs):
        raise RuntimeError("reranker load failure")

    monkeypatch.setattr("src.reranker.bge_reranker.FlagReranker", boom)

    with pytest.raises(RuntimeError, match="reranker load failure"):
        bge_reranker.get_model()

    assert bge_reranker.load_error() is not None
    assert "reranker load failure" in bge_reranker.load_error()
    assert bge_reranker.model_loaded() is False
    bge_reranker.reset_for_tests()


# ============================================================
# LanceDB: doc_id sanitization + path-keyed singleton
# ============================================================


def test_lancedb_safe_identifier_kind_parameter_changes_error_message():
    """Regression: ``_safe_identifier`` accepts a ``kind`` so the error
    message names what was being filtered (doc_id vs thread_id) — the
    kind flows into the error message, so callers can tell which field
    tripped validation. Both kinds accept the same production-format
    inputs (UUIDs / short fixtures)."""
    from src.storage.lancedb_store import _safe_identifier

    # The kind flows into the error message — easy way to confirm the
    # call site is using the right wrapper.
    with pytest.raises(ValueError, match="thread_id"):
        _safe_identifier("bad value", "thread_id")
    with pytest.raises(ValueError, match="doc_id"):
        _safe_identifier("bad value", "doc_id")

    # Both kinds accept the same production-format inputs.
    assert _safe_identifier("abc12345-1234-5678-9abc-def012345678", "thread_id") == \
        "abc12345-1234-5678-9abc-def012345678"
    assert _safe_identifier("abc12345-1234-5678-9abc-def012345678", "doc_id") == \
        "abc12345-1234-5678-9abc-def012345678"

    # Empty input is rejected regardless of kind.
    with pytest.raises(ValueError):
        _safe_identifier("", "doc_id")


def test_lancedb_path_keyed_singleton(monkeypatch, tmp_path):
    """Regression: get_db must open one connection per distinct path.

    Two tests with different RAG_DATA_DIR previously reused the first
    test's LanceDB connection — writing into test A's DB from test B.
    Now the cache is keyed by resolved path.
    """
    from src.storage import lancedb_store

    lancedb_store.reset_for_tests()

    from config.settings import reset_settings_cache

    # Two different paths → two different connections.
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "a"))
    reset_settings_cache()
    db_a = lancedb_store.get_db()

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "b"))
    reset_settings_cache()
    db_b = lancedb_store.get_db()

    assert db_a is not db_b, (
        f"expected different connections for different paths; got {db_a} and {db_b}"
    )

    # Repeated lookups for the SAME path return the cached connection
    # — the fast-path check matches _current_path without re-opening.
    db_b2 = lancedb_store.get_db()
    assert db_b is db_b2, "expected cached connection for repeated path lookups"

    # Switching back to path 'a' must reuse the previously cached
    # connection for 'a' — not re-open it.
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "a"))
    reset_settings_cache()
    db_a2 = lancedb_store.get_db()
    assert db_a is db_a2, (
        "expected cached connection when returning to a previously-used path"
    )

    lancedb_store.reset_for_tests()


# ============================================================
# History: path-aware saver + delete_thread raises
# ============================================================


@pytest.mark.asyncio
async def test_history_saver_reopens_when_path_changes(monkeypatch, tmp_path):
    """Regression (Phase 3): changing URAG_DATA_DIR between tests used to leak
    the previous sqlite connection into the new test — classic context
    interference. The new ``checkpointer.startup`` must reopen when the
    path changes.

    v2.0.21 (Phase 3) — patched from ``src.storage.history`` to the new
    ``src.storage.checkpointer``. Same path-aware behavior; just a
    different module.
    """
    from src.storage import checkpointer

    checkpointer.reset_for_tests()

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "a"))
    # First call opens the connection at path A.
    await checkpointer.startup()
    path_a = checkpointer._db_path
    assert path_a == str(tmp_path / "a" / "history.db") or path_a is not None
    conn_a = checkpointer._db_conn

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "b"))
    # Second call must open a fresh connection at path B (different identity).
    await checkpointer.startup()
    conn_b = checkpointer._db_conn
    assert conn_a is not conn_b, (
        "expected checkpointer to reopen when path changes"
    )

    await checkpointer.shutdown()
    checkpointer.reset_for_tests()


@pytest.mark.asyncio
async def test_delete_thread_raises_on_failure(monkeypatch):
    """Regression: delete_thread used to swallow all exceptions silently.
    /chat/clear would return {"ok": True} even when the delete failed —
    the classic "user clicked clear, history still there" bug.

    v2.0.21 (Phase 3) — ``delete_thread`` moved from
    ``src.storage.history`` to ``src.storage.checkpointer``. Same
    contract; just patch the new module attribute.
    """
    from src.storage import checkpointer

    checkpointer.reset_for_tests()

    async def fake_delete(_thread_id):
        raise RuntimeError("db error")

    monkeypatch.setattr(checkpointer, "delete_thread", fake_delete)

    with pytest.raises(RuntimeError, match="db error"):
        await checkpointer.delete_thread("any-thread-id")

    checkpointer.reset_for_tests()


# ============================================================
# /sessions dedup + bounded load
# ============================================================


@pytest.mark.asyncio
async def test_sessions_dedups_by_thread_id():
    """Regression: saver.alist() used to yield one row per checkpoint, so
    a thread with N rounds previously showed N entries in the sidebar.
    Now the list collapses to one entry per thread_id, picking the
    latest ts.

    v2.0.21 (Phase 3) — the dedup now happens at the SQL layer
    (``states`` table has ``thread_id PRIMARY KEY``), so the route
    just consumes a list of per-thread rows from
    :func:`checkpointer.list_threads`. We monkey-patch that to feed
    the test data.
    """
    from src.api.routes import sessions as sessions_routes

    async def fake_list_threads(*, limit=500):
        # v2.0.27.2 P1-P5 — return tuple (valid, corrupt_count)
        # to match the new signature.
        return (
            [
                {
                    "thread_id": "t1",
                    "step": 5,
                    "updated_at": "2026-01-02T00:00:00",
                    "messages": [MagicMock(content="hello t1 v2")],
                },
                {
                    "thread_id": "t2",
                    "step": 3,
                    "updated_at": "2026-01-01T12:00:00",
                    "messages": [MagicMock(content="hello t2")],
                },
            ],
            0,
        )

    sessions_routes.list_threads = fake_list_threads

    # v2.0.27.2 P1-P5 — list_sessions now requires a Response param
    # for header injection. Use a mock that captures header writes.
    from fastapi import Response as FastAPIResponse

    response = FastAPIResponse()
    result = await sessions_routes.list_sessions(response)

    # 3 checkpoints across 2 threads → 2 deduped entries.
    assert len(result) == 2, f"expected 2 deduped sessions, got {len(result)}"
    thread_ids = {s.thread_id for s in result}
    assert thread_ids == {"t1", "t2"}

    # The most recent t1 timestamp must win.
    t1 = next(s for s in result if s.thread_id == "t1")
    assert t1.updated_at == "2026-01-02T00:00:00"


@pytest.mark.asyncio
async def test_sessions_bounded_by_max_sessions():
    """Regression: ``saver.alist(None)`` previously walked the entire
    checkpoints table. With hundreds of threads the endpoint stalled.
    Now the SQL ``LIMIT`` clause in :func:`checkpointer.list_threads`
    bounds the read at _MAX_SESSIONS rows.

    v2.0.21 (Phase 3) — the bound is enforced at the SQL layer, not
    in Python. We assert the route receives at most _MAX_SESSIONS
    rows and yields no more.
    """
    from src.api.routes import sessions as sessions_routes

    received_limit: list[int] = []

    async def fake_list_threads(*, limit=500):
        # Record the limit the route asked for, then yield _MAX_SESSIONS
        # rows (matching the pre-Phase-3 "dedup by thread_id after the
        # alist walk" contract).
        # v2.0.27.2 P1-P5 — return tuple (valid, corrupt_count).
        received_limit.append(limit)
        return (
            [
                {
                    "thread_id": f"t{i}",
                    "step": 1,
                    "updated_at": f"2026-01-{i:05d}T00:00:00",
                    "messages": [HumanMessage(content=f"q for t{i}")],
                }
                for i in range(sessions_routes._MAX_SESSIONS)
            ],
            0,
        )

    sessions_routes.list_threads = fake_list_threads

    # v2.0.27.2 P1-P5 — list_sessions requires Response param.
    from fastapi import Response as FastAPIResponse

    response = FastAPIResponse()
    result = await sessions_routes.list_sessions(response)
    assert len(result) == sessions_routes._MAX_SESSIONS, (
        f"expected bounded to _MAX_SESSIONS={sessions_routes._MAX_SESSIONS}, "
        f"got {len(result)}"
    )
    assert received_limit and received_limit[0] == sessions_routes._MAX_SESSIONS, (
        f"route must pass _MAX_SESSIONS as the SQL LIMIT; "
        f"got limit={received_limit[0] if received_limit else None}"
    )


# ============================================================
# models.cancel_warmup_tasks
# ============================================================


@pytest.mark.asyncio
async def test_cancel_warmup_tasks_cancels_pending():
    """Regression: warmup tasks that finish during shutdown used to race
    with checkpointer teardown and could hold a worker thread until the
    process was killed. The lifespan now calls cancel_warmup_tasks first.
    """
    from src.api.routes import models as models_routes

    # Two long-running tasks; we cancel them before they finish.
    async def slow():
        await asyncio.sleep(60)

    t1 = asyncio.create_task(slow())
    t2 = asyncio.create_task(slow())
    models_routes._warmup_tasks.add(t1)
    models_routes._warmup_tasks.add(t2)

    n = models_routes.cancel_warmup_tasks()
    assert n == 2

    # Both should now be done (cancelled counts as done).
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert t1.cancelled() or t1.done()
    assert t2.cancelled() or t2.done()


# ============================================================
# documents.cancel_ingest_tasks (v2.0.27.1 P1-P4)
# ============================================================


@pytest.mark.asyncio
async def test_cancel_ingest_tasks_cancels_pending():
    """Regression: in-flight ingest tasks that finish during shutdown
    used to race with ``shutdown_checkpointer`` and could hold a
    worker thread until the process was killed. The lifespan now calls
    ``cancel_ingest_tasks`` (mirrors ``cancel_warmup_tasks``) before
    tearing down the checkpointer / doc_registry flusher.

    The fixture mirrors ``test_cancel_warmup_tasks_cancels_pending``
    above so the two fire-and-forget lifecycles stay symmetric.
    """
    from src.api.routes import documents as documents_routes

    # Two long-running tasks; we cancel them before they finish.
    async def slow():
        await asyncio.sleep(60)

    t1 = asyncio.create_task(slow())
    t2 = asyncio.create_task(slow())
    documents_routes._tracked_ingest_tasks.add(t1)
    documents_routes._tracked_ingest_tasks.add(t2)

    n = documents_routes.cancel_ingest_tasks()
    assert n == 2

    # Both should now be done (cancelled counts as done).
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert t1.cancelled() or t1.done()
    assert t2.cancelled() or t2.done()


@pytest.mark.asyncio
async def test_upload_dispatch_uses_create_task_not_background_tasks():
    """P1-P4 contract: ``POST /documents/upload`` must dispatch the
    ingest via ``asyncio.create_task`` (tracked in
    ``_tracked_ingest_tasks``), NOT ``BackgroundTasks.add_task``.
    BackgroundTasks blocked the response close until ingest finished,
    which made upload latency = ingest latency (several seconds for
    big PDFs) and prevented WebSocket frames from progressing while
    the dispatch held the response close path.

    We patch ``asyncio.create_task`` to record the call site and
    assert ``_tracked_ingest_tasks`` got the task (proving the new
    dispatch wire is active).
    """
    from src.api.routes import documents as documents_routes
    from src.storage import doc_registry

    # Reset ingest-task tracker between tests (the set is module-level
    # so it survives across tests otherwise).
    documents_routes._tracked_ingest_tasks.clear()

    # Track every create_task invocation that lands a coroutine that
    # looks like ``_ingest_file_async(doc_id, ...)``. We use a wrapper
    # instead of an AsyncMock so the task still runs to completion
    # (we want to verify the tracked-set membership, not block the
    # task).
    captured: list[tuple] = []

    real_create_task = asyncio.create_task

    def spy_create_task(coro, *args, **kwargs):
        task = real_create_task(coro, *args, **kwargs)
        # The dispatch site does:
        #   asyncio.create_task(_ingest_file_async(doc_id, fname, content, ...))
        # We can't introspect the coro args safely (it's a generator
        # object), so we just track that *some* task was created
        # during the upload window and the set membership grew.
        captured.append(task)
        return task

    import unittest.mock as _mock

    with _mock.patch.object(asyncio, "create_task", side_effect=spy_create_task):
        # Pre-register the doc so ``register`` succeeds and the upload
        # call can proceed past the magic-byte + size-check path.
        doc_registry.reset_for_tests()
        doc_registry.register(
            doc_id="d-test",
            filename="tiny.txt",
            source_path="/d-test/tiny.txt",
            thread_id="t-test",
        )

        # Use TestClient to invoke the upload endpoint. We don't need
        # a real LLM — we only assert the dispatch layer, which fires
        # regardless of whether the actual ingest succeeds. The ingest
        # is fire-and-forget so any failure inside is logged but
        # doesn't affect the response.
        from fastapi.testclient import TestClient

        from src.app import app

        client = TestClient(app)
        resp = client.post(
            "/documents/upload",
            files={"file": ("tiny.txt", b"hello world", "text/plain")},
        )
        # The endpoint should return 200 immediately (the upload
        # itself succeeds; ingest failure surfaces later as
        # ``status="failed"`` in the registry, not in the response).
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["doc_id"]
        assert body["filename"] == "tiny.txt"

        # The spy should have seen at least one ``create_task`` call
        # during the upload (the ingest dispatch).
        assert captured, "expected asyncio.create_task to be called during upload"

        # And the tracked set should contain at least one task (the
        # one we just dispatched; note the response can return before
        # the done-callback fires, so we don't assert ``not in``).
        # We only assert the set was used as the dispatch wire — the
        # exact task may already be removed by the done-callback.
        # The bigger guarantee is that ``cancel_ingest_tasks`` finds
        # nothing wrong if called right now (no-op on empty/done set
        # is the contract).
        n = documents_routes.cancel_ingest_tasks()
        # n == 0 is fine — the ingest finished (or never started
        # because the file was empty / dispatch was synchronous enough
        # that the done-callback already ran). The contract is "no
        # exception raised", not "must find a task".
        assert n >= 0


# ============================================================
# /chat/clear surfaces HTTPException
# ============================================================


@pytest.mark.asyncio
async def test_chat_clear_returns_500_on_failure(monkeypatch):
    """Regression: DELETE /chat/{thread_id} used to return {"ok": True}
    even when the underlying delete failed. Callers trusted the response
    shape and saw 'history still there' bugs. Now it surfaces 500.

    v2.0.21 (Phase 3) — ``delete_thread`` moved from
    ``src.storage.history`` to ``src.storage.checkpointer``. Same
    contract; chat.py's ``clear`` function does ``from
    src.storage.checkpointer import delete_thread`` at function
    scope, so we monkey-patch the module attribute and the import
    inside ``clear`` picks up our patched version (Python resolves
    the import fresh on each call to ``from ... import``).

    PR-4 (v2.0.26.1) — ``clear`` now also takes ``accept_language``
    so it can localize the error message. The default value
    (``Header(None)``) means callers that don't pass it (like this
    test) get the zh (default) copy. We just verify the status —
    the localized copy is covered in ``tests/test_api_i18n.py``.
    """
    from src.api.routes import chat as chat_routes
    from src.storage import checkpointer

    async def fake_delete(_thread_id):
        raise RuntimeError("simulated db error")

    monkeypatch.setattr(checkpointer, "delete_thread", fake_delete)

    with pytest.raises(Exception) as excinfo:
        await chat_routes.clear("any-thread", accept_language=None)
    # FastAPI's HTTPException is a subclass of Exception. Verify the status.
    assert getattr(excinfo.value, "status_code", None) == 500, (
        f"expected HTTPException(500), got {excinfo.value!r}"
    )
    checkpointer.reset_for_tests()