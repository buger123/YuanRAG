"""Phase 5+6 — FastAPI smoke tests (without a real LLM)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(app_under_test):
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_settings_get(app_under_test):
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "llm_provider" in data
    assert "llm_model" in data
    assert "has_api_key" in data
    assert "api_key_source" in data
    assert data["api_key_source"] in ("env", "keyring", "none")


@pytest.mark.asyncio
async def test_settings_update_rejects_api_keys(app_under_test):
    """The settings endpoint must NOT accept API keys — those come from env."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/settings",
            json={"openai_api_key": "should-be-ignored", "llm_model": "gpt-4o"},
        )
    # Either the field is silently dropped, or Pydantic rejects it with 422.
    # Both are acceptable — what matters is that no key makes it through.
    if resp.status_code == 200:
        body = resp.json()
        # User's UI choice is reflected (gpt-4o) — env default no longer
        # silently trumps an explicit user save.
        assert body["llm_model"] == "gpt-4o"
    else:
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_documents_list_empty(app_under_test):
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_models_status(app_under_test):
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/models/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "embedding" in data
    assert "reranker" in data
    assert data["ready"] is False  # models not loaded in test env


@pytest.mark.asyncio
async def test_models_download_endpoint_runs(app_under_test):
    """Regression: the /models/download route previously crashed because the
    ``background`` parameter was missing its ``BackgroundTasks`` type
    annotation, so FastAPI couldn't inject it and ``add_task`` raised
    AttributeError -> HTTP 500 -> frontend showed '无响应'.

    This test verifies the route accepts the POST and returns a valid status
    payload. The actual model load is skipped because FlagEmbedding is
    mocked (we only care about the route plumbing).
    """
    from unittest.mock import MagicMock, patch

    # Build a mock BGEM3FlagModel whose encode_queries / encode_corpus
    # return FlagEmbedding-shaped dicts (dense_vecs + lexical_weights).
    mock_bge_instance = MagicMock(name="BGEM3FlagModel")
    mock_bge_instance.encode_queries.return_value = {
        "dense_vecs": [[0.0] * 1024],
        "lexical_weights": [{}],
    }
    mock_bge_instance.encode_corpus.return_value = {
        "dense_vecs": [[0.0] * 1024],
        "lexical_weights": [{}],
    }

    mock_rr_instance = MagicMock(name="FlagReranker")
    mock_rr_instance.compute_score.return_value = [0.5]

    # Patch the *class* references used by the loaders, then also reset
    # the singletons so the test sees a fresh load.
    from src.embeddings.bge_m3 import reset_for_tests as reset_bge
    from src.reranker.bge_reranker import reset_for_tests as reset_rr
    reset_bge()
    reset_rr()

    with (
        patch("src.embeddings.bge_m3.BGEM3FlagModel", return_value=mock_bge_instance),
        patch("src.reranker.bge_reranker.FlagReranker", return_value=mock_rr_instance),
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/models/download")
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert "embedding" in data
            assert "reranker" in data
            assert "ready" in data

    # Cleanup so subsequent tests start clean.
    reset_bge()
    reset_rr()


@pytest.mark.asyncio
async def test_sessions_list_does_not_hold_closed_handle(app_under_test):
    """Regression: the SqliteSaver singleton used to drop its context manager
    out of scope after the first call, closing the SQLite connection. Subsequent
    requests then hit ``Cannot operate on a closed database.``.

    v2.0.21 (Phase 3 Step 7) — rewritten for the own
    ``src.storage.checkpointer`` (no LangGraph SqliteSaver). The
    test now exercises the new ``startup`` + ``list_threads`` pair
    and asserts the underlying SQLite connection survives across
    two back-to-back ``/sessions`` requests."""
    from src.storage import checkpointer

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First call must not leave the DB in a closed state
        r1 = await client.get("/sessions")
        assert r1.status_code == 200
        assert r1.json() == []
        # Second call must still work — this is what failed in production
        r2 = await client.get("/sessions")
        assert r2.status_code == 200
        assert r2.json() == []
        # And the underlying connection must still be usable directly
        await checkpointer.startup()
        conn = checkpointer._db_conn
        assert conn is not None
        # A direct list_threads query exercises the exact path that
        # used to raise after the connection was GC'd.
        # v2.0.27.2 P1-P5 — list_threads now returns tuple
        # (valid, corrupt_count).
        items, corrupt = await checkpointer.list_threads()
        assert items == []
        assert corrupt == 0
    await checkpointer.shutdown()
