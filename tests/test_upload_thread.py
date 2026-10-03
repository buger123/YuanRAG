"""Phase 6 — upload stamps the thread_id on every chunk.

The heavy ingestion path (Docling parsing, BGE-M3 embedding) is mocked
out — these tests only verify the HTTP wiring:

* Upload accepts an optional ``thread_id`` query param and stamps it on
  every ChunkRecord.
* Upload rejects an unsafe thread_id with HTTP 400.
* GET /documents filters by thread_id.
* POST /documents/clear-thread/{tid} deletes only that thread's chunks.

We don't reach the embedding model here; we mock ``BGEM3Embedder`` to
return zero-vector chunks so we can exercise the full happy path without
BGE-M3 actually loaded.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def _mock_heavy_ingestion(monkeypatch):
    """Replace Docling parsing + BGE-M3 embedding with deterministic
    stubs. The ingestion path still threads every ChunkRecord through
    ``add_chunks``, so we can verify the thread_id stamp end-to-end.
    """
    # Stub BGEM3Embedder at the import site used by routes/documents.py
    fake_embedder = MagicMock(name="BGEM3Embedder")
    fake_embedder.embed_documents.return_value = [
        {"dense": [0.1] * 1024, "sparse": {1: 0.1}},
        {"dense": [0.2] * 1024, "sparse": {2: 0.2}},
    ]
    fake_embedder.embed_query.return_value = {"dense": [0.0] * 1024, "sparse": {}}
    monkeypatch.setattr(
        "src.embeddings.bge_m3.BGEM3Embedder", lambda *a, **k: fake_embedder
    )

    # Stub run_ingestion to return 2 deterministic chunks per call
    def fake_run_ingestion(filename_or_url, content=None):
        return [
            {
                "text": f"chunk 1 for {filename_or_url}",
                "meta": {"chunk_type": "text", "page_number": 1},
            },
            {
                "text": f"chunk 2 for {filename_or_url}",
                "meta": {"chunk_type": "text", "page_number": 2},
            },
        ]

    monkeypatch.setattr(
        "src.api.routes.documents.run_ingestion", fake_run_ingestion
    )
    yield fake_embedder, fake_run_ingestion


@pytest.mark.asyncio
async def test_upload_stamps_thread_id_on_chunks(app_under_test):
    """POST /documents/upload?thread_id=... must stamp every ChunkRecord
    with that thread_id."""
    from src.storage.lancedb_store import get_table

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")}
        resp = await client.post(
            "/documents/upload?thread_id=alpha", files=files
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["thread_id"] == "alpha"

    # Wait for background ingestion to finish, then verify chunks
    import asyncio

    # Background tasks run on the FastAPI event loop; yield so they complete.
    await asyncio.sleep(0.2)

    table = get_table()
    arrow = table.to_arrow()
    rows = arrow.column("thread_id").to_pylist()
    # All chunks should be tagged "alpha"
    assert all(t == "alpha" for t in rows), f"thread_ids not stamped: {rows}"
    assert len(rows) == 2  # 2 chunks per fake_run_ingestion call


@pytest.mark.asyncio
async def test_upload_without_thread_id_stamps_empty(app_under_test):
    """Backwards-compat: omitting thread_id stamps the empty-string
    sentinel (globally visible) — does not crash."""
    from src.storage.lancedb_store import get_table

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("plain.txt", io.BytesIO(b"data"), "text/plain")}
        resp = await client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["thread_id"] is None  # None on the wire, "" in storage

    import asyncio

    await asyncio.sleep(0.2)

    table = get_table()
    arrow = table.to_arrow()
    rows = arrow.column("thread_id").to_pylist()
    assert rows == ["", ""]


@pytest.mark.asyncio
async def test_upload_rejects_unsafe_thread_id(app_under_test):
    """An injection-shaped thread_id must be rejected with HTTP 400 —
    not silently accepted and not crashed."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("x.txt", io.BytesIO(b"x"), "text/plain")}
        resp = await client.post(
            "/documents/upload",
            params={"thread_id": "a'; DROP TABLE chunks; --"},
            files=files,
        )
    assert resp.status_code == 400, resp.text


@pytest.mark.asyncio
async def test_documents_list_filtered_by_thread(app_under_test):
    """GET /documents?thread_id=alpha returns ONLY documents scoped to
    that thread."""
    import asyncio

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Upload one file on each thread
        for tid in ("alpha", "beta"):
            files = {"file": (f"{tid}.txt", io.BytesIO(b"x"), "text/plain")}
            r = await client.post(
                f"/documents/upload?thread_id={tid}", files=files
            )
            assert r.status_code == 200, r.text
        await asyncio.sleep(0.3)

        # Filter to alpha
        r_alpha = await client.get("/documents?thread_id=alpha")
        r_beta = await client.get("/documents?thread_id=beta")
        r_none = await client.get("/documents?thread_id=gamma")
        r_all = await client.get("/documents")

    assert r_alpha.status_code == 200
    assert r_beta.status_code == 200
    assert r_none.status_code == 200
    assert r_all.status_code == 200

    alpha_docs = r_alpha.json()
    beta_docs = r_beta.json()
    none_docs = r_none.json()
    all_docs = r_all.json()

    # Alpha sees only its doc, beta sees only its doc, gamma sees none,
    # unfiltered sees both.
    assert {d["filename"] for d in alpha_docs} == {"alpha.txt"}
    assert {d["filename"] for d in beta_docs} == {"beta.txt"}
    assert none_docs == []
    assert {d["filename"] for d in all_docs} == {"alpha.txt", "beta.txt"}

    # And the documents carry their thread_id in the JSON payload
    assert {d["thread_id"] for d in alpha_docs} == {"alpha"}
    assert {d["thread_id"] for d in beta_docs} == {"beta"}


@pytest.mark.asyncio
async def test_clear_thread_endpoint_removes_only_targeted_docs(app_under_test):
    """POST /documents/clear-thread/{tid} deletes ONLY that thread's
    chunks — the other thread's data survives."""
    import asyncio

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for tid in ("alpha", "beta"):
            files = {"file": (f"{tid}.txt", io.BytesIO(b"x"), "text/plain")}
            await client.post(
                f"/documents/upload?thread_id={tid}", files=files
            )
        await asyncio.sleep(0.3)

        # Sanity: both threads have data
        before_alpha = await client.get("/documents?thread_id=alpha")
        before_beta = await client.get("/documents?thread_id=beta")
        assert len(before_alpha.json()) == 1
        assert len(before_beta.json()) == 1

        # Clear alpha
        r = await client.post("/documents/clear-thread/alpha")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["thread_id"] == "alpha"
        # rows_deleted is an int (>= 0); the exact count depends on the
        # LanceDB version's count_rows availability, so we don't pin it.
        assert isinstance(body["rows_deleted"], int)
        assert body["rows_deleted"] >= 0

        # Alpha gone, beta survives
        after_alpha = await client.get("/documents?thread_id=alpha")
        after_beta = await client.get("/documents?thread_id=beta")
        assert after_alpha.json() == []
        assert len(after_beta.json()) == 1
        assert after_beta.json()[0]["filename"] == "beta.txt"


@pytest.mark.asyncio
async def test_clear_thread_rejects_empty_thread_id(app_under_test):
    """POST /documents/clear-thread/ must reject an empty thread_id —
    wiping the entire table by accident is worse than a 400."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # FastAPI treats an empty path segment as 404, but we test the
        # explicit bad-input case where thread_id is just unsafe.
        r = await client.post(
            "/documents/clear-thread/bad'; DROP TABLE chunks; --"
        )
    assert r.status_code == 400, r.text