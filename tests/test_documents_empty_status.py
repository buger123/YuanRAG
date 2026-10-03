"""Tests for the upload + ingest pipeline's empty/failed handling.

Verifies that:
- An upload whose ingest returns 0 chunks still appears in GET /documents
  with ``status="empty"`` and the error_message set.
- An upload whose ingest raises is surfaced with ``status="failed"``.
- DELETE /documents/{id} removes the registry entry, so the next
  ``GET /documents`` no longer surfaces the empty doc.
- POST /documents/clear-thread/{tid} drops registry entries too.
"""
from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
def _stub_heavy_ingestion(monkeypatch):
    """Replace the BGE-M3 embedder with a stub that returns one
    embedding per chunk. Tests that want zero embeddings override
    ``embed_documents.return_value`` inside their ``with`` block.
    """
    fake_embedder = MagicMock(name="BGEM3Embedder")
    """Replace the BGE-M3 embedder with a stub that returns one
    embedding per chunk. Tests that want zero embeddings override
    ``embed_documents.return_value`` inside their ``with`` block."""
    fake_embedder = MagicMock(name="BGEM3Embedder")

    def fake_embed(texts):
        return [
            {"dense": [0.0] * 1024, "sparse": {}}
            for _ in texts
        ]

    fake_embedder.embed_documents.side_effect = fake_embed
    fake_embedder.embed_query.return_value = {"dense": [0.0] * 1024, "sparse": {}}
    monkeypatch.setattr(
        "src.embeddings.bge_m3.BGEM3Embedder", lambda *a, **k: fake_embedder
    )
    yield fake_embedder


async def _upload(client: AsyncClient, filename: str = "doc.pdf") -> str:
    """Upload a tiny PDF-shaped payload; return the doc_id."""
    r = await client.post(
        "/documents/upload",
        files={"file": (filename, io.BytesIO(b"%PDF-fake"), "application/pdf")},
    )
    assert r.status_code == 200, r.text
    return r.json()["doc_id"]


async def _wait_for_registry(doc_id: str, timeout: float = 1.0) -> dict:
    """Poll the doc_registry until ``doc_id`` shows up with a terminal
    status (``indexed`` / ``empty`` / ``failed``) — or the test fails.

    v2.0.27.1 P1-P4 — pre-PR-2 this helper waited only for non-pending
    status (``parsing`` / ``embedding`` / terminal) because
    ``BackgroundTasks.add_task`` blocked the response close until
    ingest finished, so by the time the test polled, the entry had
    always reached terminal state. Post-PR-2 ingest is fire-and-forget
    via ``asyncio.create_task``, so the test response returns in
    milliseconds and polling catches intermediate states like
    ``embedding``. Waiting for terminal matches what every caller
    semantically wants ("did the ingest finish?") and avoids race
    against the lifespan teardown cancelling the in-flight task.
    """
    from src.storage import doc_registry

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        entries = doc_registry.list_for_thread()
        for e in entries:
            if e["doc_id"] == doc_id and e["status"] in (
                "indexed",
                "empty",
                "failed",
            ):
                return e
        await asyncio.sleep(0.02)
    pytest.fail(
        f"doc_registry never reached terminal status for {doc_id}: "
        f"{doc_registry.list_for_thread()}"
    )


@pytest.mark.asyncio
async def test_empty_doc_appears_with_status_empty(app_under_test):
    """When the parser returns 0 chunks, the doc must still surface via
    GET /documents with ``status="empty"`` and an error message."""
    from src.api.routes import documents as documents_route

    with patch.object(
        documents_route,
        "run_ingestion",
        return_value=[],  # parsing finished but produced nothing
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            doc_id = await _upload(client)
            entry = await _wait_for_registry(doc_id)
            assert entry["status"] == "empty"
            assert entry["chunk_count"] == 0
            assert "未提取到文本" in (entry["error_message"] or "")

            r = await client.get("/documents")
        assert r.status_code == 200
        docs = r.json()
        assert len(docs) == 1
        assert docs[0]["doc_id"] == doc_id
        assert docs[0]["status"] == "empty"
        assert docs[0]["chunk_count"] == 0
        assert docs[0]["error_message"] is not None
        assert "未提取到文本" in docs[0]["error_message"]


@pytest.mark.asyncio
async def test_failed_doc_appears_with_status_failed(app_under_test):
    """When the parser raises, the doc surfaces with ``status="failed"``
    and the exception message (so the user can see why)."""
    from src.api.routes import documents as documents_route

    with patch.object(
        documents_route,
        "run_ingestion",
        side_effect=RuntimeError("corrupt PDF xref"),
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            doc_id = await _upload(client)
            entry = await _wait_for_registry(doc_id)
            assert entry["status"] == "failed"
            assert "RuntimeError" in (entry["error_message"] or "")
            assert "corrupt PDF" in (entry["error_message"] or "")

            r = await client.get("/documents")
        docs = r.json()
        assert len(docs) == 1
        assert docs[0]["status"] == "failed"
        assert "corrupt PDF" in docs[0]["error_message"]


@pytest.mark.asyncio
async def test_delete_drops_registry_entry(app_under_test):
    """DELETE /documents/{id} must clear the registry entry — otherwise
    a deleted empty doc would resurrect on the next GET /documents."""
    from src.api.routes import documents as documents_route
    from src.storage import doc_registry

    with patch.object(
        documents_route, "run_ingestion", return_value=[]
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            doc_id = await _upload(client, "ephemeral.pdf")
            await _wait_for_registry(doc_id)
            assert any(
                e["doc_id"] == doc_id
                for e in doc_registry.list_for_thread()
            )

            r = await client.delete(f"/documents/{doc_id}")
        assert r.status_code == 200

        # Registry entry gone, GET /documents empty.
        assert not any(
            e["doc_id"] == doc_id
            for e in doc_registry.list_for_thread()
        )

        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.get("/documents")
        assert r.json() == []


@pytest.mark.asyncio
async def test_clear_thread_drops_registry_entries(app_under_test):
    """POST /documents/clear-thread/{tid} drops registry entries too,
    so deleting a conversation also forgets its empty/failed docs."""
    from src.api.routes import documents as documents_route

    with patch.object(
        documents_route, "run_ingestion", return_value=[]
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Upload two empty docs on thread "alpha" and one on
            # thread "beta". We can then scope the clear to alpha and
            # verify beta's docs survive.
            for i in range(2):
                r = await client.post(
                    "/documents/upload?thread_id=alpha",
                    files={
                        "file": (
                            f"alpha-{i}.pdf",
                            io.BytesIO(b"%PDF-fake"),
                            "application/pdf",
                        )
                    },
                )
                assert r.status_code == 200
            r = await client.post(
                "/documents/upload?thread_id=beta",
                files={
                    "file": (
                        "beta.pdf",
                        io.BytesIO(b"%PDF-fake"),
                        "application/pdf",
                    )
                },
            )
            assert r.status_code == 200
            await asyncio.sleep(0.3)

            before_alpha = await client.get("/documents?thread_id=alpha")
            before_beta = await client.get("/documents?thread_id=beta")
            assert len(before_alpha.json()) == 2
            assert len(before_beta.json()) == 1

            r = await client.post("/documents/clear-thread/alpha")
            assert r.status_code == 200
            body = r.json()
            assert body["ok"] is True
            assert body["registry_entries_dropped"] == 2

            after_alpha = await client.get("/documents?thread_id=alpha")
            after_beta = await client.get("/documents?thread_id=beta")
        assert after_alpha.json() == []
        # Beta's doc survives the alpha clear.
        assert len(after_beta.json()) == 1
        assert after_beta.json()[0]["filename"] == "beta.pdf"


@pytest.mark.asyncio
async def test_indexed_doc_keeps_status_indexed(app_under_test):
    """Sanity check that the happy path (chunks produced) still flows
    through with ``status="indexed"`` — the registry merge shouldn't
    regress the well-behaved case."""
    from src.api.routes import documents as documents_route

    fake_chunks = [
        {"text": "real chunk", "meta": {"page_number": 1}},
    ]
    with patch.object(
        documents_route, "run_ingestion", return_value=fake_chunks
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            doc_id = await _upload(client, "real.pdf")
            entry = await _wait_for_registry(doc_id)
            assert entry["status"] == "indexed"
            assert entry["chunk_count"] == 1

            r = await client.get("/documents")
        docs = r.json()
        assert len(docs) == 1
        assert docs[0]["status"] == "indexed"
        assert docs[0]["chunk_count"] == 1


# ============================================================
# v1.1.12 — structured ``diagnosis`` companion payload
# ============================================================
# The 4xx sniff failure path and the 200/empty status path both
# carry a structured ``diagnosis`` dict alongside the human
# ``detail`` / ``error_message`` string. Tests below pin the
# schema so the frontend (separate repo) can read ``diagnosis``
# without parsing free-text.


@pytest.mark.asyncio
async def test_empty_xlsx_carries_diagnosis_with_format_specific_hint(
    app_under_test,
):
    """A truly-empty .xlsx (zero sheets / zero rows even at the header
    level) ends up with ``status='empty'`` AND a ``diagnosis`` blob
    whose ``hint_zh`` mentions "Excel" or "sheet" — NOT the PDF
    "扫描件" copy. This is the v1.1.12 contract.

    v1.1.12b: a header-only .xlsx is no longer the right zero-chunk
    trigger — the chunker now emits a header-only chunk so the header
    text is searchable. We use a truly-vacuous workbook (one sheet
    with no header AND no data rows) to exercise the empty path.
    """
    from src.api.routes import documents as documents_route
    from src.ingestion.parsers import xlsx_parser
    from src.ingestion.parsers.xlsx_parser import parse_xlsx
    from io import BytesIO
    from openpyxl import Workbook

    # Build a truly-empty workbook (one sheet, no cells at all).
    wb = Workbook()
    ws = wb.active
    ws.title = "TrulyEmpty"
    buf = BytesIO()
    wb.save(buf)
    body = buf.getvalue()

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/documents/upload",
            files={
                "file": (
                    "truly_empty.xlsx",
                    io.BytesIO(body),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert r.status_code == 200, r.text
        doc_id = r.json()["doc_id"]
        entry = await _wait_for_registry(doc_id)
    assert entry["status"] == "empty"
    assert "diagnosis" in entry  # diagnosis field present
    diagnosis = entry["diagnosis"]
    assert diagnosis["format"] == "xlsx"
    # The .xlsx hint mentions Excel / sheet — not the PDF-specific copy.
    assert "Excel" in diagnosis["hint_zh"] or "sheet" in diagnosis["hint_zh"].lower()
    assert "扫描件" not in diagnosis["hint_zh"]


@pytest.mark.asyncio
async def test_header_only_xlsx_now_produces_chunks_via_chunk(
    app_under_test,
):
    """v1.1.12b — A header-only .xlsx now lands ``status='indexed'``,
    not ``status='empty'``: the chunker yields a header-only chunk so
    the column names are still indexed and searchable. Pre-fix this
    was the most common cause of the user-reported "未提取到文本"
    complaint on "non-empty" workbooks."""
    from io import BytesIO
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "OnlyHeader"
    ws["A1"] = "Field"
    ws["B1"] = "Value"
    buf = BytesIO()
    wb.save(buf)
    body = buf.getvalue()

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/documents/upload",
            files={
                "file": (
                    "header_only.xlsx",
                    io.BytesIO(body),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert r.status_code == 200, r.text
        doc_id = r.json()["doc_id"]
        entry = await _wait_for_registry(doc_id)
    assert entry["status"] == "indexed"
    assert entry["chunk_count"] == 1
    # No diagnosis on the success path.
    assert entry.get("diagnosis") is None


@pytest.mark.asyncio
async def test_empty_status_response_omits_diagnosis(
    app_under_test,
):
    """Sanity: when chunks DO get produced (happy path), the registry
    entry does NOT carry a diagnosis blob — diagnosis is reserved for
    the failure / empty paths."""
    from src.api.routes import documents as documents_route

    fake_chunks = [
        {"text": "real chunk", "meta": {"page_number": 1}},
    ]
    with patch.object(
        documents_route, "run_ingestion", return_value=fake_chunks
    ):
        transport = ASGITransport(app=app_under_test)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            doc_id = await _upload(client, "real.pdf")
            entry = await _wait_for_registry(doc_id)
    assert entry["status"] == "indexed"
    assert entry.get("diagnosis") is None


@pytest.mark.asyncio
async def test_sniff_failure_response_carries_diagnosis(
    app_under_test,
):
    """A 400 magic-byte sniff rejection must include a structured
    ``diagnosis`` blob alongside the human ``message`` string. The
    shape is locked so clients can read ``diagnosis.declared_ext``
    / ``diagnosis.reason`` without parsing free-text."""
    import io
    body = b"Hello, this is plain text, not a PDF."
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("evil.pdf", io.BytesIO(body), "application/pdf")}
        resp = await client.post("/documents/upload", files=files)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # The new wrapper dict has ``message`` (human) + ``diagnosis``
    # (structured) so a client that read ``detail.message`` still
    # gets the original text.
    assert "message" in detail
    assert "magic-byte" in detail["message"]
    assert "diagnosis" in detail
    diagnosis = detail["diagnosis"]
    assert diagnosis["declared_ext"] == ".pdf"
    assert diagnosis["reason"]
    # The PDF-specific reason field mentions the missing magic.
    assert "%PDF" in diagnosis["reason"] or "head_bytes" in diagnosis["reason"]


@pytest.mark.asyncio
async def test_sniff_failure_text_format_lists_tried_encodings(
    app_under_test,
):
    """For .md / .txt rejections, ``diagnosis.tried_encodings`` lists
    every encoding the validator attempted — the most useful piece of
    data for a user who wants to know why their Chinese-Windows-saved
    file got bounced."""
    import io
    body = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("binary.md", io.BytesIO(body), "text/markdown")}
        resp = await client.post("/documents/upload", files=files)
    assert resp.status_code == 400
    diagnosis = resp.json()["detail"]["diagnosis"]
    assert diagnosis["declared_ext"] == ".md"
    assert diagnosis["sniff_layer"] == "encoding"
    # Encoding list mirrors ``_looks_like_text``'s accept set.
    for enc in (
        "utf-16-le-bom",
        "utf-16-be-bom",
        "utf-8",
        "gb18030",
        "big5",
        "cp1252",
    ):
        assert enc in diagnosis["tried_encodings"]