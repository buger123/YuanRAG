"""Phase 6 — 100 MB upload cap.

The endpoint enforces ``MAX_UPLOAD_BYTES`` BEFORE kicking off background
ingestion — this is essential because reading the body of a 200 MB file
into memory just to reject it is wasteful. We verify:

* ``MAX_UPLOAD_BYTES`` is set to 100 MB (1024 * 1024 * 100).
* Files UNDER the cap are accepted (200).
* Files OVER the cap are rejected with HTTP 413 (Payload Too Large).

We don't actually upload 100 MB of bytes — that would be slow and bloat
the test suite — instead we shrink the cap to a tiny value and verify
the boundary. We additionally verify the constant's actual value via
direct assertion (cheap) so a future config drift gets caught.
"""
from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient


# ============================================================
# Constant sanity
# ============================================================


def test_max_upload_bytes_is_100mb():
    """Regression: ``MAX_UPLOAD_BYTES`` must be 100 MB. If someone bumps
    it down to recover disk space, the limit guard becomes too tight;
    if they raise it, accidental 500 MB uploads become possible. Pin
    the exact value here so the design choice is explicit."""
    from config.constants import MAX_UPLOAD_BYTES

    assert MAX_UPLOAD_BYTES == 100 * 1024 * 1024, (
        f"MAX_UPLOAD_BYTES must be 100 MiB, got {MAX_UPLOAD_BYTES}"
    )


# ============================================================
# Boundary tests (with shrunken cap to avoid 100 MB allocations)
# ============================================================


@pytest.fixture(autouse=True)
def _mock_ingestion(monkeypatch):
    """Stub the heavy ingestion path so the test doesn't need a working
    Docling / BGE-M3 stack. Returning an empty chunk list is fine — we
    only care about the size guard, not what happens after.
    """
    monkeypatch.setattr(
        "src.api.routes.documents.run_ingestion",
        lambda filename, content=None: [],
    )
    yield


@pytest.fixture
def shrunken_cap(monkeypatch):
    """Shrink ``MAX_UPLOAD_BYTES`` to 1 KiB so we can test the boundary
    without allocating 100 MB. The route reads ``MAX_UPLOAD_BYTES`` from
    ``config.constants`` at request time, so patching the module-level
    import works as long as the route module imports the name directly."""
    import config.constants as const_mod

    real = const_mod.MAX_UPLOAD_BYTES
    const_mod.MAX_UPLOAD_BYTES = 1024
    # Also patch the route module's binding (it does
    # ``from config.constants import MAX_UPLOAD_BYTES`` at module load).
    from src.api.routes import documents as docs_route

    real_route = docs_route.MAX_UPLOAD_BYTES
    docs_route.MAX_UPLOAD_BYTES = 1024
    yield 1024
    const_mod.MAX_UPLOAD_BYTES = real
    docs_route.MAX_UPLOAD_BYTES = real_route


@pytest.mark.asyncio
async def test_upload_under_cap_is_accepted(app_under_test, shrunken_cap):
    """A payload smaller than the cap must be accepted (200) and
    scheduled for background ingestion."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # shrunken_cap is 1024 → 500 bytes is well under
        files = {"file": ("small.txt", io.BytesIO(b"x" * 500), "text/plain")}
        resp = await client.post("/documents/upload", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["doc_id"]
    assert body["filename"] == "small.txt"


@pytest.mark.asyncio
async def test_upload_over_cap_rejected_with_413(app_under_test, shrunken_cap):
    """A payload strictly larger than the cap must be rejected with
    HTTP 413 Payload Too Large — BEFORE we kick off background
    ingestion. The error message should mention the cap so the user
    knows why their upload failed."""
    cap = shrunken_cap
    payload = b"x" * (cap + 1)  # 1 byte over
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("big.bin", io.BytesIO(payload), "application/octet-stream")}
        resp = await client.post("/documents/upload", files=files)

    assert resp.status_code == 413, resp.text
    # The detail should give the user a clue (mention MB or the cap)
    detail = resp.json().get("detail", "")
    assert "large" in detail.lower() or "MB" in detail


@pytest.mark.asyncio
async def test_upload_exactly_at_cap_is_accepted(app_under_test, shrunken_cap):
    """Boundary: ``len(content) > MAX_UPLOAD_BYTES`` rejects only
    strictly-larger; the cap itself is accepted."""
    cap = shrunken_cap
    payload = b"x" * cap  # NOT greater than cap
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("exact.bin", io.BytesIO(payload), "application/octet-stream")}
        resp = await client.post("/documents/upload", files=files)
    assert resp.status_code == 200, resp.text