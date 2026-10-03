"""Tests for v2.0.27.2 PR-3 checkpointer corruption contract.

Pre-PR-3 — corruption was invisible:
- ``_deserialize_state`` raised bare ``json.JSONDecodeError`` /
  ``NotImplementedError`` / etc. with no thread_id context. The
  caller (``load_latest``) bubbled those up as 500s to
  ``GET /sessions/{tid}/messages``.
- ``list_threads._to_summary`` swallowed everything with
  ``except Exception: state = {"messages": []}`` — leaving the
  sidebar looking "empty" with no operator-visible signal that
  the underlying row was corrupt. The audit noted this as
  P1-P5 ("hidden corruption invisible").

Post-PR-3 — narrow ``except`` wraps the deserialization path,
raises a typed :class:`CorruptCheckpointError` carrying
``thread_id`` / ``original_error`` / ``trace_id``. Callers
(``load_latest``, ``list_threads``) handle it explicitly and
surface a count for the operator.

The 5 tests below pin the contract.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. _deserialize_state raises CorruptCheckpointError on bad JSON
# ---------------------------------------------------------------------------


def test_deserialize_state_raises_corrupt_checkpoint_on_bad_json():
    """``_deserialize_state`` must raise :class:`CorruptCheckpointError`
    (NOT bare ``json.JSONDecodeError``) on malformed JSON. The wrapped
    exception must carry ``thread_id`` + ``trace_id`` so the operator
    can find the bad row in production logs."""
    from src.storage.checkpointer import (
        CorruptCheckpointError,
        _deserialize_state,
    )

    with pytest.raises(CorruptCheckpointError) as exc_info:
        _deserialize_state("{not valid json", thread_id="bad-thread-1")

    err = exc_info.value
    assert err.thread_id == "bad-thread-1"
    assert isinstance(err.original_error, json.JSONDecodeError)
    # trace_id is 12 hex chars (uuid4().hex[:12])
    assert len(err.trace_id) == 12
    assert all(c in "0123456789abcdef" for c in err.trace_id)
    # __cause__ preserved via `raise ... from e`
    assert err.__cause__ is err.original_error


# ---------------------------------------------------------------------------
# 2. load_latest returns None on corruption (not 500)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_latest_returns_none_on_corrupt_json(tmp_path, monkeypatch):
    """``load_latest`` must swallow :class:`CorruptCheckpointError`
    and return ``None`` (same semantics as "row not found"), so the
    caller ``GET /sessions/{tid}/messages`` returns 200 + empty list
    instead of 500-ing on the corrupt row."""
    from src.storage import checkpointer
    from src.storage.checkpointer import (
        CorruptCheckpointError,
        startup,
        shutdown,
        save,
        _db_conn,
    )

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    # Reset module state so startup() opens a fresh connection at our path.
    checkpointer.reset_for_tests()
    await startup()

    try:
        # Module-level _db_conn is assigned in startup() but it's
        # visible from the test scope via attribute lookup — read it
        # through the module reference each time (not a captured
        # value) so we always see the current state.
        conn = checkpointer._db_conn
        if conn is None:
            # Defensive: poll briefly if startup hadn't propagated
            # yet. (Usually the issue is monkeypatch timing — the
            # module imported `history_db_path` at line 74, so
            # patching `src.storage.checkpointer.history_db_path`
            # hits the right binding. If `_db_conn` is still None,
            # the patch didn't take effect — re-check by looking up
            # the actual function reference the module is using.)
            actual_fn = checkpointer.history_db_path
            actual_path = actual_fn()
            raise AssertionError(
                f"_db_conn is None after startup(); "
                f"patched history_db_path returns {db_path}, "
                f"module is using history_db_path()={actual_path}"
            )
        # First: write a valid row, save it.
        await save("good-thread", {"messages": []}, step=1)

        # Now: inject a corrupt row by writing raw bytes via the same
        # connection the public save() uses. We bypass save() because
        # save() goes through _serialize_state which would sanitize.
        # The conn is sqlite3 stdlib (not aiosqlite); thread-safe via
        # check_same_thread=False.
        conn.execute(
            "INSERT OR REPLACE INTO states (thread_id, step, state_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("bad-thread", 2, "{this is not valid json", "2026-09-21T00:00:00Z"),
        )
        conn.commit()

        # Good thread still loads fine.
        good = await checkpointer.load_latest("good-thread")
        assert good is not None
        assert good.get("messages") == []

        # Corrupt thread returns None instead of raising.
        bad = await checkpointer.load_latest("bad-thread")
        assert bad is None
    finally:
        await shutdown()


# ---------------------------------------------------------------------------
# 3. list_threads skips corrupt rows + returns corrupt count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_threads_skips_corrupt_and_returns_count(
    tmp_path, monkeypatch
):
    """``list_threads`` must skip rows that fail deserialization and
    return ``(valid, corrupt_count)``. The valid list omits the bad
    row entirely; the corrupt count is non-zero."""
    from src.storage import checkpointer
    from src.storage.checkpointer import startup, shutdown, save, _db_conn

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    checkpointer.reset_for_tests()
    await startup()

    try:
        await save("t-valid-1", {"messages": []}, step=1)
        await save("t-valid-2", {"messages": []}, step=1)
        conn = checkpointer._db_conn
        assert conn is not None, "startup() must have set _db_conn"
        # Inject two corrupt rows so we exercise the counter.
        conn.execute(
            "INSERT OR REPLACE INTO states (thread_id, step, state_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("t-corrupt-1", 1, "{bad", "2026-09-21T00:00:01Z"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO states (thread_id, step, state_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("t-corrupt-2", 1, "[1, 2,", "2026-09-21T00:00:02Z"),
        )
        conn.commit()

        valid, corrupt = await checkpointer.list_threads()
        # 2 valid rows in the result (corrupt ones omitted).
        valid_tids = {row["thread_id"] for row in valid}
        assert "t-valid-1" in valid_tids
        assert "t-valid-2" in valid_tids
        assert "t-corrupt-1" not in valid_tids
        assert "t-corrupt-2" not in valid_tids
        # Counter accurate.
        assert corrupt == 2
    finally:
        await shutdown()


# ---------------------------------------------------------------------------
# 4. Legacy _revive_not_implemented happy-path still works
# ---------------------------------------------------------------------------


def test_legacy_not_implemented_envelope_still_revives():
    """Regression guard for v2.0.23: a state_json that contains the
    Pydantic Source ``not_implemented`` envelope must still revive
    cleanly (no CorruptCheckpointError). This is the legacy-data
    happy path — the narrow ``except`` must NOT catch the inner
    ``NotImplementedError`` that the walker handles gracefully."""
    from src.storage.checkpointer import _deserialize_state

    # Minimal valid envelope shape — the walker converts it to a
    # plain dict, then ``loads()`` happily returns a state tree.
    state = {
        "messages": [],
        "sources": [
            {
                "lc": 1,
                "type": "not_implemented",
                "id": ["src", "api", "schemas", "Source"],
                "repr": "Source(index=0, chunk_id='', doc_id='web-x', url='http://x', title='t', snippet='s')",
            }
        ],
    }
    state_json = json.dumps(state, ensure_ascii=False)

    # Pre-fix path: walked + revived + returned dict successfully.
    # Post-PR-3 path: same — the inner ``except NotImplementedError``
    # in the defensive-last-resort block swallows the envelope
    # ``NotImplementedError`` BEFORE the outer narrow except sees it.
    # No CorruptCheckpointError should be raised.
    result = _deserialize_state(state_json, thread_id="legacy-thread")
    assert isinstance(result, dict)
    # The revived source is a plain dict (from the envelope).
    sources = result.get("sources") or []
    assert len(sources) == 1


# ---------------------------------------------------------------------------
# 5. sessions.py surfaces corrupt count via X-Corrupt-Thread-Count header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessions_endpoint_sets_corrupt_count_header_on_corruption(
    app_under_test, monkeypatch, tmp_path
):
    """``GET /sessions`` must set ``X-Corrupt-Thread-Count`` header
    when :func:`checkpointer.list_threads` reports corrupt rows.
    Pre-PR-3 this signal was invisible (silent swallow → empty
    sidebar with no operator notification).

    v2.0.27.2 P1-P5 — operator gets the data-loss signal via:
    (a) the response header, (b) a structured ``logger.warning``
    in the server log with the trace_ids from each corrupt row.
    """
    from httpx import ASGITransport, AsyncClient
    from src.api.routes import sessions as sessions_routes
    from src.storage.checkpointer import CorruptCheckpointError

    async def fake_list_threads_with_corruption(*, limit=500):
        # Simulate: 1 valid thread (with a message so the route's
        # ``message_count > 0 or doc_count > 0`` filter keeps it) +
        # 3 corrupt threads that the real list_threads would have
        # logged + skipped.
        return (
            [
                {
                    "thread_id": "good",
                    "step": 1,
                    "updated_at": "2026-09-21T00:00:00",
                    # Non-empty messages list — the route's final
                    # ``if message_count == 0 and doc_count == 0:
                    # continue`` filter would otherwise drop this
                    # row (no docs, no messages).
                    "messages": [MagicMock(content="hello world")],
                },
            ],
            3,  # 3 corrupt rows skipped
        )

    monkeypatch.setattr(
        sessions_routes, "list_threads", fake_list_threads_with_corruption
    )

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/sessions")

    assert r.status_code == 200
    # Body still shaped correctly — wire schema preserved.
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["thread_id"] == "good"
    # NEW: corrupt count surfaced as response header.
    assert r.headers.get("X-Corrupt-Thread-Count") == "3"


@pytest.mark.asyncio
async def test_sessions_endpoint_omits_corrupt_header_when_clean(
    app_under_test, monkeypatch
):
    """When no corruption occurs, ``X-Corrupt-Thread-Count`` is
    absent (we only set it when corrupt_count > 0). This avoids
    spurious ``X-Corrupt-Thread-Count: 0`` noise on healthy
    responses — the header is reserved for actual data loss
    events."""
    from httpx import ASGITransport, AsyncClient
    from src.api.routes import sessions as sessions_routes

    async def fake_list_threads_clean(*, limit=500):
        return ([], 0)

    monkeypatch.setattr(
        sessions_routes, "list_threads", fake_list_threads_clean
    )

    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/sessions")

    assert r.status_code == 200
    assert "X-Corrupt-Thread-Count" not in r.headers


# ---------------------------------------------------------------------------
# 6. trace_id is unique per corruption event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_id_unique_per_corruption(tmp_path, monkeypatch):
    """Each ``CorruptCheckpointError`` gets a fresh 12-hex trace_id
    so the operator can grep for one specific corruption event
    without false positives across concurrent calls."""
    from src.storage import checkpointer
    from src.storage.checkpointer import startup, shutdown, save, _db_conn

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    checkpointer.reset_for_tests()
    await startup()

    try:
        conn = checkpointer._db_conn
        assert conn is not None, "startup() must have set _db_conn"
        # Inject 5 corrupt rows; load_latest each one and capture trace_ids.
        for i in range(5):
            conn.execute(
                "INSERT OR REPLACE INTO states (thread_id, step, state_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (f"trace-{i}", 1, "{bad", "2026-09-21T00:00:00Z"),
            )
        conn.commit()

        trace_ids = set()
        for i in range(5):
            # load_latest returns None but logs a trace_id. To verify
            # uniqueness we need to either (a) capture log output or
            # (b) call _deserialize_state directly. Go with (b).
            from src.storage.checkpointer import _deserialize_state
            from src.storage.checkpointer import CorruptCheckpointError

            try:
                _deserialize_state("{bad", thread_id=f"trace-{i}")
            except CorruptCheckpointError as e:
                trace_ids.add(e.trace_id)

        # 5 corruptions → 5 unique trace_ids. (12 hex = 48 bits of
        # entropy — collision probability negligible for 5 events.)
        assert len(trace_ids) == 5
    finally:
        await shutdown()
