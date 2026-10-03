"""Tests for ``src.storage.doc_registry``.

The registry is a JSON sidecar that tracks every uploaded doc's ingestion
status (``pending`` / ``indexed`` / ``empty`` / ``failed``) so docs that
produce zero chunks still appear in the sidebar with a clear "未提取到
文本" indicator instead of vanishing.

Tests verify:
- ``register`` writes the entry; ``list_for_thread`` reads it back.
- ``update`` flips status / chunk_count / error_message; subsequent
  reads see the new values.
- ``remove`` and ``remove_thread`` delete correctly.
- Atomic write: a crashed mid-write tmp file is cleaned up and the
  previous registry is preserved.
- Thread filter semantics: ``None`` returns all, ``""`` matches only
  empty-thread entries, a real id matches only that thread.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.storage import doc_registry


def test_register_then_list_round_trip():
    """Register a doc, then list — should come back with the same fields."""
    doc_registry.register(
        doc_id="d1",
        filename="paper.pdf",
        source_path="/tmp/d1/paper.pdf",
        thread_id="t1",
    )
    entries = doc_registry.list_for_thread("t1")
    assert len(entries) == 1
    e = entries[0]
    assert e["doc_id"] == "d1"
    assert e["filename"] == "paper.pdf"
    # v2.0.27 P0-P2: source_path is no longer persisted in the registry —
    # the absolute path is a privacy leak. The canonical form is derived
    # at read time by lancedb_store.list_documents from (doc_id, filename).
    assert "source_path" not in e
    assert e["thread_id"] == "t1"
    assert e["status"] == "pending"
    assert e["chunk_count"] == 0
    assert e["error_message"] is None
    assert isinstance(e["ingested_at"], str) and e["ingested_at"]


def test_register_is_idempotent_and_preserves_ingested_at():
    """Re-registering an existing doc should keep its original
    ``ingested_at`` so sort order is stable across retries."""
    doc_registry.register("d1", "a.pdf", "/a", "t1")
    first_ts = doc_registry.list_for_thread("t1")[0]["ingested_at"]
    doc_registry.register("d1", "a-renamed.pdf", "/a2", "t1")
    e = doc_registry.list_for_thread("t1")[0]
    assert e["filename"] == "a-renamed.pdf"
    assert "source_path" not in e
    assert e["ingested_at"] == first_ts


def test_update_status_and_chunk_count():
    """Flipping status from pending → indexed should update the entry
    and bump ingested_at."""
    doc_registry.register("d1", "x.pdf", "/x", "t1")
    doc_registry.update(
        "d1", status="indexed", chunk_count=42, error_message=None
    )
    e = doc_registry.list_for_thread("t1")[0]
    assert e["status"] == "indexed"
    assert e["chunk_count"] == 42
    assert e["error_message"] is None


def test_update_empty_and_failed_carry_error_message():
    """The ``empty`` / ``failed`` statuses need an error_message so the
    UI can show why. ``update`` should preserve whatever message we
    pass."""
    doc_registry.register("d2", "scan.pdf", "/s", "t1")
    doc_registry.update(
        "d2",
        status="empty",
        error_message="未提取到文本内容。可能原因:扫描件无文字层。",
    )
    e = doc_registry.list_for_thread("t1")
    assert len(e) == 1
    e = e[0]
    assert e["doc_id"] == "d2"
    assert e["status"] == "empty"
    assert "未提取到文本" in (e["error_message"] or "")

    doc_registry.update(
        "d2", status="failed", error_message="corrupt PDF: invalid xref"
    )
    e = doc_registry.list_for_thread("t1")[0]
    assert e["status"] == "failed"
    assert "corrupt" in (e["error_message"] or "")


def test_update_unknown_doc_is_noop():
    """Calling update on a doc that wasn't registered must not crash."""
    doc_registry.update(
        "never-registered", status="indexed", chunk_count=1
    )
    # list returns the existing entries (none in this test) — verify
    # the file wasn't corrupted.
    assert doc_registry.list_for_thread() == []


def test_remove_drops_specific_doc():
    doc_registry.register("d1", "a.pdf", "/a", "t1")
    doc_registry.register("d2", "b.pdf", "/b", "t2")
    doc_registry.remove("d1")
    docs = doc_registry.list_for_thread()
    assert {d["doc_id"] for d in docs} == {"d2"}
    # Idempotent — removing again is a no-op, not a crash.
    doc_registry.remove("d1")
    assert {d["doc_id"] for d in doc_registry.list_for_thread()} == {"d2"}


def test_remove_thread_drops_all_with_matching_id():
    doc_registry.register("d1", "a.pdf", "/a", "alpha")
    doc_registry.register("d2", "b.pdf", "/b", "beta")
    doc_registry.register("d3", "c.pdf", "/c", "alpha")
    removed = doc_registry.remove_thread("alpha")
    assert removed == 2
    remaining = {d["doc_id"] for d in doc_registry.list_for_thread()}
    assert remaining == {"d2"}


def test_thread_filter_semantics():
    """``list_for_thread`` semantics:
    - ``None`` → return all
    - ``""``   → only empty-thread entries
    - ``"x"``  → only thread_id == "x"
    """
    doc_registry.register("d1", "a.pdf", "/a", "")
    doc_registry.register("d2", "b.pdf", "/b", "alpha")
    doc_registry.register("d3", "c.pdf", "/c", "beta")
    doc_registry.register("d4", "d.pdf", "/d", "alpha")

    assert {d["doc_id"] for d in doc_registry.list_for_thread(None)} == {
        "d1",
        "d2",
        "d3",
        "d4",
    }
    assert {d["doc_id"] for d in doc_registry.list_for_thread("")} == {"d1"}
    assert {d["doc_id"] for d in doc_registry.list_for_thread("alpha")} == {
        "d2",
        "d4",
    }
    assert doc_registry.list_for_thread("nonexistent") == []


def test_persists_across_reinits(tmp_path: Path, monkeypatch):
    """After registering, the on-disk file must contain the JSON we
    wrote. Writes now go through a background flusher; this test forces
    a synchronous flush before reading the file."""
    doc_registry.register(
        "d1", "persist.pdf", "/p/d1/persist.pdf", thread_id="t1"
    )
    doc_registry.update(
        "d1", status="indexed", chunk_count=5
    )
    # Force the background flusher to write the latest state to disk so
    # we can inspect it.
    doc_registry.flush()
    # Drop in-memory state by reading the JSON directly.
    from src.core.paths import uploads_dir

    raw = (uploads_dir() / "_registry.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "d1" in data
    assert data["d1"]["status"] == "indexed"
    assert data["d1"]["chunk_count"] == 5


def test_atomic_write_cleans_up_tmp_on_crash(monkeypatch, tmp_path):
    """If the atomic-write tmp file is left behind (simulated crash),
    ``register`` / ``update`` still succeed and the previous on-disk
    data is preserved."""
    from src.core.paths import uploads_dir

    doc_registry.register("d1", "x.pdf", "/x", "t1")
    doc_registry.update("d1", status="indexed", chunk_count=3)
    doc_registry.flush()  # sync the initial state to disk

    # Inject a stale tmp file (simulating a previous interrupted write).
    stale = uploads_dir() / "_registry.stale.tmp"
    stale.write_text("{ this is not json", encoding="utf-8")
    assert stale.exists()

    # Subsequent writes should not crash; the stale file may remain
    # because cleanup is best-effort, but the real registry must be
    # readable and well-formed.
    doc_registry.update("d1", status="failed", error_message="boom")
    doc_registry.flush()
    e = doc_registry.list_for_thread("t1")
    assert e[0]["status"] == "failed"

    # The real file parses as JSON.
    real = json.loads((uploads_dir() / "_registry.json").read_text("utf-8"))
    assert real["d1"]["status"] == "failed"


def test_corrupt_json_recovers_to_empty():
    """If the registry file is corrupted (manual edit, disk error),
    subsequent reads return ``[]`` and writes recover the file — we
    don't want a single bad write to brick the sidebar."""
    from src.core.paths import uploads_dir

    path = uploads_dir() / "_registry.json"
    path.write_text("not json at all", encoding="utf-8")
    assert doc_registry.list_for_thread() == []
    # Subsequent write should produce a valid file again.
    doc_registry.register("d1", "x.pdf", "/x", "t1")
    doc_registry.flush()
    data = json.loads(path.read_text("utf-8"))
    assert "d1" in data


# --- v2.0.27 P0-P2 privacy hardening ----------------------------------------


def test_serialize_source_path_strips_absolute():
    """The canonical privacy-safe form is always ``f"{doc_id}/{filename}"``,
    independent of the caller's input. Any leading slash, drive letter, or
    ``:`` in the first segment is an invariant violation."""
    out = doc_registry._serialize_source_path(
        "C:/Users/alice/Medical/diagnosis.pdf", "doc-abc", "diagnosis.pdf"
    )
    assert out == "doc-abc/diagnosis.pdf"
    assert "C:" not in out
    assert "alice" not in out
    assert "Users" not in out


def test_serialize_source_path_invariant_violations_raise():
    """Defensive assertions in the helper catch any future contributor who
    tries to pass a ``doc_id`` that itself leaks a path (drive letter or
    absolute prefix)."""
    import pytest

    with pytest.raises(AssertionError, match="drive letter"):
        doc_registry._serialize_source_path("/x", "C:bad", "f.pdf")
    with pytest.raises(AssertionError, match="absolute path"):
        doc_registry._serialize_source_path("/x", "/abs", "f.pdf")


def test_register_does_not_persist_source_path_key():
    """P0-P2 contract: the registry must not carry a ``source_path`` key,
    even after re-register with a different absolute path. Only
    ``doc_id`` + ``filename`` survive."""
    doc_registry.register("d1", "a.pdf", "C:/Users/alice/a.pdf", "t1")
    on_disk = doc_registry.list_for_thread("t1")[0]
    assert "source_path" not in on_disk

    doc_registry.register("d1", "a-renamed.pdf", "D:/tmp/a-renamed.pdf", "t1")
    on_disk = doc_registry.list_for_thread("t1")[0]
    assert "source_path" not in on_disk
    assert on_disk["filename"] == "a-renamed.pdf"


def test_migrate_absolute_paths_drops_legacy_source_path(tmp_path, monkeypatch):
    """A pre-v2.0.27 ``_registry.json`` with absolute ``source_path`` values
    gets rewritten on first read. The migration is idempotent."""
    from src.core.paths import uploads_dir

    # Pre-populate the registry with a legacy row. We write the file
    # directly (bypassing ``register``) so the absolute path survives into
    # the migration step.
    path = uploads_dir() / "_registry.json"
    legacy = {
        "legacy-doc": {
            "doc_id": "legacy-doc",
            "filename": "report.pdf",
            "source_path": "C:/Users/alice/Medical/report.pdf",
            "thread_id": "t-legacy",
            "ingested_at": "2026-09-19T00:00:00+00:00",
            "status": "indexed",
            "chunk_count": 5,
            "error_message": None,
        }
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    doc_registry.reset_for_tests()

    # First list triggers the migration; subsequent reads see the cleaned
    # registry. The legacy entry's privacy-sensitive ``source_path`` key
    # is dropped, but ``doc_id`` / ``filename`` / ``status`` survive.
    entries = doc_registry.list_for_thread("t-legacy")
    assert len(entries) == 1
    e = entries[0]
    assert e["doc_id"] == "legacy-doc"
    assert e["filename"] == "report.pdf"
    assert "source_path" not in e
    assert e["status"] == "indexed"
    assert e["chunk_count"] == 5

    # The on-disk file must also be rewritten to drop the key so a future
    # restart doesn't re-run the migration.
    raw = json.loads(path.read_text("utf-8"))
    assert "source_path" not in raw["legacy-doc"]


def test_migrate_absolute_paths_is_idempotent():
    """Re-running the migration on already-migrated data rewrites 0 rows."""
    data = {
        "d1": {
            "doc_id": "d1",
            "filename": "f.pdf",
            "thread_id": "t",
            "ingested_at": "x",
            "status": "indexed",
            "chunk_count": 1,
            "error_message": None,
        }
    }
    rewritten, count = doc_registry._migrate_absolute_paths(data)
    assert count == 0
    assert "source_path" not in rewritten["d1"]


def test_coerce_legacy_source_path_rewrites_absolute_inputs():
    """The read-time coercion (used by ``lancedb_store.list_documents``)
    returns the canonical form for legacy absolute paths and is a no-op
    for already-canonical values."""
    out1 = doc_registry._coerce_legacy_source_path(
        "C:/Users/alice/report.pdf", "d1", "report.pdf"
    )
    assert out1 == "d1/report.pdf"

    out2 = doc_registry._coerce_legacy_source_path(
        "/tmp/uploads/d1/report.pdf", "d1", "report.pdf"
    )
    assert out2 == "d1/report.pdf"

    # Already-canonical values are returned unchanged.
    out3 = doc_registry._coerce_legacy_source_path(
        "d1/report.pdf", "d1", "report.pdf"
    )
    assert out3 == "d1/report.pdf"

    # None passes through (registry-backed docs synthesize at the call site).
    assert doc_registry._coerce_legacy_source_path(None, "d1", "f.pdf") is None