"""Phase 6 — per-conversation thread isolation.

Covers:

* ``_safe_identifier`` rejects injection-shaped thread_ids / doc_ids
  (quote / semicolon / whitespace / etc.) and passes production-style
  UUIDs through — same validator used by ``lancedb_store``,
  ``hybrid_search``, and the document routes.
* ``ChunkRecord`` defaults ``thread_id=""`` so legacy fixtures keep
  working.
* ``hybrid_search`` applies ``.where(thread_id = '<safe>')`` to both
  dense and BM25 paths when ``thread_id`` is provided.
* ``hybrid_search`` falls back to an unfiltered search if the caller
  passes an unsafe thread_id (logs and continues — better than a hard
  crash).
* ``list_documents(thread_id)`` filters per-thread.
* ``delete_by_thread_id`` removes every chunk belonging to a thread.
* ``add_columns`` migration smoke: a legacy table without ``thread_id``
  gets the column added on first open.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from src.storage import lancedb_store
from src.storage.lancedb_store import (
    _safe_identifier,
    add_chunks,
    delete_by_thread_id,
    get_table,
    list_documents,
)
from src.storage.schema import LANCEDB_SCHEMA, TABLE_NAME, ChunkRecord


# ============================================================
# _safe_identifier validation
# ============================================================


def test_safe_identifier_passes_uuid_and_short_fixture():
    """Production UUIDs and short test fixtures pass through unchanged."""
    assert _safe_identifier("abc12345-1234-5678-9abc-def012345678", "x") == \
        "abc12345-1234-5678-9abc-def012345678"
    assert _safe_identifier("a" * 32, "x") == "a" * 32
    assert _safe_identifier("d1", "x") == "d1"
    assert _safe_identifier("thread-X", "x") == "thread-X"


def test_safe_identifier_rejects_dangerous_chars():
    """Anything that could break out of a single-quoted SQL literal is
    rejected — quotes, semicolons, whitespace, control chars, and the
    shell-meta set ``|&$```."""
    bad_inputs = [
        "a'; DROP TABLE chunks; --",
        "has spaces",
        "",
        "with\ttab",
        "with\nnewline",
        "with\x00null",
        "with'quote",
        'with"dquote',
        "with;semi",
        "with(paren)",
        "with|pipe",
        "with&amp",
        "with$dollar",
        "with`back",
    ]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            _safe_identifier(bad, "thread_id")


# ============================================================
# ChunkRecord default thread_id
# ============================================================


def test_chunkrecord_default_thread_id_is_empty():
    """Default ``thread_id=""`` keeps pre-extension fixtures working."""
    rec = ChunkRecord(
        chunk_id="c1",
        doc_id="d1",
        text="hello",
        vector=[0.0] * 1024,
        filename="f.txt",
        source_type="file",
        # v2.0.27 P0-P2 — canonical privacy-safe form.
        source_path="d1/f.txt",
    )
    assert rec.thread_id == ""
    # And the row carries the column too — important for legacy data
    # backfilled with an empty thread_id.
    assert rec.to_lancedb_row()["thread_id"] == ""


# ============================================================
# list_documents / delete_by_thread_id / hybrid_search filter
# ============================================================


def _make_records(doc_id: str, thread_id: str, n: int = 2) -> list[ChunkRecord]:
    """Build n random-vector ChunkRecords tagged with the given thread."""
    out = []
    for i in range(n):
        rec = ChunkRecord(
            chunk_id=f"{doc_id}-c{i}",
            doc_id=doc_id,
            text=f"text for {doc_id} #{i}",
            vector=list(
                np.random.RandomState(hash((doc_id, i)) & 0xFFFFFFFF).rand(1024).astype("float32")
            ),
            filename=f"{doc_id}.txt",
            source_type="file",
            # v2.0.27 P0-P2 — canonical privacy-safe form.
            source_path=f"{doc_id}/{doc_id}.txt",
            chunk_index=i,
            thread_id=thread_id,
        )
        out.append(rec)
    return out


def test_list_documents_filters_by_thread_id(tmp_path, monkeypatch):
    """Two documents, different threads — list_documents must return only
    the one matching the requested thread."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    from config.settings import reset_settings_cache

    reset_settings_cache()
    lancedb_store.reset_for_tests()

    # Doc A on thread "alpha", doc B on thread "beta"
    add_chunks(_make_records("doc-a", "alpha"))
    add_chunks(_make_records("doc-b", "beta"))

    # No filter — admin view sees both
    all_docs = list_documents()
    assert {d["doc_id"] for d in all_docs} == {"doc-a", "doc-b"}

    # Filter to alpha
    alpha_docs = list_documents(thread_id="alpha")
    assert len(alpha_docs) == 1
    assert alpha_docs[0]["doc_id"] == "doc-a"
    assert alpha_docs[0]["thread_id"] == "alpha"

    # Filter to beta
    beta_docs = list_documents(thread_id="beta")
    assert len(beta_docs) == 1
    assert beta_docs[0]["doc_id"] == "doc-b"

    # Filter to a thread with no documents — empty list
    none_docs = list_documents(thread_id="gamma")
    assert none_docs == []

    lancedb_store.reset_for_tests()


def test_delete_by_thread_id_removes_only_that_thread(tmp_path, monkeypatch):
    """``delete_by_thread_id`` must delete ONLY the requested thread's
    chunks; the other thread's chunks survive."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    from config.settings import reset_settings_cache

    reset_settings_cache()
    lancedb_store.reset_for_tests()

    add_chunks(_make_records("doc-a", "alpha"))
    add_chunks(_make_records("doc-b", "beta"))

    delete_by_thread_id("alpha")

    # Only beta survives
    remaining = list_documents()
    assert {d["doc_id"] for d in remaining} == {"doc-b"}

    lancedb_store.reset_for_tests()


def test_hybrid_search_thread_id_filter_shape(monkeypatch):
    """Verify the filter expression shape passed into ``.where()``.

    We monkeypatch the underlying LanceDB ``table.search`` chain to
    capture the filter string. Both dense and BM25 paths must apply
    the same per-thread filter.

    Note: ``src.retrieval.hybrid_search`` re-exports the function with
    the same name, so we import the module via ``importlib`` to avoid
    the function-shadowing-submodule ambiguity.
    """
    import importlib

    captured: list[str] = []

    class _FakeQuery:
        def __init__(self, captured_list):
            self._captured = captured_list

        def metric(self, _name):
            return self

        def where(self, expr):
            self._captured.append(expr)
            return self

        def limit(self, _n):
            return self

        def to_list(self):
            return []

    class _FakeTable:
        def search(self, _q, **_kw):
            return _FakeQuery(captured)

    hs = importlib.import_module("src.retrieval.hybrid_search")
    monkeypatch_table = _FakeTable()
    monkeypatch.setattr(hs, "get_table", lambda: monkeypatch_table)

    # Unfiltered: no .where() should be called
    captured.clear()
    hs.hybrid_search("hello", [0.0] * 1024, top_k=5)
    assert captured == []

    # Filtered: each path issues ONE .where() with the expected shape.
    # v2.0.29.4 (Phase 4 PR-2) — ``_compose_where`` now wraps the
    # thread filter in parens so it composes cleanly with the new
    # temporal filter via ``AND`` (avoids precedence pitfalls when
    # both filters are active).
    captured.clear()
    hs.hybrid_search("hello", [0.0] * 1024, top_k=5, thread_id="abc-123")
    assert len(captured) == 2  # dense + BM25
    for expr in captured:
        assert expr == "(thread_id = 'abc-123')", (
            f"expected parenthesised thread_id filter on both paths, got {expr!r}"
        )


def test_hybrid_search_unsafe_thread_id_falls_back_to_unfiltered(monkeypatch, caplog):
    """An unsafe thread_id must NOT crash — it logs a warning and falls
    back to the unfiltered path (better than a hard crash on hostile
    input)."""
    import importlib
    import logging

    captured: list[str] = []

    class _FakeQuery:
        def metric(self, _):
            return self

        def where(self, expr):
            captured.append(expr)
            return self

        def limit(self, _):
            return self

        def to_list(self):
            return []

    class _FakeTable:
        def search(self, *_a, **_k):
            return _FakeQuery()

    hs = importlib.import_module("src.retrieval.hybrid_search")
    monkeypatch.setattr(hs, "get_table", lambda: _FakeTable())

    # The hybrid_search module logger
    caplog.set_level(logging.WARNING, logger="src.retrieval.hybrid_search")

    hs.hybrid_search("x", [0.0] * 1024, thread_id="a'; DROP TABLE chunks; --")
    # No filter applied (because validation failed)
    assert captured == []


# ============================================================
# add_columns migration smoke test
# ============================================================


def test_get_table_adds_thread_id_column_to_legacy_table(tmp_path, monkeypatch):
    """If a pre-existing table lacks the ``thread_id`` column,
    ``get_table`` adds it via ``add_columns`` + backfill."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    from config.settings import reset_settings_cache

    reset_settings_cache()
    lancedb_store.reset_for_tests()

    # Manually create a legacy table WITHOUT the thread_id column.
    # v2.0.29.4 (Phase 4 PR-2) — also strip the new ``expires_at``
    # column so the test exercises a "pre-PR-2" shape. The migration
    # path should add BOTH missing columns when ``get_table()`` opens
    # the table. ``add_columns`` is idempotent — calling it twice for
    # the same column name is a no-op in LanceDB.
    db = lancedb_store.get_db()
    legacy_schema = pa.schema(
        [
            f for f in LANCEDB_SCHEMA
            if f.name not in ("thread_id", "expires_at")
        ]
    )
    legacy_table = db.create_table(
        TABLE_NAME,
        pa.table(
            {
                "chunk_id": ["c1"],
                "doc_id": ["d1"],
                "text": ["hello"],
                "vector": [[0.0] * 1024],
                "sparse": ["{}"],
                "filename": ["f.txt"],
                "source_type": ["file"],
                # v2.0.27 P0-P2 — canonical privacy-safe form.
                "source_path": ["d1/f.txt"],
                "mime_type": [""],
                "chunk_index": pa.array([0], type=pa.int32()),
                "chunk_type": [""],
                "page_number": pa.array([0], type=pa.int32()),
                "section": [""],
                "sheet": [""],
                "headers": pa.array([[]], type=pa.list_(pa.string())),
                "row_start": pa.array([0], type=pa.int32()),
                "row_end": pa.array([0], type=pa.int32()),
                "ingested_at": [""],
            },
            schema=legacy_schema,
        ),
        mode="overwrite",
    )

    # Sanity: legacy table is missing the column
    assert "thread_id" not in legacy_table.schema.names

    # Now opening via get_table must run the migration
    table = get_table()
    assert "thread_id" in table.schema.names, (
        "get_table should add thread_id to a legacy table"
    )

    # And we can immediately write a record with a thread_id — proves
    # the column is not just declared but also queryable.
    add_chunks(
        [
            ChunkRecord(
                chunk_id="c-new",
                doc_id="d-new",
                text="new text",
                vector=[0.1] * 1024,
                filename="g.txt",
                source_type="file",
                # v2.0.27 P0-P2 — canonical privacy-safe form.
                source_path="d-new/g.txt",
                thread_id="t-new",
            )
        ]
    )
    new_docs = list_documents(thread_id="t-new")
    assert {d["doc_id"] for d in new_docs} == {"d-new"}

    lancedb_store.reset_for_tests()