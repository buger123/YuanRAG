"""Regression tests for v2.0.28.18 — fix Document-namespace corruption that
hides threads from the sidebar.

User report (2026-09-26 11:39, post-v2.0.28.17 ship):

    ERROR | checkpointer._deserialize_state failed thread_id='1bca6dd2-...':
      ValueError: Deserialization of ('langchain', 'schema', 'document',
      'Document') is not allowed. The default (allowed_objects='core')
      only permits core langchain-core classes.
    ERROR | checkpointer.list_threads skipping corrupt thread_id='1bca6dd2-...'
    WARNING | list_sessions: skipped 1 corrupt thread(s) — see checkpointer
      logs for trace_ids

**Root cause**: ``_deserialize_state`` called
``loads(..., allowed_objects="messages")`` per v2.0.28.11. The
``"messages"`` scope is a STRICT subset of ``"core"`` — it allows
AIMessage / HumanMessage / ToolMessage / SystemMessage but NOT Document.
``dumpd()`` still emits the legacy
``["langchain","schema","document","Document"]`` ID for Document objects,
and that ID is not in the messages allowlist → ``loads()`` raises
``ValueError`` → thread is skipped in ``list_threads`` → invisible in
the sidebar.

**Fix** (1 LOC, the load-bearing change):

    loads(revived_json, allowed_objects="core")    # was "messages"

Verified in REPL that ``allowed_objects="core"`` accepts BOTH the legacy
``["langchain","schema","document","Document"]`` (what ``dumpd()`` emits
today for Document) AND the modern
``["langchain_core","documents","base","Document"]`` IDs, plus all the
message classes. ``"core"`` is the narrowest scope that covers every
langchain_core class we actually use (messages + documents) without
enabling partner integrations.

**Plus defense-in-depth** (also part of this fix):
- Write-time: ``_serialize_state`` mirrors the ``sources`` pre-convert
  block — Document instances become plain dicts via ``model_dump()``
  before ``dumpd()`` sees them. Eliminates the legacy envelope in NEW
  writes.
- Read-time: ``_migrate_legacy_namespace`` walker rewrites legacy
  ``["langchain","schema",X,Y]`` envelope IDs to modern
  ``["langchain_core",...]`` form. Handles hypothetical future LangChain
  versions where ``"core"`` might get stricter.

This file pins the contract end-to-end:
1. Write-time Document pre-conversion works.
2. Read-time legacy-namespace rewrite works (and is no-op on modern IDs).
3. The real corrupt thread from production DB loads successfully.
4. All 101 threads in ``data/history.db`` load without raising.
5. The pre-existing Pydantic Source revival path still works (no shadow).
"""

from __future__ import annotations

import json
import sqlite3
import shutil
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.storage import checkpointer as cp


# Reference to the real corrupt thread in the user's data dir. Copy it
# into a tmp DB for each test so we never mutate the user's data.
REAL_CORRUPT_THREAD_ID = "1bca6dd2-8768-4c36-a9b2-bb9d302d3b69"


# ---------------------------------------------------------------------------
# 1. Write-time fix — Document in state → plain dict in serialized JSON
# ---------------------------------------------------------------------------


class TestSerializeStateDocumentHandling:
    """v2.0.28.18 P1-P5 — _serialize_state pre-converts Document to plain
    dict, mirroring the existing sources pre-convert. Without this fix,
    ``dumpd()`` would emit the legacy ``["langchain","schema","document",
    "Document"]`` envelope and the round-trip would fail at load time."""

    def test_document_in_state_written_as_plain_dict(self):
        """A Document instance in state["documents"] is serialized as a
        plain dict with ``id/type/page_content/metadata`` keys, NOT as
        an ``{lc, type, id, kwargs}`` envelope."""
        state_in = {
            "messages": [],
            "documents": [Document(page_content="hello world", metadata={"src": "test"})],
            "thread_id": "t1",
        }
        out_json = cp._serialize_state(state_in)
        out = json.loads(out_json)
        docs = out.get("documents", [])
        assert len(docs) == 1, f"expected 1 doc, got {len(docs)}"
        d = docs[0]
        # Pre-fix: d would have {"lc": 1, "type": "constructor", "id": [...], "kwargs": {...}}
        # Post-fix: d is a plain dict from model_dump()
        assert "lc" not in d, (
            f"Document was not pre-converted to plain dict: {d!r}"
        )
        assert d.get("page_content") == "hello world"
        assert d.get("metadata", {}).get("src") == "test"

    def test_document_already_dict_unchanged(self):
        """If the state already carries dict-shaped documents (legacy
        plain-dict rows from before v2.0.28.18), they're passed through
        unchanged — no double-wrapping, no rewriting."""
        state_in = {
            "messages": [],
            "documents": [{"page_content": "legacy dict", "metadata": {}}],
            "thread_id": "t2",
        }
        out_json = cp._serialize_state(state_in)
        out = json.loads(out_json)
        d = out["documents"][0]
        assert "lc" not in d
        assert d.get("page_content") == "legacy dict"

    def test_multiple_documents_in_list_all_converted(self):
        """A list of mixed Document instances + dicts all survive the
        pre-convert without losing data."""
        docs_in = [
            Document(page_content="d1", metadata={"k": "v1"}),
            {"page_content": "d2", "metadata": {"k": "v2"}},  # already dict
            Document(page_content="d3", metadata={"k": "v3"}),
        ]
        state_in = {"messages": [], "documents": docs_in, "thread_id": "t3"}
        out_json = cp._serialize_state(state_in)
        out = json.loads(out_json)
        docs = out["documents"]
        assert len(docs) == 3
        for i, d in enumerate(docs):
            assert "lc" not in d, f"docs[{i}] has lc envelope: {d!r}"
            assert d.get("page_content") == f"d{i+1}"


# ---------------------------------------------------------------------------
# 2. Read-time fix — _migrate_legacy_namespace rewrites envelope IDs
# ---------------------------------------------------------------------------


class TestDeserializeStateLegacyNamespace:
    """v2.0.28.18 P1-P5 — the new ``_migrate_legacy_namespace`` walker
    rewrites legacy ``["langchain","schema",X,Y]`` envelope IDs to
    modern ``["langchain_core",...]`` form. This is defense-in-depth on
    top of the load-bearing ``allowed_objects="core"`` change in
    ``_deserialize_state``."""

    def test_legacy_document_id_is_rewritten(self):
        """Hand-crafted legacy Document envelope → id becomes
        ``['langchain_core','documents','base','Document']``."""
        raw = {
            "lc": 1, "type": "constructor",
            "id": ["langchain", "schema", "document", "Document"],
            "kwargs": {"page_content": "x", "metadata": {}},
        }
        result = cp._migrate_legacy_namespace(raw)
        assert tuple(result["id"]) == ("langchain_core", "documents", "base", "Document")

    def test_legacy_aimessage_id_is_rewritten(self):
        """Legacy AIMessage envelope → modern
        ``['langchain_core','messages','ai','AIMessage']``."""
        raw = {
            "lc": 1, "type": "constructor",
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"content": "hi"},
        }
        result = cp._migrate_legacy_namespace(raw)
        assert tuple(result["id"]) == ("langchain_core", "messages", "ai", "AIMessage")

    def test_legacy_humanmessage_id_is_rewritten(self):
        raw = {
            "lc": 1, "type": "constructor",
            "id": ["langchain", "schema", "messages", "HumanMessage"],
            "kwargs": {"content": "hi"},
        }
        result = cp._migrate_legacy_namespace(raw)
        assert tuple(result["id"]) == ("langchain_core", "messages", "human", "HumanMessage")

    def test_legacy_toolmessage_id_is_rewritten(self):
        raw = {
            "lc": 1, "type": "constructor",
            "id": ["langchain", "schema", "messages", "ToolMessage"],
            "kwargs": {"content": "out", "name": "x", "tool_call_id": "t1"},
        }
        result = cp._migrate_legacy_namespace(raw)
        assert tuple(result["id"]) == ("langchain_core", "messages", "tool", "ToolMessage")

    def test_unknown_legacy_id_is_left_alone(self):
        """A legacy-form id that isn't in the remap dict is preserved
        unchanged — ``loads()`` will raise on it and the corrupt-count
        counter still surfaces it. No silent failure."""
        raw = {
            "lc": 1, "type": "constructor",
            "id": ["langchain", "schema", "foo", "Bar"],
            "kwargs": {},
        }
        result = cp._migrate_legacy_namespace(raw)
        assert tuple(result["id"]) == ("langchain", "schema", "foo", "Bar")

    def test_nested_envelope_is_migrated(self):
        """Walker recurses into dict values AND list items. Envelope
        nested inside ``messages[0].kwargs.some_dict.nested_doc`` is
        still migrated."""
        raw = {
            "messages": [{
                "lc": 1, "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "x",
                    "some_dict": {
                        "nested_doc": {
                            "lc": 1, "type": "constructor",
                            "id": ["langchain", "schema", "document", "Document"],
                            "kwargs": {"page_content": "deep", "metadata": {}},
                        }
                    },
                    "doc_list": [{
                        "lc": 1, "type": "constructor",
                        "id": ["langchain", "schema", "document", "Document"],
                        "kwargs": {"page_content": "deep2", "metadata": {}},
                    }],
                },
            }]
        }
        result = cp._migrate_legacy_namespace(raw)
        # Top-level AIMessage migrated
        assert tuple(result["messages"][0]["id"]) == (
            "langchain_core", "messages", "ai", "AIMessage"
        )
        # Nested Document in dict migrated
        nested = result["messages"][0]["kwargs"]["some_dict"]["nested_doc"]
        assert tuple(nested["id"]) == (
            "langchain_core", "documents", "base", "Document"
        )
        # Nested Document in list migrated
        nested_list = result["messages"][0]["kwargs"]["doc_list"][0]
        assert tuple(nested_list["id"]) == (
            "langchain_core", "documents", "base", "Document"
        )


# ---------------------------------------------------------------------------
# 3. End-to-end — the real corrupt thread from data/history.db loads
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_history_db(tmp_path):
    """Copy the user's real history.db into a tmp file so tests never
    mutate production data. The tmp file is wiped automatically by
    pytest's tmp_path fixture."""
    real_db = Path("data/history.db")
    if not real_db.exists():
        pytest.skip(f"real history db missing at {real_db}")
    tmp_db = tmp_path / "history.db"
    shutil.copy(real_db, tmp_db)
    return tmp_db


class TestLoadLatestCorruptThread:
    """v2.0.28.18 P1-P5 — the highest-value test: load the EXACT corrupt
    thread the user reported (1bca6dd2-8768-4c36-a9b2-bb9d302d3b69) and
    assert it deserializes successfully with Document instances."""

    def test_real_corrupt_thread_loads(self, tmp_history_db, monkeypatch):
        # Redirect the singleton to the tmp DB path.
        import src.core.paths as paths_mod

        monkeypatch.setattr(paths_mod, "history_db_path", lambda: str(tmp_history_db))
        # Reset the connection singleton so startup() re-opens the new path.
        cp._db_conn = None
        cp._db_path = None

        # Run startup to open the tmp DB.
        import asyncio

        asyncio.get_event_loop().run_until_complete(cp.startup())

        # Now load the corrupt thread.
        state = asyncio.get_event_loop().run_until_complete(
            cp.load_latest(REAL_CORRUPT_THREAD_ID)
        )

        assert state is not None, (
            f"corrupt thread {REAL_CORRUPT_THREAD_ID} returned None "
            "(pre-fix: CorruptCheckpointError raised → load_latest returned "
            "None; post-fix: should return the full AgentState dict)"
        )
        assert isinstance(state, dict), f"load_latest returned non-dict: {type(state)}"
        # The corrupt thread had 1 message and 1 document.
        assert len(state.get("messages", [])) >= 1, (
            f"corrupt thread lost its messages: state keys = {list(state.keys())}"
        )
        assert len(state.get("documents", [])) == 1, (
            f"corrupt thread documents not restored: docs = {state.get('documents')!r}"
        )
        # The Document must be a real Document instance, not a dict
        # (because this thread was written BEFORE the v2.0.28.18
        # write-time pre-convert).
        d = state["documents"][0]
        assert isinstance(d, Document), (
            f"corrupt thread's Document was not revived as Document "
            f"instance: got {type(d).__name__}"
        )
        assert d.page_content, "Document page_content is empty"
        assert d.metadata, "Document metadata is empty"


class TestListThreadsNoCorruptionAfterFix:
    """v2.0.28.18 P1-P5 — ``list_threads()`` returns the corrupt thread
    without incrementing the corrupt_count. Pre-fix this returned
    ``(valid_count=100, corrupt_count=1)``; post-fix it must return
    ``(valid_count=101, corrupt_count=0)``."""

    def test_all_threads_listed_no_corruption(self, tmp_history_db, monkeypatch):
        import src.core.paths as paths_mod

        monkeypatch.setattr(paths_mod, "history_db_path", lambda: str(tmp_history_db))
        cp._db_conn = None
        cp._db_path = None

        import asyncio

        asyncio.get_event_loop().run_until_complete(cp.startup())

        valid, corrupt_count = asyncio.get_event_loop().run_until_complete(
            cp.list_threads(limit=500)
        )

        # The tmp DB has the same row count as production: 101 rows.
        real_count = sqlite3.connect("data/history.db").execute(
            "SELECT COUNT(*) FROM states"
        ).fetchone()[0]
        assert len(valid) == real_count, (
            f"expected {real_count} valid threads, got {len(valid)} — "
            f"corrupt_count = {corrupt_count}"
        )
        assert corrupt_count == 0, (
            f"corrupt_count must be 0 post-fix, got {corrupt_count}"
        )
        # And the corrupt thread specifically MUST be in the valid list.
        valid_ids = {t["thread_id"] for t in valid}
        assert REAL_CORRUPT_THREAD_ID in valid_ids, (
            f"corrupt thread {REAL_CORRUPT_THREAD_ID} not in list_threads "
            f"output — the user's reported symptom is still present"
        )


# ---------------------------------------------------------------------------
# 4. No regression — the existing Pydantic Source revival path still works
# ---------------------------------------------------------------------------


class TestNoRegressionOfPydanticSourceRevival:
    """v2.0.28.18 P1-P5 — the new ``_migrate_legacy_namespace`` walker
    is disjoint from ``_revive_not_implemented``. Pydantic Source
    envelopes (``{"lc":1,"type":"not_implemented","repr":"Source(...)"}``)
    must still be revived into plain dicts via the existing walker."""

    def test_pydantic_source_envelope_still_revived(self):
        """A Pydantic Source envelope goes through ``_revive_not_implemented``
        (the OLD walker) — not ``_migrate_legacy_namespace``. After both
        walkers run, the source kwargs are accessible as a plain dict."""
        from src.api.schemas import Source

        src_obj = Source(
            index=0,
            url="https://example.com",
            text="a snippet",
            filename="example.html",
            source_kind="web",
        )
        state_in = {
            "messages": [],
            "documents": [],
            "sources": [src_obj],
            "thread_id": "t_src",
        }
        out_json = cp._serialize_state(state_in)
        # Re-parse and feed through full _deserialize_state.
        state_back = cp._deserialize_state(out_json, "t_src")
        sources = state_back.get("sources", [])
        assert len(sources) == 1
        s = sources[0]
        # The pre-convert at write time means sources are already plain
        # dicts here (per v2.0.23); _revive_not_implemented's no-op for
        # plain dicts is the expected behavior. We pin that the round
        # trip preserves all fields.
        if isinstance(s, dict):
            assert s.get("url") == "https://example.com"
            assert s.get("text") == "a snippet"
            assert s.get("source_kind") == "web"
        else:
            assert s.url == "https://example.com"
            assert s.text == "a snippet"
            assert s.source_kind == "web"