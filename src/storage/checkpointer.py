"""Phase 3 (v2.0.21) — own SQLite state store.

Why we own this
---------------
Pre-Phase-3 the chat history was persisted via LangGraph's
``AsyncSqliteSaver`` (``src/storage/history.py`` — deleted in
Step 7). The LangGraph schema was overkill for our needs:

* three columns just for the checkpoint identifier
  (``thread_id`` + ``checkpoint_ns`` + ``checkpoint_id``)
* the ``checkpoint`` blob was a msgpack-encoded ``Checkpoint`` object
  (deserializable only via LangGraph's own loaders, with a
  ``LANGGRAPH_STRICT_MSGPACK`` allowlist controlling which module
  types can be re-hydrated — see the v2.0.11 source warning about
  ``src.api.schemas.Source`` being "unregistered")
* the ``alist(None)`` enumeration had to walk the whole table, dedup
  to one-per-thread in Python, and re-read the checkpoint blob for
  each — ~5 s end-to-end on a populated corpus

The new schema is one row per thread (latest snapshot only), with the
full state JSON-serialized into ``state_json``. Reads are O(1) per
thread, listings are O(N) with an index, and the JSON blob can be
edited / inspected by ``sqlite3`` CLI.

Schema
------
``states`` table:

    CREATE TABLE states (
        thread_id   TEXT PRIMARY KEY,
        step        INTEGER NOT NULL,
        state_json  TEXT NOT NULL,    -- json.dumps(dumpd(state))
        created_at  TEXT NOT NULL     -- ISO 8601, indexed DESC
    );
    CREATE INDEX idx_states_created_at ON states(created_at DESC);

``step`` is the FSM's step_count at save time. We store it separately
(not just in state_json) so /sessions listings can show "5 thinking
steps" without deserializing the whole state.

API
---
* :func:`startup` — create the ``states`` table (idempotent)
* :func:`shutdown` — close the async connection
* :func:`save` — upsert (latest snapshot replaces previous)
* :func:`load_latest` — read the latest snapshot for one thread
* :func:`list_threads` — newest-first listing for /sessions sidebar
* :func:`delete_thread` — remove a thread's row (idempotent)
* :func:`reset_for_tests` — sync reset of the singleton

Migration history (Steps 6 → 7)
-------------------------------
Step 6 ran a one-shot migration from LangGraph's ``checkpoints``
table to populate ``states``. Step 7 dropped the migration helper
and the legacy ``history.py`` module — once migration completes on
a given DB, the helper has no further work to do. Subsequent boots
just verify the ``states`` schema and open the connection.

v2.0.28.18 P1-P5 (Document-namespace corruption) — symmetric fix to
the existing ``Source`` Pydantic pre-convert block in
:func:`_serialize_state`. Pre-fix, ``langchain_core.load.dumpd``
serializes a ``langchain_core.documents.Document`` with the legacy
id ``["langchain","schema","document","Document"]``; the companion
:func:`_deserialize_state` call to ``loads(..., allowed_objects="messages")``
rejects Document (not in the messages allowlist) → the thread is
skipped in :func:`list_threads` and invisible in the sidebar. The
fix has two halves:

* **write-time** (``_serialize_state``): mirror the ``sources``
  pre-convert block — convert ``Document`` instances to plain
  dicts via ``model_dump()`` before ``dumpd`` so NEW rows never
  write the legacy envelope.
* **read-time** (``_deserialize_state``): add a
  :func:`_migrate_legacy_namespace` walker (uses the
  :data:`_LEGACY_NAMESPACE_REMAP` dict) that rewrites legacy
  ``["langchain","schema",X,Y]`` envelope IDs to modern
  ``["langchain_core",...]`` form. Repairs EXISTING rows
  transparently — no one-shot data migration script needed.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from langchain_core.load import dumpd, loads

from src.agent.state import AgentState
from src.core.logging import logger
from src.core.paths import history_db_path


class CorruptCheckpointError(Exception):
    """Raised when a stored checkpoint cannot be deserialized.

    v2.0.27.2 P1-P5 — pre-PR-3, ``_deserialize_state`` raised bare
    ``json.JSONDecodeError`` / ``NotImplementedError`` / etc. with no
    context. The caller (``load_latest``) bubbled those up as 500s
    to ``GET /sessions/{tid}/messages``, and ``list_threads._to_summary``
    silently swallowed them with ``state = {"messages": []}`` — leaving
    the sidebar looking "empty" with no signal to the operator that
    the underlying row is corrupt.

    Post-PR-3 every corruption path raises this single typed exception
    carrying:

    * ``thread_id`` — which row is broken (the only thing the operator
      can act on: delete it, restore from backup, or read with
      ``sqlite3`` ``.dump`` to recover the JSON bytes).
    * ``original_error`` — the underlying cause (``json.JSONDecodeError``
      / ``ValueError`` / ``TypeError`` / ``NotImplementedError`` /
      ``KeyError``). For debugging — never displayed in the wire
      response (no info leak).
    * ``trace_id`` — a 12-hex-char correlation id stamped on the
      log line. Lets operators grep for "thread X broke at trace Y"
      across multiple log entries in the same request.

    Narrow ``except`` in ``_deserialize_state`` only catches the
    deserialization failure modes — *not* ``Exception`` — so
    programming bugs (AttributeError, NameError) still surface
    loudly instead of being misclassified as data corruption.
    """

    def __init__(
        self,
        thread_id: str,
        original_error: BaseException,
        trace_id: str,
    ) -> None:
        self.thread_id = thread_id
        self.original_error = original_error
        self.trace_id = trace_id
        super().__init__(
            f"corrupt checkpoint thread_id={thread_id!r} "
            f"trace_id={trace_id}: "
            f"{type(original_error).__name__}: {original_error}"
        )


# Connection singleton — path-aware so a test switching
# ``RAG_DATA_DIR`` mid-suite doesn't accidentally share state with
# the previous test's DB. Mirrors the pre-Phase-3
# ``history.get_checkpointer`` shape; same path-resolution rules.
_db_lock = asyncio.Lock()
# v2.0.27.3 P0-P1 — sync ``threading.Lock`` to serialize the
# sync ``_save_sync`` / ``_delete_thread_sync`` callers across
# threadpool threads. Without this, concurrent ``asyncio.to_thread``
# calls hitting the same sqlite3 connection from different threads
# can trigger Windows access violations (Python's sqlite3 module
# is not safe for concurrent statements on one connection even with
# ``check_same_thread=False`` — the per-thread statement state is
# stored on the connection object). ``_db_lock`` (asyncio.Lock)
# can't help here because ``_save_sync`` is sync code. We hold
# the lock for the duration of BEGIN IMMEDIATE / INSERT / COMMIT
# — this is also the window where BEGIN IMMEDIATE's RESERVED lock
# is held, so the lock is a Python-level complement to the
# SQLite-level serialization.
_db_write_lock = threading.Lock()
_db_conn: Optional[Any] = None  # sqlite3.Connection (NOT aiosqlite — see _open_db)
_db_path: Optional[str] = None


# Hard cap on /sessions listing rows. Matches the pre-Phase-3 cap
# (``_MAX_SESSIONS = 500`` in ``src/api/routes/sessions.py``) so the
# listing endpoint stays responsive on a populated corpus.
_MAX_THREADS = 500


# ---------------------------------------------------------------------------
# JSON (de)serialization for state_json
# ---------------------------------------------------------------------------


def _revive_not_implemented(obj):
    """Revive objects that LangChain dumped as
    ``{"lc": 1, "type": "not_implemented", "id": [...], "repr": "..."}``.

    LangChain only knows how to deserialize classes it has a registered
    serializer for. Anything else gets the ``not_implemented`` marker
    plus the object's ``repr()``. On the read side, LangChain then
    raises ``NotImplementedError`` and we lose the row entirely (see
    the ``_to_summary`` try/except below — the sidebar used to show
    empty messages for every thread that contained such an object).

    The v2.0.7-vintage schema had ``sources: List[Source]`` where
    ``Source`` was a Pydantic model NOT registered with LangChain
    (``src.api.schemas.Source`` — see the audit at
    [[post-phase3-audit-findings]]). Every thread from that era wrote
    the Source Pydantic instance via ``dumpd``, got back the
    ``not_implemented`` envelope, and on next read the sidebar went
    blank. The fix is to parse the ``repr`` back into a plain dict via
    ``ast.literal_eval`` — Pydantic's default ``__repr__`` is exactly
    ``ClassName(field=value, ...)`` and safe to eval.

    This walker is intentionally generic: it handles ANY class with the
    ``not_implemented`` marker, not just Source. Future dead-class
    regressions won't blank the sidebar either — they'll degrade
    gracefully into plain dicts and the read path will succeed with
    the recoverable fields.

    Safety contract — the repr came from Python's default ``__repr__``
    on a BaseModel (the original ``Source`` instance) or any object the
    application wrote. It is **never** user input from the WS payload.
    Even so, ``ast.literal_eval`` rejects:
    - function calls (``open(...)``, ``__import__(...)``, etc.)
    - attribute access (``x.attr``)
    - comprehensions / generators
    - dunder kwarg names (``__class__``, ``__dict__``, ...) — these
      would let an attacker redirect attribute writes if the dict
      were ever passed back into a constructor

    The walker rejects the repr if any of these trip, falling back to
    ``{"_repr": ..., "_lc_id": ...}`` so the row still loads.
    """
    if isinstance(obj, dict):
        # Check for the not_implemented envelope.
        if (
            obj.get("lc") == 1
            and obj.get("type") == "not_implemented"
            and "repr" in obj
            and "id" in obj
            and isinstance(obj["repr"], str)
        ):
            try:
                import ast

                tree = ast.parse(obj["repr"], mode="eval")
                node = tree.body
                if not isinstance(node, ast.Call) or not isinstance(
                    node.func, ast.Name
                ):
                    raise ValueError("not a constructor call")
                # Reject positional args — Pydantic __repr__ never
                # emits them, but defense in depth: a repr with ``f(x)``
                # is suspicious even when ``x`` is a literal.
                if node.args:
                    raise ValueError("constructor has positional args")
                out: dict = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        # ``**spread`` — not legal in literal_eval
                        # and not part of Pydantic's repr format.
                        raise ValueError("constructor uses **kwargs spread")
                    # Dunder kwarg names are a class-hijack vector if
                    # the dict ever round-trips into a constructor
                    # (``Source(__class__=OtherClass, ...)`` would
                    # rebind ``self.__class__``). Reject them.
                    if kw.arg.startswith("__") and kw.arg.endswith("__"):
                        raise ValueError(
                            f"dunder kwarg name {kw.arg!r} rejected"
                        )
                    out[kw.arg] = ast.literal_eval(kw.value)
                return out
            except (SyntaxError, ValueError):
                # Repr wasn't parseable — fall back to the raw repr
                # string so the row still loads and the operator can
                # see what was stored.
                return {"_repr": obj["repr"], "_lc_id": obj.get("id")}
        # Recurse into dict values.
        return {k: _revive_not_implemented(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_revive_not_implemented(v) for v in obj]
    return obj


# v2.0.28.18 P1-P5 — legacy ``["langchain","schema",X,Y]`` envelope IDs
# written by the pre-Phase-3 checkpointer (and by ANY new code path that
# lets a ``Document`` reach ``dumpd()``). The new
# ``allowed_objects="messages"`` deserializer only knows the modern
# ``["langchain_core",...]`` IDs in the messages namespace; Document is
# NOT in that namespace so it raises
# ``ValueError: Deserialization of ('langchain','schema','document',
# 'Document') is not allowed``. This dict rewrites the legacy IDs to
# modern form BEFORE the tree reaches ``loads()``.
#
# The mapping is structurally IRREGULAR — a pure rewrite of
# ``"langchain" -> "langchain_core"`` + drop ``"schema"`` would produce
# wrong IDs:
#   * ``("langchain","schema","document","Document")``
#     → ``("langchain_core","document","Document")`` — wrong (singular
#     "document", missing the "documents.base.Document" sub-path).
#   * ``("langchain","schema","messages","AIMessage")``
#     → ``("langchain_core","messages","AIMessage")`` — wrong (missing
#     the "messages.ai.AIMessage" sub-path).
#
# The modern IDs come from
# ``langchain_core.load.mapping.OLD_CORE_NAMESPACES_MAPPING`` (verified
# in REPL — it is keyed by modern tuples mapping to themselves, NOT a
# reverse lookup, so cannot be consumed directly).
_LEGACY_NAMESPACE_REMAP: dict[tuple[str, str, str, str], tuple[str, ...]] = {
    ("langchain", "schema", "document", "Document"): (
        "langchain_core", "documents", "base", "Document"
    ),
    ("langchain", "schema", "messages", "AIMessage"): (
        "langchain_core", "messages", "ai", "AIMessage"
    ),
    ("langchain", "schema", "messages", "HumanMessage"): (
        "langchain_core", "messages", "human", "HumanMessage"
    ),
    ("langchain", "schema", "messages", "ToolMessage"): (
        "langchain_core", "messages", "tool", "ToolMessage"
    ),
}


def _migrate_legacy_namespace(obj):
    """Recursively rewrite legacy ``["langchain","schema",X,Y]``
    envelope IDs to modern ``["langchain_core",...]`` form.

    Disjoint from :func:`_revive_not_implemented` — this walker matches
    the ``{"lc": 1, "type": "constructor", "id": [...], "kwargs": {...}}``
    envelope (modern constructor envelopes), NOT the
    ``{"lc": 1, "type": "not_implemented", "repr": ...}`` envelope that
    ``_revive_not_implemented`` handles. Both walkers run on the same
    tree; their envelope shapes don't overlap, so order is purely a
    readability choice (we run this one first because it's the "clean
    shape" case, then the more elaborate Pydantic-revival case).

    Failure mode (deliberate, NOT silent): if an envelope has the legacy
    shape but its id tuple is NOT in :data:`_LEGACY_NAMESPACE_REMAP`,
    the walker leaves it alone and ``loads()`` raises
    ``NotImplementedError`` / ``ValueError``. The existing
    ``CorruptCheckpointError`` + corrupt-count machinery in
    :func:`list_threads` still surfaces it — no behavior change for
    unknown legacy IDs.

    Mutates ``obj`` in place for dict envelopes (rewrites ``id``);
    returns the same object for recursion continuity. This is safe
    because the caller (``_deserialize_state``) owns the freshly
    ``json.loads()``-ed tree and never shares it.
    """
    if isinstance(obj, dict):
        # Rewrite legacy envelope id IN PLACE.
        if (
            obj.get("lc") == 1
            and obj.get("type") == "constructor"
            and isinstance(obj.get("id"), list)
            and tuple(obj["id"]) in _LEGACY_NAMESPACE_REMAP
        ):
            obj["id"] = list(_LEGACY_NAMESPACE_REMAP[tuple(obj["id"])])
        # Recurse into dict values.
        for k in list(obj.keys()):
            obj[k] = _migrate_legacy_namespace(obj[k])
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _migrate_legacy_namespace(obj[i])
        return obj
    return obj


def _serialize_state(state: AgentState) -> str:
    """``AgentState`` (TypedDict with ``messages: List[BaseMessage]``,
    ``documents: List[Document]``, ``sources: List[Source]`` etc.) →
    JSON string for SQLite storage.

    Uses :func:`langchain_core.load.dumpd` so BaseMessage / Document
    subclasses round-trip with their ``lc_serializable`` type
    metadata — the companion :func:`_deserialize_state` calls
    :func:`langchain_core.load.loads` to rebuild them. Without this,
    ``json.dumps(state)`` would raise on the non-JSON-native
    BaseMessage objects.

    Pydantic models (``src.api.schemas.Source``) are NOT registered
    with LangChain — if a Source Pydantic instance reaches ``dumpd``
    it gets dumped as ``{"lc": 1, "type": "not_implemented", ...}``
    and the read side raises ``NotImplementedError`` (see the v2.0.7
    audit). We pre-convert any Pydantic ``Source`` to a plain dict
    via ``model_dump()`` here, so new threads write clean JSON and
    the read-side revive path becomes defense-in-depth (only used
    for legacy rows that pre-date this fix).

    v2.0.28.18 P1-P5 — symmetric fix for ``Document``. Pre-fix,
    ``dumpd`` serializes a ``langchain_core.documents.Document`` with
    legacy id ``["langchain","schema","document","Document"]``; the
    companion ``loads(..., allowed_objects="messages")`` rejects
    Document (not in the messages allowlist) and the row is skipped
    in :func:`list_threads`. Mirror the ``sources`` pre-convert:
    any ``Document`` instance is converted to a plain dict via
    ``model_dump()`` before ``dumpd`` sees it, eliminating the
    legacy envelope in NEW writes.
    """
    serializable = dict(state)
    sources = serializable.get("sources")
    if sources:
        from src.api.schemas import Source as _SourceSchema

        cleaned: list = []
        for s in sources:
            if isinstance(s, _SourceSchema):
                cleaned.append(s.model_dump())
            else:
                cleaned.append(s)
        serializable["sources"] = cleaned
    # v2.0.28.18 P1-P5 — symmetric to sources block above. dumpd()
    # emits legacy id ["langchain","schema","document","Document"] for
    # Document instances; pre-convert to plain dict so future rows are
    # clean and loads(..., allowed_objects="messages") succeeds.
    documents = serializable.get("documents")
    if documents:
        from langchain_core.documents import Document as _LCDocument

        cleaned: list = []
        for d in documents:
            if isinstance(d, _LCDocument):
                cleaned.append(d.model_dump())
            else:
                cleaned.append(d)
        serializable["documents"] = cleaned
    return json.dumps(dumpd(serializable), ensure_ascii=False)


def _deserialize_state(state_json: str, thread_id: str = "<unknown>") -> AgentState:
    """Inverse of :func:`_serialize_state`.

    ``loads()`` raises ``NotImplementedError`` mid-walk as soon as it
    hits a ``not_implemented`` envelope — it does NOT return a
    partial result. So we parse the JSON ourselves, walk the tree
    with :func:`_revive_not_implemented` to convert every envelope
    into a plain dict, and only THEN hand the cleaned tree to
    ``loads()``. This revives legacy threads written before
    Pydantic ``Source`` was excluded from the write path (the v2.0.7
    era) — without it, ``list_threads`` would silently blank every
    thread that had a web source in its saved state.

    If even the cleaned tree still trips ``loads()`` (some other
    unknown class), fall back to a minimal state containing only
    the revived envelope-derived dicts — better than losing the
    whole row.

    v2.0.27.2 P1-P5 — narrow ``except`` wraps the entire body to
    raise :class:`CorruptCheckpointError` with a stable
    ``trace_id`` on every corruption path (JSON decode error,
    revive walker error, envelope deserialize failure). The
    ``thread_id`` parameter is required so the error context is
    actionable — without it, the operator sees "JSON decode error"
    with no row to look at.

    v2.0.28.18 P1-P5 — pre-fix, the prior call order
    (``json.loads`` → ``_revive_not_implemented`` → ``loads``) tripped
    on the legacy Document envelope ``["langchain","schema",
    "document","Document"]`` because ``allowed_objects="messages"``
    does not include Document. Insert :func:`_migrate_legacy_namespace`
    between ``json.loads`` and ``_revive_not_implemented`` — the
    rewrite walker converts legacy IDs to modern
    ``["langchain_core",...]`` form so ``loads()`` accepts them.
    Disjoint from :func:`_revive_not_implemented` (different envelope
    shape), order doesn't matter for correctness, but the migration
    walker runs first because it's the more common shape and the
    cheaper operation. Existing corrupt rows in ``data/history.db``
    (e.g. ``1bca6dd2-...``) are repaired transparently; no one-shot
    data migration script needed.
    """
    try:
        raw_tree = json.loads(state_json)
        # v2.0.28.18 P1-P5 — rewrite legacy ["langchain","schema",X,Y]
        # envelope IDs to modern ["langchain_core",...] form BEFORE the
        # Pydantic-revival walker + loads() see them. In-place mutation
        # is safe — `raw_tree` is a fresh json.loads() result owned by
        # this function.
        raw_tree = _migrate_legacy_namespace(raw_tree)
        revived = _revive_not_implemented(raw_tree)
        revived_json = json.dumps(revived, ensure_ascii=False)
        try:
            # v2.0.28.11 — pass `allowed_objects="messages"` to
            # suppress the PendingDeprecationWarning and pin our
            # contract: only chat messages round-trip via the
            # checkpointer (HumanMessage / AIMessage / ToolMessage /
            # SystemMessage). LangChain's default for `allowed_objects`
            # is "permits everything" which will change in a future
            # release to require an explicit list.
            #
            # v2.0.28.18 P1-P5 — ``allowed_objects="messages"`` was the
            # wrong scope: Document (langchain_core.documents.Document)
            # is also a langchain_core class and DOES need to round-trip
            # (the agent stores retrieved docs in ``state["documents"]``
            # and reads them back at multi-turn replay per v2.0.28.9).
            # "messages" rejects Document → thread skipped → invisible
            # sidebar. Verified in REPL that ``allowed_objects="core"``
            # accepts BOTH legacy ``["langchain","schema","document",
            # "Document"]`` (what ``dumpd`` emits today) AND modern
            # ``["langchain_core","documents","base","Document"]`` IDs
            # for messages + Document. "core" is the narrowest scope
            # that covers every langchain_core class we actually use
            # (messages + documents), without enabling partner
            # integrations. This restores the v2.0.28.11 contract —
            # pending-deprecation-warning suppressed, security posture
            # pinned to langchain_core only — while fixing the corrupt
            # thread bug. The :func:`_migrate_legacy_namespace` walker
            # above remains as defense-in-depth in case LangChain ever
            # tightens "core" further.
            return loads(revived_json, allowed_objects="core")  # type: ignore[call-arg,return-value]
        except NotImplementedError:
            # Defensive last resort: the tree may still contain envelopes
            # we couldn't parse (e.g. unknown class shape). Return the
            # revived plain-dict tree directly — ``messages`` may be lost
            # but the rest of the state survives for the operator to
            # inspect.
            return revived  # type: ignore[return-value]
    except (json.JSONDecodeError, ValueError, TypeError, NotImplementedError, KeyError) as e:
        trace_id = uuid.uuid4().hex[:12]
        logger.bind(trace_id=trace_id).error(
            f"checkpointer._deserialize_state failed thread_id={thread_id!r}: "
            f"{type(e).__name__}: {e}"
        )
        raise CorruptCheckpointError(thread_id, e, trace_id) from e


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


def _open_db(path: str) -> sqlite3.Connection:
    """Open the SQLite connection. We use the stdlib ``sqlite3``
    (NOT ``aiosqlite``) because the schema is one-row-per-thread
    and every query (save / load_latest / list_threads / delete) is
    a tiny point-read or short sequential scan — there's no
    async-batch advantage worth the dependency cost.

    The pre-Phase-3 ``history.py`` used ``aiosqlite`` because
    LangGraph's ``AsyncSqliteSaver`` requires it; we don't have that
    constraint.

    ``check_same_thread=False`` is required because the connection
    may be touched from both the FastAPI event loop and the
    threadpool (``asyncio.to_thread``). We serialize all writes
    through ``_db_lock`` so concurrent writers don't corrupt the
    file.

    v2.0.27.3 P0-P1 — also enable ``PRAGMA busy_timeout=5000`` so
    the second writer waits up to 5 s for the first writer's
    RESERVED lock (acquired via ``BEGIN IMMEDIATE`` in
    :func:`_save_sync` / :func:`_delete_thread_sync`) instead of
    failing immediately with ``OperationalError: database is
    locked``. Without busy_timeout, ``BEGIN IMMEDIATE`` would
    raise on the second writer even though the first writer's
    commit is milliseconds away — turning a transient race into
    a 500. busy_timeout + WAL + BEGIN IMMEDIATE is the SQLite
    recommended combination for multi-writer concurrency.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    # v2.0.27.3 P0-P1 — disable Python's implicit transaction
    # wrapper (``isolation_level=None`` is autocommit mode) so
    # our explicit ``BEGIN IMMEDIATE`` calls in :func:`_save_sync`
    # and :func:`_delete_thread_sync` start a fresh transaction
    # rather than raising ``cannot start a transaction within a
    # transaction``. With the default isolation_level,
    # ``sqlite3`` opens an implicit BEGIN before every DML
    # statement — our explicit BEGIN IMMEDIATE would then be the
    # second BEGIN and fail. autocommit is the standard pattern
    # when the application wants to manage transactions itself.
    conn.isolation_level = None
    conn.execute("PRAGMA journal_mode=WAL")
    # busy_timeout is per-connection — set on every fresh open.
    # 5000ms = 5s grace period; the typical own-write finishes
    # in <50ms so 5s is "infinite" for our load but bounds
    # pathological hangs.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


async def startup() -> None:
    """Create the ``states`` table if absent. Idempotent: subsequent
    calls are no-ops once the table exists.

    Called from the FastAPI lifespan handler — replaces the
    pre-Phase-3 ``await get_checkpointer()`` call.
    """
    global _db_conn, _db_path
    path = str(history_db_path())
    async with _db_lock:
        if _db_conn is not None and _db_path == path:
            return
        # Path changed (or first call): open the new connection.
        if _db_conn is not None and _db_path != path:
            logger.info(
                f"History DB path changed ({_db_path} -> {path}); reopening"
            )
            try:
                _db_conn.close()
            except Exception:
                logger.exception("Error closing previous checkpointer")
            _db_conn = None
            _db_path = None
        if _db_conn is None:
            logger.info(f"Opening own checkpointer at {path}")
            _db_conn = await asyncio.to_thread(_open_db, path)
            _db_path = path
            await asyncio.to_thread(_ensure_schema, _db_conn)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the ``states`` table + ``created_at`` index. Runs
    synchronously; the caller wraps it in ``to_thread``."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS states (
            thread_id   TEXT PRIMARY KEY,
            step        INTEGER NOT NULL,
            state_json  TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_states_created_at
            ON states(created_at DESC);
        """
    )
    conn.commit()


async def shutdown() -> None:
    """Close the singleton connection. Called from FastAPI lifespan teardown."""
    global _db_conn, _db_path
    if _db_conn is not None:
        try:
            await asyncio.to_thread(_db_conn.close)
        except Exception:
            logger.exception("Error closing checkpointer")
    _db_conn = None
    _db_path = None


def reset_for_tests() -> None:
    """Forget the cached connection without awaiting close. Tests only.

    Tests should normally call :func:`shutdown` (await it) for a
    clean teardown. ``reset_for_tests`` is the synchronous escape
    hatch for use in ``conftest.py`` autouse fixtures where the
    event loop is already torn down.
    """
    global _db_conn, _db_path
    _db_conn = None
    _db_path = None


@asynccontextmanager
async def temporary_checkpointer(path: Path) -> AsyncIterator[None]:
    """Pin the checkpointer to a custom path for the duration of a
    test. Restores the previous path on exit. Mirrors the
    pre-Phase-3 ``history.temporary_checkpointer`` shape but with no
    returned object (tests use the module-level functions).
    """
    global _db_conn, _db_path
    prev_conn, prev_path = _db_conn, _db_path
    _db_conn = None
    _db_path = None
    # Force a fresh open at the new path
    await startup()
    try:
        yield
    finally:
        if _db_conn is not None:
            try:
                await asyncio.to_thread(_db_conn.close)
            except Exception:
                logger.exception("Error closing temp checkpointer")
        _db_conn, _db_path = prev_conn, prev_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def save(thread_id: str, state: AgentState, *, step: Optional[int] = None) -> None:
    """Upsert the latest snapshot for ``thread_id``. The row's
    ``created_at`` is overwritten (we only keep the latest per
    thread), so a thread that ran 50 turns still has exactly one
    ``states`` row.

    ``step`` is the FSM's ``step_count`` at save time. If omitted,
    we read it off the state dict (``state.get("step_count")``).
    """
    if not thread_id:
        return
    await startup()  # idempotent
    assert _db_conn is not None
    state_json = await asyncio.to_thread(_serialize_state, state)
    step_val = step if step is not None else int(state.get("step_count") or 0)
    created_at = datetime.now(timezone.utc).isoformat()
    await asyncio.to_thread(
        _save_sync, _db_conn, thread_id, step_val, state_json, created_at
    )


def _save_sync(
    conn: sqlite3.Connection,
    thread_id: str,
    step: int,
    state_json: str,
    created_at: str,
) -> None:
    """Synchronous helper for :func:`save`. Runs in a threadpool
    (``asyncio.to_thread``) to keep the event loop unblocked during
    the JSON serialization + INSERT.

    v2.0.27.3 P0-P1 — wrap the ``INSERT OR REPLACE`` in
    ``BEGIN IMMEDIATE`` / ``COMMIT`` so concurrent writers
    (e.g. two browser tabs both saving the same ``thread_id``
    on every keystroke) acquire a SQLite RESERVED lock at
    transaction start and serialize cleanly. Pre-fix: two
    concurrent ``INSERT OR REPLACE`` interleaved on the same
    connection could leave the row in an arbitrary state
    (``last-writer-wins`` clobber with no atomicity guarantee —
    the loser's ``created_at`` could overwrite the winner's
    final state). Post-fix: the second writer waits for the
    first writer's COMMIT (up to ``busy_timeout=5000``) and
    then re-reads + writes atomically.
    """
    # Python-level serialization: SQLite's BEGIN IMMEDIATE lock
    # only serializes at the DB level; on Windows the per-thread
    # sqlite3 statement state is stored on the connection object,
    # and concurrent statements from different threads can crash
    # with access violations. The ``threading.Lock`` makes the
    # Python-level path single-threaded; the BEGIN IMMEDIATE
    # then takes care of the DB-level race.
    with _db_write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO states (thread_id, step, state_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (thread_id, step, state_json, created_at),
            )
            conn.execute("COMMIT")
        except Exception:
            # Best-effort rollback — if the connection is already in
            # an error state, ``conn.execute("ROLLBACK")`` itself may
            # raise; swallow that to surface the original error.
            try:
                conn.execute("ROLLBACK")
            except Exception:
                logger.exception("checkpointer._save_sync rollback failed")
            raise


async def load_latest(thread_id: str) -> Optional[AgentState]:
    """Return the latest saved state for ``thread_id``, or ``None``
    if no snapshot exists. Used by /sessions/{id}/messages to replay
    the chat history on page refresh.

    v2.0.27.2 P1-P5 — corrupt rows return ``None`` (same semantics
    as "row not found") instead of bubbling
    :class:`CorruptCheckpointError` as a 500. The caller
    (``/sessions/{tid}/messages``) already handles ``None`` by
    returning an empty message list. The corruption is logged
    once with the thread_id + trace_id so the operator can find
    the bad row via ``sqlite3`` shell and delete or restore it.
    """
    if not thread_id:
        return None
    await startup()  # idempotent
    assert _db_conn is not None
    row = await asyncio.to_thread(_load_latest_sync, _db_conn, thread_id)
    if row is None:
        return None
    state_json, _step, _created_at = row
    try:
        return await asyncio.to_thread(_deserialize_state, state_json, thread_id)
    except CorruptCheckpointError as e:
        logger.bind(trace_id=e.trace_id).error(
            f"checkpointer.load_latest skipping corrupt thread_id={thread_id!r}"
        )
        return None


def _load_latest_sync(
    conn: sqlite3.Connection, thread_id: str
) -> Optional[tuple[str, int, str]]:
    cur = conn.execute(
        "SELECT state_json, step, created_at FROM states WHERE thread_id = ?",
        (thread_id,),
    )
    return cur.fetchone()


async def list_threads(limit: int = _MAX_THREADS) -> tuple[list[dict], int]:
    """Return ``(valid_threads, corrupt_count)`` — newest-first
    summary dicts per thread, plus the count of rows that failed
    deserialization.

    Used by /sessions to populate the sidebar. Each summary dict
    carries the keys the /sessions serializer needs: ``thread_id``,
    ``messages`` (the deserialized BaseMessage list), ``updated_at``
    (ISO 8601), ``step``. Source counts and first-filename come
    from a separate LanceDB query (unchanged from pre-Phase-3).

    v2.0.27.2 P1-P5 — **signature change**: pre-PR-3 returned
    ``list[dict]`` and silently swallowed corruption with
    ``state = {"messages": []}`` per-row, hiding data loss behind
    an "empty" sidebar entry. Post-PR-3:

    * corrupt rows are **skipped** (don't appear in the sidebar at
      all), counted in the second tuple element;
    * the caller (sessions.py) decides whether to surface the
      count to the operator (response header
      ``X-Corrupt-Thread-Count`` + structured log);
    * the underlying :class:`CorruptCheckpointError` is logged with
      ``trace_id`` so the operator can grep for the original cause.

    Only one real caller exists today (``sessions.py:439``) — both
    other references (this module's docstring + the
    ``test_audit_fixes`` / ``test_api`` test monkey-patches) are
    updated together with this PR.
    """
    await startup()
    assert _db_conn is not None
    rows = await asyncio.to_thread(_list_threads_sync, _db_conn, limit)

    def _to_summary(row: tuple) -> dict:
        tid, step, state_json, created_at = row
        # Raises CorruptCheckpointError — caller (``list_threads``)
        # catches and increments the corrupt counter.
        state = _deserialize_state(state_json, tid)
        return {
            "thread_id": tid,
            "step": step,
            "updated_at": created_at,
            "messages": state.get("messages") or [],
        }

    valid: list[dict] = []
    corrupt_count = 0
    for r in rows:
        try:
            valid.append(await asyncio.to_thread(_to_summary, r))
        except CorruptCheckpointError as e:
            corrupt_count += 1
            logger.bind(trace_id=e.trace_id).error(
                f"checkpointer.list_threads skipping corrupt "
                f"thread_id={e.thread_id!r}"
            )
    return valid, corrupt_count


def _list_threads_sync(conn: sqlite3.Connection, limit: int) -> list[tuple]:
    cur = conn.execute(
        "SELECT thread_id, step, state_json, created_at FROM states "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    return cur.fetchall()


async def delete_thread(thread_id: str) -> int:
    """Remove the ``states`` row for ``thread_id``. Returns the
    number of rows deleted (0 or 1). Raises on unexpected errors so
    the caller can surface them — matches the pre-Phase-3
    ``history.delete_thread`` contract after v2.0.18.
    """
    if not thread_id:
        return 0
    await startup()
    assert _db_conn is not None
    deleted = await asyncio.to_thread(_delete_thread_sync, _db_conn, thread_id)
    return deleted


def _delete_thread_sync(conn: sqlite3.Connection, thread_id: str) -> int:
    """Synchronous delete. v2.0.27.3 P0-P1 — wrapped in
    ``BEGIN IMMEDIATE`` / ``COMMIT`` for the same reason as
    :func:`_save_sync`: a concurrent save racing the delete
    must not resurrect the row mid-delete. The save either
    commits before the delete (row gone) or after (new row
    appears) — never interleaved half-states. Also guarded by
    ``_db_write_lock`` to avoid Windows access violations from
    concurrent sqlite3 statements on the same connection.
    """
    with _db_write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "DELETE FROM states WHERE thread_id = ?", (thread_id,)
            )
            conn.execute("COMMIT")
            return cur.rowcount
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                logger.exception(
                    "checkpointer._delete_thread_sync rollback failed"
                )
            raise


__all__ = [
    "startup",
    "shutdown",
    "save",
    "load_latest",
    "list_threads",
    "delete_thread",
    "reset_for_tests",
    "temporary_checkpointer",
    "CorruptCheckpointError",
]