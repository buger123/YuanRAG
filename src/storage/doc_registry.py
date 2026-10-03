"""JSON-backed registry for document ingestion state.

Why this exists
---------------
LanceDB only stores ``ChunkRecord`` rows. If a parse / chunk step produces
**zero** rows (e.g. an image-only PDF where Docling's OCR pipeline returns
empty, or any other parser failure), the doc silently disappears from the
sidebar — the user uploads a file, the upload progress row vanishes, and
nothing appears in the documents list. Nothing tells them "this file was
parsed but no text was extracted" vs "the upload never finished."

This module persists a tiny per-doc status record alongside the chunk store
so empty/failed/pending docs are queryable the same way indexed docs are.

Storage
-------
A single ``_registry.json`` file under ``uploads_dir()``. Volume is tiny
(one row per uploaded file), and per-key write serialization via a single
``threading.Lock`` is enough. We do atomic writes via ``tempfile`` +
``os.replace`` so a crash mid-write can't corrupt the file.

Write batching
--------------
``register`` (the "pending" sentinel added when an upload returns) and
``update`` (the terminal status flip once ingestion finishes) used to each
trigger an immediate atomic disk write. For an upload that registers +
indexes in <50 ms, that's 2 writes — and the second supersedes the first,
so the first is wasted. A single-doc upload saves one ``tempfile + os.replace``
round-trip (~5-10 ms on slow disks) plus a brief hold on ``_lock``.

Both calls now update an in-memory ``_state`` dict synchronously (so the
sidebar's next ``list_for_thread`` immediately sees the change) and mark
the cache dirty. A background daemon thread flushes every 200 ms if dirty.
Crash semantics: a hard kill before the next flush loses the most recent
mutation — for a single doc that's the difference between "pending"
(visible briefly in the sidebar) and "indexed" (the terminal state). Since
both are recoverable from the chunk store + registry, this is acceptable.

Schema
------
``{doc_id: {"doc_id", "filename", "thread_id",
            "ingested_at", "status", "chunk_count", "error_message"}, ...}``

(v2.0.27 — P0-P2 privacy hardening: the ``source_path`` key is **no longer
written** to the registry. The wire surface (LanceDB + ``GET /documents`` +
``DocumentSummary``) returns ``f"{doc_id}/{filename}"` instead of an
absolute path. Legacy ``_registry.json`` files with the old absolute path
are rewritten on first read by ``_migrate_absolute_paths``; legacy
LanceDB chunk rows are coerced at read time by
``_coerce_legacy_source_path``.)

``status`` is one of ``pending`` / ``parsing`` / ``embedding`` /
``indexed`` / ``empty`` / ``failed``:
- ``pending`` — registered before background ingestion starts
- ``parsing`` (v2.0.5) — ``run_ingestion`` is running (Docling /
  text-parser / etc.). The "30-90 s pending" UX bug was that the
  sidebar showed only a generic "处理中" chip with no
  per-stage signal; this state lets the frontend render
  "解析中…" while the parser is the active worker.
- ``embedding`` (v2.0.5) — parser finished, BGE-M3 is computing
  embeddings. The chip can show "向量化中… (N/N 块)" if the
  frontend wants fine-grained progress.
- ``indexed`` — at least one chunk landed in LanceDB
- ``empty`` — ingestion completed but no chunks were produced
- ``failed`` — ingestion raised an exception (e.g. corrupt PDF)
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from src.core.logging import logger
from src.core.paths import uploads_dir

DocStatus = Literal[
    "pending",
    "parsing",
    "embedding",
    "indexed",
    "empty",
    "failed",
    # v2.0.29.4 (Phase 4 PR-2) — "superseded" marks a doc that has
    # been re-uploaded (or otherwise replaced) by a newer version.
    # ``list_for_thread`` default-excludes rows in this state (callers
    # pass ``include_superseded=True`` to opt in, e.g. for an admin /
    # "history" view). Retrieval also drops hits whose doc_id points at
    # a superseded entry so the LLM never cites stale content. The
    # terminal "indexed" state is intentionally preserved alongside
    # superseded — a doc that was once retrievable can later become
    # retrievable-no-longer without rewriting its original status.
    "superseded",
]


def _serialize_source_path(source_path: str, doc_id: str, filename: str) -> str:
    """Coerce any caller-supplied path into the canonical privacy-safe form.

    v2.0.27 (Item 9 PR-1, P0-P2): absolute server-side filesystem paths must
    never be persisted in ``_registry.json`` or returned over the API.
    Instead, store ``f"{doc_id}/{filename}"`` — just enough context for an
    operator to locate the orphan raw file under ``uploads_dir()/<doc_id>/``
    if a doc ends up in the ``empty`` / ``failed`` state and they need to
    investigate the original bytes (which are retained only for that
    debugging path; the QW-#12 success path deletes raw bytes on ingest).

    The invariant checks here are defensive: any future contributor who tries
    to write a leading ``/``, a Windows drive letter, or a path with a colon
    in the first segment will get an ``AssertionError`` immediately rather
    than silently regressing the privacy guarantee.
    """
    assert source_path and isinstance(source_path, str)
    # The output is always doc_id/filename — independent of the caller's
    # input. We carry the parameter only to keep the helper's signature
    # self-documenting at call sites.
    first_segment = doc_id.split("/")[0] if "/" in doc_id else doc_id
    assert ":" not in first_segment, f"doc_id leaks drive letter: {doc_id!r}"
    assert not doc_id.startswith("/"), f"doc_id leaks absolute path: {doc_id!r}"
    out = f"{doc_id}/{filename}"
    return out


def _coerce_legacy_source_path(source_path: Optional[str], doc_id: str, filename: str) -> Optional[str]:
    """Read-time coercion for legacy ``source_path`` values.

    Pre-v2.0.27 entries have the full absolute path on disk. We rewrite
    them to the canonical privacy-safe form on first read so the wire
    surface stops leaking. Idempotent: if the value is already
    ``doc_id/filename``-relative (new writes or already-migrated entries)
    it's returned unchanged.

    Used by ``list_documents`` (LanceDB-side) and by ``_ensure_loaded``
    (registry-side migration).
    """
    if source_path is None:
        return None
    if "/" in source_path or "\\" in source_path:
        first = source_path.replace("\\", "/").split("/")[0]
        # Heuristic: an absolute path either starts with "/" or contains a
        # colon (Windows drive letter) or is a known absolute prefix on
        # POSIX (e.g. "/home", "/Users", "/tmp", "C:\\" → after replace
        # becomes "C:").
        looks_absolute = (
            source_path.startswith("/")
            or first.endswith(":")
            or source_path.startswith("~")
        )
        if looks_absolute:
            return _serialize_source_path(source_path, doc_id, filename)
    return source_path

# Single process-wide lock. ``register`` / ``update`` / ``remove`` /
# ``remove_thread`` all read-modify-write the in-memory state, and the
# flusher reads the same dict, so concurrent callers would race without
# it. The lock is held only for the duration of an in-memory dict mutation
# (the flush itself happens inside the lock to keep disk + memory in sync),
# so contention is minimal.
_lock = threading.Lock()

# In-memory source of truth. ``register`` / ``update`` mutate this dict
# synchronously under ``_lock`` so ``list_for_thread`` (which also reads
# under the lock) sees the change immediately. The flusher copies it to
# disk on a 200 ms cadence.
_state: dict[str, dict] = {}
_initialized = False
_dirty = False

# Background flusher: wakes on either a dirty notify or the 200 ms tick.
_flush_event = threading.Event()
_flush_thread: Optional[threading.Thread] = None
_shutdown = False

FLUSH_INTERVAL_SEC = 0.2


def _path() -> Path:
    return uploads_dir() / "_registry.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_disk() -> dict[str, dict]:
    """Load the on-disk registry. Tolerates missing / corrupt files by
    returning ``{}`` and logging — a bad file should not brick the sidebar."""
    p = _path()
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8") or "{}"
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning(f"doc_registry: {p} has non-dict top level; ignoring")
            return {}
        return data
    except Exception:
        logger.exception(f"doc_registry: failed to read {p}")
        return {}


def _migrate_absolute_paths(data: dict[str, dict]) -> tuple[dict[str, dict], int]:
    """v2.0.27 P0-P2 privacy migration: drop legacy ``source_path`` keys.

    Older ``_registry.json`` files store the absolute server-side path that
    the upload landed at (e.g. ``D:\\...\\uploads\\<doc_id>\\<filename>``).
    That information is not necessary for the service to function and is a
    privacy leak — anyone with read access to ``_registry.json`` (or the
    sidebar's ``GET /documents`` response that proxies it) learns the
    project's storage layout, the OS user, and the volume of activity.

    For each entry that has a legacy ``source_path`` key pointing at an
    absolute path, we drop the key. The other fields are kept; the row's
    wire representation (``doc_id`` + ``filename`` + status) is unchanged.

    Returns the rewritten dict and the count of rows rewritten so the
    caller can log an audit trail. The migration is idempotent: re-running
    on already-migrated data rewrites 0 rows.
    """
    rewritten = 0
    for doc_id, entry in list(data.items()):
        if "source_path" not in entry:
            continue
        # Pre-v2.0.27 stored the absolute path. Post-v2.0.27 entries don't
        # have the key at all. Drop unconditionally — the canonical form
        # for ``GET /documents`` is computed at read time from doc_id and
        # filename (see ``lancedb_store.list_documents``).
        del entry["source_path"]
        rewritten += 1
    return data, rewritten


def _write_disk(data: dict[str, dict]) -> None:
    """Atomic write of the entire registry. ``tempfile`` + ``os.replace``
    so a crash mid-write leaves the previous file intact."""
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix="_registry.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _ensure_loaded() -> None:
    """Lazy init — read the on-disk file into ``_state`` on first access.

    Double-checked locking: the common case (state already loaded) skips
    the lock entirely so the hot path stays lock-free.

    v2.0.27 P0-P2: also runs ``_migrate_absolute_paths`` on first load to
    rewrite any pre-privacy-hardening rows in place. The migration is
    idempotent; a previously-migrated registry short-circuits with 0
    rewrites and no disk write.
    """
    global _initialized, _state, _dirty
    if _initialized:
        return
    with _lock:
        if _initialized:
            return
        loaded = _read_disk()
        migrated, count = _migrate_absolute_paths(loaded)
        _state = migrated
        _initialized = True
        if count:
            logger.info(
                f"doc_registry: migrated {count} legacy absolute-path "
                f"entries (v2.0.27 P0-P2 privacy hardening)"
            )
            # Persist the rewritten registry immediately so subsequent
            # restarts don't re-log the migration.
            _dirty = True
            try:
                _write_disk(_state)
                _dirty = False
            except Exception:
                logger.exception(
                    "doc_registry: failed to persist migrated registry; "
                    "will retry on next flush"
                )


def _start_flush_thread() -> None:
    """Start the background flusher. Idempotent. The thread is a daemon
    so it never blocks process exit."""
    global _flush_thread
    if _flush_thread is not None and _flush_thread.is_alive():
        return
    _flush_thread = threading.Thread(
        target=_flush_loop, daemon=True, name="doc-registry-flush"
    )
    _flush_thread.start()


def _flush_loop() -> None:
    """Daemon loop: flush when either an explicit notify arrives or the
    200 ms tick fires. Exits cleanly when ``_shutdown`` is set."""
    global _dirty
    while True:
        # Wait up to FLUSH_INTERVAL_SEC for an event; if none arrives, we
        # still tick through to handle the case where a notify raced
        # against the clear.
        triggered = _flush_event.wait(timeout=FLUSH_INTERVAL_SEC)
        if _shutdown:
            break
        if triggered:
            _flush_event.clear()
        # Flush under the lock so memory and disk stay coherent — the
        # mutation side reads ``_state`` under the same lock.
        with _lock:
            if not _dirty:
                continue
            try:
                _write_disk(_state)
                _dirty = False
            except Exception:
                logger.exception("doc_registry: background flush failed")


def _mark_dirty() -> None:
    """Mark state dirty and notify the flusher. Caller must hold ``_lock``."""
    global _dirty
    _dirty = True
    _flush_event.set()


def register(
    doc_id: str,
    filename: str,
    source_path: str,
    thread_id: str,
) -> None:
    """Pre-register a freshly-uploaded doc as ``pending``.

    Idempotent — calling again with the same ``doc_id`` overwrites the
    filename / thread_id (used if a re-upload lands on the same id, e.g.
    retry path). The ``ingested_at`` timestamp is preserved across
    re-registers so the "first upload" time stays stable.

    v2.0.27 P0-P2: ``source_path`` is no longer persisted in the registry
    (the absolute server path is a privacy leak). The parameter is kept
    on the public signature for caller compatibility — callers may pass
    the raw target path, but it is intentionally discarded here. The
    canonical wire value (``f"{doc_id}/{filename}"``) is computed by
    ``lancedb_store.list_documents`` at read time.

    v2.0.29.4 (Phase 4 PR-2) — versioning: each register call stamps
    ``version`` on the row. The first register for a doc_id gets
    ``version=1``. Re-registering the same doc_id preserves the
    version (we don't bump on re-register; ``update(superseded_by=...)``
    is the explicit version-bump path). ``superseded_by`` defaults to
    None so new docs are not flagged superseded.

    Disk write is deferred to the background flusher — the in-memory
    ``_state`` is updated synchronously so ``list_for_thread`` sees the
    entry immediately.
    """
    _ensure_loaded()
    _start_flush_thread()
    with _lock:
        prev = _state.get(doc_id) or {}
        _state[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "thread_id": thread_id,
            "ingested_at": prev.get("ingested_at") or _now_iso(),
            "status": "pending",
            "chunk_count": 0,
            "error_message": None,
            # v2.0.29.4 (Phase 4 PR-2) — versioning fields. Preserve
            # existing version on re-register (treat the re-register
            # as "same upload, retrying" rather than "new version").
            # Bump via ``update(superseded_by=...)`` for true version
            # transitions.
            "version": prev.get("version", 1),
            "superseded_by": prev.get("superseded_by"),
        }
        _mark_dirty()


def supersede(
    old_doc_id: str,
    new_doc_id: str,
) -> None:
    """v2.0.29.4 (Phase 4 PR-2) — mark ``old_doc_id`` as superseded by ``new_doc_id``.

    Flips ``old_doc_id``'s ``status`` to ``"superseded"`` and stamps
    ``superseded_by=new_doc_id``. Idempotent: calling again with the
    same arguments is a no-op (the row is already superseded). No-op
    if ``old_doc_id`` isn't registered (defensive — re-upload flows
    may register the new id before the old one is known).

    The new version's ``version`` field is NOT touched here; that's
    the responsibility of the caller (``register(new_doc_id, ...)``
    will stamp it as 1 unless the caller threads version through).
    Most re-upload flows treat the new doc as ``version=1`` of a
    fresh doc_id; the old doc keeps its own ``version`` for history.

    Disk write is deferred to the background flusher — the in-memory
    ``_state`` is updated synchronously so ``list_for_thread`` sees
    the flip immediately.
    """
    _ensure_loaded()
    _start_flush_thread()
    with _lock:
        old = _state.get(old_doc_id)
        if old is None:
            return
        # Idempotent: already-superseded → no change, no dirty mark.
        if old.get("status") == "superseded" and old.get("superseded_by") == new_doc_id:
            return
        old["status"] = "superseded"
        old["superseded_by"] = new_doc_id
        # Bump ``ingested_at`` so the sidebar sorts the superseded
        # row to the top of "most recently touched" — useful for
        # debugging a re-upload flow.
        old["ingested_at"] = _now_iso()
        _mark_dirty()


def update(
    doc_id: str,
    *,
    status: DocStatus,
    chunk_count: int = 0,
    error_message: Optional[str] = None,
    diagnosis: Optional[dict] = None,
) -> None:
    """Update ingestion status for an already-registered doc.

    No-op if the doc isn't registered — defensive against the call
    happening before ``register`` (e.g. if someone refactors the
    upload route and forgets the pre-register step).

    ``diagnosis`` (v1.1.12) is a free-form structured payload that
    callers can attach alongside ``error_message``. Used today for
    "未提取到文本" status to carry a per-format hint (Excel "每个
    sheet 至少需要一行数据", PDF "扫描件无文字层" etc.) so a
    UI can render the right copy without parsing the free-text
    ``error_message``. Optional — backward compatible with the
    pre-v1.1.12 call sites that only pass ``error_message``.
    """
    _ensure_loaded()
    _start_flush_thread()
    with _lock:
        if doc_id not in _state:
            return
        _state[doc_id]["status"] = status
        _state[doc_id]["chunk_count"] = chunk_count
        _state[doc_id]["error_message"] = error_message
        # v1.1.12 — only set ``diagnosis`` when provided so the
        # registry's default value stays ``None`` for callers that
        # don't use the new field. This keeps the on-disk JSON clean
        # and avoids spurious diffs in snapshots.
        if diagnosis is not None:
            _state[doc_id]["diagnosis"] = diagnosis
        # Bump ``ingested_at`` on every status change so the sidebar can
        # sort "most recently touched" first even for empty docs (the
        # chunk-store aggregation doesn't see them, so we have no other
        # signal).
        _state[doc_id]["ingested_at"] = _now_iso()
        _mark_dirty()


def remove(doc_id: str) -> None:
    """Drop the registry entry for ``doc_id``. No-op if missing."""
    _ensure_loaded()
    _start_flush_thread()
    with _lock:
        if doc_id in _state:
            _state.pop(doc_id)
            _mark_dirty()


def remove_thread(thread_id: str) -> int:
    """Drop every entry whose ``thread_id`` matches. Returns count removed."""
    _ensure_loaded()
    _start_flush_thread()
    with _lock:
        removed = 0
        # Build the kept dict (don't mutate while iterating).
        kept = {
            k: v for k, v in _state.items() if v.get("thread_id") != thread_id
        }
        removed = len(_state) - len(kept)
        if removed:
            _state.clear()
            _state.update(kept)
            _mark_dirty()
        return removed


def list_for_thread(
    thread_id: Optional[str] = None,
    *,
    include_superseded: bool = False,
) -> list[dict]:
    """Return a copy of registry entries, optionally filtered by thread_id.

    - ``thread_id=None`` → return everything (admin / legacy view).
    - ``thread_id=""`` → match only entries with an empty ``thread_id``.
    - ``thread_id="abc"`` → match only entries tagged with ``"abc"``.

    v2.0.29.4 (Phase 4 PR-2) — ``include_superseded=False`` (the
    default) DROPS rows whose ``status == "superseded"``. Most callers
    want the active set; the explicit ``True`` opt-in is for the
    sidebar's "show history" toggle, an admin endpoint, or tests.

    Each returned dict is a fresh shallow copy so callers can mutate
    without disturbing the on-disk state.
    """
    _ensure_loaded()
    with _lock:
        snapshot = dict(_state)
    out: list[dict] = []
    for entry in snapshot.values():
        if thread_id is not None and entry.get("thread_id") != thread_id:
            continue
        # v2.0.29.4 (Phase 4 PR-2) — default-exclude superseded rows.
        if not include_superseded and entry.get("status") == "superseded":
            continue
        out.append(dict(entry))
    return out


def flush() -> None:
    """Force any pending in-memory changes to disk synchronously.

    Used by tests that need to inspect the on-disk file after a mutation,
    and by the FastAPI shutdown handler to make sure no final-state write
    is lost. Safe to call when there's nothing to flush.
    """
    global _dirty
    _ensure_loaded()
    with _lock:
        if not _dirty:
            return
        _write_disk(_state)
        _dirty = False


def shutdown() -> None:
    """Stop the background flusher and write any pending changes.

    Called from the FastAPI lifespan exit handler.
    """
    global _shutdown
    _ensure_loaded()
    flush()
    _shutdown = True
    _flush_event.set()


def reset_for_tests() -> None:
    """Drop in-memory state and force a clean re-load from disk on next
    access. Tests only — does not touch the on-disk file."""
    global _initialized, _state, _dirty
    with _lock:
        _initialized = False
        _state = {}
        _dirty = False


__all__ = [
    "DocStatus",
    "register",
    "update",
    "remove",
    "remove_thread",
    "list_for_thread",
    "supersede",
    "flush",
    "shutdown",
    "reset_for_tests",
    "_serialize_source_path",
    "_coerce_legacy_source_path",
    "_migrate_absolute_paths",
]