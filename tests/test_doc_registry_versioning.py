"""v2.0.29.4 (Phase 4 PR-2) — doc_registry versioning + superseded contract.

Pins the contract for the 4 new fields/behaviours introduced in
Phase 4 PR-2:

  1. ``version`` field — defaults to 1 on register; preserved on
     re-register of the same doc_id.
  2. ``superseded_by`` field — defaults to None on register; stamped
     by ``supersede(old, new)``.
  3. ``DocStatus`` gains ``"superseded"`` literal — set automatically
     by ``supersede()``.
  4. ``list_for_thread(thread_id)`` (default) drops superseded rows;
     ``include_superseded=True`` recovers them.

Invariants
----------
* ``supersede()`` is idempotent (calling twice is a no-op the
  second time — same args → same row → no dirty mark).
* ``supersede()`` of an unknown doc_id is a no-op (defensive against
  re-upload flows where the new id was registered before the old
  one was known).
* The on-disk JSON keys mirror the in-memory dict so a registry
  restart preserves versioning state.
"""
from __future__ import annotations

from src.storage import doc_registry


def test_register_defaults_version_to_1():
    """A freshly-registered doc starts at ``version=1``."""
    doc_registry.reset_for_tests()
    doc_registry.register("doc1", "file.pdf", "/x", "t1")
    entries = doc_registry.list_for_thread("t1", include_superseded=True)
    assert len(entries) == 1
    assert entries[0]["version"] == 1
    assert entries[0]["superseded_by"] is None


def test_register_preserves_version_on_reregister():
    """Re-registering the same doc_id preserves version (retry path).

    The semantic distinction: a re-register (same upload, retrying)
    keeps the original ``version``. A version BUMP requires the
    explicit ``supersede()`` path.
    """
    doc_registry.reset_for_tests()
    doc_registry.register("doc1", "file.pdf", "/x", "t1")
    # Operator bumps version via supersede from "self" — defensive
    # contract that version lives across re-register.
    doc_registry.register("doc1", "file.pdf", "/x", "t1")  # re-register
    entries = doc_registry.list_for_thread("t1", include_superseded=True)
    assert entries[0]["version"] == 1  # still 1; re-register is idempotent


def test_supersede_stamps_new_doc_id():
    """``supersede(old, new)`` flips ``old``'s status and stamps superseded_by."""
    doc_registry.reset_for_tests()
    doc_registry.register("v1", "old.pdf", "/x", "t1")
    doc_registry.update("v1", status="indexed", chunk_count=3)
    doc_registry.register("v2", "new.pdf", "/x", "t1")
    doc_registry.update("v2", status="indexed", chunk_count=3)
    doc_registry.supersede("v1", "v2")

    # Default list: v1 hidden, v2 active
    active = doc_registry.list_for_thread("t1")
    assert [e["doc_id"] for e in active] == ["v2"]

    # Opt-in list: both, with v1 superseded
    all_entries = doc_registry.list_for_thread("t1", include_superseded=True)
    by_id = {e["doc_id"]: e for e in all_entries}
    assert by_id["v1"]["status"] == "superseded"
    assert by_id["v1"]["superseded_by"] == "v2"
    assert by_id["v2"]["status"] == "indexed"
    assert by_id["v2"]["superseded_by"] is None


def test_supersede_is_idempotent():
    """Calling ``supersede(old, new)`` twice with the same args is a no-op."""
    doc_registry.reset_for_tests()
    doc_registry.register("v1", "old.pdf", "/x", "t1")
    doc_registry.register("v2", "new.pdf", "/x", "t1")
    doc_registry.supersede("v1", "v2")
    # Second call with same args must not crash, must not change anything.
    doc_registry.supersede("v1", "v2")

    all_entries = doc_registry.list_for_thread("t1", include_superseded=True)
    by_id = {e["doc_id"]: e for e in all_entries}
    assert by_id["v1"]["superseded_by"] == "v2"
    assert by_id["v1"]["status"] == "superseded"


def test_supersede_unknown_doc_id_is_noop():
    """``supersede`` of an unregistered doc_id is a defensive no-op."""
    doc_registry.reset_for_tests()
    # No crash, no state mutation
    doc_registry.supersede("does_not_exist", "anything")
    assert doc_registry.list_for_thread(None, include_superseded=True) == []