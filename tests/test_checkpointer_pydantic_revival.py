"""Tests for v2.0.22+ checkpointer Pydantic Source revival (post-Phase-3 hotfix).

Background — the v2.0.7-vintage bug
-----------------------------------
``src.api.schemas.Source`` is a Pydantic ``BaseModel``. Pydantic models
are **not** registered with LangChain's serializer registry — there's
no ``lc_serializable`` declaration, and LangChain falls back to the
``not_implemented`` envelope:

.. code-block:: python

    {
        "lc": 1,
        "type": "not_implemented",
        "id": ["src", "api", "schemas", "Source"],
        "repr": "Source(index=1, chunk_id='', doc_id='web-abc', ...)",
    }

When ``_serialize_state`` walked an ``AgentState`` whose
``sources`` field contained a Pydantic ``Source``, ``dumpd`` produced
that envelope. On the read side, ``loads()`` raised
``NotImplementedError``, ``_to_summary`` caught it and returned
``{"messages": []}``, and the sidebar showed an empty title for
every thread that ever had a web source saved.

Every thread with even one web search was affected. That's why the
error log in production looked like an "all threads failing" wall —
the same shape repeated hundreds of times.

The fix
--------
Two layers (defense in depth):

1. **Write-side** — ``_serialize_state`` walks ``state["sources"]``
   and converts any Pydantic ``Source`` instance to a plain dict via
   ``model_dump()`` BEFORE handing the state to ``dumpd``. New
   threads now write clean JSON.

2. **Read-side** — ``_deserialize_state`` post-processes the
   ``loads()`` result with ``_revive_not_implemented``, which walks
   the tree and parses any ``{"lc": 1, "type": "not_implemented",
   "repr": "..."}`` envelope back into a plain dict via
   ``ast.literal_eval``. Pydantic's default ``__repr__`` is exactly
   ``Source(field=value, ...)`` — safe under ``literal_eval``
   (no code execution, only literal kwargs).

The walker is intentionally generic: any class dumped as
``not_implemented`` gets revived, not just ``Source``. Future dead-
class regressions degrade gracefully instead of blanking the sidebar.

These tests pin all three of:
- write-side Pydantic→dict conversion
- read-side legacy envelope revival (the recovery path for existing
  corrupted threads)
- the walker's robustness against nested structures + unparseable
  reprs (graceful fallback to ``{"_repr": ..., "_lc_id": ...}``).
"""
from __future__ import annotations

import ast
import json


# ---------------------------------------------------------------------------
# 1. Write-side: Pydantic Source → plain dict before dumpd
# ---------------------------------------------------------------------------


def test_serialize_state_converts_pydantic_source_to_dict():
    """``_serialize_state`` must strip Pydantic Source wrappers BEFORE
    handing state to ``dumpd``. Without this, every new thread would
    write the ``not_implemented`` envelope and the bug would
    silently come back even with the read-side revival in place.
    """
    from src.api.schemas import Source
    from src.storage.checkpointer import _serialize_state

    state = {
        "messages": [],
        "sources": [
            Source(
                index=1,
                chunk_id="",
                doc_id="web-abc",
                filename="Example",
                page=None,
                section=None,
                sheet=None,
                text="hello",
                score=1.0,
                url="https://example.com",
                domain="example.com",
                source_kind="web",
            )
        ],
    }
    out = _serialize_state(state)
    parsed = json.loads(out)
    sources = parsed["sources"]
    assert len(sources) == 1
    # The Pydantic instance must become a plain dict BEFORE dumpd sees
    # it — verify by inspecting the wire shape.
    assert isinstance(sources[0], dict), (
        f"sources[0] must be a plain dict after serialization, got "
        f"{type(sources[0]).__name__}"
    )
    assert sources[0]["doc_id"] == "web-abc"
    assert sources[0]["url"] == "https://example.com"
    # No LangChain envelope — the dict went straight to JSON.
    assert "lc" not in sources[0]


def test_serialize_state_preserves_dict_sources_unchanged():
    """If ``state["sources"]`` already contains plain dicts (the post-fix
    normal shape), the writer must NOT touch them. Verify by passing a
    pre-baked dict and confirming it round-trips byte-identical.
    """
    from src.storage.checkpointer import _serialize_state

    raw = {"index": 1, "chunk_id": "", "doc_id": "d", "filename": "f"}
    state = {"messages": [], "sources": [raw]}
    out = _serialize_state(state)
    parsed = json.loads(out)
    assert parsed["sources"][0] == raw


def test_serialize_state_handles_missing_sources_field():
    """``_serialize_state`` must not crash if ``sources`` is absent
    (greeting / simple-fact threads don't write it).
    """
    from src.storage.checkpointer import _serialize_state

    state = {"messages": []}
    out = _serialize_state(state)
    parsed = json.loads(out)
    assert "sources" not in parsed or parsed.get("sources") in (None, [])


def test_serialize_state_handles_empty_sources_list():
    from src.storage.checkpointer import _serialize_state

    state = {"messages": [], "sources": []}
    out = _serialize_state(state)
    parsed = json.loads(out)
    assert parsed["sources"] == []


# ---------------------------------------------------------------------------
# 2. Read-side: revive ``not_implemented`` envelopes from repr
# ---------------------------------------------------------------------------


def test_repr_parser_handles_typical_pydantic_source():
    """The Pydantic default ``__repr__`` is ``Class(field=value, ...)``
    with all-literal kwargs. ``ast.literal_eval`` must parse it into
    a plain dict. Pin the exact format so a future Pydantic upgrade
    that changes repr() is caught here.
    """
    from src.storage.checkpointer import _revive_not_implemented

    envelope = {
        "lc": 1,
        "type": "not_implemented",
        "id": ["src", "api", "schemas", "Source"],
        "repr": (
            "Source(index=1, chunk_id='', doc_id='web-abc', "
            "filename='Example', page=None, section=None, sheet=None, "
            "text='hello', score=1.0, url='https://example.com', "
            "domain='example.com', source_kind='web')"
        ),
    }
    out = _revive_not_implemented(envelope)
    assert out == {
        "index": 1,
        "chunk_id": "",
        "doc_id": "web-abc",
        "filename": "Example",
        "page": None,
        "section": None,
        "sheet": None,
        "text": "hello",
        "score": 1.0,
        "url": "https://example.com",
        "domain": "example.com",
        "source_kind": "web",
    }


def test_revival_walks_nested_structures():
    """Legacy threads stored the envelope at arbitrary depth — e.g.
    ``state["sources"][0]`` AND inside a ToolMessage ``additional_kwargs``.
    The walker must descend into both dicts and lists, not just top-
    level.
    """
    from src.storage.checkpointer import _revive_not_implemented

    tree = {
        "sources": [
            {
                "lc": 1,
                "type": "not_implemented",
                "id": ["src", "api", "schemas", "Source"],
                "repr": "Source(index=1, doc_id='a', source_kind='local')",
            }
        ],
        "additional_kwargs": {
            "nested": [
                {
                    "lc": 1,
                    "type": "not_implemented",
                    "id": ["x", "y", "Z"],
                    "repr": "Z(value=42, name='q')",
                }
            ]
        },
    }
    out = _revive_not_implemented(tree)
    assert out["sources"][0] == {"index": 1, "doc_id": "a", "source_kind": "local"}
    assert out["additional_kwargs"]["nested"][0] == {"value": 42, "name": "q"}


def test_revival_passes_through_normal_dicts():
    """Plain dicts (no envelope marker) must round-trip unchanged. The
    walker should never accidentally rewrite non-envelope dicts.
    """
    from src.storage.checkpointer import _revive_not_implemented

    plain = {"index": 1, "doc_id": "abc", "text": "hello"}
    assert _revive_not_implemented(plain) is plain or _revive_not_implemented(plain) == plain


def test_revival_passes_through_normal_lists():
    """Plain lists (no envelopes inside) must round-trip unchanged.

    Pin the contract: when the input is a list with NO envelope
    members, the output equals the input element-for-element. This
    is the hot-path for any thread without dead-class state.
    """
    from src.storage.checkpointer import _revive_not_implemented

    plain = [1, 2, {"a": "b"}, [3, 4]]
    out = _revive_not_implemented(plain)
    assert out == plain


def test_revival_handles_unparseable_repr_gracefully():
    """If the envelope's ``repr`` field isn't a parseable constructor
    call (e.g. corrupted storage, a class whose ``__repr__`` uses
    angle brackets, etc.), the walker must NOT raise — it falls back
    to ``{"_repr": ..., "_lc_id": ...}`` so the row still loads.
    The sidebar then shows the thread instead of an empty entry.
    """
    from src.storage.checkpointer import _revive_not_implemented

    envelope = {
        "lc": 1,
        "type": "not_implemented",
        "id": ["some", "weird", "Class"],
        "repr": "ThisIs<NotAValidCall>(at all)",
    }
    out = _revive_not_implemented(envelope)
    assert out == {
        "_repr": "ThisIs<NotAValidCall>(at all)",
        "_lc_id": ["some", "weird", "Class"],
    }


def test_revival_handles_partial_envelope():
    """Defensive: if some envelope keys are missing (e.g. ``repr``
    absent), the walker must not crash. Fallback to the raw dict so
    the row loads.
    """
    from src.storage.checkpointer import _revive_not_implemented

    # Missing repr field — the envelope marker is incomplete.
    incomplete = {"lc": 1, "type": "not_implemented", "id": ["x"]}
    # Should NOT raise — either revive to a fallback dict, or pass
    # through unchanged.
    out = _revive_not_implemented(incomplete)
    assert out == incomplete


def test_revival_is_not_affected_by_repr_substring_match():
    """A normal dict whose fields happen to contain the substring
    ``"Source(index=...)"`` in some value must NOT be mistaken for an
    envelope. Pin the exact marker check (``lc == 1 AND type ==
    "not_implemented"``), not a substring / repr-match heuristic.
    """
    from src.storage.checkpointer import _revive_not_implemented

    # A dict that just happens to have the text "Source(index=1)" in
    # one of its values — must pass through unchanged.
    decoy = {
        "text": "Source(index=1, doc_id='abc') appears in this snippet",
        "extra": "another Source(index=2) mention",
    }
    assert _revive_not_implemented(decoy) == decoy


# ---------------------------------------------------------------------------
# 3. End-to-end: round-trip a state that contains a Pydantic Source
# ---------------------------------------------------------------------------


def test_round_trip_pydantic_source_via_serialize_deserialize():
    """End-to-end: write a state with a Pydantic ``Source``, then read
    it back. The revived state must have a plain dict in
    ``state["sources"][0]`` with all fields intact. This is the
    exact scenario that was failing in production.
    """
    from src.api.schemas import Source
    from src.storage.checkpointer import _serialize_state, _deserialize_state

    original_source = Source(
        index=1,
        chunk_id="",
        doc_id="web-0dac1d973cc8",
        filename="广州百科",
        page=None,
        section=None,
        sheet=None,
        text="广州市",
        score=1.0,
        url="https://baike.baidu.com/item/广州市",
        domain="baike.baidu.com",
        source_kind="web",
    )
    state = {"messages": [], "sources": [original_source]}
    state_json = _serialize_state(state)
    revived = _deserialize_state(state_json)

    sources = revived.get("sources") or []
    assert len(sources) == 1
    assert isinstance(sources[0], dict)
    assert sources[0]["doc_id"] == "web-0dac1d973cc8"
    assert sources[0]["url"] == "https://baike.baidu.com/item/广州市"
    assert sources[0]["source_kind"] == "web"


def test_round_trip_legacy_envelope_via_deserialize():
    """End-to-end recovery: a legacy ``state_json`` containing a
    ``not_implemented`` envelope (the actual shape from production
    log dumps) must round-trip back to a state with a plain dict in
    ``state["sources"][0]``. This is the recovery path that
    retroactively fixes the broken sidebar.
    """
    from src.storage.checkpointer import _deserialize_state

    # Construct the exact envelope shape from the production logs.
    legacy_state = {
        "messages": [],
        "sources": [
            {
                "lc": 1,
                "type": "not_implemented",
                "id": ["src", "api", "schemas", "Source"],
                "repr": (
                    "Source(index=1, chunk_id='', doc_id='web-abc', "
                    "filename='X', page=None, section=None, sheet=None, "
                    "text='hello', score=1.0, url='https://x.com', "
                    "domain='x.com', source_kind='web')"
                ),
            }
        ],
    }
    state_json = json.dumps(legacy_state, ensure_ascii=False)
    revived = _deserialize_state(state_json)

    sources = revived.get("sources") or []
    assert len(sources) == 1
    assert sources[0] == {
        "index": 1,
        "chunk_id": "",
        "doc_id": "web-abc",
        "filename": "X",
        "page": None,
        "section": None,
        "sheet": None,
        "text": "hello",
        "score": 1.0,
        "url": "https://x.com",
        "domain": "x.com",
        "source_kind": "web",
    }


# ---------------------------------------------------------------------------
# 4. AST safety — repr must be parsed safely (no code execution)
# ---------------------------------------------------------------------------


def test_revival_does_not_execute_arbitrary_code():
    """``ast.literal_eval`` only accepts literals — it must NOT eval
    function calls or attribute access. This is a critical safety
    pin: even if someone manages to inject a malicious ``repr``
    string (e.g. by tampering with the SQLite file), the walker
    cannot execute it.
    """
    from src.storage.checkpointer import _revive_not_implemented

    # Each of these would execute code if we used eval() instead of
    # ast.literal_eval(). All must fall back gracefully.
    malicious_reprs = [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd').read()",
        "Source(__class__=None)",  # attr access
        "[x for x in range(10)]",  # comprehension
    ]
    for bad_repr in malicious_reprs:
        envelope = {
            "lc": 1,
            "type": "not_implemented",
            "id": ["x"],
            "repr": bad_repr,
        }
        out = _revive_not_implemented(envelope)
        # Must NOT have executed the repr. Fallback to dict form.
        assert isinstance(out, dict)
        assert "_repr" in out, f"failed gracefully for {bad_repr!r}"


# ---------------------------------------------------------------------------
# 5. Module surface
# ---------------------------------------------------------------------------


def test_revival_and_helpers_exported():
    """Pin the public surface of the helpers so a future refactor that
    renames them breaks this test loudly instead of silently.
    """
    import src.storage.checkpointer as cp

    assert callable(getattr(cp, "_revive_not_implemented", None))
    assert callable(getattr(cp, "_serialize_state", None))
    assert callable(getattr(cp, "_deserialize_state", None))
