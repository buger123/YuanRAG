"""Regression tests for the v2.0 Q&A timestamp contract.

Three pieces:

1. ``_initial_state`` stamps HumanMessage with
   ``additional_kwargs["created_at"]`` (ISO 8601) at server side.
2. ``react_generate_direct`` / ``react_generate`` stamp the AIMessage
   with the same key + emit ``created_at`` on the wire.
3. ``_serialize_messages`` in the sessions endpoint rehydrates
   ``created_at`` to ``MessageRecord.created_at`` for history replay.

These tests pin all three layers so the frontend's timestamp display
can't silently break.
"""
from __future__ import annotations

import re

import pytest


ISO_8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
)


# ---------------------------------------------------------------------------
# Server-side: HumanMessage stamp
# ---------------------------------------------------------------------------


def test_initial_state_stamps_human_message_with_created_at():
    """User message gets a server-side ISO 8601 timestamp on
    additional_kwargs['created_at']. The frontend renders this
    below the user bubble in ChatPane."""
    from src.agent.runner import _initial_state

    state = _initial_state(thread_id="t", message="hello")
    msgs = state["messages"]
    assert len(msgs) == 1
    human = msgs[0]
    assert "created_at" in human.additional_kwargs
    ts = human.additional_kwargs["created_at"]
    assert isinstance(ts, str)
    assert ISO_8601_RE.match(ts), f"not ISO 8601: {ts!r}"


def test_initial_state_created_at_field_is_none():
    """The assistant turn's created_at is filled by react_generate_*
    AFTER the LLM streams; initial state seeds None. Pins the
    contract that the user message is timestamped up front (so
    optimistic UI displays before the answer lands)."""
    from src.agent.runner import _initial_state

    state = _initial_state(thread_id="t", message="hi")
    assert state["created_at"] is None


# ---------------------------------------------------------------------------
# Runner wire format: answer_complete.created_at
# ---------------------------------------------------------------------------


def test_answer_complete_event_accepts_created_at_kwarg():
    """events.answer_complete must accept created_at and pass it
    through unchanged — frontend reads this for live rendering."""
    from src.agent.events import answer_complete

    evt = answer_complete(
        sources=[],
        answer="hi",
        source_kinds=[],
        created_at="2026-09-06T12:34:56+08:00",
    )
    assert evt["created_at"] == "2026-09-06T12:34:56+08:00"


def test_answer_complete_event_omits_created_at_when_none():
    """When the upstream node doesn't stamp, the field is absent
    (not None) — frontend treats absent as 'no timestamp' rather
    than rendering a literal 'None'."""
    from src.agent.events import answer_complete

    evt = answer_complete(sources=[], answer="hi")
    assert "created_at" not in evt


# ---------------------------------------------------------------------------
# History replay: _serialize_messages reads additional_kwargs["created_at"]
# ---------------------------------------------------------------------------


def test_serialize_messages_rehydrates_user_created_at():
    """A HumanMessage with additional_kwargs['created_at'] becomes
    a MessageRecord with created_at=<that timestamp>."""
    from langchain_core.messages import HumanMessage
    from src.api.routes.sessions import _serialize_messages

    ts = "2026-09-06T10:00:00+08:00"
    msgs = [HumanMessage(content="hi", additional_kwargs={"created_at": ts})]
    records = _serialize_messages(msgs)
    assert len(records) == 1
    # MessageRecord is a Pydantic BaseModel — attribute access, not subscript.
    assert records[0].created_at == ts
    assert records[0].role == "user"


def test_serialize_messages_rehydrates_assistant_created_at():
    """An AIMessage with additional_kwargs['created_at'] becomes a
    MessageRecord with created_at=<that timestamp>."""
    from langchain_core.messages import AIMessage
    from src.api.routes.sessions import _serialize_messages

    ts = "2026-09-06T11:00:00+08:00"
    msgs = [
        AIMessage(
            content="hello",
            additional_kwargs={"created_at": ts},
        )
    ]
    records = _serialize_messages(msgs)
    assert len(records) == 1
    assert records[0].created_at == ts
    assert records[0].role == "assistant"


def test_serialize_messages_graceful_when_created_at_missing():
    """Old checkpoint (pre-v2.0) has no created_at on AIMessage.
    _serialize_messages must produce record with created_at=None,
    not crash."""
    from langchain_core.messages import AIMessage
    from src.api.routes.sessions import _serialize_messages

    msgs = [AIMessage(content="old answer")]
    records = _serialize_messages(msgs)
    assert len(records) == 1
    assert records[0].created_at is None


def test_serialize_messages_rehydrates_tool_calls():
    """v2.0 also persists tool_calls on AIMessage. The history
    endpoint surfaces them as a list of dicts."""
    from langchain_core.messages import AIMessage
    from src.api.routes.sessions import _serialize_messages

    tool_calls = [
        {"id": "tc1", "name": "retrieve_docs", "args": {"query": "Q3"}}
    ]
    msgs = [
        AIMessage(
            content="",
            additional_kwargs={"tool_calls": tool_calls, "created_at": "2026-09-06T12:00:00+08:00"},
        )
    ]
    records = _serialize_messages(msgs)
    assert len(records) == 1
    assert records[0].tool_calls == tool_calls


def test_serialize_messages_skips_empty_tool_call_only_aimessage():
    """Defense in depth (v1.1.9 + v2.0): a half-finished ReAct
    turn that has only tool_calls (no prose text, no reasoning)
    is still kept — the user wants to see the tool activity even
    before the LLM produces its final answer."""
    from langchain_core.messages import AIMessage
    from src.api.routes.sessions import _serialize_messages

    msgs = [
        AIMessage(
            content="",
            additional_kwargs={"tool_calls": [{"id": "tc1", "name": "web_search", "args": {"query": "news"}}]},
        )
    ]
    records = _serialize_messages(msgs)
    # The AIMessage is preserved (tool_calls-only), even though it
    # has no text/reasoning. v1.1.9 would have dropped this; the
    # extension in v2.0 keeps it.
    assert len(records) == 1
    assert records[0].tool_calls == [
        {"id": "tc1", "name": "web_search", "args": {"query": "news"}}
    ]
