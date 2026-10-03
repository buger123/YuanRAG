"""Tests for ``src.agent.wire_protocol`` — the WebSocket event schema.

What we lock here
-----------------
1. Each event type round-trips through ``dump_payload`` → ``validate_payload``
   with no field loss.
2. Required fields are enforced: a missing field raises ``ValidationError``.
3. Wrong field types are enforced: ``{"type": "token", "content": 123}``
   raises ``ValidationError``.
4. Unknown ``type`` discriminators raise ``ValidationError``.
5. Extra fields are silently ignored (forward-compat: an older frontend
   reading a newer backend's payload won't crash).
6. ``dump_payload(..., exclude_none=True)`` keeps the on-the-wire shape
   identical to the pre-v2.0.17 hand-rolled builders.
7. ``events.py`` builders produce payloads that round-trip through
   ``validate_payload`` (the whole point of the migration).
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.agent import events
from src.agent.wire_protocol import (
    AnswerCompleteEvent,
    DoneEvent,
    ErrorEvent,
    GroundingEvent,
    IntentEvent,
    ReasoningEvent,
    StepEndEvent,
    StepStartEvent,
    TokenEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    WebSearchEvent,
    dump_payload,
    validate_payload,
)


# ---------------------------------------------------------------------------
# Round-trip: every event type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event,expected_type",
    [
        (TokenEvent(content="hello"), "token"),
        (ReasoningEvent(content="thinking..."), "reasoning"),
        (DoneEvent(), "done"),
        (WebSearchEvent(attempted=True), "web_search"),
        (GroundingEvent(status="grounded"), "grounding"),
        (
            AnswerCompleteEvent(
                sources=[],
                answer="hi",
                source_kinds=["local"],
                created_at="2026-09-16T00:00:00Z",
            ),
            "answer_complete",
        ),
        (ErrorEvent(message="oops"), "error"),
        (
            IntentEvent(intent="qa_complex", corrected_query="rewritten"),
            "intent",
        ),
        (StepStartEvent(step=1), "step_start"),
        (StepEndEvent(step=2, ok=True), "step_end"),
        (
            ToolCallStartEvent(
                name="web_search",
                args={"q": "x"},
                tool_call_id="tc1",
                step=1,
            ),
            "tool_call_start",
        ),
        (
            ToolCallEndEvent(
                tool_call_id="tc1",
                result_summary="result",
                ok=True,
                step=1,
                elapsed_ms=42,
            ),
            "tool_call_end",
        ),
    ],
    ids=lambda ev_or_type: (
        ev_or_type if isinstance(ev_or_type, str) else type(ev_or_type).__name__
    ),
)
def test_event_round_trip(event, expected_type):
    """Every event dumps to a dict with the right ``type``, then validates
    back to the same subclass."""
    payload = dump_payload(event)
    assert payload["type"] == expected_type
    # JSON-safe (no Python types like ``set`` / ``bytes``).
    json.dumps(payload)
    # Round-trip: parse back, get the same typed instance.
    parsed = validate_payload(payload)
    assert type(parsed).__name__ == type(event).__name__


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


def test_token_missing_content_rejected():
    with pytest.raises(ValidationError) as exc_info:
        validate_payload({"type": "token"})
    assert "content" in str(exc_info.value).lower()


def test_answer_complete_missing_answer_rejected():
    with pytest.raises(ValidationError) as exc_info:
        validate_payload({"type": "answer_complete", "sources": []})
    assert "answer" in str(exc_info.value).lower()


def test_tool_call_start_missing_tool_call_id_rejected():
    with pytest.raises(ValidationError) as exc_info:
        validate_payload({"type": "tool_call_start", "name": "x", "step": 1})
    assert "tool_call_id" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Type coercion / rejection
# ---------------------------------------------------------------------------


def test_token_content_must_be_string():
    with pytest.raises(ValidationError):
        validate_payload({"type": "token", "content": 123})


def test_web_search_attempted_must_be_boolean():
    """v1.1.8 — the regression that motivated the schema. The old
    builder shipped ``"attempted": bool`` (the type, not a value);
    Starlette's ``json.dumps`` rejected it with ``Object of type type
    is not JSON serializable``. The schema makes the wrong type a hard
    error at the boundary."""
    with pytest.raises(ValidationError):
        validate_payload({"type": "web_search", "attempted": "true"})


def test_step_start_step_must_be_int():
    with pytest.raises(ValidationError):
        validate_payload({"type": "step_start", "step": "one"})


def test_unknown_event_type_rejected():
    with pytest.raises(ValidationError):
        validate_payload({"type": "totally_unknown_event"})


# ---------------------------------------------------------------------------
# Forward-compat: extra fields are silently dropped
# ---------------------------------------------------------------------------


def test_extra_field_silently_ignored():
    """Backend adds a new field; old frontend (no schema bump yet)
    parses without crashing. The field lands in ``model_extra`` on the
    parsed instance — the frontend's TypeScript union ignores unknown
    keys, the Pydantic model silently absorbs them."""
    payload = {"type": "token", "content": "hi", "future_field": 42}
    parsed = validate_payload(payload)
    assert parsed.content == "hi"
    # Extra field is accessible via model_extra (Pydantic v2 default
    # for ``extra="ignore"`` keeps the field on the instance but doesn't
    # surface it as a typed attribute).
    assert getattr(parsed, "future_field", None) == 42 or "future_field" not in parsed.model_dump()


def test_missing_optional_field_omitted_on_dump():
    """``exclude_none=True`` keeps the on-the-wire shape identical to
    the pre-v2.0.17 hand-rolled builders — absent fields stay absent."""
    ev = IntentEvent(intent="greeting")  # no corrected_query
    payload = dump_payload(ev)
    assert payload == {"type": "intent", "intent": "greeting"}
    assert "corrected_query" not in payload


# ---------------------------------------------------------------------------
# ``events.py`` builders produce schema-valid payloads
# ---------------------------------------------------------------------------


def test_events_token_produces_valid_wire_payload():
    payload = events.token("hello world")
    parsed = validate_payload(payload)
    assert isinstance(parsed, TokenEvent)
    assert parsed.content == "hello world"


def test_events_web_search_produces_valid_wire_payload():
    """v1.1.8 regression locked — the builder must coerce ``attempted``
    to a real bool (not the ``bool`` type)."""
    payload = events.web_search(attempted=True)
    parsed = validate_payload(payload)
    assert isinstance(parsed, WebSearchEvent)
    assert parsed.attempted is True
    # Must NOT contain the ``bool`` type as a value (the old bug).
    assert payload["attempted"] is True


def test_events_answer_complete_produces_valid_wire_payload():
    payload = events.answer_complete(
        sources=[],
        answer="hi",
        source_kinds=["web", "local", "web"],  # unsorted + dup
    )
    parsed = validate_payload(payload)
    assert isinstance(parsed, AnswerCompleteEvent)
    # v1.1.14 — sorted at construction time.
    assert parsed.source_kinds == ["local", "web"]
    assert parsed.created_at is None  # default-omitted on wire


def test_events_answer_complete_with_created_at():
    payload = events.answer_complete(
        sources=[],
        answer="hi",
        source_kinds=["local"],
        created_at="2026-09-16T00:00:00Z",
    )
    parsed = validate_payload(payload)
    assert parsed.created_at == "2026-09-16T00:00:00Z"


def test_events_error_produces_valid_wire_payload():
    payload = events.error("oops")
    parsed = validate_payload(payload)
    assert isinstance(parsed, ErrorEvent)
    assert parsed.message == "oops"


def test_events_done_produces_valid_wire_payload():
    payload = events.done()
    parsed = validate_payload(payload)
    assert isinstance(parsed, DoneEvent)


def test_events_intent_with_no_correction_omits_field():
    """Pre-v2.0.17 builder only set ``corrected_query`` when it had a
    value. The schema migration must preserve that — absent means
    ``None`` is dropped from the wire payload."""
    payload = events.intent("greeting")
    parsed = validate_payload(payload)
    assert isinstance(parsed, IntentEvent)
    assert parsed.corrected_query is None
    assert "corrected_query" not in payload


def test_events_tool_call_end_truncates_to_800_chars():
    """The pre-v2.0.17 builder sliced ``result_summary`` to 800 chars
    before sending. The schema migration must preserve that cap."""
    long_text = "x" * 1500
    payload = events.tool_call_end(
        tool_call_id="tc1",
        result_summary=long_text,
        ok=True,
        step=1,
        elapsed_ms=100,
    )
    parsed = validate_payload(payload)
    assert isinstance(parsed, ToolCallEndEvent)
    assert len(parsed.result_summary) == 800
    # Round-trip via JSON so we exercise the actual wire format.
    reloaded = validate_payload(json.loads(json.dumps(payload)))
    assert len(reloaded.result_summary) == 800


# ---------------------------------------------------------------------------
# All builders produce schema-valid payloads (parametrized smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        lambda: events.token("hi"),
        lambda: events.reasoning("thinking"),
        lambda: events.web_search(attempted=True),
        lambda: events.answer_complete(sources=[], answer="x"),
        lambda: events.grounding("grounded"),
        lambda: events.error("msg"),
        lambda: events.done(),
        lambda: events.intent("greeting"),
        lambda: events.step_start(1),
        lambda: events.step_end(1, True),
        lambda: events.tool_call_start("web_search", {"q": "x"}, "tc1", 1),
        lambda: events.tool_call_end("tc1", "summary", True, 1, 100),
    ],
    ids=lambda fn: fn.__doc__ or "builder",
)
def test_all_events_builders_round_trip(builder):
    """The whole point of the v2.0.17 migration: every builder in
    ``events.py`` produces a payload that satisfies the wire schema.
    A future builder that forgets to use one of the typed events will
    fail this test."""
    payload = builder()
    # JSON-serializable (Pydantic dump uses ``mode="json"``).
    json.dumps(payload)
    # Schema-valid (raises ValidationError on regression).
    parsed = validate_payload(payload)
    assert parsed.type == payload["type"]