"""WebSocket event payload builders.

Each builder constructs a typed :class:`WireEvent` subclass and dumps
it to a wire-format dict via :func:`dump_payload`. The dict shape
on the wire is unchanged from the v2.0.16 hand-rolled builders — only
the construction path now goes through Pydantic, so type errors
(wrong field type, missing required field) are caught at the moment
of construction instead of leaking through to the frontend.

v2.0.17 — wire-protocol Pydantic migration. See
``src/agent/wire_protocol.py`` for the canonical schema and the
schema guard (the runner's send boundary validates every payload
via :func:`validate_payload` before writing the frame).

v2.0 — ReAct rewrite. Five new event types were added:

  * ``{"type": "intent", "intent": ..., "corrected_query": ...}``
    Emitted when ``intent_analysis`` finishes classification.

  * ``{"type": "step_start", "step": N}`` /
    ``{"type": "step_end", "step": N, "ok": bool}``
    One pair per ``react_agent`` invocation.

  * ``{"type": "tool_call_start", "name": ..., "args": ...,
     "tool_call_id": ..., "step": N}``
    Emitted when ``react_agent`` produces an AIMessage with
    ``tool_calls``.

  * ``{"type": "tool_call_end", "tool_call_id": ...,
     "result_summary": ..., "ok": bool, "step": N, "elapsed_ms": M}``
    Emitted when the ``ToolNode`` produces a matching ``ToolMessage``.

  * ``answer_complete.created_at`` (ISO 8601) on the existing
    ``answer_complete`` event.

Existing events (token / reasoning / web_search / grounding /
error / done) are unchanged.

Why we keep emitting ``web_search`` even though it's tool-ified
---------------------------------------------------------------
The frontend's ``webSearching`` flag (the "🔎 正在联网搜索…" pill)
is keyed off this event in App.tsx. We can't change the frontend's
signal without breaking the UX, so the runner fires it on every
``web_search`` tool invocation — keeping the pill in sync with the
tool card.
"""
from __future__ import annotations

from typing import Any, Optional, Union

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
    VerificationResultEvent,
    WebSearchEvent,
    dump_payload,
)
from src.api.schemas import Source as SourceWire


def token(content: str) -> dict[str, Any]:
    return dump_payload(TokenEvent(content=content))


def reasoning(content: str) -> dict[str, Any]:
    return dump_payload(ReasoningEvent(content=content))


def web_search(attempted: bool) -> dict[str, Any]:
    """v1.1.8 — return ``bool(attempted)`` (the value), NOT ``bool``
    (the class itself). The previous line shipped ``"attempted": bool``,
    which put the Python ``bool`` type into the WS payload as the value
    — Starlette's ``json.dumps`` rejects it with ``Object of type type
    is not JSON serializable``. The Pydantic ``WebSearchEvent.attempted:
    bool`` field makes the type a hard error at construction."""
    return dump_payload(WebSearchEvent(attempted=bool(attempted)))


def answer_complete(
    sources: list[Union[SourceWire, dict]],
    answer: str,
    *,
    source_kinds: Optional[list[str]] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """``source_kinds`` (v1.1.14) drives the banner; ``created_at``
    (v2.0) is the server-side ISO 8601 timestamp on the assistant
    turn. Both default to None / empty for backward compatibility.

    v2.0.7 SoT-1: ``sources`` may be a list of Pydantic ``SourceWire``
    instances or raw dicts (history replay still yields dicts from
    the SqliteSaver). Pydantic sources are dumped via ``model_dump``
    so the JSON payload shape is unchanged for either input form.

    v2.0.17 — sources are normalized into a list of ``Source`` (the
    wire-protocol Source) before construction, so
    ``AnswerCompleteEvent.sources`` always validates.
    """
    if source_kinds is None:
        source_kinds = []
    wire_sources: list[SourceWire] = []
    for s in sources:
        if isinstance(s, SourceWire):
            wire_sources.append(s)
        else:
            # Defensive: history replay may pass dicts (the SqliteSaver
            # round-trip — see ``MessageRecord``). Wrap into the wire
            # ``Source`` model so the list validates.
            wire_sources.append(SourceWire.model_validate(s))
    return dump_payload(
        AnswerCompleteEvent(
            sources=wire_sources,
            answer=answer,
            source_kinds=sorted(set(source_kinds)),
            created_at=created_at,
        )
    )


def grounding(status: str) -> dict[str, Any]:
    return dump_payload(GroundingEvent(status=status))


def verification_result(
    *,
    consistent: bool = True,
    mismatches_count: int = 0,
    regenerated: bool = False,
    fallback_refusal: bool = False,
    skipped: bool = False,
    reason: Optional[str] = None,
    mismatches: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """v2.0.29.7 (Phase 6) — second-pass verification wire event.

    Mirrors :class:`src.agent.wire_protocol.VerificationResultEvent`.
    The FSM yields ``("verification_result", payload_dict)`` and
    the runner maps that to the WS frame via this factory.

    Backward-compat: omitted fields default to "no problem" (consistent
    = True, count = 0, etc.) so the Pydantic ``exclude_none=True`` dump
    matches the shape the frontend expects to see when verification
    was a no-op (skipped path).
    """
    return dump_payload(
        VerificationResultEvent(
            consistent=bool(consistent),
            mismatches_count=int(mismatches_count),
            regenerated=bool(regenerated),
            fallback_refusal=bool(fallback_refusal),
            skipped=bool(skipped),
            reason=reason,
            mismatches=list(mismatches) if mismatches else [],
        )
    )


def error(message: str, **extra: Any) -> dict[str, Any]:
    """Error event. ``extra`` kwargs are ignored by the wire schema
    (see :class:`wire_protocol.ErrorEvent`) — only ``message`` is on
    the contract. The runner's redactor (``redact_exception``) injects
    a ``trace_id`` at the send boundary; downstream consumers read
    it via ``ErrorEvent.model_extra`` if needed."""
    return dump_payload(ErrorEvent(message=message))


def done() -> dict[str, Any]:
    return dump_payload(DoneEvent())


# ---- v2.0 new builders ---------------------------------------------------


def intent(intent: str, corrected_query: Optional[str] = None) -> dict[str, Any]:
    """Pre-ReAct classification result.

    ``intent`` is one of ``"greeting"`` / ``"summary"`` /
    ``"simple_fact"`` / ``"qa_complex"``. ``corrected_query`` is the
    cheap-model rewrite when the LLM detected typos; ``None`` when no
    rewrite was needed (the frontend should treat its absence as "no
    change").
    """
    return dump_payload(
        IntentEvent(intent=intent, corrected_query=corrected_query)
    )


def step_start(step: int) -> dict[str, Any]:
    """A new ReAct ``react_agent`` step is starting.

    ``step`` is 1-based (the first LLM call of a qa_complex turn is
    step 1, the first tool-result re-entry is step 2, etc.).
    """
    return dump_payload(StepStartEvent(step=int(step)))


def step_end(step: int, ok: bool = True) -> dict[str, Any]:
    """The ReAct step finished (``react_agent`` returned).

    ``ok=False`` if the LLM call raised — the runner follows up
    with an ``error`` event in that case.
    """
    return dump_payload(StepEndEvent(step=int(step), ok=bool(ok)))


def tool_call_start(
    name: str,
    args: dict,
    tool_call_id: str,
    step: int,
) -> dict[str, Any]:
    """A tool call is being dispatched.

    Emitted by the runner the moment ``react_agent`` returns an
    AIMessage with ``tool_calls``. The matching ``tool_call_end``
    fires when the ToolNode produces its ``ToolMessage``. Both
    events share ``tool_call_id`` so the frontend can pair them.

    ``args`` is the LLM's raw call payload (LangChain guarantees
    this by the time ``ToolNode`` runs — but the runner reads it
    off the AIMessage BEFORE ToolNode runs, so the args here are
    the LLM's raw call payload, not the validated version. For
    tool_call_end the runner re-reads the validated args off the
    ToolMessage metadata.)
    """
    return dump_payload(
        ToolCallStartEvent(
            name=str(name),
            args=dict(args) if isinstance(args, dict) else {},
            tool_call_id=str(tool_call_id),
            step=int(step),
        )
    )


def tool_call_end(
    tool_call_id: str,
    result_summary: str,
    ok: bool,
    step: int,
    elapsed_ms: int,
) -> dict[str, Any]:
    """The tool finished.

    ``result_summary`` is a preview of the ToolMessage content —
    truncated to 800 chars to keep the WS frame small. Full content
    is persisted on the ToolMessage in the checkpointer; history
    replay can surface it via the ToolMessage.

    ``ok=False`` means the tool raised (Pydantic validation,
    embedder crash, etc.). The frontend flips the card to red and
    shows the error string.
    """
    return dump_payload(
        ToolCallEndEvent(
            tool_call_id=str(tool_call_id),
            result_summary=str(result_summary or "")[:800],
            ok=bool(ok),
            step=int(step),
            elapsed_ms=int(elapsed_ms),
        )
    )


__all__ = [
    "token",
    "reasoning",
    "web_search",
    "answer_complete",
    "grounding",
    "error",
    "done",
    # v2.0 new
    "intent",
    "step_start",
    "step_end",
    "tool_call_start",
    "tool_call_end",
    # v2.0.29.7 (Phase 6) — second-pass verification.
    "verification_result",
]