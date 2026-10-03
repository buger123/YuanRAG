"""WebSocket wire-event schema — single source of truth for the
backend ↔ frontend contract.

v2.0.17 — every event the backend emits over the WebSocket must
satisfy one of the discriminated-union members defined here. The
runner validates each payload at the send boundary so a regression
that produces, say, ``{"type": "token", "content": 123}`` (number
instead of string) fails LOUDLY inside the server (logged +
``error`` event emitted) rather than arriving at the frontend as a
malformed frame and silently corrupting state.

Why a discriminated union (not a single BaseModel with optional
fields)?

  * Each event has its own field set. ``answer_complete.sources`` is
    a list of ``Source`` objects; ``tool_call_start.args`` is a dict;
    ``done`` has zero fields. A single-model approach with
    everything-optional collapses into ``dict[str, Any]`` —
    equivalent to no schema at all.
  * Pydantic v2 ``model_validate(payload)`` on a discriminated
    union picks the right branch in one shot. The runner's
    pre-validation cost is one match-and-construct per event
    (microseconds), negligible next to the JSON serialization cost.
  * The 12 ``UnionMember`` classes make it easy for a reader to see
    the wire shape without grepping ``events.py``.

Why ``model_config = ConfigDict(extra="ignore")`` (not ``"forbid"``)

  * The frontend evolves faster than the backend. If the backend
    adds a new optional field, the frontend's old (un-updated)
    ``AgentEvent`` union still parses the payload (it ignores the
    unknown field) — no broken replay on a user's stale tab.
  * ``"forbid"`` would reject any new field, turning a benign
    backend change into a wire breakage on the next deploy. The
    trade-off is: extra fields are silently dropped (the backend
    is canonical, not the frontend), missing required fields are
    rejected (the wire must carry them).

Discriminator

  * All 12 subclasses set ``type: Literal["<name>"]`` so Pydantic
    routes the validation by the ``type`` field. A payload with
    an unknown ``type`` raises ``ValidationError`` at the send
    boundary — that's the desired loud-failure signal.

Dropping ``None`` fields on dump

  * ``model_dump(mode="json", exclude_none=True)`` keeps the
    on-the-wire shape identical to the pre-v2.0.17 hand-rolled
    dict builders (those builders only set a field when it had a
    value). The frontend's TypeScript discriminated union can
    therefore treat each variant as having exactly the right set
    of fields — ``answer_complete.corrected_query`` is just absent
    if no correction was made, not ``null``.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from src.api.schemas import Source


class _WireBase(BaseModel):
    """Common Pydantic config for every wire event.

    ``extra="ignore"`` (see module docstring) and ``populate_by_name``
    so callers can construct events via either kwarg or alias.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------------------
# Token / reasoning / done
# ---------------------------------------------------------------------------


class TokenEvent(_WireBase):
    """An incremental content chunk from the LLM.

    Streamed many times per assistant turn (Anthropic ``message_delta``
    events). The frontend appends ``content`` to the in-flight message
    bubble.
    """

    type: Literal["token"] = "token"
    content: StrictStr


class ReasoningEvent(_WireBase):
    """An incremental reasoning chunk (Anthropic extended thinking).

    Streamed alongside ``TokenEvent`` on the same channel. The
    frontend renders it inside a collapsible "💭 thinking" drawer.
    """

    type: Literal["reasoning"] = "reasoning"
    content: StrictStr


class DoneEvent(_WireBase):
    """Final event of every turn (success or error).

    Emitted by the runner after ``react_generate`` (or its greeting /
    summary fast-paths) has flushed the canonical ``answer_complete``
    (or an ``error`` event). The frontend uses this as the end-of-turn
    marker that closes the in-flight state and unfreezes the input.
    """

    type: Literal["done"] = "done"


# ---------------------------------------------------------------------------
# Search / grounding
# ---------------------------------------------------------------------------


class WebSearchEvent(_WireBase):
    """The agent is / was searching the web.

    Emitted once per ``web_search`` tool invocation. Drives the
    "🔎 正在联网搜索…" pill in the UI — see
    ``src/frontend/src/chat-utils.tsx`` for the consumer.

    v1.1.8 — ``attempted`` is a real boolean (not the ``bool`` type).
    The previous hand-rolled builder shipped ``"attempted": bool`` —
    Starlette's ``json.dumps`` rejected it with ``Object of type type
    is not JSON serializable``. The Pydantic type stops the regression
    at construction time.
    """

    type: Literal["web_search"] = "web_search"
    attempted: StrictBool


class GroundingEvent(_WireBase):
    """Result of the post-answer grounding check.

    ``status`` is one of:

      * ``"grounded"`` — the answer's citations are supported by the
        retrieved documents. Frontend renders the answer normally.
      * ``"ungrounded"`` — at least one citation is unsupported. The
        answer body should display "⚠️ 部分内容可能与文档不符".
      * ``"skipped"`` — no documents to check against (greeting /
        direct / summary with no docs). The banner is hidden.
    """

    type: Literal["grounding"] = "grounding"
    status: Literal["grounded", "ungrounded", "skipped"]


class VerificationResultEvent(_WireBase):
    """v2.0.29.7 (Phase 6) — second-pass verification outcome.

    Emitted by :class:`src.agent.nodes.verify_answer.verify_answer_node`
    after the FSM routes through the verification layer. The
    frontend can render a ✓/✗ icon based on ``consistent``;
    ``regenerated`` signals the user that the answer was rewritten
    by the verification pass. ``fallback_refusal=True`` means the
    answer was REPLACED with the Phase 1
    ``REFUSAL_TEMPLATES["low_relevance"]`` template (consistent=false,
    regenerated=false). ``skipped=True`` is the zero-cost path —
    the rule engine + verifier LLM did not run because the answer
    had no extractable fields.

    PR-1 ships the wire event only — the frontend currently ignores
    ``verification_result`` (no UI). PR-2 / later will add ✓/✗
    rendering. Backward-compat: omitting the field is safe
    (frontend's existing discriminated union ignores unknown
    types); extra fields are ignored by the backend's
    ``extra="ignore"`` config.
    """

    type: Literal["verification_result"] = "verification_result"
    consistent: StrictBool = True
    mismatches_count: StrictInt = 0
    regenerated: StrictBool = False
    fallback_refusal: StrictBool = False
    skipped: StrictBool = False
    reason: StrictStr | None = None
    mismatches: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Answer complete
# ---------------------------------------------------------------------------


class AnswerCompleteEvent(_WireBase):
    """The canonical, post-stream answer payload.

    Emitted once per assistant turn after the stream has finished.
    The frontend uses ``answer`` as the source-of-truth body (it
    may overwrite the accumulated streaming text — see the
    v2.0.13 canonical-whitespace fix in
    ``src/frontend/src/App.tsx``).

    ``source_kinds`` (v1.1.14) drives the banner. ``created_at``
    (v2.0) is the server-side ISO 8601 timestamp on the assistant
    turn — rehydrated by history replay from
    ``additional_kwargs["created_at"]`` on the persisted AIMessage.

    v2.0.7 SoT-1: ``sources`` carries Pydantic ``Source`` instances
    (validated by ``Source.model_dump`` on dump). History replay may
    also pass raw dicts (the SqliteSaver round-trip — see
    ``src/api/schemas.py``'s ``MessageRecord``). Both forms are
    accepted here.
    """

    type: Literal["answer_complete"] = "answer_complete"
    sources: list[Source] = Field(default_factory=list)
    answer: StrictStr
    # v1.1.14 — banner. Sorted+deduped at construction time so the
    # frontend's ``source_kinds.includes("web")`` is order-stable.
    source_kinds: list[str] = Field(default_factory=list)
    # v2.0 — server-side ISO 8601 timestamp.
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorEvent(_WireBase):
    """A user-facing error.

    ``message`` is the human-readable text (the UI shows it in the
    in-flight message bubble as the assistant's last word).
    Additional fields are open (``trace_id`` from
    ``redact_exception``, custom error codes, etc.). ``extra="ignore"``
    on the base config means unknown fields round-trip through
    validation silently — the backend owns the wire shape, not the
    consumer.
    """

    type: Literal["error"] = "error"
    message: StrictStr


# ---------------------------------------------------------------------------
# v2.0 ReAct events
# ---------------------------------------------------------------------------


class IntentEvent(_WireBase):
    """Pre-ReAct intent classification result.

    Emitted exactly once per turn, right after ``intent_analysis``
    returns. ``intent`` is one of ``"greeting"`` / ``"summary"`` /
    ``"simple_fact"`` / ``"qa_complex"``. ``corrected_query`` is
    the cheap-model typo rewrite (omitted entirely when no rewrite
    was needed — the frontend should treat its absence as "no
    correction").

    See ``src/agent/nodes/intent_analysis.py`` for the classifier.
    """

    type: Literal["intent"] = "intent"
    intent: Literal["greeting", "summary", "simple_fact", "qa_complex"]
    corrected_query: str | None = None


class StepStartEvent(_WireBase):
    """A ReAct ``react_agent`` step is starting.

    ``step`` is 1-based: the first LLM call of a qa_complex turn is
    step 1, the first tool-result re-entry is step 2, etc. Drives
    the frontend's "Step N" pill.
    """

    type: Literal["step_start"] = "step_start"
    step: StrictInt


class StepEndEvent(_WireBase):
    """The ReAct step finished.

    ``ok=False`` if the LLM call raised — the runner follows up
    with an ``error`` event in that case.
    """

    type: Literal["step_end"] = "step_end"
    step: StrictInt
    ok: StrictBool = True


class ToolCallStartEvent(_WireBase):
    """A tool call is being dispatched.

    Emitted by the runner the moment ``react_agent`` returns an
    AIMessage with ``tool_calls``. The matching ``tool_call_end``
    fires when the ToolNode produces its ``ToolMessage``. Both
    events share ``tool_call_id`` so the frontend can pair them.

    ``args`` is the LLM's RAW call payload (NOT the validated
    Pydantic args — that's surfaced on the matching ``tool_call_end``
    once the ToolNode runs).
    """

    type: Literal["tool_call_start"] = "tool_call_start"
    name: StrictStr
    args: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: StrictStr
    step: StrictInt


class ToolCallEndEvent(_WireBase):
    """The tool finished.

    ``result_summary`` is a preview of the ToolMessage content —
    truncated to 800 chars to keep the WS frame small. Full content
    is persisted on the ToolMessage in the checkpointer; history
    replay can surface it via the ToolMessage.

    ``ok=False`` means the tool raised (Pydantic validation,
    embedder crash, etc.). The frontend flips the card to red and
    shows the error string.
    """

    type: Literal["tool_call_end"] = "tool_call_end"
    tool_call_id: StrictStr
    # Truncated at construction time (see ``events.tool_call_end``).
    result_summary: StrictStr
    ok: StrictBool
    step: StrictInt
    elapsed_ms: StrictInt


# ---------------------------------------------------------------------------
# Discriminated union + helpers
# ---------------------------------------------------------------------------


# Build the union + discriminator. We use ``TypeAdapter`` (not an
# ``Annotated[Union[...], Field(discriminator=...)]`` type alias)
# because Pydantic v2's union dispatch needs an actual validator
# instance — an annotated alias has no ``model_validate`` to call.
WireEvent = Annotated[
    Union[
        TokenEvent,
        ReasoningEvent,
        DoneEvent,
        WebSearchEvent,
        GroundingEvent,
        # v2.0.29.7 (Phase 6) — second-pass verification outcome.
        VerificationResultEvent,
        AnswerCompleteEvent,
        ErrorEvent,
        IntentEvent,
        StepStartEvent,
        StepEndEvent,
        ToolCallStartEvent,
        ToolCallEndEvent,
    ],
    Field(discriminator="type"),
]


def dump_payload(event) -> dict[str, Any]:
    """Convert a WireEvent instance into the wire-format dict.

    Uses ``mode="json"`` so Pydantic converts non-JSON-native types
    (datetimes, sets, etc.) to their JSON equivalents at dump time.
    ``exclude_none=True`` keeps the on-the-wire shape identical to
    the pre-v2.0.17 hand-rolled dict builders — absent fields stay
    absent (rather than showing up as ``null``).

    Returns a ``dict`` ready for ``WebSocket.send_json``.
    """
    return event.model_dump(mode="json", exclude_none=True)


def validate_payload(payload: dict[str, Any]):
    """Validate a wire-format dict and return the typed event.

    Raises ``pydantic.ValidationError`` on any contract violation
    (missing required field, wrong type, unknown ``type`` discriminator,
    etc.). The runner's send boundary catches this and emits an
    ``error`` event in place of the bad payload.
    """
    from pydantic import TypeAdapter

    # Pydantic v2 discriminated-union parse via TypeAdapter. The
    # ``type`` field is the discriminator; an unknown ``type`` raises
    # immediately.
    return TypeAdapter(WireEvent).validate_python(payload)


__all__ = [
    "WireEvent",
    "dump_payload",
    "validate_payload",
    # Event classes (re-exported for callers that want to construct
    # events directly without going through ``events.py`` builders).
    "TokenEvent",
    "ReasoningEvent",
    "DoneEvent",
    "WebSearchEvent",
    "GroundingEvent",
    # v2.0.29.7 (Phase 6) — verification_result wire event.
    "VerificationResultEvent",
    "AnswerCompleteEvent",
    "ErrorEvent",
    "IntentEvent",
    "StepStartEvent",
    "StepEndEvent",
    "ToolCallStartEvent",
    "ToolCallEndEvent",
]