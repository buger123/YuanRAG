"""Regression tests for v2.0.24 (Item 8 P0-F4) — historical conversations
must show ALL ReAct rounds, not just the final answer.

Root cause
----------
Intermediate ReAct AIMessages (one per round of the
react_agent ↔ tools loop) carry their reasoning INSIDE
``AIMessage.content`` as Anthropic ``{"type": "thinking",
"thinking": "..."}`` blocks and their tool calls via the langchain
built-in ``AIMessage.tool_calls`` attribute. They do NOT have:

* a ``text`` content block
* an ``additional_kwargs["reasoning"]`` entry (only stamped by
  ``react_generate`` on the FINAL answer)
* an ``additional_kwargs["tool_calls"]`` entry (the per-round log
  ``_build_tool_call_log`` produces is also stamped only on the
  final answer)

The previous ``_serialize_messages`` filter at the bottom of the
function dropped any AIMessage with all three of those empty:

    if not text and not reasoning and not tool_calls:
        continue

That dropped every intermediate round on history replay, so the
user reopened a multi-round thread and only saw their question +
the final answer. Symptom reported by the user as "每个历史对话中
只能看到最近一次问答, 之前的消息不见了".

These tests pin the new contract:

* The serializer synthesizes a minimal ``tool_calls`` card list from
  ``AIMessage.tool_calls`` (langchain native attr) for intermediate
  AIMessages that lack ``additional_kwargs["tool_calls"]``.
* The final AIMessage's rich ``additional_kwargs["tool_calls"]`` log
  wins (it has full result previews + timing data).
* SystemMessage / ToolMessage filtering is unchanged.
* AIMessages with no text AND no reasoning AND no tool_calls still
  get dropped (genuinely empty stub defense).

v2.0.28.15 — reasoning field on intermediate AIMessages is now
``None`` (the ``_extract_thinking_text`` fallback that lifted
content-block thinking onto the wire was removed in
``sessions.py``). Intermediate AIMessages still SURVIVE in the
wire because their tool_calls card list is non-empty, but their
``reasoning`` field carries no content-block thinking text — the
LLM's planning-step meta-commentary no longer leaks into the
``<ThinkingDrawer>`` on history replay. The Anthropic
extended-thinking continuity is unaffected because
``_sanitize_history_for_generate`` reads ``AIMessage.content``
directly, not the wire payload.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.api.routes.sessions import _serialize_messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ai(
    content,
    *,
    additional_kwargs=None,
    tool_calls=None,
    id_="lc_run--test",
):
    """Build an AIMessage with optional content list, additional_kwargs,
    and native ``tool_calls`` (langchain built-in attribute)."""
    kwargs = additional_kwargs if additional_kwargs is not None else {}
    msg = AIMessage(
        content=content,
        additional_kwargs=kwargs,
        id=id_,
    )
    # langchain_core.messages.AIMessage stores ``tool_calls`` as a
    # private attribute; setting it via the constructor doesn't work,
    # so we set it on the instance after construction.
    if tool_calls is not None:
        msg.tool_calls = tool_calls
    return msg


def _thinking(text):
    """Build a single Anthropic thinking block."""
    return {"type": "thinking", "thinking": text}


def _text(text):
    return {"type": "text", "text": text}


def _tool_use(name, call_id, args=None):
    return {
        "type": "tool_use",
        "name": name,
        "id": call_id,
        "input": args or {},
    }


# ---------------------------------------------------------------------------
# _extract_thinking_text (the helper that fixes the bug)
# ---------------------------------------------------------------------------


class TestExtractThinkingText:
    """Pin the helper that surfaces Anthropic ``thinking`` blocks
    on intermediate ReAct rounds."""

    def test_string_content_returns_empty(self):
        """HumanMessage-style string content — no thinking to extract."""
        from src.api.routes.sessions import _extract_thinking_text
        assert _extract_thinking_text(HumanMessage(content="hi")) == ""

    def test_empty_list_returns_empty(self):
        from src.api.routes.sessions import _extract_thinking_text
        assert _extract_thinking_text(_ai(content=[])) == ""

    def test_only_text_blocks_returns_empty(self):
        """We don't count text as thinking — that's the ``text`` field."""
        from src.api.routes.sessions import _extract_thinking_text
        msg = _ai(content=[_text("hello")])
        assert _extract_thinking_text(msg) == ""

    def test_single_thinking_block(self):
        from src.api.routes.sessions import _extract_thinking_text
        msg = _ai(content=[_thinking("I should call retrieve_docs first.")])
        assert _extract_thinking_text(msg) == "I should call retrieve_docs first."

    def test_multiple_thinking_blocks_joined_with_blank_line(self):
        """The ThinkingDrawer renders the string verbatim, so multiple
        blocks must read as separate paragraphs."""
        from src.api.routes.sessions import _extract_thinking_text
        msg = _ai(content=[
            _thinking("Round 1 thought."),
            _tool_use("retrieve_docs", "c1"),
            _thinking("Round 2 thought."),
        ])
        out = _extract_thinking_text(msg)
        assert out == "Round 1 thought.\n\nRound 2 thought."

    def test_tool_use_only_returns_empty(self):
        """A tool_use-only AIMessage (no thinking block) returns empty
        — the serializer relies on tool_calls to surface it instead."""
        from src.api.routes.sessions import _extract_thinking_text
        msg = _ai(content=[_tool_use("web_search", "c1", {"query": "x"})])
        assert _extract_thinking_text(msg) == ""

    def test_thinking_with_empty_string_is_skipped(self):
        """Defensive: a thinking block with empty string is skipped
        rather than producing a leading blank line."""
        from src.api.routes.sessions import _extract_thinking_text
        msg = _ai(content=[_thinking(""), _thinking("real thought")])
        assert _extract_thinking_text(msg) == "real thought"

    def test_thinking_with_non_string_is_skipped(self):
        """Defensive: malformed block where ``thinking`` isn't a string
        (shouldn't happen but graceful)."""
        from src.api.routes.sessions import _extract_thinking_text
        msg = _ai(content=[{"type": "thinking", "thinking": 123}])
        assert _extract_thinking_text(msg) == ""


# ---------------------------------------------------------------------------
# _serialize_messages — intermediate ReAct AIMessage survival
# ---------------------------------------------------------------------------


class TestSerializeIntermediateReact:
    """The bug fix: intermediate AIMessages (tool_use only or
    thinking + tool_use) must survive the serializer."""

    def test_thinking_plus_tool_use_ai_message_survives(self):
        """The exact shape from production thread eb45c632 [1]:
        content = [thinking, tool_use, tool_use], no additional_kwargs."""
        msgs = [
            HumanMessage(content="广州今天天气怎么样?"),
            _ai(
                content=[
                    _thinking("I need to get current time then search weather."),
                    _tool_use("get_current_time", "c1", {"timezone": "Asia/Shanghai"}),
                    _tool_use("web_search", "c2", {"query": "广州天气"}),
                ],
                tool_calls=[
                    {"id": "c1", "name": "get_current_time", "args": {"timezone": "Asia/Shanghai"}},
                    {"id": "c2", "name": "web_search", "args": {"query": "广州天气"}},
                ],
            ),
        ]
        records = _serialize_messages(msgs)
        assert len(records) == 2
        assert records[0].role == "user"
        assert records[1].role == "assistant"
        # v2.0.28.15 — content-block reasoning lift was removed.
        # The AIMessage still SURVIVES in the wire (tool_calls is
        # non-empty → passes the empty-message filter), so reload
        # still shows the get_current_time + web_search tool cards.
        # But the reasoning field is None — the LLM's planning-step
        # meta-commentary ("I need to get current time then search
        # weather.") no longer leaks onto the wire → no longer
        # surfaces into <ThinkingDrawer>. See
        # tests/test_v2_0_28_15_bugfixes.py::TestSerializeMessagesNoReasoningFromContentBlocks
        # for the explicit pin.
        assert records[1].reasoning is None, (
            f"intermediate round reasoning should be None "
            f"(v2.0.28.15 dropped content-block lift); got "
            f"{records[1].reasoning!r}"
        )
        # Tool calls surfaced from native .tool_calls attribute
        assert records[1].tool_calls is not None
        assert len(records[1].tool_calls) == 2
        assert records[1].tool_calls[0]["name"] == "get_current_time"
        assert records[1].tool_calls[1]["name"] == "web_search"
        # content is empty (the AIMessage had no text block)
        assert records[1].content == ""

    def test_multiple_intermediate_rounds_all_survive(self):
        """Full multi-round ReAct trace — every intermediate AIMessage
        appears, plus the final answer."""
        msgs = [
            HumanMessage(content="summarize this doc"),
            _ai(
                content=[
                    _thinking("Round 1: load the doc."),
                    _tool_use("retrieve_docs", "r1", {"query": "summary"}),
                ],
                tool_calls=[
                    {"id": "r1", "name": "retrieve_docs", "args": {"query": "summary"}},
                ],
            ),
            ToolMessage(content="doc text...", tool_call_id="r1"),
            _ai(
                content=[
                    _thinking("Round 2: also check web for context."),
                    _tool_use("web_search", "w1", {"query": "context"}),
                ],
                tool_calls=[
                    {"id": "w1", "name": "web_search", "args": {"query": "context"}},
                ],
            ),
            ToolMessage(content="web results...", tool_call_id="w1"),
            _ai(
                content=[_text("Here is the summary...")],
                additional_kwargs={
                    "sources": [],
                    "reasoning": "Final synthesis.",
                    "source_kinds": ["doc"],
                    "created_at": "2026-09-19T00:00:00+00:00",
                    "tool_calls": [
                        {"id": "r1", "name": "retrieve_docs", "args": {}, "result": "doc text...", "ok": True},
                        {"id": "w1", "name": "web_search", "args": {}, "result": "web results...", "ok": True},
                    ],
                },
            ),
        ]
        records = _serialize_messages(msgs)
        # 2 user/assistant (input + final) + 2 intermediate rounds
        # (ToolMessages get dropped, that's the v2.0.3 contract)
        assert len(records) == 4
        roles = [r.role for r in records]
        assert roles == ["user", "assistant", "assistant", "assistant"]
        # v2.0.28.15 — intermediate round reasoning is None
        # (the `_extract_thinking_text` fallback that lifted
        # content-block thinking onto the wire was removed in
        # sessions.py). The final AIMessage keeps its
        # `additional_kwargs["reasoning"]` stamp ("Final
        # synthesis.") which goes through unchanged via the
        # `kwargs.get("reasoning")` branch (line 375-377 of
        # sessions.py). The intermediate AIMessages still SURVIVE
        # in the wire — their tool_calls card list is non-empty
        # so they pass the `if not text and not reasoning and
        # not tool_calls: continue` filter — but their reasoning
        # field is None.
        assert records[1].reasoning is None, (
            f"intermediate round 1 reasoning should be None "
            f"(v2.0.28.15 dropped content-block lift); got "
            f"{records[1].reasoning!r}"
        )
        assert records[2].reasoning is None, (
            f"intermediate round 2 reasoning should be None "
            f"(v2.0.28.15 dropped content-block lift); got "
            f"{records[2].reasoning!r}"
        )
        # Final synthesis AIMessage: reasoning stamp survives
        # because it lives in additional_kwargs (not content
        # blocks). The runner still stamps it via the synthesis
        # path; the v2.0.28.15 fix removed the explicit
        # `additional_kwargs["reasoning"] = reasoning_text`
        # line in react_generate.py + fsm.py so NEW synthesis
        # AIMessages won't carry it, but for THIS test the final
        # AIMessage's additional_kwargs is hand-set with
        # "reasoning": "Final synthesis." — so the field IS
        # preserved here. This is intentional: it lets us pin
        # the additional_kwargs branch (which still works) and
        # pin the new contract that NEW runs will not stamp
        # reasoning on synthesis AIMessage (covered in
        # tests/test_v2_0_28_15_bugfixes.py).
        assert records[3].reasoning == "Final synthesis.", (
            f"final AIMessage additional_kwargs['reasoning'] "
            f"should be preserved by serializer; got "
            f"{records[3].reasoning!r}"
        )
        # Final message uses the rich additional_kwargs log (full
        # result previews + timing), NOT the synthesized native attr
        # cards (which would lack result preview).
        assert len(records[3].tool_calls) == 2
        assert records[3].tool_calls[0].get("result") == "doc text..."

    def test_final_ai_uses_rich_additional_kwargs_tool_log(self):
        """The runner stamps a rich per-call log on the final answer
        via ``additional_kwargs["tool_calls"]`` (with result preview +
        ok + timing). The synthesizer must NOT replace it with the
        minimal native attr cards."""
        rich_log = [
            {"id": "x", "name": "retrieve_docs", "args": {}, "result": "preview", "ok": True, "elapsed_ms": 42, "step": 0},
        ]
        msgs = [
            HumanMessage(content="q"),
            _ai(
                content=[_text("answer")],
                additional_kwargs={"tool_calls": rich_log, "reasoning": "final"},
                tool_calls=[{"id": "x", "name": "retrieve_docs", "args": {}}],
            ),
        ]
        records = _serialize_messages(msgs)
        assert records[1].tool_calls == rich_log  # exact identity — the rich log won

    def test_ai_with_text_and_tool_calls_synthesizes_cards(self):
        """Mixed content [text, tool_use] — text goes into content,
        native tool_calls goes into tool_calls cards."""
        msgs = [
            HumanMessage(content="q"),
            _ai(
                content=[_text("I'll look that up."), _tool_use("web_search", "w1", {"q": "x"})],
                tool_calls=[{"id": "w1", "name": "web_search", "args": {"q": "x"}}],
            ),
        ]
        records = _serialize_messages(msgs)
        assert len(records) == 2
        assert records[1].content == "I'll look that up."
        assert records[1].tool_calls is not None
        assert records[1].tool_calls[0]["name"] == "web_search"

    def test_ai_with_only_text_unchanged(self):
        """Pre-v2.0.24 contract: AIMessage with text content + no
        tool_calls keeps going through unchanged."""
        msgs = [
            HumanMessage(content="q"),
            _ai(content=[_text("hi")], additional_kwargs={"reasoning": "I thought."}),
        ]
        records = _serialize_messages(msgs)
        assert len(records) == 2
        assert records[1].content == "hi"
        assert records[1].reasoning == "I thought."
        assert records[1].tool_calls is None

    def test_genuinely_empty_ai_message_still_dropped(self):
        """Defense in depth: AIMessage with empty content AND no
        tool_calls AND no reasoning must still be dropped (the
        pre-existing guard, not relaxed too much)."""
        msgs = [
            HumanMessage(content="q"),
            _ai(content=[], additional_kwargs={}),
            _ai(content=[_thinking("")], additional_kwargs={}),
            _ai(content="", tool_calls=[]),
            _ai(content=[_text("answer")]),
        ]
        records = _serialize_messages(msgs)
        # user + final answer; the 3 empty AIMessages get dropped
        assert len(records) == 2
        assert records[0].role == "user"
        assert records[1].content == "answer"

    def test_tool_messages_still_dropped(self):
        """v2.0.3 contract unchanged — ToolMessages don't become
        user-visible assistant messages on reload."""
        msgs = [
            HumanMessage(content="q"),
            _ai(
                content=[_thinking("call tool"), _tool_use("retrieve_docs", "t1", {})],
                tool_calls=[{"id": "t1", "name": "retrieve_docs", "args": {}}],
            ),
            ToolMessage(content="tool result", tool_call_id="t1"),
            _ai(content=[_text("answer")]),
        ]
        records = _serialize_messages(msgs)
        roles = [r.role for r in records]
        assert "tool" not in roles
        assert len(records) == 3

    def test_system_message_still_dropped(self):
        """Pre-existing contract — SystemMessages never appear."""
        msgs = [
            SystemMessage(content="you are a helpful assistant"),
            HumanMessage(content="q"),
            _ai(content=[_text("answer")]),
        ]
        records = _serialize_messages(msgs)
        assert len(records) == 2
        assert all(r.role != "system" for r in records)

    def test_native_tool_calls_malformed_entries_skipped(self):
        """Defensive: if some native ``tool_calls`` entries are not
        dicts (shouldn't happen, but malformed checkpoints exist),
        skip them rather than crashing."""
        msgs = [
            HumanMessage(content="q"),
            _ai(
                content=[_thinking("call something"), _tool_use("retrieve_docs", "r1", {})],
                tool_calls=[
                    "not a dict",  # garbage
                    {"id": "r1", "name": "retrieve_docs", "args": {}},
                    42,  # also garbage
                ],
            ),
        ]
        records = _serialize_messages(msgs)
        # Malformed entries dropped, valid one survives
        assert records[1].tool_calls is not None
        assert len(records[1].tool_calls) == 1
        assert records[1].tool_calls[0]["name"] == "retrieve_docs"

    def test_native_tool_calls_empty_dict_entry_dropped(self):
        """Defensive: a native ``tool_calls`` entry that's an empty
        dict (no id / name / args) carries no information for the
        frontend, so we skip it. If ALL entries are empty dicts the
        resulting card list is empty / None and the AIMessage gets
        dropped by the line 301 filter — same as a genuinely-empty
        AIMessage."""
        msgs = [
            HumanMessage(content="q"),
            _ai(
                content=[_tool_use("retrieve_docs", "r1", {})],
                tool_calls=[{}],  # empty dict — no id / name / args
            ),
        ]
        records = _serialize_messages(msgs)
        # The empty-dict entry was filtered out by ``if card``, so
        # tool_calls is None and the AIMessage is dropped (line 301).
        # Only the user message survives.
        assert len(records) == 1
        assert records[0].role == "user"


# ---------------------------------------------------------------------------
# Backward-compat: messages endpoint still returns [] for unknown threads
# ---------------------------------------------------------------------------


class TestSerializeEmpty:
    """Pin the no-op paths so future refactors don't accidentally
    crash on empty input."""

    def test_empty_list_returns_empty(self):
        assert _serialize_messages([]) == []

    def test_only_system_messages_returns_empty(self):
        msgs = [
            SystemMessage(content="sys1"),
            SystemMessage(content="sys2"),
        ]
        assert _serialize_messages(msgs) == []

    def test_only_tool_messages_returns_empty(self):
        """Pure-tool-messages input — none visible to user."""
        msgs = [
            ToolMessage(content="result1", tool_call_id="t1"),
            ToolMessage(content="result2", tool_call_id="t2"),
        ]
        assert _serialize_messages(msgs) == []

# ---------------------------------------------------------------------------
# v2.0.28.13 — planning-step AIMessage tool_call cards must show
# ok=True on reload (not pending forever).
# ---------------------------------------------------------------------------


class TestSerializePlanningStepToolCallCompleted:
    """v2.0.28.13 — synthesized tool_call cards for intermediate
    planning-step AIMessages must mark ``ok=True`` (and populate
    ``result``) when a matching ToolMessage follows in the messages
    list. Otherwise the frontend's ToolCallCard renders the card as
    ⏳ pending forever (``running = call.ok === undefined``), causing
    the user to see TWO cards per turn on reload: one pending and one
    completed — looks like the tool call is "long-running" / stuck.

    Pre-fix (per the user's 2026-09-23 reload report):
      TURN 2 message [3] (planning-step AIMessage from react_agent)
        content=""
        tool_calls=[{id, name, args}]   ← ok undefined, no result
      TURN 2 message [4] (synthesis AIMessage from react_generate)
        content="今天 是..."
        tool_calls=[{id, name, args, result, ok:true, step, ...}]

    Post-fix: [3]'s tool_calls[0] now carries ``ok=True`` and the
    ToolMessage content as ``result``, so the frontend renders both
    cards as ✓ completed (consistent with the live stream).
    """

    def test_planning_step_card_marked_ok_when_tool_message_follows(self):
        from src.api.routes.sessions import _serialize_messages

        # Two-AIMessage shape: planning-step + synthesis (the v2.0.28.10
        # contract). Planning-step has native tool_calls; synthesis has
        # the rich additional_kwargs["tool_calls"] log.
        tm_result = (
            "2026-09-23T18:33:06.849037+08:00 (Asia/Shanghai)\n"
            "2026年9月23日 周三 18:33"
        )
        msgs = [
            HumanMessage(content="现在几点了"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_test_123",
                    "name": "get_current_time",
                    "args": {"timezone": "Asia/Shanghai"},
                }],
            ),
            ToolMessage(
                content=tm_result,
                tool_call_id="call_test_123",
                name="get_current_time",
            ),
            AIMessage(
                content="今天是 **星期三**。",
                additional_kwargs={
                    "tool_calls": [{
                        "id": "call_test_123",
                        "name": "get_current_time",
                        "args": {"timezone": "Asia/Shanghai"},
                        "result": tm_result,
                        "ok": True,
                        "step": 3,
                        "started_at": "2026-09-23T10:33:06+00:00",
                        "ended_at": "2026-09-23T10:33:07+00:00",
                        "elapsed_ms": 0,
                    }],
                },
            ),
        ]

        records = _serialize_messages(msgs)
        # Filter to assistant records.
        asst = [r for r in records if r.role == "assistant"]
        assert len(asst) == 2, (
            f"expected 2 assistant records (planning-step + synthesis), "
            f"got {len(asst)}"
        )

        # The FIRST assistant record is the planning-step.
        planning = asst[0]
        assert planning.tool_calls is not None
        assert len(planning.tool_calls) == 1
        plan_card = planning.tool_calls[0]
        # THE FIX: ok=True and result populated from following TM.
        assert plan_card.get("ok") is True, (
            f"v2.0.28.13 fix broken: planning-step card has ok="
            f"{plan_card.get('ok')!r} (expected True). The frontend "
            "would render this as ⏳ pending forever on reload."
        )
        assert plan_card.get("result") == tm_result, (
            f"planning-step card result mismatch: "
            f"got {plan_card.get('result')!r}, expected {tm_result!r}"
        )

        # The SECOND assistant record is the synthesis (rich log wins).
        synthesis = asst[1]
        assert synthesis.tool_calls is not None
        assert len(synthesis.tool_calls) == 1
        synth_card = synthesis.tool_calls[0]
        assert synth_card.get("ok") is True
        assert synth_card.get("step") == 3  # only the rich log has step
        assert synth_card.get("elapsed_ms") == 0

    def test_planning_step_card_marked_error_when_tool_message_missing(self):
        """No following ToolMessage → ok=False (frontend shows ✗, not
        infinite ⏳ spinner)."""
        from src.api.routes.sessions import _serialize_messages

        msgs = [
            HumanMessage(content="你是什么?"),
            # Planning-step AIMessage with tool_calls but NO following
            # ToolMessage (tool was killed mid-flight or never ran).
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_killed",
                    "name": "retrieve_docs",
                    "args": {"q": "test"},
                }],
            ),
            AIMessage(content="直接回答。"),
        ]

        records = _serialize_messages(msgs)
        asst = [r for r in records if r.role == "assistant"]
        # Both AIMessages survive (the second one has text; the first
        # has tool_calls → survive filter at line 410).
        assert len(asst) >= 1
        planning = asst[0]
        assert planning.tool_calls is not None
        assert planning.tool_calls[0].get("ok") is False, (
            f"v2.0.28.13 fix: missing TM should set ok=False; got "
            f"{planning.tool_calls[0].get('ok')!r}"
        )
        assert "result" not in planning.tool_calls[0]
