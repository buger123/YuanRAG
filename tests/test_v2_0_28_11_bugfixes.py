"""v2.0.28.11 — 「上一轮的回复简要总结:...」 meta-text synthesis bug regression tests.

Bug history
-----------
User reported (2026-09-23, post-v2.0.28.10 + DB-clear):

    Q: "现在是几点"
    A: "上一轮的回复简要总结:\n\n- **当前时间**:2026 年 9 月 23 日(周三)**上午 09:49**..."

The actual time IS correct, but the framing is wrong — the synthesis
LLM wrote "上一轮的回复简要总结:" (brief summary of previous round's
reply) instead of a direct answer like "现在是上午 09:49". The user
expects a clean direct answer.

真根因 (root cause)
--------------------
``_sanitize_history_for_generate`` (src/agent/nodes/react_generate.py)
keeps react_agent's text-only "draft" AIMessage in the synthesis
prompt. ``_HISTORY_FRAME_HINT`` (src/agent/nodes/_history_frame_hint.py)
then mislabels it as "[AIMessage] 是你之前的回答" → the synthesis LLM
thinks it has already replied with the time, and produces a meta-
summary of its prior "reply".

The synthesis LLM should NOT see react_agent's draft — it's not
conversation history, it's intermediate reasoning that v2.0.28.10
will overwrite via ``replace_last_message``.

修法 (fix)
----------
Drop the trailing text-only AIMessage in ``_sanitize_history_for_generate``.
The ``after_react_agent`` predicate (src/agent/fsm.py:247-260) GUARANTEES
that the last AIMessage at react_generate entry has no ``tool_calls`` —
that AIMessage is react_agent's "draft". The synthesis LLM sees only
[HM, AIM1(tool_calls), TM] and synthesizes a fresh answer from
the user's question + the tool result.

These tests pin the contract so a future refactor that re-introduces
the draft to the synthesis prompt fails CI immediately.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ============================================================================
# Helpers — small message builders
# ============================================================================


def _hm(content: str) -> HumanMessage:
    return HumanMessage(content=content)


def _aim_tool_calls(content: str = " ", tool_name: str = "get_current_time") -> AIMessage:
    """AIMessage with a tool_calls entry — the planning step of a tool round."""
    return AIMessage(
        content=content,
        tool_calls=[{"id": "tc-1", "name": tool_name, "args": {}}],
    )


def _aim_text(content: str) -> AIMessage:
    """Text-only AIMessage — the "draft" that v2.0.28.11 strips."""
    return AIMessage(content=content)


def _tm(content: str, name: str = "get_current_time", tcid: str = "tc-1") -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=tcid)


# ============================================================================
# 1. _sanitize_history_for_generate drops the trailing text-only AIMessage
# ============================================================================


class TestV2_0_28_11DraftStrippedFromSynthesis:
    """The sanitizer must drop the trailing text-only AIMessage so
    the synthesis LLM doesn't read it as "your previous reply" via
    ``_HISTORY_FRAME_HINT`` and produce meta-summary text.

    ``after_react_agent`` predicate contract guarantees the last
    AIMessage at react_generate entry is the draft (no tool_calls);
    the strip is safe by construction."""

    def _sanitize(self, messages):
        from src.agent.nodes.react_generate import _sanitize_history_for_generate
        return _sanitize_history_for_generate(list(messages))

    def test_drops_trailing_text_only_ai_message(self):
        """Basic case (the reported bug):
        [HM, AIM1(tool_calls), TM, AIM_draft] → [HM, AIM1(tool_calls), TM].
        The draft is GONE — the synthesis LLM only sees the user's
        question + the tool_call → tool_result pair."""
        msgs = [
            _hm("现在是几点"),
            _aim_tool_calls(" ", "get_current_time"),
            _tm("2026-09-23T09:49 上午 09:49"),
            _aim_text("现在是2026年9月23日上午 09:49"),  # the draft
        ]
        out = self._sanitize(msgs)
        contents = [(type(m).__name__, getattr(m, "content", "")[:30]) for m in out]
        assert contents == [
            ("HumanMessage", "现在是几点"),
            ("AIMessage", " "),  # the tool-call AIMessage is preserved (parent of preserved TM)
            ("ToolMessage", "2026-09-23T09:49 上午 09:49"),
        ], (
            f"v2.0.28.11: trailing text-only AIMessage must be stripped "
            f"so the synthesis LLM doesn't read it as 'previous reply'. "
            f"Got {contents!r}"
        )

    def test_keeps_trailing_ai_with_tool_calls(self):
        """Defensive: if a tool-calls AIMessage is the trailing
        message (hypothetical — would never happen under the current
        ``after_react_agent`` predicate contract, but defend against
        future FSM changes), the new strip must NOT drop it — that
        would break tool_use ↔ tool_result pairing.

        Use ``get_current_time`` as the tool so the AIMessage survives
        the prior passes (parent of preserved ToolMessage). The new
        strip then sees ``out[-1].tool_calls`` is truthy and skips the
        ``[:-1]`` slice. Result: all 3 messages preserved."""
        msgs = [
            _hm("现在是几点"),
            _aim_tool_calls(" ", "get_current_time"),  # parent AIM
            _tm("2026-09-23T09:49"),  # preserved ToolMessage
        ]
        out = self._sanitize(msgs)
        contents = [(type(m).__name__, getattr(m, "content", "")[:30]) for m in out]
        assert contents == [
            ("HumanMessage", "现在是几点"),
            ("AIMessage", " "),  # tool-calls AIM with parent-of-preserved
            ("ToolMessage", "2026-09-23T09:49"),
        ], (
            f"v2.0.28.11: trailing AIMessage WITH tool_calls must be "
            f"preserved (strip checks `not tool_calls` guard). Got "
            f"{contents!r}"
        )
        # Sanity: the tool-calls AIM (parent of the preserved TM) is
        # still in the output — would have been wrongly stripped if
        # the new logic didn't guard on `not tool_calls`.
        aim_msgs = [m for m in out if isinstance(m, AIMessage)]
        assert aim_msgs, "tool-calls AIM dropped — broken"
        assert any(getattr(m, "tool_calls", None) for m in aim_msgs), (
            "the tool-calls AIM must still have tool_calls metadata "
            "(strip should NOT have removed it)"
        )

    def test_keeps_trailing_human_message(self):
        """Defensive: if state["messages"] ends with a HumanMessage
        (multi-turn where the user just spoke), we don't touch it.
        HM is real input — the synthesis LLM needs to see the latest
        question."""
        msgs = [
            _hm("turn 1"),
            _aim_text("previous answer"),
            _hm("现在是几点"),  # last entry is HM — multi-turn user message
        ]
        out = self._sanitize(msgs)
        assert isinstance(out[-1], HumanMessage), (
            f"trailing HumanMessage must NOT be stripped — the "
            f"synthesis LLM needs the user's latest question. "
            f"Got trailing type {type(out[-1]).__name__}"
        )
        assert out[-1].content == "现在是几点"
        # The previous-turn AIMessage (real synthesis, not a draft)
        # MUST be preserved — that's legitimate history.
        aim_msgs = [m for m in out if isinstance(m, AIMessage)]
        assert len(aim_msgs) == 1
        assert aim_msgs[0].content == "previous answer"

    def test_keeps_prior_turn_synthesis_aimessage(self):
        """Multi-turn shape: a prior turn's saved synthesis must
        SURVIVE the strip — only the CURRENT turn's draft is dropped.

        Shape: [HM_t1, AIM_t1_synthesis, HM_t2, AIM1_t2(tool_calls), TM_t2, AIM_t2_draft]
        Output: [HM_t1, AIM_t1_synthesis, HM_t2, AIM1_t2(tool_calls), TM_t2]

        The strip targets only the LAST entry. AIM_t1_synthesis is
        buried in history (not the trailing entry) → preserved."""
        msgs = [
            _hm("Q1 turn"),
            _aim_text("A1 turn synthesis"),  # prior turn's saved synthesis
            _hm("Q2 turn"),
            _aim_tool_calls(" ", "get_current_time"),
            _tm("2026-09-23T09:49"),
            _aim_text("A2 draft"),  # current turn's draft — to be stripped
        ]
        out = self._sanitize(msgs)
        contents = [(type(m).__name__, getattr(m, "content", "")[:30]) for m in out]
        assert contents == [
            ("HumanMessage", "Q1 turn"),
            ("AIMessage", "A1 turn synthesis"),  # preserved (not last)
            ("HumanMessage", "Q2 turn"),
            ("AIMessage", " "),  # tool-call AIM (parent of preserved TM)
            ("ToolMessage", "2026-09-23T09:49"),
            # NO trailing draft
        ], (
            f"v2.0.28.11: must drop ONLY the trailing text-only AIMessage "
            f"(current turn's draft); prior turn's synthesis must survive. "
            f"Got {contents!r}"
        )
        assert len(out) == 5, f"expected 5 messages after strip; got {len(out)}"

    def test_strips_when_only_hm_and_draft(self):
        """Edge case: single-round no-tool turn with only HM + draft.
        [HM, AIM_draft] → [HM].

        This is what react_agent looks like when the LLM answers in a
        single round without any tool calls (e.g. trivial factual
        recall). The strip removes the draft → synthesis sees only
        HM. Same as ``react_generate_direct`` behavior — the System
        prompt carries the synthesis instructions."""
        msgs = [
            _hm("你好"),
            _aim_text("draft text-only answer"),
        ]
        out = self._sanitize(msgs)
        contents = [(type(m).__name__, getattr(m, "content", "")[:30]) for m in out]
        assert contents == [("HumanMessage", "你好")], (
            f"v2.0.28.11: HM + draft → HM only (draft stripped). "
            f"Got {contents!r}"
        )

    def test_keeps_trailing_thinking_only_ai_message(self):
        """v1.1.9 regression guard — thinking-only AIMessages have
        content as a LIST of thinking blocks (Anthropic's extended
        thinking). They are NOT drafts; they are intermediate
        reasoning that the synthesis LLM should carry forward.

        The strip must distinguish "draft" (content is a string)
        from "thinking" (content is a list of blocks). Stripping
        thinking would break extended-thinking forward carry.

        Pin: tests/test_v2_0_4_bugfixes.py
        ``TestP04SanitizeHistory::test_keeps_thinking_only_ai_messages``
        already tests the sanitizer's behavior on a single thinking-
        only AIM; this test pins that the NEW v2.0.28.11 strip
        doesn't accidentally drop it."""
        thinking_content = [
            {"type": "thinking", "thinking": "internal reasoning..."}
        ]
        msgs = [
            _hm("q"),
            AIMessage(
                content=thinking_content,
                additional_kwargs={"reasoning": "internal reasoning..."},
            ),
        ]
        out = self._sanitize(msgs)
        assert len(out) == 2, (
            f"v2.0.28.11: thinking-only AIMessage must NOT be stripped "
            f"(content is a list of thinking blocks, NOT a draft string). "
            f"Got {[type(m).__name__ for m in out]!r}"
        )
        assert isinstance(out[0], HumanMessage)
        assert out[0].content == "q"
        assert isinstance(out[1], AIMessage)
        assert out[1].content == thinking_content, (
            f"thinking-only AIMessage content must be preserved verbatim; "
            f"got {out[1].content!r}"
        )

    def test_strips_when_empty(self):
        """Defensive: empty input → empty output. No crash."""
        out = self._sanitize([])
        assert out == [], f"empty messages should yield empty sanitized list; got {out!r}"

    def test_preserves_trailing_text_aim_when_not_last_synthesis_target(self):
        """The bug only manifests when the LAST AIMessage at synthesis
        time IS the draft. This test pins that the strip correctly
        identifies the LAST entry — NOT any earlier text-only
        AIMessage.

        Shape: [HM, AIM_text_real, AIM_text_draft]. The real AIM (an
        earlier turn's synthesis that has text) is preserved; the
        draft (last entry) is stripped.

        Wait — this shape doesn't actually occur in practice (real
        synthesis is always followed by a new HM in multi-turn). Let
        me reshape: [HM, AIM_text_with_tool_calls_actually_no_tool_calls,
        AIM_text_draft]. The first AIM has text → not a draft; the
        last is a draft → strip.

        Better: [HM_t1, AIM_t1_synth, HM_t2, AIM_draft_t2]. Here the
        draft IS the last entry → strip. The t1 synthesis is
        preserved (not last)."""
        msgs = [
            _hm("Q1"),
            _aim_text("A1 real synthesis"),
            _hm("Q2"),
            _aim_text("A2 draft"),  # last entry → strip
        ]
        out = self._sanitize(msgs)
        contents = [(type(m).__name__, getattr(m, "content", "")[:30]) for m in out]
        assert contents == [
            ("HumanMessage", "Q1"),
            ("AIMessage", "A1 real synthesis"),  # preserved (not last)
            ("HumanMessage", "Q2"),
            # NO trailing draft
        ], (
            f"v2.0.28.11: must drop the LAST AIMessage only. "
            f"Got {contents!r}"
        )
        assert len(out) == 3


# ============================================================================
# 2. Source-level pin — the strip block lives in react_generate.py
# ============================================================================


class TestV2_0_28_11SourceCodePin:
    """Pin the source code location so a refactor that moves the
    strip elsewhere (or breaks the predicate) fails CI."""

    def test_sanitizer_module_has_v2_0_28_11_comment(self):
        """``src/agent/nodes/react_generate.py`` must contain the
        v2.0.28.11 comment block explaining why the strip is
        necessary. Without the comment a future refactor might
        delete the strip thinking it's dead code."""
        from pathlib import Path
        path = Path("D:/prep for work/Projects/YuanRAG/src/agent/nodes/react_generate.py")
        text = path.read_text(encoding="utf-8")
        assert "v2.0.28.11" in text, (
            "react_generate.py missing the v2.0.28.11 comment block. "
            "A future refactor might delete the trailing-strip thinking "
            "it's dead code if there's no comment marking the rationale."
        )
        assert "上一轮" in text or "上一轮" in text.encode().decode("utf-8", errors="replace"), (
            "v2.0.28.11 comment must mention the bug symptom (meta-text "
            "'上一轮的回复简要总结:...') so the next maintainer understands "
            "WHY the strip is here."
        )

    def test_sanitizer_strips_trailing_text_only_aim(self):
        """Pinned at source level: the strip logic itself must
        check ``isinstance(out[-1], AIMessage)`` AND
        ``not getattr(out[-1], "tool_calls", None)`` AND
        ``isinstance(content, str)`` — not just
        ``isinstance(out[-1], AIMessage)`` (which would break
        tool_use ↔ tool_result pairing AND drop thinking-only
        AIMessages)."""
        from pathlib import Path
        import re
        path = Path("D:/prep for work/Projects/YuanRAG/src/agent/nodes/react_generate.py")
        text = path.read_text(encoding="utf-8")
        # The new strip block must reference tool_calls check.
        idx = text.find("v2.0.28.11")
        assert idx != -1, "v2.0.28.11 comment block missing"
        body = text[idx:idx + 2500]
        assert "tool_calls" in body, (
            "v2.0.28.11 strip must check `tool_calls` attr — without "
            "this guard, a trailing tool-calls AIMessage (broken "
            "tool_use ↔ tool_result pairing) would be incorrectly stripped."
        )
        # Pin the isinstance(out[-1], AIMessage) check.
        assert re.search(r"isinstance\(out\[-1\],\s*AIMessage\)", body), (
            "v2.0.28.11 strip must check isinstance(out[-1], AIMessage)"
        )
        # Pin the isinstance(content, str) check (distinguishes drafts
        # from thinking-only AIMessages — v1.1.7/v1.1.9 contract).
        assert re.search(r"isinstance\(content,\s*str\)", body), (
            "v2.0.28.11 strip must check isinstance(content, str) — "
            "without this, thinking-only AIMessages (content is a "
            "list of thinking blocks per v1.1.9) would be incorrectly "
            "dropped, breaking Anthropic extended-thinking forward carry."
        )
        assert "out = out[:-1]" in body, (
            "v2.0.28.11 strip must actually slice off the last entry"
        )

    def test_after_react_agent_predicate_guarantees_strip_is_safe(self):
        """The ``after_react_agent`` predicate contract in fsm.py
        MUST continue to ensure the last AIMessage at react_generate
        entry has no ``tool_calls``. If a future FSM refactor changes
        this contract, v2.0.28.11's strip would silently drop
        legitimate history (a tool-calls AIM that some prior step
        appended)."""
        from pathlib import Path
        import re
        path = Path("D:/prep for work/Projects/YuanRAG/src/agent/fsm.py")
        text = path.read_text(encoding="utf-8")
        # Locate the def of the predicate (skip earlier docstring /
        # narrative references that also contain the name).
        idx = text.find("def after_react_agent")
        assert idx != -1, "fsm.py missing `def after_react_agent`"
        # Look at the predicate body — must check last AIM has no
        # tool_calls to route to react_generate.
        body = text[idx:idx + 1500]
        # Predicate routes to react_generate only when last AIM has
        # no tool_calls.
        assert "react_generate" in body, (
            "after_react_agent must reference react_generate in its route"
        )
        # Find the no-tool_calls condition. The actual predicate
        # reads ``getattr(last, "tool_calls", None)`` then checks
        # truthiness — accept any reasonable variant.
        has_no_tools = (
            "tool_calls" in body
            and re.search(
                r"getattr\(\s*last\s*,\s*['\"]tool_calls['\"]"
                r"|not\s+last\s*\.?\s*tool_calls"
                r"|tool_calls\s+is\s+None"
                r"|tool_calls\s*==\s*\[\]",
                body,
            )
        )
        assert has_no_tools, (
            "after_react_agent predicate must check that the last AIMessage "
            "has no tool_calls before routing to react_generate. "
            "Without this check, v2.0.28.11's strip could drop "
            "legitimate tool-calls history."
        )