"""Regression tests for v2.0.28.15 — synthesis reasoning leak fix.

User reported (2026-09-23, post-v2.0.28.14 ship): when asking
"现在几点了" (reload or live), the assistant bubble's
``<ThinkingDrawer>`` showed the synthesis LLM's meta-commentary
instead of the direct time answer. Sample observed text:

    The user is asking "现在几点了" (What time is it now?). This is
    a time-sensitive question, so I need to call get_current_time
    to get the accurate current time.

And (Run 1 of probe — worse case):

    The user is asking "现在几点了" (What time is it now?). I
    already provided the answer in my previous response with the
    time 2026年9月23日 19:09. There's no new question here - the
    conversation history just sh...

The second sample is a hallucinated prior-turn claim by the
synthesis LLM, actively misleading the user.

**Root cause**: the synthesis LLM emits extended-thinking blocks.
Pre-v2.0.28.15 these were:

1. Yielded as ``("reasoning", ...)`` events → ``reasoningHandler``
   → ``<ThinkingDrawer text={m.reasoning}/>`` (default open) →
   user saw the meta-commentary above the direct answer.
2. Stamped as ``additional_kwargs["reasoning"]`` on the synthesis
   AIMessage → saved to DB → on reload, ``restoreMessages`` and
   ``mergeAdjacentAssistantTurns`` carried it forward to the
   merged bubble's ``reasoning`` field → ``<ThinkingDrawer>``
   rendered it again.

**Fix**:
* ``src/agent/nodes/react_generate.py`` — drop the ``yield
  ("reasoning", ...)`` event AND drop the
  ``additional_kwargs["reasoning"]`` stamp on synthesis AIMessage.
* ``src/agent/fsm.py`` (the ``react_generate_direct`` greeting
  branch) — same drops.
* ``src/frontend/src/chat-utils.tsx`` —
  ``mergeAdjacentAssistantTurns`` now does
  ``reasoning: prev.reasoning`` (no fallback to synthesis
  reasoning). Pre-fix DB rows are repaired on reload because the
  no-fallback rule drops synthesis reasoning even when it's
  present in the wire payload.

These tests pin all four contracts:

1. Synthesis streaming yields NO ``("reasoning", ...)`` events.
2. Synthesis AIMessage has NO ``additional_kwargs["reasoning"]``.
3. ``react_generate_direct`` (greeting path) also has NO
   ``additional_kwargs["reasoning"]``.
4. v2.0.28.11's preservation of thinking-only AIMessages in
   ``_sanitize_history_for_generate`` is NOT regressed (planning-
   step blocks must still survive, so Anthropic extended-thinking
   continuity is preserved).
"""
from __future__ import annotations

import re
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# TestReactGenerateNoReasoningYielded
# ============================================================================


class TestReactGenerateNoReasoningYielded:
    """Pin the streaming-side drop: synthesis LLM must NOT emit
    ``("reasoning", ...)`` events.

    Pre-fix path: ``react_generate.py:711-713`` did
    ``yield ("reasoning", {"text": thinking})`` for every
    extended-thinking block. The runner forwarded as
    ``events.reasoning`` WS frames; the frontend
    ``reasoningHandler`` accumulated them into
    ``messages[lastIdx].reasoning``; ``<ThinkingDrawer>`` rendered
    the meta-commentary above the direct answer.

    Post-fix: thinking blocks are detected by
    ``split_text_and_thinking`` but the pass does nothing
    (``pass``). No event is yielded.
    """

    @pytest.mark.asyncio
    async def test_synthesis_no_reasoning_event_yielded(self):
        """Drive ``react_generate`` with a stub LLM that streams
        thinking blocks. Capture every yielded event. Assert NO
        ``("reasoning", ...)`` event was emitted.
        """
        from langchain_core.messages import HumanMessage

        # Two chunks: one thinking-only, one text-only.
        # Anthropic streaming format: content is a list of dicts.
        async def fake_astream(msgs):
            for chunk in [
                MagicMock(
                    content=[
                        {
                            "type": "thinking",
                            "thinking": (
                                "The user is asking the time. I "
                                "should write a direct answer."
                            ),
                        }
                    ]
                ),
                MagicMock(content=[{"type": "text", "text": "现在是 19:11。"}]),
            ]:
                yield chunk

        synth_model = MagicMock()
        synth_model.astream = fake_astream

        import src.agent.nodes.react_generate as rg

        original_build = rg.build_chat_model
        rg.build_chat_model = lambda **kw: synth_model
        try:
            from langchain_core.messages import AIMessage, HumanMessage as HM

            state = {
                "messages": [HM(content="现在几点了")],
                "current_query": "现在几点了",
                "original_query": "现在几点了",
                "thread_id": "test-v2_0_28_15-no-reasoning-event",
                "step_count": 0,
                "intent": "qa_complex",
                "needs_current_time": True,
                "corrected_query": None,
                "created_at": None,
                "merged_documents": [],
                "reranked_documents": [],
                "web_search_status": None,
                "grounding_status": None,
                "route_decision": "qa_complex",
            }

            captured_events = []
            captured_aimessage = None
            async for kind, payload in rg.react_generate(state, step=0):
                captured_events.append((kind, payload))
                if kind == "__delta__":
                    # v2.0.28.10 — synthesis AIMessage yielded under
                    # ``replace_last_message`` (not ``messages``).
                    if "replace_last_message" in payload:
                        candidate = payload["replace_last_message"]
                        if isinstance(candidate, AIMessage):
                            captured_aimessage = candidate
                    else:
                        msgs = payload.get("messages") or []
                        for m in msgs:
                            if isinstance(m, AIMessage):
                                captured_aimessage = m
                                break
                    break
        finally:
            rg.build_chat_model = original_build

        # (1) NO reasoning event was yielded.
        reasoning_events = [
            (k, p) for k, p in captured_events if k == "reasoning"
        ]
        assert reasoning_events == [], (
            f"BUG: synthesis yielded {len(reasoning_events)} "
            f"('reasoning', ...) events; expected 0 (v2.0.28.15 "
            f"drop). First event: {reasoning_events[0] if reasoning_events else 'n/a'}"
        )

        # (2) Token events still yielded (don't regress the visible
        # answer streaming path).
        token_events = [
            (k, p) for k, p in captured_events if k == "token"
        ]
        assert len(token_events) >= 1, (
            f"token events dropped — synthesis answer streaming "
            f"regressed. Events: {captured_events}"
        )


# ============================================================================
# TestReactGenerateNoReasoningStamped
# ============================================================================


class TestReactGenerateNoReasoningStamped:
    """Pin the save-side drop: synthesis AIMessage MUST NOT carry
    ``additional_kwargs["reasoning"]``.
    """

    @pytest.mark.asyncio
    async def test_synthesis_aimessage_no_reasoning_additional_kwargs(self):
        """Drive ``react_generate`` and inspect the resulting
        AIMessage's ``additional_kwargs``. Assert NO key
        ``"reasoning"``.
        """
        from langchain_core.messages import AIMessage, HumanMessage as HM

        async def fake_astream(msgs):
            for chunk in [
                MagicMock(
                    content=[
                        {
                            "type": "thinking",
                            "thinking": "Plan: write direct time answer.",
                        }
                    ]
                ),
                MagicMock(content=[{"type": "text", "text": "现在是 19:12 北京时间。"}]),
            ]:
                yield chunk

        synth_model = MagicMock()
        synth_model.astream = fake_astream

        import src.agent.nodes.react_generate as rg

        original_build = rg.build_chat_model
        rg.build_chat_model = lambda **kw: synth_model
        try:
            state = {
                "messages": [HM(content="现在几点了")],
                "current_query": "现在几点了",
                "original_query": "现在几点了",
                "thread_id": "test-v2_0_28_15-no-reasoning-stamp",
                "step_count": 0,
                "intent": "qa_complex",
                "needs_current_time": True,
                "corrected_query": None,
                "created_at": None,
                "merged_documents": [],
                "reranked_documents": [],
                "web_search_status": None,
                "grounding_status": None,
                "route_decision": "qa_complex",
            }

            captured_aimessage = None
            async for kind, payload in rg.react_generate(state, step=0):
                if kind == "__delta__":
                    # v2.0.28.10 — synthesis AIMessage yielded under
                    # ``replace_last_message``.
                    if "replace_last_message" in payload:
                        candidate = payload["replace_last_message"]
                        if isinstance(candidate, AIMessage):
                            captured_aimessage = candidate
                    else:
                        msgs = payload.get("messages") or []
                        for m in msgs:
                            if isinstance(m, AIMessage):
                                captured_aimessage = m
                                break
                    break
        finally:
            rg.build_chat_model = original_build

        assert captured_aimessage is not None, (
            "react_generate did not yield an AIMessage delta"
        )
        # THE assertion: NO "reasoning" key on additional_kwargs.
        assert "reasoning" not in (captured_aimessage.additional_kwargs or {}), (
            f"BUG: synthesis AIMessage still has "
            f"additional_kwargs['reasoning'] = "
            f"{captured_aimessage.additional_kwargs.get('reasoning')!r}; "
            f"v2.0.28.15 must drop the reasoning stamp."
        )

    def test_synthesis_source_no_reasoning_line(self):
        """Source-pin: ``react_generate.py`` must NOT contain
        ``additional_kwargs[\"reasoning\"] = reasoning_text`` (the
        stamp) OR ``yield (\"reasoning\", ...)`` (the event).

        Pre-fix had both. Post-fix must have neither.
        """
        from src.agent.nodes import react_generate

        src = open(react_generate.__file__, "r", encoding="utf-8").read()

        # The stamp line should be gone.
        stamp_pattern = re.compile(
            r'additional_kwargs\["reasoning"\]\s*=\s*reasoning_text',
        )
        stamp_hits = stamp_pattern.findall(src)
        assert stamp_hits == [], (
            f"BUG: react_generate.py still contains the reasoning "
            f"stamp line; v2.0.28.15 must drop it. "
            f"Found: {stamp_hits[:3]}"
        )

        # The yield event should be gone too. Search for the
        # exact event payload structure.
        yield_pattern = re.compile(
            r'yield\s*\(\s*"reasoning"\s*,\s*',
        )
        yield_hits = yield_pattern.findall(src)
        assert yield_hits == [], (
            f"BUG: react_generate.py still yields "
            f"('reasoning', ...) events; v2.0.28.15 must drop them. "
            f"Found: {yield_hits[:3]}"
        )


# ============================================================================
# TestReactGenerateDirectNoReasoningStamped
# ============================================================================


class TestReactGenerateDirectNoReasoningStamped:
    """Pin the greeting path drop: ``react_generate_direct``
    AIMessage must NOT carry ``additional_kwargs["reasoning"]``.

    ``react_generate_direct`` is the ``route_decision == "direct"``
    branch (greeting / simple_fact / summary). Same synthesis LLM
    emits extended-thinking; same drop applies.
    """

    @pytest.mark.asyncio
    async def test_direct_aimessage_no_reasoning_additional_kwargs(self):
        """Drive ``react_generate_direct`` and inspect the resulting
        AIMessage's ``additional_kwargs``.
        """
        from langchain_core.messages import AIMessage, HumanMessage as HM

        async def fake_astream(msgs):
            for chunk in [
                MagicMock(
                    content=[
                        {
                            "type": "thinking",
                            "thinking": "Greeting the user warmly.",
                        }
                    ]
                ),
                MagicMock(content=[{"type": "text", "text": "晚上好呀！😄"}]),
            ]:
                yield chunk

        synth_model = MagicMock()
        synth_model.astream = fake_astream

        import src.agent.fsm as fsm_mod

        original_build = fsm_mod.build_chat_model
        fsm_mod.build_chat_model = lambda **kw: synth_model
        try:
            state = {
                "messages": [HM(content="你好")],
                "current_query": "你好",
                "original_query": "你好",
                "thread_id": "test-v2_0_28_15-direct-no-reasoning",
                "step_count": 0,
                "intent": "greeting",
                "needs_current_time": False,
                "corrected_query": None,
                "created_at": None,
                "merged_documents": [],
                "reranked_documents": [],
                "web_search_status": None,
                "grounding_status": None,
                "route_decision": "direct",
            }

            captured_aimessage = None
            async for kind, payload in fsm_mod.react_generate_direct(
                state, step=0
            ):
                if kind == "__delta__":
                    if "replace_last_message" in payload:
                        candidate = payload["replace_last_message"]
                        if isinstance(candidate, AIMessage):
                            captured_aimessage = candidate
                    else:
                        msgs = payload.get("messages") or []
                        for m in msgs:
                            if isinstance(m, AIMessage):
                                captured_aimessage = m
                                break
                    break
        finally:
            fsm_mod.build_chat_model = original_build

        assert captured_aimessage is not None, (
            "react_generate_direct did not yield an AIMessage delta"
        )
        assert "reasoning" not in (captured_aimessage.additional_kwargs or {}), (
            f"BUG: react_generate_direct AIMessage still has "
            f"additional_kwargs['reasoning'] = "
            f"{captured_aimessage.additional_kwargs.get('reasoning')!r}; "
            f"v2.0.28.15 must drop the reasoning stamp on the "
            f"direct path too."
        )

    def test_fsm_source_no_reasoning_yield(self):
        """Source-pin: ``fsm.py`` must NOT contain
        ``yield (\"reasoning\", ...)`` OR
        ``additional_kwargs[\"reasoning\"] = reasoning_text``.

        Both used to be in the ``react_generate_direct`` block.
        """
        from src.agent import fsm

        src = open(fsm.__file__, "r", encoding="utf-8").read()

        yield_pattern = re.compile(
            r'yield\s*\(\s*"reasoning"\s*,\s*',
        )
        yield_hits = yield_pattern.findall(src)
        assert yield_hits == [], (
            f"BUG: fsm.py still yields ('reasoning', ...) events "
            f"in the react_generate_direct branch; v2.0.28.15 must "
            f"drop them. Found: {yield_hits[:3]}"
        )

        stamp_pattern = re.compile(
            r'additional_kwargs\["reasoning"\]\s*=\s*reasoning_text',
        )
        stamp_hits = stamp_pattern.findall(src)
        assert stamp_hits == [], (
            f"BUG: fsm.py still has the reasoning stamp line in "
            f"react_generate_direct; v2.0.28.15 must drop it. "
            f"Found: {stamp_hits[:3]}"
        )


# ============================================================================
# TestSanitizeHistoryPreservesPlanningStepThinkingBlocks
# ============================================================================


class TestSanitizeHistoryPreservesPlanningStepThinkingBlocks:
    """Regression guard for v2.0.28.11: ``_sanitize_history_for_generate``
    must STILL preserve thinking-only AIMessage blocks (so Anthropic
    extended-thinking continuity is maintained). v2.0.28.15 cannot
    regress this — we drop the synthesis LLM's reasoning at the
    save path, NOT at the sanitize path.
    """

    def test_thinking_only_aimessage_preserved_in_sanitized_history(self):
        """Construct history with:
        * HM
        * AIM (planning-step): tool_calls + thinking-block content
        * TM
        * AIM (synthesis draft, text-only) — to be stripped

        Run ``_sanitize_history_for_generate`` and assert:
        * Planning AIMessage is still in ``out`` (with tool_calls
          AND thinking-block content intact).
        * Trailing text-only draft AIMessage is stripped (v2.0.28.11
          contract preserved).
        """
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            ToolMessage,
        )

        from src.agent.nodes.react_generate import (
            _sanitize_history_for_generate,
        )

        planning_content = [
            {"type": "thinking", "thinking": "I need to call get_current_time."},
        ]
        planning_ai = AIMessage(
            content=planning_content,
            tool_calls=[
                {"id": "tc-1", "name": "get_current_time", "args": {}}
            ],
        )
        tm = ToolMessage(
            content="2026-09-23T19:11:00+08:00 (Asia/Shanghai)\n现在是2026年9月23日 周三 19:11",
            tool_call_id="tc-1",
            name="get_current_time",
        )
        draft_ai = AIMessage(content="DRAFT — synthesis will replace this.")

        history = [
            HumanMessage(content="现在几点了"),
            planning_ai,
            tm,
            draft_ai,
        ]

        out = _sanitize_history_for_generate(history)

        # (1) Planning AIMessage preserved with thinking blocks intact.
        plan_kept = [m for m in out if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]
        assert len(plan_kept) == 1, (
            f"v2.0.28.11 regression — planning AIMessage with "
            f"tool_calls was stripped (Anthropic extended-thinking "
            f"continuity broken). out has {len(plan_kept)} tool-calls "
            f"AIMessage(s). out = {out!r}"
        )
        assert plan_kept[0].content == planning_content, (
            f"v2.0.28.11 regression — planning AIMessage's "
            f"thinking-block content was coerced. expected "
            f"{planning_content!r}, got {plan_kept[0].content!r}"
        )

        # (3) Trailing text-only draft AIMessage was stripped.
        drafts_kept = [
            m for m in out
            if isinstance(m, AIMessage)
            and not getattr(m, "tool_calls", None)
            and isinstance(getattr(m, "content", None), str)
        ]
        assert drafts_kept == [], (
            f"v2.0.28.11 regression — trailing text-only draft "
            f"AIMessage was NOT stripped; v2.0.28.15 must not "
            f"regress this. drafts_kept = {drafts_kept!r}"
        )

    def test_sanitize_source_pins_no_reasoning_drop(self):
        """Source-pin: ``_sanitize_history_for_generate`` must NOT
        drop AIMessages with tool_calls (only text-only AIMessages
        are stripped, per v2.0.28.11).
        """
        from src.agent.nodes import react_generate

        src = open(react_generate.__file__, "r", encoding="utf-8").read()
        # Look for the v2.0.28.11 strip pattern. Pre-v2.0.28.11 was
        # `not has_text AND tool_calls` — kept drafting + drop
        # tool_calls. Post-v2.0.28.11 is the inverse:
        # `not tool_calls AND isinstance(content, str)` — drops
        # trailing text-only draft, preserves tool-calls.
        # Verify the post-fix condition is present.
        assert "isinstance(content, str)" in src, (
            "v2.0.28.11 contract regressed — _sanitize_history_for_generate "
            "lost its `isinstance(content, str)` guard. Without this, "
            "Anthropic extended-thinking thinking-block AIMessages would "
            "be wrongly stripped, breaking multi-turn extended thinking."
        )


# ============================================================================
# TestSerializeMessagesNoReasoningFromContentBlocks
# ============================================================================


class TestSerializeMessagesNoReasoningFromContentBlocks:
    """Pin the wire-side drop: ``_serialize_messages`` must NOT
    populate ``MessageRecord.reasoning`` from content-block thinking
    text for any AIMessage.

    Pre-fix the `_extract_thinking_text` fallback at line 398-399
    (sessions.py) lifted the planning-step AIMessage's raw Anthropic
    thinking blocks onto the wire as ``MessageRecord.reasoning``.
    This was the THIRD leak source — the synthesis-side stamp was
    dropped in react_generate.py and the greeting-side stamp was
    dropped in fsm.py, but the content-block fallback still fed
    every intermediate ReAct AIMessage's thinking text into the
    drawer. Pre-fix user report (post-v2.0.28.14 ship):
    "现在没有出现直接回答结果" — the LLM's "The user is asking ...
    I need to call get_current_time" reasoning was rendered into the
    assistant bubble's <ThinkingDrawer> above the direct time answer.

    Post-fix: every AIMessage has wire ``reasoning=None`` regardless
    of how the LLM held its reasoning (additional_kwargs or content
    blocks). The Anthropic extended-thinking continuity is preserved
    because ``_sanitize_history_for_generate`` reads
    ``AIMessage.content`` directly (not the wire payload).
    """

    def test_serialize_aimessage_no_reasoning_from_content_blocks(self):
        """Construct an AIMessage with ``content`` as a thinking-
        block list (the planning-step shape). Run
        ``_serialize_messages`` and assert the resulting
        ``MessageRecord.reasoning`` is None.
        """
        from langchain_core.messages import AIMessage

        from src.api.routes.sessions import _serialize_messages

        planning_ai = AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": (
                        "The user is asking what time it is. I "
                        "should call get_current_time to get the "
                        "accurate current time."
                    ),
                }
            ],
            tool_calls=[
                {"id": "tc-1", "name": "get_current_time", "args": {}}
            ],
        )

        records = _serialize_messages([planning_ai])
        assert len(records) == 1, (
            f"expected 1 record from planning AIMessage, got "
            f"{len(records)}"
        )
        record = records[0]
        # v2.0.28.15 — wire reasoning must be None (NOT the
        # content-block thinking text).
        assert record.reasoning is None, (
            f"BUG: _serialize_messages still lifts content-block "
            f"thinking onto wire `reasoning` = {record.reasoning!r}; "
            f"v2.0.28.15 must drop the _extract_thinking_text "
            f"fallback. This is the planning-step leak source — "
            f"the <ThinkingDrawer> would otherwise render the LLM's "
            f"meta-commentary above the direct answer."
        )

    def test_serialize_source_no_extract_thinking_fallback(self):
        """Source-pin: ``_serialize_messages`` must NOT contain the
        ``if reasoning is None and thinking_text: reasoning =
        thinking_text`` fallback that v2.0.28.15 removed.
        """
        from src.api.routes import sessions

        src = open(sessions.__file__, "r", encoding="utf-8").read()
        # The v2.0.24 fallback line (removed in v2.0.28.15).
        fallback_pattern = re.compile(
            r"if\s+reasoning\s+is\s+None\s+and\s+thinking_text\s*:",
        )
        fallback_hits = fallback_pattern.findall(src)
        assert fallback_hits == [], (
            f"BUG: _serialize_messages still contains the "
            f"`if reasoning is None and thinking_text: reasoning = "
            f"thinking_text` fallback; v2.0.28.15 must drop it. "
            f"Found: {fallback_hits[:3]}"
        )