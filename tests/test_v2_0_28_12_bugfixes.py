"""Regression tests for v2.0.28.12 — multi-turn "现在几点了" bug fixes.

User reported (2026-09-23, post-v2.0.28.11 polish): the LLM in turn 2 of
a multi-turn session replied with "我没有获取实时信息的能力, … 需要
你重新发送一次才能触发检索流程" verbatim from DIRECT_SYSTEM instead of
calling the ``get_current_time`` tool. Two distinct root causes, one
shared symptom:

1. **Intent misroute** — the cheap-model classifier in
   ``intent_analysis`` sometimes rounds "现在几点了" to
   ``intent="greeting"`` (no "time" category, picks the closest
   conversational shape). The pre-v2.0.28.12 override only forced
   qa_complex for simple_fact / summary; the greeting branch was
   unprotected, so the path landed in ``react_generate_direct`` (no
   tools bound) and the LLM hallucinated DIRECT_SYSTEM's "re-send to
   trigger retrieval" line.

2. **Synthesis-LLM confusion** — even when intent correctly routed
   to qa_complex → react_agent → react_generate, the synthesis
   prompt's ``_HISTORY_FRAME_HINT`` labelled every AIMessage as
   "你之前的回答" (your prior reply). The planning-step AIMessage
   (the LLM's "I'll call get_current_time" round) was kept in the
   sanitized history as the parent of the preserved ToolMessage, so
   the synthesis LLM saw a confusing "previous reply says 'let me
   check', tool returned the time, no new user question" flow and
   produced meta-text like "（等待你的下一条消息～）" or fell back to
   the denial language from DIRECT_SYSTEM.

The fix touches three files:

* ``src/agent/nodes/_history_frame_hint.py`` — clarify the
  AIMessage taxonomy (text-only = final reply; tool_calls =
  planning step).
* ``src/agent/nodes/intent_analysis.py`` — guard fast-path 1
  (greeting) with ``not needs_time`` AND extend the slow-path
  override to cover ``"greeting"``.
* ``src/agent/nodes/react_agent.py`` — strengthen the time-tool
  hint from "建议先调用" (recommend) to "必须先调用" (must call).

These tests pin all three contracts.
"""
from __future__ import annotations

import asyncio
import re
import sys

import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# 1. Intent analysis — greeting fast-path is guarded by needs_time
# ============================================================================


class TestV2_0_28_12IntentGreetingGuard:
    """Pin the v2.0.28.12 fix: greeting fast-path skips time queries."""

    def test_time_query_skips_greeting_fast_path(self):
        """``现在几点了`` matches needs_time AND the greeting regex is
        permitted to match — the guard ensures we still go through the
        slow-path LLM classification (not the greeting fast-path).

        Pre-fix: greeting fast-path fired first → intent="greeting" →
        path went through ``react_generate_direct`` (no tools bound).
        Post-fix: needs_time guard skips the greeting fast-path; the
        cheap-model LLM classifies, and the override at line 389 forces
        qa_complex when the LLM still picks greeting.
        """
        from src.agent.nodes import intent_analysis

        query = "现在几点了"
        # Sanity: the cheap pre-check confirms the time marker.
        from src.agent.nodes.intent_analysis import _query_needs_time

        assert _query_needs_time(query), (
            "precondition: regex should match time markers in the query"
        )

        # The greeting fast-path branch should NOT yield when
        # needs_time is True. Drive the async generator with a stubbed
        # state and assert the FIRST intent event is NOT "greeting"
        # (which would mean the fast-path fired and bypassed everything
        # else).
        from langchain_core.messages import HumanMessage

        async def _drive():
            state = {
                "current_query": query,
                "original_query": query,
                "messages": [HumanMessage(content=query)],
                "step_count": 0,
                "intent": None,
                "corrected_query": None,
                "needs_current_time": None,
            }
            intents_seen = []
            async for kind, payload in intent_analysis.intent_analysis(state):
                if kind == "intent":
                    intents_seen.append(payload.get("intent_value"))
                if kind == "__delta__":
                    return intents_seen
            return intents_seen

        # Run in isolation; the cheap LLM may fail to classify (no API
        # key in CI), but the fallback at line 342-357 still yields
        # intent="qa_complex" — never greeting.
        intents = asyncio.run(_drive())
        assert "greeting" not in intents, (
            f"BUG: greeting fast-path fired for time query {query!r}; "
            f"intents seen: {intents}"
        )
        assert intents[-1] == "qa_complex", (
            f"expected final intent to be qa_complex for {query!r} "
            f"(override should fire when LLM returns greeting); "
            f"got {intents}"
        )

    def test_pure_greeting_still_fast_path(self):
        """A pure greeting (no time markers) still hits the greeting
        fast-path — the v2.0.28.12 guard MUST NOT regress the fast-path
        for non-time queries.
        """
        from src.agent.nodes import intent_analysis
        from langchain_core.messages import HumanMessage

        async def _drive():
            state = {
                "current_query": "你好",
                "original_query": "你好",
                "messages": [HumanMessage(content="你好")],
                "step_count": 0,
                "intent": None,
                "corrected_query": None,
                "needs_current_time": None,
            }
            intents_seen = []
            async for kind, payload in intent_analysis.intent_analysis(state):
                if kind == "intent":
                    intents_seen.append(payload.get("intent_value"))
                if kind == "__delta__":
                    return intents_seen
            return intents_seen

        intents = asyncio.run(_drive())
        assert intents == ["greeting"], (
            f"pure greeting should still hit fast-path 1; got {intents}"
        )

    def test_overrride_includes_greeting_in_value_set(self):
        """Pin the override's accepted-intent set so a future refactor
        can't silently drop the ``"greeting"`` case. We grep the source
        for the override branch (cheap guard; the production override
        logic is exercised by the slow-path test above).
        """
        from src.agent.nodes import intent_analysis

        src = open(intent_analysis.__file__, "r", encoding="utf-8").read()
        # The override condition MUST list "greeting", "simple_fact",
        # and "summary" — otherwise the time-need guard has regressed.
        m = re.search(
            r"if needs_time and intent in \(([^)]+)\):",
            src,
        )
        assert m, "could not find the time-need override condition in intent_analysis.py"
        listed = [s.strip().strip("\"'") for s in m.group(1).split(",")]
        for required in ("greeting", "simple_fact", "summary"):
            assert required in listed, (
                f"override condition missing {required!r}; "
                f"listed: {listed}. Pre-v2.0.28.12 regression."
            )


# ============================================================================
# 2. _HISTORY_FRAME_HINT — clarifies AIMessage taxonomy
# ============================================================================


class TestV2_0_28_12HistoryFrameHintTaxonomy:
    """Pin the updated ``_HISTORY_FRAME_HINT`` wording so a future
    refactor can't regress the synthesis-LLM's understanding of
    tool-calls AIMessages as planning steps (not prior replies).
    """

    def test_hint_distinguishes_tool_call_aimessage(self):
        """The hint MUST mention ``tool_calls`` explicitly and label
        such AIMessages as planning steps (NOT prior replies).
        """
        from src.agent.nodes._history_frame_hint import _HISTORY_FRAME_HINT

        assert "tool_calls" in _HISTORY_FRAME_HINT, (
            "hint must mention 'tool_calls' so the LLM distinguishes "
            "planning-step AIMessages from prior final replies"
        )
        # Must NOT contain the pre-v2.0.28.12 oversimplification that
        # labelled EVERY AIMessage as "你之前的回答" — that's the very
        # phrase that confused the synthesis LLM.
        assert "[AIMessage] 是你之前的回答" not in _HISTORY_FRAME_HINT, (
            "hint must NOT lump every AIMessage together as 'your "
            "prior reply' — that oversimplification is what caused "
            "the synthesis LLM to misread planning steps as final "
            "answers. The taxonomy must split text-only vs tool_call "
            "AIMessages."
        )

    def test_hint_names_tool_message_role(self):
        """The hint MUST mention ``[ToolMessage]`` so the synthesis LLM
        knows the data right after the planning-step AIMessage is the
        tool result it must use.
        """
        from src.agent.nodes._history_frame_hint import _HISTORY_FRAME_HINT

        assert "[ToolMessage]" in _HISTORY_FRAME_HINT, (
            "hint must label [ToolMessage] as the tool-result data "
            "so the synthesis LLM uses it to answer the latest HM"
        )

    def test_hint_directs_synthesis_to_latest_human_message(self):
        """The hint MUST direct the synthesis LLM to answer the LATEST
        HumanMessage — without this the LLM may refuse to write a fresh
        answer (the user's reported "（等待你的下一条消息～）" symptom).
        """
        from src.agent.nodes._history_frame_hint import _HISTORY_FRAME_HINT

        assert "最近的 HumanMessage" in _HISTORY_FRAME_HINT or "最新" in _HISTORY_FRAME_HINT, (
            "hint must direct the synthesis LLM to answer the LATEST "
            "HumanMessage; otherwise the LLM may decide 'no new user "
            "message' and produce meta-text"
        )


# ============================================================================
# 3. react_agent — strengthened time-tool hint
# ============================================================================


class TestV2_0_28_12TimeToolHintDirective:
    """Pin the strengthened time-tool hint wording."""

    def test_hint_is_directive_not_recommendation(self):
        """Pre-v2.0.28.12: ``建议先调用`` (recommend). Post-v2.0.28.12:
        ``必须先调用`` (must). The pre-v2.0.28.12 wording let the LLM
        ignore the hint on stochastic turns; the post-v2.0.28.12 wording
        makes the tool call mandatory when needs_current_time=True.
        """
        from src.agent.nodes.react_agent import _TIME_TOOL_HINT

        assert "必须" in _TIME_TOOL_HINT, (
            "time-tool hint must use '必须' (must) wording, not just "
            "'建议' (recommend), so the LLM reliably calls "
            "get_current_time when needs_current_time=True"
        )
        # Defensive: the old recommendation-only wording must be gone.
        # We grep for "建议先调用" specifically (the pre-v2.0.28.12
        # phrase that allowed stochastic skips).
        assert "建议先调用" not in _TIME_TOOL_HINT, (
            "old '建议先调用' (recommend-first-call) wording still "
            "present; v2.0.28.12 must replace with '必须先调用' (must-"
            "first-call) so the LLM cannot skip the tool on a "
            "stochastic turn"
        )

    def test_hint_explicitly_disallows_training_knowledge_fallback(self):
        """The hint MUST tell the LLM NOT to fall back to "I don't
        know" or training knowledge for time queries. This closes the
        DIRECT_SYSTEM denial-language leak that surfaced in the user's
        bug report ("重新发送一次才能触发检索流程").
        """
        from src.agent.nodes.react_agent import _TIME_TOOL_HINT

        assert "不允许" in _TIME_TOOL_HINT or "不要" in _TIME_TOOL_HINT, (
            "time-tool hint must explicitly forbid the training-"
            "knowledge / 'I don't know' fallback so the LLM cannot "
            "emit DIRECT_SYSTEM-style denial language"
        )


# ============================================================================
# 4. End-to-end synthesis-prompt probe (Bug B regression guard)
# ============================================================================


class TestV2_0_28_12SynthesisPromptCorrectness:
    """End-to-end: drive the FSM with a mock synthesis LLM and assert
    the prompt sent to it. Pins the entire chain of fixes.

    Mirrors ``tests/debug_now_multiturn.py`` structure but as a
    formal regression test (re-runnable in CI, no DEBUG-only output).
    """

    @pytest.mark.asyncio
    async def test_synthesis_prompt_clarifies_tool_call_aimessage(
        self, tmp_path, monkeypatch
    ):
        """The synthesis LLM prompt MUST contain the updated
        ``_HISTORY_FRAME_HINT`` AND distinguish tool-calls AIMessages
        from prior replies. Pre-v2.0.28.12 the hint lumped them
        together, causing the synthesis LLM to confuse planning steps
        with prior final answers.
        """
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from src.storage import checkpointer

        db_path = tmp_path / "history.db"
        monkeypatch.setattr(
            "src.storage.checkpointer.history_db_path", lambda: db_path
        )
        checkpointer.reset_for_tests()
        await checkpointer.startup()

        try:
            thread_id = "test-v2_0_28_12-synthesis-probe"

            async def fake_intent(state, *, step=0):
                yield ("__delta__", {
                    "intent": "qa_complex",
                    "needs_current_time": True,
                })

            # react_agent round 1: AIM with tool_calls (planning step)
            # round 2: text-only draft (the typical multi-round flow).
            round1_response = AIMessage(
                content="让我查一下当前时间。",
                tool_calls=[{"id": "tc-1", "name": "get_current_time", "args": {}}],
            )
            round2_draft = AIMessage(content="DRAFT placeholder")
            fake_model = MagicMock()
            fake_model.bind_tools = MagicMock(return_value=fake_model)
            fake_model.ainvoke = AsyncMock(side_effect=[round1_response, round2_draft])

            import src.agent.nodes.react_agent as ra
            monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

            async def fake_tools(state, *, step=0):
                history = list(state.get("messages") or [])
                last = history[-1] if history else None
                if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        yield ("tool_call_end", {
                            "tool_call_id": tc.get("id", ""),
                            "tool_name": tc.get("name", ""),
                            "ok": True,
                            "elapsed_ms": 10,
                            "step": step,
                        })
                tm = ToolMessage(
                    content="2026-09-23T12:40:13.300697+08:00",
                    tool_call_id="tc-1",
                    name="get_current_time",
                )
                yield ("__delta__", {"messages": [tm]})

            captured = []
            chunk = MagicMock()
            chunk.content = "现在是 12:40 北京时间。"

            async def fake_astream(msgs):
                captured.append(list(msgs))
                yield chunk

            synth_model = MagicMock()
            synth_model.bind_tools = MagicMock(return_value=synth_model)
            synth_model.astream = fake_astream

            import src.agent.nodes.react_generate as rg
            monkeypatch.setattr(rg, "build_chat_model", lambda **kw: synth_model)

            import src.agent.fsm as fsm_mod
            orig_intent = dict(fsm_mod.NODES["intent_analysis"])
            orig_tools = dict(fsm_mod.NODES["tools"])
            try:
                fsm_mod.NODES["intent_analysis"] = {
                    **orig_intent, "fn": fake_intent,
                }
                fsm_mod.NODES["tools"] = {**orig_tools, "fn": fake_tools}

                state = {
                    "messages": [
                        HumanMessage(content="你是谁"),
                        AIMessage(content="你好！我是源 RAG..."),
                        HumanMessage(content="现在几点了"),
                    ],
                    "thread_id": thread_id,
                    "original_query": "现在几点了",
                    "current_query": "现在几点了",
                    "step_count": 0,
                    "intent": None,
                    "corrected_query": None,
                    "needs_current_time": True,
                    "created_at": None,
                }

                from src.agent.fsm import run_fsm
                async for _ in run_fsm(state, thread_id=thread_id, max_steps=10):
                    pass

                assert captured, "synthesis LLM was never called"
                msgs = captured[0]

                # Find the SystemMessage carrying _HISTORY_FRAME_HINT.
                from src.agent.nodes._history_frame_hint import _HISTORY_FRAME_HINT
                hint_indices = [
                    i for i, m in enumerate(msgs)
                    if isinstance(getattr(m, "content", None), str)
                    and _HISTORY_FRAME_HINT[:30] in getattr(m, "content", "")
                ]
                assert hint_indices, (
                    "_HISTORY_FRAME_HINT not found in synthesis prompt"
                )

                # The hint MUST mention tool_calls (v2.0.28.12 fix).
                hint_text = msgs[hint_indices[0]].content
                assert "tool_calls" in hint_text, (
                    f"synthesis LLM prompt's _HISTORY_FRAME_HINT must "
                    f"mention 'tool_calls' (v2.0.28.12 fix); got: "
                    f"{hint_text!r}"
                )
            finally:
                fsm_mod.NODES["intent_analysis"] = orig_intent
                fsm_mod.NODES["tools"] = orig_tools
        finally:
            await checkpointer.shutdown()


# Helper imports for the e2e probe test (avoids polluting module top
# with MagicMock + AsyncMock for the other 3 test classes).
from unittest.mock import AsyncMock, MagicMock  # noqa: E402


class TestV2_0_28_12_1ThinkingOnlyFallbackOrdering:
    """v2.0.28.12.1 — synthesis-LLM thinking-only fallback must run
    BEFORE the ``"(no answer text generated)"`` placeholder is set.

    Pre-v2.0.28.12.1 the placeholder was assigned right after the
    streaming loop finished (line 714-715 of
    ``src/agent/nodes/react_generate.py``). The TM-data fallback (line
    773+) checks ``if not answer_text`` — but with the placeholder
    string already in ``answer_text``, the check is FALSE and the
    fallback never fires. Result: the user sees ``"(no answer text
    generated)"`` even though the conversation history contains a
    perfectly good ``get_current_time`` ToolMessage.

    Confirmed via ``tests/e2e/debug-now-bug2.py`` (2026-09-23):
    TURN 2 saved assistant message [4] had content
    ``"(no answer text generated)"`` instead of the constructed
    fallback like ``"根据刚刚查询,当前时间是 2026-09-23T13:09:43+08:00
    (Asia/Shanghai)"``.

    The fix moves the placeholder assignment AFTER the fallback block,
    so the fallback has a chance to populate ``answer_text`` first.
    """

    def test_no_answer_text_placeholder_runs_after_fallback(self):
        """Source-pin: ``answer_text = "(no answer text generated)"``
        MUST appear AFTER the get_current_time fallback block, not
        before it."""
        from pathlib import Path

        src = Path("src/agent/nodes/react_generate.py").read_text(
            encoding="utf-8"
        )
        # Find the index of the placeholder assignment.
        placeholder_idx = src.find(
            'answer_text = "(no answer text generated)"'
        )
        assert placeholder_idx != -1, (
            "placeholder assignment not found — was it renamed?"
        )
        # Find the index of the get_current_time fallback.
        fallback_idx = src.find(
            'isinstance(_m, ToolMessage)\n                and '
            '(getattr(_m, "name", "") or "") == "get_current_time"'
        )
        assert fallback_idx != -1, (
            "get_current_time TM fallback block not found — was it renamed?"
        )
        # Find the streaming-loop ``if not answer_text:`` end (line 715).
        # The placeholder MUST appear AFTER the fallback block.
        assert fallback_idx < placeholder_idx, (
            "v2.0.28.12.1 fix broken: placeholder assignment "
            "(answer_text = '(no answer text generated)') MUST come "
            f"AFTER the get_current_time TM fallback block "
            f"(placeholder at {placeholder_idx}, fallback at "
            f"{fallback_idx}). Reverting to pre-fix order "
            "short-circuits the fallback."
        )

    def test_fallback_fires_when_streaming_produces_no_text(self):
        """Unit: when ``answer_text`` stays empty after streaming AND
        history contains a get_current_time TM, the fallback block
        populates ``answer_text`` with a time-stamped string (NOT the
        ``"(no answer text generated)"`` placeholder)."""
        import asyncio
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        from src.agent.nodes.react_generate import (
            react_generate,
            _sanitize_history_for_generate,
        )

        # 1. Build a state with thinking-only history.
        history = [
            HumanMessage(content="现在几点了"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_test_1",
                    "name": "get_current_time",
                    "args": {"timezone": "Asia/Shanghai"},
                }],
            ),
            ToolMessage(
                content="2026-09-23T13:30:00+08:00 (Asia/Shanghai)\n"
                "2026年9月23日 周三 13:30",
                tool_call_id="call_test_1",
                name="get_current_time",
            ),
        ]

        async def run():
            captured_msgs: list = []
            from src.agent.nodes import react_generate as rg_mod

            original_build = rg_mod.build_chat_model

            def fake_build(*args, **kwargs):
                # Capture msgs at astream time.
                class _Stub:
                    async def astream(self, msgs):
                        captured_msgs.extend(msgs)
                        # Stream ONLY thinking blocks, no visible text.
                        for chunk_text in [
                            "thinking: 用户问了当前时间。",
                            "thinking: 我有 get_current_time 数据。",
                        ]:
                            yield AIMessage(
                                content=[{
                                    "type": "thinking",
                                    "thinking": chunk_text,
                                }]
                            )

                return _Stub()

            rg_mod.build_chat_model = fake_build
            try:
                state = {
                    "messages": history,
                    "route_decision": "qa_complex",
                    "needs_current_time": True,
                }
                events: list = []
                async for ev in react_generate(state):
                    events.append(ev)
                return captured_msgs, events
            finally:
                rg_mod.build_chat_model = original_build

        captured, events = asyncio.run(run())

        # 2. The streaming loop was entered (proves astream was reached).
        assert len(captured) > 0, "synthesis LLM astream was not called"

        # 3. The LAST delta event should carry the constructed fallback
        # answer (NOT the placeholder string).
        last_delta = None
        for ev in events:
            if isinstance(ev, tuple) and ev[0] == "__delta__":
                last_delta = ev
        assert last_delta is not None, "no __delta__ event emitted"
        delta = last_delta[1]
        assert delta.get("answer") is not None, "delta missing 'answer'"
        answer = delta["answer"]

        # The fallback MUST have fired.
        assert "(no answer text generated)" not in answer, (
            f"v2.0.28.12.1 fix broken: synthesis fallback did not fire; "
            f"answer still contains the placeholder: {answer!r}"
        )
        # And it MUST include the time data.
        assert "2026" in answer and ("13:30" in answer or "周三" in answer), (
            f"constructed fallback answer missing time data: {answer!r}"
        )