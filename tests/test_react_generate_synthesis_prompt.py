"""v2.0.28.11 — assert the synthesis LLM prompt EXCLUDES react_agent's draft.

Bug history
-----------
User reported (2026-09-23, post-v2.0.28.10 + DB-clear):

    Q: "现在是几点"
    A: "上一轮的回复简要总结:\n\n- **当前时间**:2026 年 9 月 23 日..."

The synthesis LLM was reading react_agent's text-only draft AIMessage
(labeled as "your previous reply" by ``_HISTORY_FRAME_HINT``) and
producing meta-summary text instead of a direct answer.

These tests drive the REAL react_generate end-to-end with a mock LLM
that captures the synthesis prompt (the ``msgs`` arg to
``model.astream(msgs)``). We then assert:

  1. The draft text is NOT in any message content of the prompt.
  2. The ToolMessage result IS in the prompt (regression guard for
     v2.0.9 — the time string the LLM needs to make a direct answer).

Pattern
-------
Mirrors ``tests/test_react_generate_replace.py`` (v2.0.28.10) — same
``tmp_path`` + ``monkeypatch`` direct test params + inline
``checkpointer.reset_for_tests() / startup() / shutdown()`` per the
v2.0.28.10 trap #6 (``pytest.PytestRemovedIn9Warning`` from
``@pytest.fixture async def``).
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_state(thread_id: str = "") -> Dict[str, Any]:
    """Minimal AgentState for FSM tests — matches runner._initial_state shape."""
    return {
        "messages": [HumanMessage(content="seed")],
        "thread_id": thread_id,
        "original_query": "seed",
        "current_query": "seed",
        "step_count": 0,
        "intent": None,
        "corrected_query": None,
        "needs_current_time": False,
        "created_at": None,
    }


def _make_fake_llm(*responses: AIMessage) -> MagicMock:
    """react_agent's LLM (round-robin). The synthesis LLM is patched
    separately inside each test."""
    fake = MagicMock()
    fake.bind_tools = MagicMock(return_value=fake)
    if len(responses) == 1:
        fake.ainvoke = AsyncMock(return_value=responses[0])
    else:
        fake.ainvoke = AsyncMock(side_effect=list(responses))
    return fake


async def _drain_run_fsm(state, **kwargs):
    from src.agent.fsm import run_fsm

    async for _ in run_fsm(state, **kwargs):
        pass


def _patch_node(name: str, fn) -> None:
    import src.agent.fsm as fsm_mod

    spec = dict(fsm_mod.NODES[name])
    spec["fn"] = fn
    fsm_mod.NODES[name] = spec


def _original_node_fn(name: str):
    import src.agent.fsm as fsm_mod

    return fsm_mod.NODES[name]["fn"]


def _make_capture_synthesis_model(captured_msgs: List[List[Any]],
                                   answer_text: str) -> MagicMock:
    """Build a MagicMock that the synthesis LLM (react_generate's
    model) uses. ``astream(msgs)`` records ``msgs`` and yields one
    chunk whose ``.content`` is ``answer_text``."""

    async def fake_astream(msgs):
        captured_msgs.append(list(msgs))
        chunk = MagicMock()
        chunk.content = answer_text
        yield chunk

    fake = MagicMock()
    fake.bind_tools = MagicMock(return_value=fake)
    fake.astream = fake_astream
    return fake


# ---------------------------------------------------------------------------
# 1. The synthesis prompt excludes the draft text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_llm_prompt_excludes_draft_text(tmp_path, monkeypatch):
    """Drive a qa_complex turn with 1 tool-call round + 1 text-only
    round (current bug scenario). The synthesis LLM is the one inside
    ``react_generate`` — capture the prompt and assert the draft
    text does NOT appear in any message content.

    Pre-v2.0.28.11 the draft ("现在是2026年9月23日上午 09:49") would
    appear as the LAST AIMessage content in the synthesis prompt,
    causing the LLM to misinterpret the turn as a request to
    summarize its prior reply.

    Post-v2.0.28.11 the trailing text-only AIMessage is stripped by
    ``_sanitize_history_for_generate`` — the prompt only contains the
    HM, the tool-calls AIMessage (parent of preserved TM), and the
    TM itself.
    """
    from src.storage import checkpointer

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    checkpointer.reset_for_tests()
    await checkpointer.startup()

    try:
        thread_id = "test-synthesis-prompt-1"

        # intent_analysis routes qa_complex → react_agent
        async def fake_intent_analysis(state, *, step=0):
            yield ("__delta__", {"intent": "qa_complex"})

        # react_agent round 1: tool_calls. Round 2: text-only draft.
        draft_text = "现在是2026年9月23日上午 09:49 (FSM draft)"
        round1_response = AIMessage(
            content=" ",
            tool_calls=[{
                "id": "tc-time",
                "name": "get_current_time",
                "args": {},
            }],
        )
        round2_response = AIMessage(content=draft_text)
        fake_model = _make_fake_llm(round1_response, round2_response)

        import src.agent.nodes.react_agent as ra
        monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

        # _tools_step: emit tool_call_end + ToolMessage for get_current_time.
        async def fake_tools_step(state, *, step=0):
            history = list(state.get("messages") or [])
            last = history[-1]
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
                content="2026-09-23T09:49:00+08:00 (Asia/Shanghai)\n2026年9月23日 周三 上午 09:49",
                tool_call_id="tc-time",
                name="get_current_time",
            )
            yield ("__delta__", {"messages": [tm]})

        # Capture what react_generate passes to model.astream(msgs).
        captured_msgs: List[List[Any]] = []
        synthesis_model = _make_capture_synthesis_model(
            captured_msgs,
            "现在时间是2026年9月23日 周三 上午 09:49",
        )
        import src.agent.nodes.react_generate as rg
        monkeypatch.setattr(
            rg, "build_chat_model", lambda **kw: synthesis_model
        )

        # Patch ONLY intent + tools; let the REAL react_generate run
        # so it actually invokes build_chat_model + astream(msgs).
        original_intent = _original_node_fn("intent_analysis")
        original_tools = _original_node_fn("tools")
        try:
            _patch_node("intent_analysis", fake_intent_analysis)
            _patch_node("tools", fake_tools_step)

            state = _initial_state(thread_id=thread_id)
            state["messages"] = [HumanMessage(content="现在是几点")]

            await _drain_run_fsm(state, thread_id=thread_id, max_steps=10)

            assert captured_msgs, (
                "synthesis LLM (react_generate's model.astream) was "
                "never called — the test wiring is broken"
            )

            msgs = captured_msgs[0]

            # THE CRITICAL ASSERTION: the draft text must NOT appear in
            # ANY message content of the synthesis prompt.
            for m in msgs:
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    assert draft_text not in content, (
                        f"v2.0.28.11 fix: synthesis LLM prompt must NOT "
                        f"contain the react_agent draft text. Found it in "
                        f"a {type(m).__name__} with content={content!r}. "
                        f"Pre-fix this would cause meta-summary text like "
                        f"'上一轮的回复简要总结:...'"
                    )
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            block_text = block.get("text", "") or ""
                            assert draft_text not in block_text, (
                                f"v2.0.28.11 fix: draft text found in a "
                                f"content block: {block_text!r}"
                            )

            # Sanity: the prompt DOES contain the user's actual question.
            user_question_seen = any(
                isinstance(getattr(m, "content", ""), str)
                and "现在是几点" in getattr(m, "content", "")
                for m in msgs
            )
            assert user_question_seen, (
                "synthesis LLM prompt must include the user's question "
                "'现在是几点' somewhere — the test wiring is broken if absent"
            )
        finally:
            _patch_node("intent_analysis", original_intent)
            _patch_node("tools", original_tools)
    finally:
        await checkpointer.shutdown()


# ---------------------------------------------------------------------------
# 2. The synthesis prompt INCLUDES the time ToolMessage
#    (regression guard for v2.0.9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_llm_prompt_includes_tool_result(tmp_path, monkeypatch):
    """v2.0.9 regression guard — the time ToolMessage MUST still be
    in the synthesis prompt after the v2.0.28.11 strip. Without
    this the LLM has no source material and would hallucinate
    "抱歉,我无法确定现在...".

    The strip targets text-only AIMessages only; ToolMessages with
    whitelisted names (``_STRING_RETURNING_TOOLS``) are preserved by
    the existing v2.0.9 / v2.0.10.1 logic.
    """
    from src.storage import checkpointer

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    checkpointer.reset_for_tests()
    await checkpointer.startup()

    try:
        thread_id = "test-synthesis-prompt-2"

        async def fake_intent_analysis(state, *, step=0):
            yield ("__delta__", {"intent": "qa_complex"})

        round1_response = AIMessage(
            content=" ",
            tool_calls=[{"id": "tc-time", "name": "get_current_time", "args": {}}],
        )
        round2_response = AIMessage(content="draft text")
        fake_model = _make_fake_llm(round1_response, round2_response)

        import src.agent.nodes.react_agent as ra
        monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

        time_string = "2026-09-23T09:49:00+08:00 (Asia/Shanghai)"

        async def fake_tools_step(state, *, step=0):
            history = list(state.get("messages") or [])
            last = history[-1]
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
                content=time_string,
                tool_call_id="tc-time",
                name="get_current_time",
            )
            yield ("__delta__", {"messages": [tm]})

        captured_msgs: List[List[Any]] = []
        synthesis_model = _make_capture_synthesis_model(
            captured_msgs, "现在是09:49"
        )
        import src.agent.nodes.react_generate as rg
        monkeypatch.setattr(
            rg, "build_chat_model", lambda **kw: synthesis_model
        )

        original_intent = _original_node_fn("intent_analysis")
        original_tools = _original_node_fn("tools")
        try:
            _patch_node("intent_analysis", fake_intent_analysis)
            _patch_node("tools", fake_tools_step)

            state = _initial_state(thread_id=thread_id)
            state["messages"] = [HumanMessage(content="现在是几点")]

            await _drain_run_fsm(state, thread_id=thread_id, max_steps=10)

            assert captured_msgs, "synthesis LLM was not called"
            msgs = captured_msgs[0]

            tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
            assert tool_msgs, (
                "v2.0.9 regression guard: time ToolMessage missing from "
                "synthesis prompt. The LLM would have no source material "
                "and would hallucinate 'I can't determine the current time'."
            )
            assert any(
                time_string in (getattr(m, "content", "") or "")
                for m in tool_msgs
            ), (
                f"time string {time_string!r} not found in any ToolMessage "
                f"content {[getattr(m, 'content', '')[:50] for m in tool_msgs]!r}"
            )

            # The parent AIMessage(tool_calls=[...]) must also survive
            # (Anthropic's tool_use ↔ tool_result pairing invariant).
            aim_msgs = [m for m in msgs if isinstance(m, AIMessage)]
            assert aim_msgs, "no AIMessage in synthesis prompt — broken"
            parent_aim = [
                m for m in aim_msgs
                if getattr(m, "tool_calls", None)
            ]
            assert parent_aim, (
                "v2.0.10.1 regression guard: parent AIMessage(tool_calls=[...]) "
                "missing from synthesis prompt. Anthropic would reject the "
                "request with 'tool call result does not follow tool call (2013)'."
            )
        finally:
            _patch_node("intent_analysis", original_intent)
            _patch_node("tools", original_tools)
    finally:
        await checkpointer.shutdown()