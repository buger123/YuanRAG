"""Probe — drive the user's exact 2-turn flow in-process and capture
the synthesis LLM prompt so we can see WHY the LLM thinks "no new user
message".

Run with: cd "D:/prep for work/Projects/YuanRAG" && python -m pytest tests/debug_now_multiturn.py -v -s 2>&1 | head -80
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _initial_state(thread_id: str = "") -> Dict[str, Any]:
    return {
        "messages": [],
        "thread_id": thread_id,
        "original_query": "现在几点了",
        "current_query": "现在几点了",
        "step_count": 0,
        "intent": None,
        "corrected_query": None,
        "needs_current_time": True,
        "created_at": None,
    }


@pytest.mark.asyncio
async def test_user_multiturn_synthesis_prompt(tmp_path, monkeypatch, capsys):
    """Drive the user's exact scenario: turn 1 = intro, turn 2 = time.

    Pre-seed state with turn 1's messages, then run FSM for turn 2.
    Capture the synthesis LLM's prompt and dump it for analysis.
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
        thread_id = "test-multiturn-probe-1"

        # intent_analysis → qa_complex
        async def fake_intent_analysis(state, *, step=0):
            yield ("__delta__", {"intent": "qa_complex", "needs_current_time": True})

        # react_agent: 1 tool-call round (text-only round skipped —
        # for the probe, the LLM only calls get_current_time and
        # the synthesis runs).
        round1_response = AIMessage(
            content="让我查一下当前时间。",
            tool_calls=[{"id": "tc-time-1", "name": "get_current_time", "args": {}}],
        )
        # No round 2 — react_agent emits text-only draft is the
        # common case but we want to test multi-turn so let's also
        # emit a draft (the typical flow).
        round2_draft = AIMessage(content="DRAFT: 让我准备一个简洁的回答。")
        fake_model = MagicMock()
        fake_model.bind_tools = MagicMock(return_value=fake_model)
        fake_model.ainvoke = AsyncMock(side_effect=[round1_response, round2_draft])

        import src.agent.nodes.react_agent as ra
        monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

        # _tools_step
        async def fake_tools_step(state, *, step=0):
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
                content="2026-09-23T12:40:13.300697+08:00 (Asia/Shanghai)\n2026年9月23日 周三 12:40",
                tool_call_id="tc-time-1",
                name="get_current_time",
            )
            yield ("__delta__", {"messages": [tm]})

        # Capture synthesis LLM prompt
        captured_msgs: List[List[Any]] = []
        chunk = MagicMock()
        chunk.content = "（等待你的下一条消息～）"  # mimic the user's bug

        async def fake_astream(msgs):
            captured_msgs.append(list(msgs))
            yield chunk

        synthesis_model = MagicMock()
        synthesis_model.bind_tools = MagicMock(return_value=synthesis_model)
        synthesis_model.astream = fake_astream

        import src.agent.nodes.react_generate as rg
        monkeypatch.setattr(rg, "build_chat_model", lambda **kw: synthesis_model)

        # Patch only intent + tools; let real react_generate run.
        import src.agent.fsm as fsm_mod
        original_intent = dict(fsm_mod.NODES["intent_analysis"])
        original_tools = dict(fsm_mod.NODES["tools"])
        try:
            fsm_mod.NODES["intent_analysis"] = {**original_intent, "fn": fake_intent_analysis}
            fsm_mod.NODES["tools"] = {**original_tools, "fn": fake_tools_step}

            state = _initial_state(thread_id=thread_id)
            # Pre-seed TURN 1's messages (intro Q + intro A).
            state["messages"] = [
                HumanMessage(content="你是谁"),
                AIMessage(
                    content=(
                        "你好！我是源 RAG（Yuan RAG），一个本地优先的 RAG 助手。"
                        "我的主要工作是帮你理解和分析你上传的文档，也可以回答一般性的知识问题，"
                        "必要时还能联网搜索最新信息。\n\n有什么我可以帮你的吗？😊"
                    ),
                ),
                # TURN 2 — current turn's user question
                HumanMessage(content="现在几点了"),
            ]

            from src.agent.fsm import run_fsm

            async for _ in run_fsm(state, thread_id=thread_id, max_steps=10):
                pass

            assert captured_msgs, "synthesis LLM was never called"
            msgs = captured_msgs[0]

            # Dump the captured prompt.
            print("\n========== SYNTHESIS LLM PROMPT (msgs to model.astream) ==========")
            for i, m in enumerate(msgs):
                role = type(m).__name__
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    content_repr = "[" + ", ".join(
                        str(b)[:80] for b in content
                    ) + "]"
                else:
                    content_repr = str(content)
                tool_calls = getattr(m, "tool_calls", None) or []
                tool_call_names = [tc.get("name") for tc in tool_calls]
                print(
                    f"  [{i:2d}] {role} tool_calls={tool_call_names}\n"
                    f"       content={content_repr[:300]!r}"
                )

            # Sanity checks
            user_question_seen = any(
                isinstance(getattr(m, "content", ""), str)
                and "现在几点了" in getattr(m, "content", "")
                for m in msgs
            )
            print(f"\n>>> user question '现在几点了' in prompt: {user_question_seen}")

            tool_result_seen = any(
                isinstance(m, ToolMessage)
                and "2026-09-23T12:40" in getattr(m, "content", "")
                for m in msgs
            )
            print(f">>> tool result '2026-09-23T12:40' in prompt: {tool_result_seen}")
        finally:
            fsm_mod.NODES["intent_analysis"] = original_intent
            fsm_mod.NODES["tools"] = original_tools
    finally:
        await checkpointer.shutdown()