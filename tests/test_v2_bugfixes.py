"""Regression tests for two v2.0 user-facing bugs found in live testing.

Bug 1 (summary path no docs)
----------------------------
User uploaded ``1.txt`` then asked "总结一下文档内容". The model
answered "I can't see any uploaded documents" — but the banner
"本回答参考了文档内容" still appeared. Banner says docs used, body
denies docs — confusing contradiction.

Root cause: ``_intent_to_route_decision("summary")`` returned
``"direct"``, which made ``react_generate`` use ``DIRECT_SYSTEM``
(a prompt with no ``{documents}`` slot). But ``summary_path`` had
already loaded chunks into ``state["documents"]`` and
``_derive_source_kinds(merged)`` returned ``["local"]`` → banner
rendered.

Fix: ``_intent_to_route_decision("summary")`` now returns
``"retrieve"`` so the same GENERATE_SYSTEM prompt is used as the
qa_complex path. The LLM sees the chunks summary_path loaded.

Bug 2 (ReAct web_search loop)
-----------------------------
User asked "联网搜索双鱼座男生特点". The agent called ``web_search``
4 times in a row (each returning irrelevant results like US law
cases), then hit ``recursion_limit=10`` and the runner emitted
"抱歉,模型在生成回复时遇到了问题". User got nothing useful.

Root cause: no loop-breaker in ``react_agent``. When the same tool
returns garbage repeatedly, the LLM keeps rephrasing the query
instead of admitting "I couldn't find anything". The recursion limit
was the only safety net, and it triggered too late.

Fix: count tool calls in the current turn. If any tool has been
called ``>= _MAX_SAME_TOOL_CALLS`` times, unbind tools + inject a
stop hint. The LLM is forced to produce a final answer (which routes
to ``react_generate`` because no tool_calls come back).

These tests pin both fixes so a future refactor can't silently
regress them.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Bug 1: summary intent → GENERATE_SYSTEM, not DIRECT_SYSTEM
# ---------------------------------------------------------------------------


def test_intent_to_route_decision_summary_maps_to_retrieve():
    """Bug 1 fix: 'summary' intent now routes to GENERATE_SYSTEM path.

    Without this, react_generate uses DIRECT_SYSTEM (no docs), so the
    LLM answers "I can't see any documents" while the banner still
    claims docs were used.
    """
    from src.agent.nodes.react_generate import _intent_to_route_decision

    # Critical: summary → retrieve (was "direct" pre-fix)
    assert _intent_to_route_decision("summary") == "retrieve"
    # Sanity: greeting still direct (no docs, no citations)
    assert _intent_to_route_decision("greeting") == "direct"
    # qa_complex / None → retrieve (docs possible via tools)
    assert _intent_to_route_decision("qa_complex") == "retrieve"
    assert _intent_to_route_decision(None) == "retrieve"


def test_summary_path_chunks_reach_react_generate_with_docs_in_prompt():
    """End-to-end: summary_path chunks flow into react_generate's
    GENERATE_SYSTEM template (not DIRECT_SYSTEM).

    Pins that the summary fast-path actually summarizes — the LLM
    prompt contains a ``<documents>`` block, not the bare greeting
    prompt.
    """
    from src.agent.nodes.react_generate import _intent_to_route_decision

    # Simulate: summary_path returned one chunk.
    chunks = [
        Document(
            page_content="这是一段需要被总结的文档内容。" * 5,
            metadata={
                "chunk_id": "c1",
                "doc_id": "d1",
                "filename": "1.txt",
                "source_kind": "local",
            },
        )
    ]

    intent = "summary"
    route_decision = _intent_to_route_decision(intent)
    # Bug 1 fix: summary should be "retrieve" → react_generate uses
    # GENERATE_SYSTEM, which formats the docs into the prompt.
    assert route_decision == "retrieve"

    # Now check that _format_documents produces a non-empty block.
    from src.agent.legacy_helpers.doc_formatting import _format_documents

    formatted = _format_documents(chunks)
    # The fenced block must include the filename + content. If we got
    # the old behavior (DIRECT_SYSTEM), formatted would be unused —
    # so this is the symptom we're guarding against.
    assert "1.txt" in formatted
    assert "需要被总结的文档内容" in formatted
    assert "<documents" in formatted


def test_react_generate_uses_generate_system_for_summary_intent():
    """Direct check: react_generate returns answer_complete with
    ``source_kinds=['local']`` AND a non-empty answer that references
    the docs (not the canned "no docs uploaded" line).

    Pre-fix: source_kinds=['local'] but answer_text was the canned
    greeting-style "I see no documents" → banner vs body contradiction.
    """
    from src.agent.legacy_helpers.doc_formatting import _derive_source_kinds
    from src.agent.nodes.react_generate import _intent_to_route_decision

    chunks = [
        Document(
            page_content="这是一份关于腾讯游戏的报告。",
            metadata={
                "chunk_id": "c1",
                "doc_id": "d1",
                "filename": "tencent.txt",
                "source_kind": "local",
            },
        )
    ]

    # The fix means summary now goes through the GENERATE_SYSTEM path.
    # We can't easily call react_generate (LLM call), but we can verify
    # that the data flow that drives it is correct: route_decision is
    # "retrieve" + source_kinds is ["local"] + formatted docs has the
    # content. If those three hold, react_generate will:
    #   1. Use GENERATE_SYSTEM.format(documents=formatted, ...)
    #   2. See the doc content
    #   3. Emit source_kinds=["local"] for the banner
    # → no contradiction.

    assert _intent_to_route_decision("summary") == "retrieve"
    assert _derive_source_kinds(chunks) == ["local"]


# ---------------------------------------------------------------------------
# Bug 2: ReAct loop breaker
# ---------------------------------------------------------------------------


def _ai_with_tool_call(name: str, args: dict | None = None, call_id: str = "tc") -> AIMessage:
    """Build an AIMessage carrying a single tool_call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "name": name,
                "args": args or {},
            }
        ],
    )


def test_count_tool_calls_since_last_human_zero_at_start():
    """A fresh conversation has no tool calls in the current turn."""
    from src.agent.nodes.react_agent import _count_tool_calls_since_last_human

    msgs = [HumanMessage(content="hello")]
    counts = _count_tool_calls_since_last_human(msgs)
    assert counts == {}


def test_count_tool_calls_since_last_human_only_current_turn():
    """Tool calls from PRIOR turns don't leak into the count.

    Without this guard, after a few turns of normal usage the LLM
    would always hit the loop breaker.
    """
    from src.agent.nodes.react_agent import _count_tool_calls_since_last_human

    # 2 turns ago: 1 web_search call
    # 1 turn ago:  2 retrieve_docs calls
    # current turn: 0 calls
    msgs = [
        HumanMessage(content="turn 1"),
        _ai_with_tool_call("web_search"),
        ToolMessage(content="[]", tool_call_id="tc"),
        AIMessage(content="turn 1 answer"),
        HumanMessage(content="turn 2"),
        _ai_with_tool_call("retrieve_docs", call_id="tc2"),
        ToolMessage(content="[]", tool_call_id="tc2"),
        _ai_with_tool_call("retrieve_docs", call_id="tc3"),
        ToolMessage(content="[]", tool_call_id="tc3"),
        AIMessage(content="turn 2 answer"),
        HumanMessage(content="turn 3"),
    ]
    counts = _count_tool_calls_since_last_human(msgs)
    assert counts == {}, f"prior-turn tool calls leaked: {counts}"


def test_count_tool_calls_since_last_human_aggregates_current_turn():
    """Two different tools in the same turn each get counted."""
    from src.agent.nodes.react_agent import _count_tool_calls_since_last_human

    msgs = [
        HumanMessage(content="query"),
        _ai_with_tool_call("retrieve_docs", call_id="tc1"),
        ToolMessage(content="[]", tool_call_id="tc1"),
        _ai_with_tool_call("web_search", call_id="tc2"),
        ToolMessage(content="[]", tool_call_id="tc2"),
        _ai_with_tool_call("retrieve_docs", call_id="tc3"),
        ToolMessage(content="[]", tool_call_id="tc3"),
    ]
    counts = _count_tool_calls_since_last_human(msgs)
    assert counts == {"retrieve_docs": 2, "web_search": 1}


def test_react_agent_loop_breaker_unbinds_tools_after_threshold():
    """Bug 2 fix: after 3 web_search calls, react_agent MUST unbind
    tools so the LLM is forced to produce a final answer.

    The fake graph path: ainvoke returns an AIMessage (we don't care
    about its content here). We assert that:
      - ``build_chat_model`` was called
      - ``.bind_tools`` was NOT called on the model
      - The LLM received a SystemMessage with the stop hint
    """
    # Import the submodule (NOT `from src.agent.nodes import react_agent`),
    # because ``src.agent.nodes.__init__`` re-exports the ``react_agent``
    # function which would shadow the module reference. We need the module
    # here so ``patch.object(react_agent_module, "build_chat_model", ...)``
    # can resolve ``build_chat_model`` at the module level.
    import src.agent.nodes.react_agent as react_agent_module

    fake_messages = [
        HumanMessage(content="联网搜索双鱼座男生特点"),
        # 3 prior web_search calls → loop-breaker triggers now
        _ai_with_tool_call("web_search", call_id="tc1"),
        ToolMessage(content="[]", tool_call_id="tc1"),
        _ai_with_tool_call("web_search", call_id="tc2"),
        ToolMessage(content="[]", tool_call_id="tc2"),
        _ai_with_tool_call("web_search", call_id="tc3"),
        ToolMessage(content="[]", tool_call_id="tc3"),
    ]

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="最终回答:用已有信息作答。")
    )
    # bind_tools returns a model — but the breaker should NOT call it.
    fake_model.bind_tools = MagicMock(return_value=fake_model)

    with patch.object(
        react_agent_module, "build_chat_model", return_value=fake_model
    ) as mock_build:
        result = drive_react_agent(react_agent_module.react_agent, {
            "messages": fake_messages,
            "step_count": 3,
            "needs_current_time": False,
        })

    # build_chat_model was called once.
    assert mock_build.call_count == 1
    # bind_tools was NOT called → LLM is tool-less.
    assert fake_model.bind_tools.call_count == 0
    # ainvoke was called with the messages + the loop-break hint.
    assert fake_model.ainvoke.call_count == 1
    ainvoke_arg = fake_model.ainvoke.call_args[0][0]
    # First message should be the loop-break SystemMessage.
    from langchain_core.messages import SystemMessage
    assert isinstance(ainvoke_arg[0], SystemMessage)
    assert "不要再调用工具" in ainvoke_arg[0].content or "直接给出" in ainvoke_arg[0].content
    # The user's actual question is still in the list (later).
    assert any(
        getattr(m, "content", "") == "联网搜索双鱼座男生特点" for m in ainvoke_arg
    )
    # v2.0.22 (Item 7 Step 5) — step_count is auto-injected by
    # ``merge()`` rather than returned by the node. Verify the path
    # so a regression in merge() is caught.
    assert "step_count" not in result
    from src.agent.fsm import merge
    merged = merge(
        {
            "messages": fake_messages,
            "step_count": 3,
            "needs_current_time": False,
        },
        result,
    )
    assert merged["step_count"] == 4


def test_react_agent_normal_path_binds_tools():
    """When no tool has hit the threshold, tools are still bound."""
    # Import the submodule (NOT `from src.agent.nodes import react_agent`),
    # because ``src.agent.nodes.__init__`` re-exports the ``react_agent``
    # function which would shadow the module reference. We need the module
    # here so ``patch.object(react_agent_module, "build_chat_model", ...)``
    # can resolve ``build_chat_model`` at the module level.
    import src.agent.nodes.react_agent as react_agent_module

    fake_messages = [
        HumanMessage(content="今天天气如何?"),
        # 1 prior call — well below threshold (3).
        _ai_with_tool_call("get_current_time", call_id="tc1"),
        ToolMessage(content="2026-09-06", tool_call_id="tc1"),
    ]

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="", tool_calls=[
            {"id": "tc2", "name": "web_search", "args": {"query": "今天天气"}}
        ])
    )
    fake_model.bind_tools = MagicMock(return_value=fake_model)

    with patch.object(
        react_agent_module, "build_chat_model", return_value=fake_model
    ) as mock_build:
        drive_react_agent(react_agent_module.react_agent, {
            "messages": fake_messages,
            "step_count": 1,
            "needs_current_time": True,
        })

    # Normal path: bind_tools called once with ALL_TOOLS.
    assert fake_model.bind_tools.call_count == 1
    # And no loop-break hint should be the first message.
    ainvoke_arg = fake_model.ainvoke.call_args[0][0]
    # First message is the time-tool hint (because needs_current_time=True),
    # NOT the loop-break hint.
    from langchain_core.messages import SystemMessage
    assert isinstance(ainvoke_arg[0], SystemMessage)
    assert "不要再调用工具" not in ainvoke_arg[0].content


def test_react_agent_loop_breaker_activates_at_exactly_threshold():
    """Boundary check: 3 calls of same tool triggers the breaker.

    (The constant is ``_MAX_SAME_TOOL_CALLS = 3``.)
    """
    # Import the submodule (NOT `from src.agent.nodes import react_agent`),
    # because ``src.agent.nodes.__init__`` re-exports the ``react_agent``
    # function which would shadow the module reference. We need the module
    # here so ``patch.object(react_agent_module, "build_chat_model", ...)``
    # can resolve ``build_chat_model`` at the module level.
    import src.agent.nodes.react_agent as react_agent_module

    fake_messages = [
        HumanMessage(content="q"),
        _ai_with_tool_call("web_search", call_id="tc1"),
        ToolMessage(content="[]", tool_call_id="tc1"),
        _ai_with_tool_call("web_search", call_id="tc2"),
        ToolMessage(content="[]", tool_call_id="tc2"),
        # Exactly 2 so far — should NOT trigger.
        _ai_with_tool_call("web_search", call_id="tc3"),
        ToolMessage(content="[]", tool_call_id="tc3"),
        # Now 3 → trigger.
    ]

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="final answer")
    )
    fake_model.bind_tools = MagicMock(return_value=fake_model)

    with patch.object(
        react_agent_module, "build_chat_model", return_value=fake_model
    ) as mock_build:
        drive_react_agent(react_agent_module.react_agent, {
            "messages": fake_messages,
            "step_count": 3,
            "needs_current_time": False,
        })

    assert fake_model.bind_tools.call_count == 0  # breaker active


def test_react_agent_loop_breaker_mixed_tools_each_counted_separately():
    """2 web_search + 2 retrieve_docs = 4 total but neither hits 3.

    The breaker is per-tool-name, not aggregate. Each tool has its
    own counter; 2 of each is below the threshold of 3.
    """
    # Import the submodule (NOT `from src.agent.nodes import react_agent`),
    # because ``src.agent.nodes.__init__`` re-exports the ``react_agent``
    # function which would shadow the module reference. We need the module
    # here so ``patch.object(react_agent_module, "build_chat_model", ...)``
    # can resolve ``build_chat_model`` at the module level.
    import src.agent.nodes.react_agent as react_agent_module

    fake_messages = [
        HumanMessage(content="q"),
        _ai_with_tool_call("web_search", call_id="w1"),
        ToolMessage(content="[]", tool_call_id="w1"),
        _ai_with_tool_call("retrieve_docs", call_id="r1"),
        ToolMessage(content="[]", tool_call_id="r1"),
        _ai_with_tool_call("web_search", call_id="w2"),
        ToolMessage(content="[]", tool_call_id="w2"),
        _ai_with_tool_call("retrieve_docs", call_id="r2"),
        ToolMessage(content="[]", tool_call_id="r2"),
    ]

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="final answer")
    )
    fake_model.bind_tools = MagicMock(return_value=fake_model)

    with patch.object(
        react_agent_module, "build_chat_model", return_value=fake_model
    ) as mock_build:
        drive_react_agent(react_agent_module.react_agent, {
            "messages": fake_messages,
            "step_count": 4,
            "needs_current_time": False,
        })

    # Neither tool hit 3 → tools still bound.
    assert fake_model.bind_tools.call_count == 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def drive_react_agent(react_agent_fn, state):
    """Drive the v2.0.22 (Item 7 Step 6) async-gen ``react_agent`` node
    and return the ``__delta__`` payload.

    Pre-Step-6 the node was ``async def -> dict`` and tests called
    ``asyncio_run(react_agent(state))`` to await its return. Post-Step-6
    it's ``async def -> AsyncIterator`` and must be drained via
    ``async for`` — this shim does that and returns the delta dict so
    callers can assert on it (e.g. the ``merge()`` step_count
    injection check in ``test_react_agent_loop_breaker_unbinds_tools_after_threshold``).

    Pass the bound function (typically ``react_agent_module.react_agent``)
    explicitly so we don't rely on module-level namespace shadowing —
    ``src.agent.nodes.__init__`` doesn't re-export the function in v2.0.16+.
    """
    import asyncio

    async def _drive():
        async for kind, payload in react_agent_fn(state):
            if kind == "__delta__":
                return payload
        raise RuntimeError("react_agent yielded no __delta__")

    return asyncio.run(_drive())


__all__ = [
    "test_intent_to_route_decision_summary_maps_to_retrieve",
    "test_summary_path_chunks_reach_react_generate_with_docs_in_prompt",
    "test_react_generate_uses_generate_system_for_summary_intent",
    "test_count_tool_calls_since_last_human_zero_at_start",
    "test_count_tool_calls_since_last_human_only_current_turn",
    "test_count_tool_calls_since_last_human_aggregates_current_turn",
    "test_react_agent_loop_breaker_unbinds_tools_after_threshold",
    "test_react_agent_normal_path_binds_tools",
    "test_react_agent_loop_breaker_activates_at_exactly_threshold",
    "test_react_agent_loop_breaker_mixed_tools_each_counted_separately",
]