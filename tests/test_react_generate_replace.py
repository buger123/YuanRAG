"""v2.0.28.10 — end-to-end FSM tests pinning the 1-AIMessage-per-turn contract.

Pre-v2.0.28.10 the FSM saved 2 AIMessages per turn when no tool calls
were involved: react_agent's text-only "draft" + react_generate's
"synthesis". The frontend's ``answerCompleteHandler`` canonical
overwrite (src/frontend/src/chat/eventHandlers/answerCompleteHandler.ts:116-143)
hid the draft in the live UI, but the DB had both → reload showed
extra assistant bubbles that were never seen live.

The fix: react_generate yields ``("__delta__", {"replace_last_message":
response})`` instead of ``"messages": [response]``; ``merge()``
(src/agent/fsm.py) handles the new key by replacing the last message
in state["messages"].

These tests drive ``run_fsm`` end-to-end with mock LLM + mock tools
and assert on the SAVED state (via ``checkpointer.load_latest``) to
pin the message-list shape after a complete turn.

Note on FSM state visibility
----------------------------
``merge()`` returns a NEW dict (shallow copy of the user's state), so
the user's ``state["messages"]`` reference is NOT mutated by the FSM
when the delta carries ``messages`` or ``replace_last_message``. The
ONLY way to inspect the post-FSM state is via the checkpointer
(``load_latest(thread_id)``), which is what these tests do.

Mirrors ``tests/test_fsm_tool_lifecycle.py`` patterns:
- MagicMock + AsyncMock for the LLM.
- ``tmp_path`` + ``monkeypatch`` directly as test parameters (not as a
  fixture), with ``checkpointer.reset_for_tests()`` + ``await
  startup()`` inlined per test, then ``await shutdown()`` in finally.
  Same pattern as tests/test_checkpointer_corruption_contract.py:65-87.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_state(thread_id: str = "") -> Dict[str, Any]:
    """Minimal AgentState for FSM tests — matches runner._initial_state shape."""
    return {
        "messages": [HumanMessage(content="seed")],  # start with 1 HM
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
    """Build a MagicMock with the contract ``react_agent`` expects:
    ``build_chat_model(**kw).bind_tools(tools).ainvoke(msgs) → response``.

    Each ``ainvoke`` call returns the next response (round-robin). If
    only one is given, every call returns it.
    """
    fake = MagicMock()
    fake.bind_tools = MagicMock(return_value=fake)

    if len(responses) == 1:
        fake.ainvoke = AsyncMock(return_value=responses[0])
    else:
        # Round-robin via side_effect list
        fake.ainvoke = AsyncMock(side_effect=list(responses))

    return fake


async def _drain_run_fsm(state, **kwargs):
    """Run ``run_fsm`` and exhaust the async generator."""
    from src.agent.fsm import run_fsm

    async for _ in run_fsm(state, **kwargs):
        pass


def _patch_node(name: str, fn) -> None:
    """Replace ``NODES[name]["fn"]`` with ``fn``."""
    import src.agent.fsm as fsm_mod

    spec = dict(fsm_mod.NODES[name])
    spec["fn"] = fn
    fsm_mod.NODES[name] = spec


def _original_node_fn(name: str):
    """Snapshot of the original node fn for try/finally restoration."""
    import src.agent.fsm as fsm_mod

    return fsm_mod.NODES[name]["fn"]


# ---------------------------------------------------------------------------
# 1) react_generate replaces react_agent's draft — single-round, no tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_generate_replaces_react_agent_draft(tmp_path, monkeypatch):
    """Drive a qa_complex turn (no tool calls). The DB after the turn
    should have exactly 1 AIMessage from this turn — the synthesis.
    react_agent's text-only "draft" AIMessage is REPLACED, not appended.

    Pre-v2.0.28.10 this test would fail: saved state would have
    [HM, AIM_draft, AIM_synthesis] (3 messages from 1 turn).
    Post-v2.0.28.10: saved state = [HM, AIM_synthesis] (2 messages).
    """
    from src.storage import checkpointer

    import src.agent.nodes.react_agent as ra

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    checkpointer.reset_for_tests()
    await checkpointer.startup()

    try:
        thread_id = "test-thread-1"

        # intent_analysis routes qa_complex → react_agent → react_generate
        async def fake_intent_analysis(state, *, step=0):
            yield ("__delta__", {"intent": "qa_complex"})

        # react_agent returns text-only AIMessage (no tool_calls → routes to react_generate)
        draft_aimessage = AIMessage(content="USA answer draft")
        fake_model = _make_fake_llm(draft_aimessage)
        monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

        # react_generate yields the canonical synthesis via replace_last_message
        synthesis_aimessage = AIMessage(
            content="synthesis answer",
            additional_kwargs={
                "sources": [{"id": "doc-1"}],
                "created_at": "2026-09-23T10:00:00Z",
            },
        )

        async def fake_react_generate(state, *, step=0):
            yield ("answer_complete", {
                "answer": "synthesis answer",
                "sources": [{"id": "doc-1"}],
                "source_kinds": ["local"],
                "created_at": "2026-09-23T10:00:00Z",
                "route_decision": "retrieve",
            })
            yield ("__delta__", {
                "answer": "synthesis answer",
                "sources": [{"id": "doc-1"}],
                "source_kinds": ["local"],
                "documents": [],
                "replace_last_message": synthesis_aimessage,
                "created_at": "2026-09-23T10:00:00Z",
            })

        # Patch the two nodes we're testing
        original_intent = _original_node_fn("intent_analysis")
        original_rg = _original_node_fn("react_generate")
        try:
            _patch_node("intent_analysis", fake_intent_analysis)
            _patch_node("react_generate", fake_react_generate)

            state = _initial_state(thread_id=thread_id)
            state["messages"] = [HumanMessage(content="美国首都是哪里?")]

            await _drain_run_fsm(state, thread_id=thread_id, max_steps=10)

            # Read what was saved to the checkpointer
            saved = await checkpointer.load_latest(thread_id)
            assert saved is not None, "FSM should have saved the turn state"
            messages = saved.get("messages") or []

            # The CRITICAL assertion: exactly 2 messages after the turn —
            # the HumanMessage + the synthesis (which replaced the draft).
            assert len(messages) == 2, (
                f"expected 2 saved messages (HM + synthesis); got {len(messages)}: "
                f"{[(type(m).__name__, getattr(m, 'content', '')[:30]) for m in messages]}"
            )
            assert isinstance(messages[0], HumanMessage)
            assert messages[0].content == "美国首都是哪里?"
            assert isinstance(messages[1], AIMessage)
            # The synthesis is at index 1 — react_agent's draft was REPLACED.
            assert messages[1].content == "synthesis answer"
            # The draft ("USA answer draft") is GONE from saved state.
            assert all(
                getattr(m, "content", "") != "USA answer draft"
                for m in messages
            ), "react_agent's draft must be replaced by react_generate's synthesis"
        finally:
            _patch_node("intent_analysis", original_intent)
            _patch_node("react_generate", original_rg)
    finally:
        await checkpointer.shutdown()


# ---------------------------------------------------------------------------
# 2) react_generate_direct still APPENDS (no replace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_generate_direct_still_appends(tmp_path, monkeypatch):
    """react_generate_direct runs for greeting/simple_fact WITHOUT a
    prior react_agent. state["messages"] last entry is the HumanMessage.
    react_generate_direct must APPEND (not replace) — replacing would
    clobber the user's question.

    Pinning this prevents future regressions where someone copies the
    ``replace_last_message`` change from react_generate into
    react_generate_direct by mistake.
    """
    from src.storage import checkpointer

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    checkpointer.reset_for_tests()
    await checkpointer.startup()

    try:
        thread_id = "test-thread-2"

        # The actual react_generate_direct streams from a real LLM. We
        # can't drive it without a real LLM, so we patch the FSM node.
        direct_aimessage = AIMessage(content="direct greeting response")

        async def fake_react_generate_direct(state, *, step=0):
            yield ("answer_complete", {
                "answer": "direct greeting response",
                "sources": [],
                "source_kinds": [],
                "created_at": "2026-09-23T10:00:00Z",
                "route_decision": "direct",
            })
            yield ("__delta__", {
                "answer": "direct greeting response",
                "sources": [],
                "source_kinds": [],
                "documents": [],
                "messages": [direct_aimessage],  # APPEND, not replace
                "created_at": "2026-09-23T10:00:00Z",
            })

        async def fake_intent_analysis(state, *, step=0):
            yield ("__delta__", {"intent": "greeting"})

        async def fake_check_hallucination(state, *, step=0):
            yield ("__delta__", {})

        original_intent = _original_node_fn("intent_analysis")
        original_rgd = _original_node_fn("react_generate_direct")
        original_ch = _original_node_fn("check_hallucination")
        try:
            _patch_node("intent_analysis", fake_intent_analysis)
            _patch_node("react_generate_direct", fake_react_generate_direct)
            _patch_node("check_hallucination", fake_check_hallucination)

            state = _initial_state(thread_id=thread_id)
            state["messages"] = [HumanMessage(content="你好")]

            await _drain_run_fsm(state, thread_id=thread_id, max_steps=10)

            saved = await checkpointer.load_latest(thread_id)
            assert saved is not None
            messages = saved.get("messages") or []

            # Greeting path: HumanMessage + AIMessage (appended). 2 messages total.
            assert len(messages) == 2, (
                f"expected 2 saved messages (HM + appended AIMessage); got {len(messages)}"
            )
            assert isinstance(messages[0], HumanMessage)
            assert messages[0].content == "你好"
            assert isinstance(messages[1], AIMessage)
            # The user's question is PRESERVED (not clobbered by replace).
            assert messages[1].content == "direct greeting response"
        finally:
            _patch_node("intent_analysis", original_intent)
            _patch_node("react_generate_direct", original_rgd)
            _patch_node("check_hallucination", original_ch)
    finally:
        await checkpointer.shutdown()


# ---------------------------------------------------------------------------
# 3) Multi-round (tool call + final text) — tool AIMessage preserved, draft replaced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_round_preserves_tool_aimessage(tmp_path, monkeypatch):
    """Drive a qa_complex turn with one tool-call round + one text-only
    round. Expected saved messages after the turn:
        [HM, AIM_tool_calls, TM, AIM_synthesis]
    NOT:
        [HM, AIM_tool_calls, TM, AIM_draft, AIM_synthesis]  ← pre-fix shape

    The tool-call AIMessage must stay (paired with its ToolMessage for
    ToolCallCard rendering). Only the text-only "draft" AIMessage
    (AIM_draft) is replaced by react_generate's synthesis.
    """
    from src.storage import checkpointer

    db_path = tmp_path / "history.db"
    monkeypatch.setattr(
        "src.storage.checkpointer.history_db_path", lambda: db_path
    )
    checkpointer.reset_for_tests()
    await checkpointer.startup()

    try:
        thread_id = "test-thread-3"

        # intent_analysis routes qa_complex → react_agent
        async def fake_intent_analysis(state, *, step=0):
            yield ("__delta__", {"intent": "qa_complex"})

        # react_agent round 1: tool_calls. Round 2: text-only.
        round1_response = AIMessage(
            content="",
            tool_calls=[{"id": "tc-1", "name": "retrieve_docs", "args": {"q": "capitals"}}],
        )
        round2_response = AIMessage(content="USA answer draft")
        fake_model = _make_fake_llm(round1_response, round2_response)

        import src.agent.nodes.react_agent as ra
        monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

        # _tools_step: emit tool_call_end + ToolMessage (no real LanceDB).
        async def fake_tools_step(state, *, step=0):
            history = list(state.get("messages") or [])
            last = history[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    yield ("tool_call_end", {
                        "tool_call_id": tc.get("id", ""),
                        "tool_name": tc.get("name", ""),
                        "ok": True,
                        "elapsed_ms": 50,
                        "step": step,
                    })
            tm = ToolMessage(
                content='[{"page_content": "capital info", "metadata": {"doc_id": "d1", "chunk_id": "c1"}}]',
                tool_call_id="tc-1",
                name="retrieve_docs",
            )
            yield ("__delta__", {"messages": [tm]})

        # react_generate with replace_last_message
        synthesis_aimessage = AIMessage(
            content="synthesis with doc citation",
            additional_kwargs={
                "sources": [{"id": "d1"}],
                "source_kinds": ["local"],
                "created_at": "2026-09-23T10:00:00Z",
            },
        )

        async def fake_react_generate(state, *, step=0):
            yield ("answer_complete", {
                "answer": "synthesis with doc citation",
                "sources": [{"id": "d1"}],
                "source_kinds": ["local"],
                "created_at": "2026-09-23T10:00:00Z",
                "route_decision": "retrieve",
            })
            yield ("__delta__", {
                "answer": "synthesis with doc citation",
                "sources": [{"id": "d1"}],
                "source_kinds": ["local"],
                "documents": [],
                "replace_last_message": synthesis_aimessage,
                "created_at": "2026-09-23T10:00:00Z",
            })

        original_intent = _original_node_fn("intent_analysis")
        original_tools = _original_node_fn("tools")
        original_rg = _original_node_fn("react_generate")
        try:
            _patch_node("intent_analysis", fake_intent_analysis)
            _patch_node("tools", fake_tools_step)
            _patch_node("react_generate", fake_react_generate)

            state = _initial_state(thread_id=thread_id)
            state["messages"] = [HumanMessage(content="美国首都?")]

            await _drain_run_fsm(state, thread_id=thread_id, max_steps=10)

            saved = await checkpointer.load_latest(thread_id)
            assert saved is not None
            messages = saved.get("messages") or []

            # Expected: HM + AIM_tool_calls + TM + AIM_synthesis (4 messages)
            # NOT: HM + AIM_tool_calls + TM + AIM_draft + AIM_synthesis (5 — pre-fix)
            assert len(messages) == 4, (
                f"expected 4 saved messages (HM + tool AIM + TM + synthesis); got "
                f"{len(messages)}: "
                f"{[(type(m).__name__, getattr(m, 'content', '')[:30]) for m in messages]}"
            )
            assert isinstance(messages[0], HumanMessage)
            # Tool-call AIMessage preserved (has tool_calls metadata)
            assert isinstance(messages[1], AIMessage)
            assert getattr(messages[1], "tool_calls", None), (
                "tool_call AIMessage must be preserved with its tool_calls metadata "
                "for ToolCallCard rendering on history replay"
            )
            # ToolMessage preserved
            assert isinstance(messages[2], ToolMessage)
            assert messages[2].tool_call_id == "tc-1"
            # Synthesis replaced the draft
            assert isinstance(messages[3], AIMessage)
            assert messages[3].content == "synthesis with doc citation"
            # The draft ("USA answer draft") must NOT be in messages
            assert all(
                getattr(m, "content", "") != "USA answer draft"
                for m in messages
            ), "react_agent's text-only draft must be replaced by react_generate's synthesis"
        finally:
            _patch_node("intent_analysis", original_intent)
            _patch_node("tools", original_tools)
            _patch_node("react_generate", original_rg)
    finally:
        await checkpointer.shutdown()


# ---------------------------------------------------------------------------
# 4) regression guard — legacy "messages" key still in react_agent's __delta__
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_legacy_messages_still_in_delta(monkeypatch):
    """Sanity check: react_agent's __delta__ uses ``messages: [response]``
    (NOT ``replace_last_message``). This pins the contract — if react_agent
    ever starts using replace_last_message, react_generate would have
    nothing to replace and the user's question would be clobbered.
    """
    import src.agent.nodes.react_agent as ra

    response = AIMessage(content="draft answer")
    fake_model = _make_fake_llm(response)
    monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

    state = _initial_state()
    state["messages"] = [HumanMessage(content="hi")]
    state["thread_id"] = "t"

    delta = None
    async for kind, payload in ra.react_agent(state, step=1):
        if kind == "__delta__":
            delta = payload
            break

    assert delta is not None, "react_agent must yield __delta__"
    assert "messages" in delta, (
        "react_agent must use the 'messages' key (append), not 'replace_last_message'. "
        "If this fails, react_agent's text-only draft would be unreplaceable by "
        "react_generate (because nothing to replace) and the user's HumanMessage "
        "would be clobbered."
    )
    assert "replace_last_message" not in delta, (
        "react_agent must NOT use replace_last_message — that key is reserved "
        "for synthesis nodes (react_generate) that supersede the prior AIMessage"
    )
    assert delta["messages"] == [response]


# ---------------------------------------------------------------------------
# 5) react_generate_direct legacy "messages" key still in __delta__
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_generate_direct_legacy_messages_still_in_delta(monkeypatch):
    """Pinning the contract for react_generate_direct — uses the legacy
    'messages' key (append), NOT 'replace_last_message'. Same
    rationale as test_react_agent_legacy_messages_still_in_delta.
    """
    from src.agent.fsm import react_generate_direct

    state = _initial_state()
    state["messages"] = [HumanMessage(content="hi")]
    state["thread_id"] = "t"

    delta = None
    async for kind, payload in react_generate_direct(state, step=1):
        if kind == "__delta__":
            delta = payload
            break

    assert delta is not None
    assert "messages" in delta, "react_generate_direct must use the 'messages' key (append)"
    assert "replace_last_message" not in delta, (
        "react_generate_direct must NOT use replace_last_message — there is no "
        "prior AIMessage to replace; only the HumanMessage (which would be clobbered)."
    )
    assert len(delta["messages"]) == 1
    assert isinstance(delta["messages"][0], AIMessage)


# ---------------------------------------------------------------------------
# v2.0.29.4 (Phase 4 PR-1) — _dedupe_documents contract pin
#
# Pre-PR-1 the dedupe key was ``(doc_id, chunk_id)`` only. A
# re-uploaded doc with same ``doc_id`` + ``chunk_id`` but different
# content silently collapsed to the stale version — the LLM would
# read the wrong chunk for "current" queries.
#
# Post-PR-1 the key is ``(doc_id, chunk_id, page_content[:200])``;
# different content keeps BOTH docs so the LLM reads the most
# recent content. The full dedupe test suite lives in
# ``tests/test_retrieval_filters.py`` (which exercises the helper
# at the unit level); this test pins the contract at the public
# import site that ``react_generate`` uses.
# ---------------------------------------------------------------------------


def test_dedupe_documents_keeps_distinct_content():
    """Same (doc_id, chunk_id) but different content → 2 docs.
    This is the load-bearing 1-line contract the dedupe helper
    honors. If a future refactor reverts to the 2-tuple key, the
    LLM would silently read stale content for re-uploaded docs.
    """
    from langchain_core.documents import Document

    from src.agent.nodes.react_generate import _dedupe_documents

    docs = [
        Document(
            page_content="old version: salary cap is $50k",
            metadata={"doc_id": "d1", "chunk_id": "c1"},
        ),
        Document(
            page_content="new version: salary cap is $75k",
            metadata={"doc_id": "d1", "chunk_id": "c1"},
        ),
    ]
    out = _dedupe_documents(docs)
    assert len(out) == 2, (
        f"different content should produce 2 distinct docs; got {len(out)}. "
        "Pre-PR-1 the key was 2-tuple (doc_id, chunk_id) and would silently drop one."
    )