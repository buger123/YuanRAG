"""v2.0.28.10 — ``merge()`` supports ``replace_last_message`` key.

Pre-v2.0.28.10 the FSM saved 2 AIMessages per turn when no tool
calls were involved (react_agent's text-only "draft" + react_generate's
"synthesis"). reload showed extra assistant bubbles that the user
never saw live.

The fix is a new ``replace_last_message`` key in the delta dict,
handled by ``merge()``. Synthesis nodes (``react_generate``) yield
this key instead of ``messages: [...]`` so the prior AIMessage is
replaced rather than appended.

These tests pin:

1. ``replace_last_message`` replaces the last message in state.messages.
2. Empty-state fallback appends the new message (defensive).
3. Other delta keys (answer, sources, etc.) still merge correctly.
4. Prior messages in state.messages are byte-identical after replace.
5. ``messages`` key (legacy append path) still works — no regression.
6. Step count auto-increment contract is preserved.
"""
from __future__ import annotations

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.fsm import merge


def _initial_state() -> dict:
    """Minimal AgentState shape — mirrors tests/test_fsm_tool_lifecycle.py:_initial_state."""
    return {
        "messages": [],
        "thread_id": "",
        "original_query": "q",
        "current_query": "q",
        "step_count": 0,
        "intent": None,
        "corrected_query": None,
        "needs_current_time": False,
        "created_at": None,
    }


# ---------------------------------------------------------------------------
# 1) replace_last_message replaces the last message
# ---------------------------------------------------------------------------


def test_merge_replace_last_message_replaces_last():
    """When delta carries ``replace_last_message: msg`` and state has
    at least one message, the LAST message becomes ``msg`` and the
    prior messages are unchanged. Length is unchanged (replace, not append).
    """
    h1 = HumanMessage(content="hello")
    a1_old = AIMessage(content="draft answer")
    a1_new = AIMessage(content="synthesis answer")

    state = _initial_state()
    state["messages"] = [h1, a1_old]
    state["step_count"] = 5

    out = merge(state, {"replace_last_message": a1_new})

    assert out["messages"] == [h1, a1_new]
    assert len(out["messages"]) == 2  # not 3 (no append)
    # Original state not mutated (snapshot semantics per docstring)
    assert state["messages"] == [h1, a1_old]


def test_merge_replace_last_message_only_replaces_last_index():
    """Prior messages are byte-identical after replace. Identity check
    (not just equality) so a future implementation can't accidentally
    reconstruct prior messages from copy/clone pipelines.
    """
    h1 = HumanMessage(content="hello")
    a1 = AIMessage(content="first answer")
    h2 = HumanMessage(content="follow-up")
    a2_old = AIMessage(content="old draft")
    a2_new = AIMessage(content="new synthesis")

    state = _initial_state()
    state["messages"] = [h1, a1, h2, a2_old]

    out = merge(state, {"replace_last_message": a2_new})

    assert out["messages"][0] is h1
    assert out["messages"][1] is a1
    assert out["messages"][2] is h2
    assert out["messages"][3] is a2_new


# ---------------------------------------------------------------------------
# 2) Empty-state fallback
# ---------------------------------------------------------------------------


def test_merge_replace_last_message_empty_state_appends():
    """When state["messages"] is empty, ``replace_last_message`` falls
    back to ``[v]`` so we never silently drop the message. Defensive
    only — runner always seeds a HumanMessage so this should never
    trigger in production. Test pins the contract.
    """
    a1 = AIMessage(content="only synthesis")

    state = _initial_state()
    state["messages"] = []

    out = merge(state, {"replace_last_message": a1})

    assert out["messages"] == [a1]
    assert len(out["messages"]) == 1


# ---------------------------------------------------------------------------
# 3) Other delta keys still merge
# ---------------------------------------------------------------------------


def test_merge_replace_last_message_preserves_other_keys():
    """When delta carries BOTH ``replace_last_message`` and other keys
    (``answer``, ``sources``, ``source_kinds``, ``documents``,
    ``created_at``), they all apply correctly. This is the actual
    shape react_generate yields (see nodes/react_generate.py:746-758).
    """
    h1 = HumanMessage(content="q")
    a_old = AIMessage(content="draft")
    a_new = AIMessage(content="synthesis")

    state = _initial_state()
    state["messages"] = [h1, a_old]
    state["step_count"] = 3

    delta = {
        "answer": "synthesis text",
        "sources": [{"id": "doc-1"}, {"id": "doc-2"}],
        "source_kinds": ["local"],
        "documents": [{"page_content": "doc body", "metadata": {}}],
        "replace_last_message": a_new,
        "created_at": "2026-09-23T10:00:00Z",
    }

    out = merge(state, delta)

    assert out["answer"] == "synthesis text"
    assert out["sources"] == [{"id": "doc-1"}, {"id": "doc-2"}]
    assert out["source_kinds"] == ["local"]
    assert out["documents"] == [{"page_content": "doc body", "metadata": {}}]
    assert out["created_at"] == "2026-09-23T10:00:00Z"
    assert out["messages"][-1] is a_new
    # Step count NOT in delta → auto-injected (v2.0.22 contract preserved)
    assert out["step_count"] == 4


# ---------------------------------------------------------------------------
# 4) Legacy ``messages`` key still appends
# ---------------------------------------------------------------------------


def test_merge_messages_key_still_appends():
    """Backward compat: the legacy ``messages: [msg]`` key still
    appends. Every other node (react_agent, react_generate_direct,
    _tools_step) uses this key. No behavior change for those call sites.
    """
    h1 = HumanMessage(content="hello")
    a1 = AIMessage(content="draft")

    state = _initial_state()
    state["messages"] = [h1]
    state["step_count"] = 2

    out = merge(state, {"messages": [a1]})

    assert out["messages"] == [h1, a1]
    assert len(out["messages"]) == 2  # appended, not replaced


def test_merge_messages_key_with_multiple_messages():
    """``messages: [a, b, c]`` appends all three in order."""
    state = _initial_state()
    state["messages"] = [HumanMessage(content="q")]

    a = AIMessage(content="a")
    b = AIMessage(content="b")
    c = AIMessage(content="c")
    out = merge(state, {"messages": [a, b, c]})

    assert out["messages"][0].content == "q"
    assert out["messages"][1] is a
    assert out["messages"][2] is b
    assert out["messages"][3] is c


# ---------------------------------------------------------------------------
# 5) Mixed keys in same delta — regression guard
# ---------------------------------------------------------------------------


def test_merge_replace_last_does_not_double_apply_with_messages_key():
    """If a delta carries BOTH ``messages`` and ``replace_last_message``
    (a bug that would never happen in production but pins the merge
    semantics), ``messages`` is processed first (concatenated) then
    ``replace_last_message`` overrides the LAST message. Net effect:
    replace_last wins. This is the order-preserving semantics from
    dict iteration order — pins the contract so future refactors
    don't silently change it.
    """
    state = _initial_state()
    state["messages"] = [HumanMessage(content="q")]
    state["step_count"] = 0

    a_draft = AIMessage(content="draft")
    a_legacy = AIMessage(content="legacy append")
    a_synthesis = AIMessage(content="synthesis")

    # NOTE: in practice this delta would never be yielded — react_generate
    # uses replace_last_message, react_generate_direct uses messages.
    # Pinning the merge order so future node changes can't introduce
    # both keys accidentally.
    out = merge(state, {
        "messages": [a_draft, a_legacy],  # append 2
        "replace_last_message": a_synthesis,  # replace last (a_legacy)
    })

    assert len(out["messages"]) == 3  # q + draft + synthesis
    assert out["messages"][0].content == "q"
    assert out["messages"][1] is a_draft
    assert out["messages"][2] is a_synthesis


# ---------------------------------------------------------------------------
# 6) ToolMessage preservation (regression for ToolCallCard contract)
# ---------------------------------------------------------------------------


def test_merge_replace_last_does_not_clobber_toolmessage():
    """Sanity check: if the last message in state.messages is a
    ToolMessage (which shouldn't happen in normal flow because
    after_react_agent only routes to react_generate when the LAST
    AIMessage has no tool_calls), replace_last_message would
    replace it. This test documents the contract — ToolMessage at
    the last position should never happen in production. If a future
    change ever triggers this, the test would fail loudly so the
    dev can investigate.

    Pinning the contract: in the current FSM, the last message at
    react_generate entry is ALWAYS a text-only AIMessage (because
    after_react_agent routes to react_generate only when
    tool_calls is empty on the last AIMessage).
    """
    h1 = HumanMessage(content="q")
    a_with_tools = AIMessage(
        content="",
        tool_calls=[{"id": "tc-1", "name": "web_search", "args": {}}],
    )
    tm = ToolMessage(content="search result", tool_call_id="tc-1")
    a_after_tools = AIMessage(content="follow-up text")
    a_synthesis = AIMessage(content="synthesis answer")

    state = _initial_state()
    state["messages"] = [h1, a_with_tools, tm, a_after_tools]

    # Normal flow: react_generate replaces the LAST AIMessage (a_after_tools).
    # The tool-call AIMessage (a_with_tools) and its ToolMessage (tm) stay intact.
    out = merge(state, {"replace_last_message": a_synthesis})

    assert out["messages"][0] is h1
    assert out["messages"][1] is a_with_tools  # preserved (has tool_calls)
    assert out["messages"][2] is tm            # preserved (ToolMessage)
    assert out["messages"][3] is a_synthesis    # replaced (text-only draft)


# ---------------------------------------------------------------------------
# 7) Step count contract (regression for v2.0.22 Step 5)
# ---------------------------------------------------------------------------


def test_merge_replace_last_message_auto_injects_step_count():
    """replace_last_message doesn't carry step_count, so the v2.0.22
    auto-increment branch must fire. Step budget unaffected by the new key.
    """
    state = _initial_state()
    state["step_count"] = 7

    out = merge(state, {"replace_last_message": AIMessage(content="x")})

    assert out["step_count"] == 8


def test_merge_replace_last_message_honors_explicit_step_count():
    """If a node carries an explicit step_count in its delta (escape
    hatch for helpers like _handle_tool_error), it must be honored
    even when replace_last_message is present.
    """
    state = _initial_state()
    state["step_count"] = 7

    out = merge(state, {
        "replace_last_message": AIMessage(content="x"),
        "step_count": 99,  # explicit override
    })

    assert out["step_count"] == 99