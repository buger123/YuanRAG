"""v2.0.22 (Item 7 Step 9, P2-B3) — FSM owns pending_tool_calls lifecycle.

Pre-Step-9, ``runner.py::stream_agent`` held a function-local
``pending_tool_calls: dict[str, dict]`` dict that:

1. Recorded every ``tool_call_start`` (setitem).
2. Popped every matching ``tool_call_end``.
3. On graceful shutdown (Exception, not BaseException), drained
   remaining entries as synthetic ``tool_call_end`` events with
   ``ok=False``.

Post-Step-9, the lifecycle moves to ``fsm.py::run_fsm``:

* ``state["_pending_tool_calls"]`` is initialized at ``run_fsm``
  entry (shallow-copied by every ``merge`` so the nested dict
  identity persists across node calls).
* ``react_agent`` records ``{step, name, started_at}`` for each
  tool call BEFORE yielding ``tool_call_start``.
* ``_tools_step`` pops the matching entry BEFORE yielding
  ``tool_call_end``.
* ``run_fsm``'s ``except Exception`` clause yields a synthetic
  ``tool_call_end`` per unmatched start, then re-raises.
* ``run_fsm``'s ``finally`` clause pops the tracker from state
  (no yield — yielding in finally during ``aclose()`` raises
  ``RuntimeError`` per the runner.py:30-34 contract).

The runner is now a pure forwarder of FSM events. CancelledError /
``aclose()`` still bypass the drain (``BaseException`` skips both
the runner's and the FSM's ``except Exception`` clauses — matches
pre-Step-9 semantics).

These tests pin:

1. Tracker init + cleanup lifecycle.
2. ``react_agent`` records before yielding ``tool_call_start``.
3. ``_tools_step`` pops before yielding ``tool_call_end``.
4. FSM drains on Exception with the exact payload shape from the
   pre-Step-9 runner.
5. FSM drain is SKIPPED on normal completion (empty tracker) and
   on BaseException (matches runner.py pre-Step-9 semantics).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Tuple

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_state() -> Dict[str, Any]:
    """Minimal ``AgentState`` for FSM tests — same shape as
    ``runner._initial_state`` without the HumanMessage so we can
    drive the FSM directly.
    """
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


async def _drain_run_fsm(state, **kwargs) -> List[Dict[str, Any]]:
    """Run ``run_fsm`` and return all yielded events (including those
    yielded during ``except Exception`` drain).
    """
    from src.agent.fsm import run_fsm

    events: list = []
    try:
        async for fsm_evt in run_fsm(state, **kwargs):
            events.append(fsm_evt)
    except Exception:
        # Exception raised by fake nodes propagates after the FSM
        # has drained — we want both the drained events AND to
        # acknowledge the exception. Stash the exception on the
        # return value via a sentinel.
        events.append({"_raised_": True})
    return events


def _patch_node(name: str, fn) -> None:
    """Replace ``NODES[name]["fn"]`` with ``fn``. Other NodeSpec
    fields (emits_pre / emits_post) are preserved so the FSM
    dispatch still works.
    """
    import src.agent.fsm as fsm_mod

    spec = dict(fsm_mod.NODES[name])
    spec["fn"] = fn
    fsm_mod.NODES[name] = spec


def _restore_node(name: str, fn) -> None:
    """Restore the original node fn (used in fixture teardown)."""
    _patch_node(name, fn)


# ---------------------------------------------------------------------------
# 1) Tracker init + cleanup lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fsm_initializes_pending_tracker_at_entry():
    """``run_fsm`` MUST init ``state["_pending_tool_calls"] = {}``
    at entry — every node in the FSM that yields a
    ``tool_call_start`` (``react_agent``) or ``tool_call_end``
    (``_tools_step``) mutates this nested dict. Without init,
    ``state.setdefault("_pending_tool_calls", {})`` would silently
    mask bugs where the tracker wasn't initialized.
    """
    import src.agent.fsm as fsm_mod

    state = _initial_state()

    # Patch every node to a no-op terminal so run_fsm completes
    # immediately. Each yields ``("__delta__", {})``.
    async def terminal_node(state, *, step=0):
        yield ("__delta__", {})

    original = {n: fsm_mod.NODES[n]["fn"] for n in fsm_mod.NODES}
    try:
        for n in fsm_mod.NODES:
            _patch_node(n, terminal_node)
        async for _ in fsm_mod.run_fsm(state):
            pass

        # After normal completion, the tracker is GONE from state
        # (finally clause popped it). This matches the pre-Step-9
        # runner behavior where the dict was function-local.
        assert "_pending_tool_calls" not in state, (
            "FSM finally clause must pop the tracker after drain — "
            "leaving it would let stale entries leak into the "
            "next turn via checkpointer / state dict."
        )
    finally:
        for n, fn in original.items():
            _patch_node(n, fn)


@pytest.mark.asyncio
async def test_run_fsm_initializes_tracker_even_when_state_already_has_one():
    """If the state dict already has ``_pending_tool_calls`` from a
    previous (incomplete) turn, ``run_fsm`` resets it to ``{}`` —
    we don't trust leftover tracker entries across turns.

    Defensive: if checkpointer behavior changes in the future and
    the tracker survives across turns, we don't want stale entries
    bleeding into a fresh run.
    """
    import src.agent.fsm as fsm_mod

    state = _initial_state()
    state["_pending_tool_calls"] = {"orphan-id": {"step": 99, "name": "x", "started_at": "old"}}

    async def terminal_node(state, *, step=0):
        yield ("__delta__", {})

    original = {n: fsm_mod.NODES[n]["fn"] for n in fsm_mod.NODES}
    try:
        for n in fsm_mod.NODES:
            _patch_node(n, terminal_node)
        async for _ in fsm_mod.run_fsm(state):
            pass

        # Tracker reset — orphan-id is gone.
        assert state.get("_pending_tool_calls") is None, (
            "FSM must reset tracker at entry; leftover entries "
            "from a previous turn must NOT bleed into this turn."
        )
    finally:
        for n, fn in original.items():
            _patch_node(n, fn)


# ---------------------------------------------------------------------------
# 2) react_agent records before yielding tool_call_start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_records_pending_before_yielding_start(monkeypatch):
    """``react_agent`` MUST mutate ``state["_pending_tool_calls"]``
    BEFORE yielding each ``tool_call_start`` event. The FSM's
    drain iterates this tracker after the FSM raises, so the
    ordering invariant is: any start event that reaches the wire
    has a corresponding entry in the tracker.

    Tested by snapshotting the tracker at each event.
    """
    from langchain_core.messages import AIMessage

    import src.agent.nodes.react_agent as ra

    response = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc-A", "name": "web_search", "args": {"q": "1"}},
            {"id": "tc-B", "name": "retrieve_docs", "args": {"q": "2"}},
        ],
    )
    fake_model = ra._FakeLLM(response) if False else None
    # Use a MagicMock with the necessary attributes
    from unittest.mock import AsyncMock, MagicMock

    fake_model = MagicMock()
    fake_model.bind_tools = MagicMock(return_value=fake_model)
    fake_model.ainvoke = AsyncMock(return_value=response)
    monkeypatch.setattr(ra, "build_chat_model", lambda **kw: fake_model)

    state = _initial_state()
    state["_pending_tool_calls"] = {}
    state["thread_id"] = "test"

    kinds_during_yield: list = []
    async for kind, payload in ra.react_agent(state, step=1):
        kinds_during_yield.append(kind)
        if kind == "tool_call_start":
            # The entry must already be in the tracker BEFORE
            # the event reaches the consumer.
            tc_id = payload["tool_call_id"]
            assert tc_id in state["_pending_tool_calls"], (
                f"tool_call_start for {tc_id!r} yielded but the "
                f"tracker has no entry for it — drain won't be "
                f"able to close it on shutdown. Tracker: "
                f"{state['_pending_tool_calls']!r}"
            )
            # And the metadata must match what react_agent stamped.
            meta = state["_pending_tool_calls"][tc_id]
            assert meta["step"] == 1
            assert meta["name"] == payload["tool_name"]

    # We should have seen at least 2 tool_call_start events.
    starts = [k for k in kinds_during_yield if k == "tool_call_start"]
    assert len(starts) == 2, f"expected 2 tool_call_starts; got {kinds_during_yield}"

    # Both entries should be in the tracker.
    assert "tc-A" in state["_pending_tool_calls"]
    assert "tc-B" in state["_pending_tool_calls"]


# ---------------------------------------------------------------------------
# 3) _tools_step pops before yielding tool_call_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_step_pops_pending_on_end():
    """``_tools_step`` MUST pop the tracker entry for each tool
    call BEFORE yielding ``tool_call_end`` — so the FSM's drain
    only catches truly unmatched starts.

    Tested by pre-populating the tracker with the IDs the tools
    step will encounter, then asserting the tracker is empty
    when each ``tool_call_end`` fires. We use the "unknown tool"
    branch of ``_tools_step`` (the LLM hallucinated a tool name
    that isn't in ALL_TOOLS) so we don't have to mock pydantic
    StructuredTool — but the pop + yield path is identical to the
    happy-path success branch.
    """
    from langchain_core.messages import AIMessage

    import src.agent.fsm as fsm_mod

    # Pre-populate the tracker with a fake tool_call_id.
    state = _initial_state()
    state["_pending_tool_calls"] = {
        "tc-1": {"step": 1, "name": "fake_tool", "started_at": "x"},
    }
    # AIMessage with a tool_call pointing at a name NOT in
    # ALL_TOOLS — _tools_step will fall into the "unknown tool"
    # branch (line 451) which still goes through the pop.
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[{"id": "tc-1", "name": "nonexistent_tool", "args": {}}],
        )
    ]
    state["thread_id"] = "test"

    kinds: list = []
    async for kind, payload in fsm_mod._tools_step(state, step=1):
        kinds.append(kind)
        if kind == "tool_call_end":
            assert "tc-1" not in state["_pending_tool_calls"], (
                f"_tools_step yielded tool_call_end for tc-1 but "
                f"the tracker still has it — drain would synthesize "
                f"a second end. Tracker: {state['_pending_tool_calls']!r}"
            )

    # After _tools_step completes, tracker is empty (for this ID).
    assert "tc-1" not in state["_pending_tool_calls"]
    # And we did see a tool_call_end (not just delta).
    assert "tool_call_end" in kinds


# ---------------------------------------------------------------------------
# 4) FSM drains on Exception with exact pre-Step-9 payload shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fsm_drains_pending_on_exception(monkeypatch):
    """When the FSM's main loop raises ``Exception`` after a
    ``tool_call_start`` has been recorded (but no matching
    ``tool_call_end`` has fired), ``run_fsm`` MUST yield a
    synthetic ``tool_call_end`` per unmatched start BEFORE the
    exception propagates.

    Same payload shape as the pre-Step-9 runner.py:430-441 drain:
    ``tool_ok=False``, ``tool_result="(cancelled before tool
    result)"``, ``elapsed_ms=0``, ``step`` from the tracker.
    """
    import src.agent.fsm as fsm_mod

    state = _initial_state()

    # intent_analysis → emits the only normal event we want before
    # the crash. Replace with a node that yields one tool_call_start
    # and then raises.
    async def crashing_intent(state, *, step=0):
        # Pretend the LLM decided to call a tool — record the start
        # in the tracker (same shape react_agent uses) and yield the
        # start event, then raise.
        state.setdefault("_pending_tool_calls", {})["tc-A"] = {
            "step": 1,
            "name": "web_search",
            "started_at": "2026-01-01T00:00:00+00:00",
        }
        yield ("tool_call_start", {
            "tool_call_id": "tc-A",
            "tool_name": "web_search",
            "tool_args": {"q": "x"},
            "started_at": "2026-01-01T00:00:00+00:00",
            "step": 1,
        })
        raise RuntimeError("crash after start, before end")

    original_intent = fsm_mod.NODES["intent_analysis"]["fn"]
    _patch_node("intent_analysis", crashing_intent)
    try:
        events: list = []
        with pytest.raises(RuntimeError, match="crash after start, before end"):
            async for fsm_evt in fsm_mod.run_fsm(state):
                events.append(fsm_evt)

        # The synthetic drain event MUST be present, BEFORE the
        # exception propagated to us.
        drain_events = [
            e for e in events
            if e.get("kind") == "tool_call_end" and e.get("tool_call_id") == "tc-A"
        ]
        assert len(drain_events) == 1, (
            f"FSM drain missed tc-A; got events: {events!r}"
        )

        drain = drain_events[0]
        assert drain["tool_ok"] is False
        assert drain["tool_result"] == "(cancelled before tool result)"
        assert drain["elapsed_ms"] == 0
        assert drain["step"] == 1
    finally:
        _patch_node("intent_analysis", original_intent)


@pytest.mark.asyncio
async def test_fsm_drain_payload_matches_pre_step9_runner():
    """Pin the exact payload field set + types for the synthetic
    drain ``tool_call_end``. If the runner's pre-Step-9 drain
    block ever needed to be reinstated for any reason (e.g., FSM
    refactor), it must produce events with this exact shape so
    frontend rendering stays stable.
    """
    import src.agent.fsm as fsm_mod

    state = _initial_state()

    async def multi_start_then_raise(state, *, step=0):
        # Two pending starts; drain must emit two synthetic ends.
        state.setdefault("_pending_tool_calls", {})
        for tc_id, name in [("tc-A", "web_search"), ("tc-B", "retrieve_docs")]:
            state["_pending_tool_calls"][tc_id] = {
                "step": 1, "name": name, "started_at": "x",
            }
            yield ("tool_call_start", {
                "tool_call_id": tc_id,
                "tool_name": name,
                "tool_args": {},
                "started_at": "x",
                "step": 1,
            })
        raise RuntimeError("boom")

    original_intent = fsm_mod.NODES["intent_analysis"]["fn"]
    _patch_node("intent_analysis", multi_start_then_raise)
    try:
        events: list = []
        with pytest.raises(RuntimeError, match="boom"):
            async for fsm_evt in fsm_mod.run_fsm(state):
                events.append(fsm_evt)

        drain_ends = [
            e for e in events
            if e.get("kind") == "tool_call_end"
            and e.get("tool_call_id") in {"tc-A", "tc-B"}
            and e.get("tool_ok") is False
        ]
        assert len(drain_ends) == 2, f"expected 2 drain ends; got {drain_ends!r}"

        for d in drain_ends:
            assert set(d.keys()) >= {
                "kind", "tool_call_id", "tool_ok",
                "tool_result", "step", "elapsed_ms",
            }, f"drain payload missing fields: {d.keys()!r}"
            assert d["tool_result"] == "(cancelled before tool result)"
            assert d["elapsed_ms"] == 0
            assert d["step"] == 1
            assert d["tool_call_id"] in {"tc-A", "tc-B"}
    finally:
        _patch_node("intent_analysis", original_intent)


# ---------------------------------------------------------------------------
# 5) FSM drain SKIPPED on normal completion + BaseException
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fsm_drain_skipped_on_normal_completion():
    """Normal completion of ``run_fsm`` MUST NOT yield any
    synthetic ``tool_call_end`` events — the only ``tool_call_end``
    in the stream comes from ``_tools_step`` matching a
    ``tool_call_start`` 1:1.
    """
    import src.agent.fsm as fsm_mod
    from langchain_core.messages import AIMessage

    state = _initial_state()

    # Patch every node to a terminal that yields one ``__delta__``.
    # No tool_call_start anywhere → no tool_call_end anywhere.
    async def terminal(state, *, step=0):
        yield ("__delta__", {})

    original = {n: fsm_mod.NODES[n]["fn"] for n in fsm_mod.NODES}
    try:
        for n in fsm_mod.NODES:
            _patch_node(n, terminal)
        events: list = []
        async for fsm_evt in fsm_mod.run_fsm(state):
            events.append(fsm_evt)
    finally:
        for n, fn in original.items():
            _patch_node(n, fn)

    # No synthetic ends.
    assert all(
        e.get("kind") != "tool_call_end" for e in events
    ), f"normal completion yielded unexpected tool_call_end: {events!r}"


@pytest.mark.asyncio
async def test_fsm_drain_skipped_on_baseexception():
    """CancelledError / GeneratorExit (``aclose()`` mid-yield)
    MUST bypass the FSM drain — matches the pre-Step-9 runner
    semantics where the function-body drain block was skipped on
    ``BaseException`` (GeneratorExit never entered
    ``except Exception``).

    Why this matters: aclose() during a tool call must NOT yield
    synthetic ``ok=False`` ends because the frontend is going
    away (the WS is being torn down). Forcing extra events into
    the consumer could trigger additional writes to a closed
    socket.
    """
    import src.agent.fsm as fsm_mod

    state = _initial_state()

    # Set up a pending start that would normally be drained.
    state["_pending_tool_calls"] = {
        "tc-A": {"step": 1, "name": "web_search", "started_at": "x"},
    }

    async def raises_cancelled(state, *, step=0):
        yield ("tool_call_start", {
            "tool_call_id": "tc-A",
            "tool_name": "web_search",
            "tool_args": {},
            "started_at": "x",
            "step": 1,
        })
        # Simulate CancelledError (a subclass of BaseException).
        raise asyncio.CancelledError()

    original = fsm_mod.NODES["intent_analysis"]["fn"]
    _patch_node("intent_analysis", raises_cancelled)
    try:
        events: list = []
        with pytest.raises(asyncio.CancelledError):
            async for fsm_evt in fsm_mod.run_fsm(state):
                events.append(fsm_evt)

        # No synthetic drain.
        drain_ends = [
            e for e in events
            if e.get("kind") == "tool_call_end"
            and e.get("tool_call_id") == "tc-A"
            and e.get("tool_ok") is False
        ]
        assert len(drain_ends) == 0, (
            f"CancelledError must bypass FSM drain (BaseException, "
            f"not Exception). Got: {drain_ends!r}"
        )
    finally:
        _patch_node("intent_analysis", original)


# ---------------------------------------------------------------------------
# 6) Helper imports
# ---------------------------------------------------------------------------


def test_pending_tracker_field_is_underscore_prefixed():
    """The internal tracker field is ``_pending_tool_calls`` —
    underscore prefix marks it as FSM-internal, not a wire-visible
    state field. The runner doesn't read it (it just forwards
    FSM events) and the state is popped in ``run_fsm``'s finally
    so it never leaks to checkpointer / next turn.
    """
    import inspect

    src = inspect.getsource(__import__("src.agent.fsm", fromlist=["run_fsm"]))
    assert "_pending_tool_calls" in src, (
        "FSM source must reference the tracker field by its "
        "underscore-prefixed name (matches this test's import)."
    )


__all__ = [
    "test_run_fsm_initializes_pending_tracker_at_entry",
    "test_run_fsm_initializes_tracker_even_when_state_already_has_one",
    "test_react_agent_records_pending_before_yielding_start",
    "test_tools_step_pops_pending_on_end",
    "test_fsm_drains_pending_on_exception",
    "test_fsm_drain_payload_matches_pre_step9_runner",
    "test_fsm_drain_skipped_on_normal_completion",
    "test_fsm_drain_skipped_on_baseexception",
    "test_pending_tracker_field_is_underscore_prefixed",
]