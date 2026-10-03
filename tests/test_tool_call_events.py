"""Regression tests for v2.0 tool-call event emission.

The runner exposes ``tool_call_start`` / ``tool_call_end`` events
over the WebSocket so the frontend can render per-call UI cards.
These tests pin the wire shape and the graceful-shutdown drain
behavior.

v2.0.19 (Phase 3) — the FSM owns tool-call event emission
directly (see ``src/agent.fsm._tools_step`` and the
``tool_call_start`` / ``tool_call_end`` yields in
``run_fsm``).

v2.0.22 (Item 7 Step 9) — the ``pending_tool_calls`` lifecycle
moved from ``runner.py`` to ``fsm.py`` (P2-B3). The runner is now
a pure forwarder of FSM events; the FSM's ``except Exception``
clause in ``run_fsm`` drains unmatched starts with synthetic
``tool_call_end`` events (same payload shape as the pre-Step-9
runner drain). CancelledError / ``aclose()`` still bypass the
drain (matches pre-Step-9 semantics).

We no longer test the pre-Phase-3 ``_ToolCallTracker`` /
``_handle_react_agent`` / ``_handle_tools_node`` helpers — those
moved into the FSM. Instead, we test:

1. The wire-shape builders in ``src.agent.events`` (unchanged).
2. The runner forwards the FSM's synthetic end events
   (replacement for ``_ToolCallTracker.close_all_pending`` —
   the drain now lives in the FSM, the runner just forwards).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List

import pytest


# ---------------------------------------------------------------------------
# events.* builders — wire shape
# ---------------------------------------------------------------------------


def test_tool_call_start_event_shape():
    from src.agent.events import tool_call_start

    evt = tool_call_start(
        name="retrieve_docs",
        args={"query": "Q3", "top_k": 5},
        tool_call_id="tc-1",
        step=2,
    )
    assert evt["type"] == "tool_call_start"
    assert evt["name"] == "retrieve_docs"
    assert evt["args"] == {"query": "Q3", "top_k": 5}
    assert evt["tool_call_id"] == "tc-1"
    assert evt["step"] == 2


def test_tool_call_end_event_shape():
    from src.agent.events import tool_call_end

    evt = tool_call_end(
        tool_call_id="tc-1",
        result_summary="[doc1 chunk 3] revenue = 1.2B",
        ok=True,
        step=2,
        elapsed_ms=420,
    )
    assert evt["type"] == "tool_call_end"
    assert evt["tool_call_id"] == "tc-1"
    assert evt["result_summary"] == "[doc1 chunk 3] revenue = 1.2B"
    assert evt["ok"] is True
    assert evt["step"] == 2
    assert evt["elapsed_ms"] == 420


def test_intent_event_shape():
    from src.agent.events import intent

    evt = intent(intent="qa_complex", corrected_query="Q3 营收")
    assert evt["type"] == "intent"
    assert evt["intent"] == "qa_complex"
    assert evt["corrected_query"] == "Q3 营收"


def test_step_start_step_end_shape():
    from src.agent.events import step_start, step_end

    assert step_start(3) == {"type": "step_start", "step": 3}
    assert step_end(3, ok=True) == {"type": "step_end", "step": 3, "ok": True}
    assert step_end(3, ok=False)["ok"] is False


# ---------------------------------------------------------------------------
# Runner pending-tool-calls drain (Phase-3 replacement for
# ``_ToolCallTracker.close_all_pending``)
# ---------------------------------------------------------------------------


def _patch_runner_run_fsm(monkeypatch, fsm_events: List[Dict[str, Any]]):
    """Wire ``runner.run_fsm`` to a one-shot async generator that
    yields the test's pre-built FSMEvent list. Bypasses the
    "models still loading" gate."""
    from src.agent import runner
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
        for evt in fsm_events:
            yield evt

    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)
    monkeypatch.setattr(runner, "run_fsm", fake_run_fsm)


@pytest.mark.asyncio
async def test_runner_closes_pending_tool_calls_on_exception(monkeypatch):
    """If the FSM raises mid-ReAct-loop (after ``tool_call_start``
    but before ``tool_call_end``), the runner must forward the
    synthetic end events the FSM emits in its drain (ok=False) so
    the frontend's "running" cards flip to red.

    v2.0.22 (Item 7 Step 9) — the drain moved from the runner to
    the FSM (``run_fsm``'s ``except Exception`` clause yields
    synthetic ``tool_call_end`` events for every pending start
    before re-raising). The runner is now a pure forwarder. The
    pre-Step-9 version of this test drove a ``fake_run_fsm`` that
    raised BEFORE yielding synthetic ends and asserted the runner
    synthesized them. Post-Step-9 the fake_run_fsm yields the
    drain events itself (mirroring the real ``run_fsm``), and the
    runner just forwards.
    """
    from src.agent import runner

    async def failing_run_fsm(inputs, *, thread_id="", max_steps=None):
        # Step 9 — ``run_fsm`` initializes the pending tracker on
        # the state dict at entry. The fake_run_fsm mirrors that
        # so this test exercises the post-Step-9 contract exactly.
        inputs["_pending_tool_calls"] = {}

        # Two real tool_call_starts — react_agent records both in
        # the pending tracker before yielding each event (mirrors
        # ``src/agent/nodes/react_agent.py`` Step 9 mutation).
        for tool_call_id, tool_name, step, args in [
            ("tc-A", "web_search", 1, {}),
            ("tc-B", "retrieve_docs", 2, {"query": "x"}),
        ]:
            inputs["_pending_tool_calls"][tool_call_id] = {
                "step": step,
                "name": tool_name,
                "started_at": "2024-01-01T00:00:00+00:00",
            }
            yield {
                "kind": "tool_call_start",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": args,
                "step": step,
                "started_at": "2024-01-01T00:00:00+00:00",
            }

        # ``run_fsm``'s except-clause drain — synthetic closes for
        # every entry the FSM's pending tracker still holds. Same
        # payload shape as the pre-Step-9 runner drain.
        try:
            raise RuntimeError("fsm exploded mid-ReAct")
        except Exception:
            for tool_call_id, meta in list(
                inputs.get("_pending_tool_calls", {}).items()
            ):
                yield {
                    "kind": "tool_call_end",
                    "tool_call_id": tool_call_id,
                    "tool_ok": False,
                    "tool_result": "(cancelled before tool result)",
                    "step": meta.get("step", 1),
                    "elapsed_ms": 0,
                }
            raise  # re-raise so the runner's except catches it

    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)
    monkeypatch.setattr(runner, "run_fsm", failing_run_fsm)

    events = []
    async for evt in runner.stream_agent("t", "q"):
        events.append(evt)

    # Two start events flow through.
    starts = [e for e in events if e["type"] == "tool_call_start"]
    assert len(starts) == 2
    start_ids = {s["tool_call_id"] for s in starts}
    assert start_ids == {"tc-A", "tc-B"}

    # Two synthetic close events arrive (forwarded from the FSM's
    # drain, NOT synthesized by the runner).
    ends = [e for e in events if e["type"] == "tool_call_end"]
    assert len(ends) == 2
    end_ids = {e["tool_call_id"] for e in ends}
    assert end_ids == {"tc-A", "tc-B"}
    # All synthesized closes are failures.
    assert all(e["ok"] is False for e in ends)
    # Cancellation summary matches the pre-Phase-3 contract.
    for e in ends:
        assert "cancelled" in e["result_summary"].lower()

    # The error event + done still fire.
    types = [e["type"] for e in events]
    assert "error" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_runner_paired_start_end_emits_no_synthetic_close(monkeypatch):
    """When every ``tool_call_start`` has a matching
    ``tool_call_end``, the runner emits NO synthetic close — the
    real one carries the result.
    """
    from src.agent import runner

    fsm_events = [
        {
            "kind": "tool_call_start",
            "tool_call_id": "tc-A",
            "tool_name": "web_search",
            "tool_args": {},
            "step": 1,
            "started_at": "2024-01-01T00:00:00+00:00",
        },
        {
            "kind": "tool_call_end",
            "tool_call_id": "tc-A",
            "tool_ok": True,
            "elapsed_ms": 420,
            "step": 1,
            "tool_result": "ok result",
        },
        {
            "kind": "answer_complete",
            "answer": "done",
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
    ]
    _patch_runner_run_fsm(monkeypatch, fsm_events)

    events = []
    async for evt in runner.stream_agent("t", "q"):
        events.append(evt)

    ends = [e for e in events if e["type"] == "tool_call_end"]
    # Exactly one end — the real one (ok=True), no synthetic close.
    assert len(ends) == 1
    assert ends[0]["tool_call_id"] == "tc-A"
    assert ends[0]["ok"] is True


@pytest.mark.asyncio
async def test_runner_step_paired_with_tool_call(monkeypatch):
    """``step_start`` fires BEFORE the tool_call_start that the
    step produces; ``step_end`` fires AFTER — so the frontend's
    pill renders around the tool cards.

    This is the Phase-3 replacement for
    ``test_handle_react_agent_emits_step_start_and_end`` — the
    FSM owns the ordering, we verify it via ``stream_agent``.
    """
    from src.agent import runner

    fsm_events = [
        # FSM emits step_start before react_agent runs (no tool_call
        # yet).
        {"kind": "step_start", "node": "react_agent", "step": 1},
        # After react_agent produces an AIMessage with tool_calls,
        # the FSM emits tool_call_start for each tool call.
        {
            "kind": "tool_call_start",
            "tool_call_id": "tc-1",
            "tool_name": "retrieve_docs",
            "tool_args": {"query": "Q3"},
            "step": 1,
            "started_at": "2024-01-01T00:00:00+00:00",
        },
        # Then tool_call_end fires when the tool returns.
        {
            "kind": "tool_call_end",
            "tool_call_id": "tc-1",
            "tool_ok": True,
            "elapsed_ms": 100,
            "step": 1,
            "tool_result": "ok",
        },
        # step_end closes the step pill.
        {"kind": "step_end", "node": "react_agent", "step": 1},
        {
            "kind": "answer_complete",
            "answer": "done",
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
    ]
    _patch_runner_run_fsm(monkeypatch, fsm_events)

    events = []
    async for evt in runner.stream_agent("t", "q"):
        events.append(evt)

    types = [e["type"] for e in events]
    # step_start is first, step_end is after the tool_call pair.
    assert types[0] == "step_start"
    assert types[-1] == "done"
    # Index of step_end must be after tool_call_start.
    step_end_idx = types.index("step_end")
    tool_call_start_idx = types.index("tool_call_start")
    assert step_end_idx > tool_call_start_idx