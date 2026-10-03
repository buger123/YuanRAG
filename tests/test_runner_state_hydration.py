"""Regression tests for v2.0.28.9 — runner hydration of prior turns.

Background
----------
The pre-v2.0.28.9 ``runner.stream_agent`` always started each turn with a
fresh ``_initial_state`` containing only the current ``HumanMessage``.
The end-of-turn ``save`` (``INSERT OR REPLACE`` on a ``thread_id``
primary key in ``src/storage/checkpointer.py``) then clobbered the
previous snapshot. The result: every thread only ever showed its latest
Q&A pair on replay; earlier Q&A pairs in the same thread were lost.

The fix: before calling ``_initial_state``, the runner calls
``await load_latest(thread_id)`` and prepends the prior
``state["messages"]`` (if any) to the inputs. This file pins that
contract so a future regression that drops the hydration (e.g. someone
"simplifying" the runner back to a fresh-only state) is caught in CI.

Why this is its own file rather than folded into ``test_initial_state.py``:
the hydration logic lives in ``stream_agent``, not ``_initial_state``;
``_initial_state`` is still a pure function (no DB access) and its
existing tests must keep passing unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.conftest import bypass_models_loaded_gate


async def _fake_run_fsm_capture_inputs(
    inputs: Dict[str, Any], *, thread_id: str = "", max_steps: int | None = None
):
    """Capture ``inputs`` so the test can inspect the messages list
    that the runner actually handed to the FSM. Yields a single
    ``answer_complete`` so ``stream_agent`` returns cleanly."""
    # Stash on the generator's closure — see test body for retrieval.
    _fake_run_fsm_capture_inputs.last_inputs = inputs  # type: ignore[attr-defined]
    yield {
        "kind": "answer_complete",
        "answer": "ok",
        "sources": [],
        "source_kinds": [],
        "created_at": "2024-01-01T00:00:00+00:00",
        "route_decision": None,
    }


@pytest.mark.asyncio
async def test_stream_agent_hydrates_prior_messages(monkeypatch):
    """When ``load_latest`` returns a prior state with messages,
    ``stream_agent`` prepends them to ``inputs["messages"]`` so the
    new turn sees the full conversation history."""
    from src.agent import runner
    from src.storage import checkpointer

    prior_messages = [
        HumanMessage(content="first question", additional_kwargs={"created_at": "t0"}),
        AIMessage(content="first answer", additional_kwargs={"created_at": "t1"}),
        HumanMessage(content="second question", additional_kwargs={"created_at": "t2"}),
        AIMessage(content="second answer", additional_kwargs={"created_at": "t3"}),
    ]
    prior_state: Dict[str, Any] = {
        "messages": prior_messages,
        "original_query": "first question",
        "thread_id": "thread-A",
    }

    async def fake_load_latest(thread_id: str):
        assert thread_id == "thread-A"
        return prior_state

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", _fake_run_fsm_capture_inputs)
    monkeypatch.setattr(checkpointer, "load_latest", fake_load_latest)

    agen = runner.stream_agent("thread-A", "third question")
    async for _ in agen:
        pass  # drain

    inputs = _fake_run_fsm_capture_inputs.last_inputs  # type: ignore[attr-defined]
    msgs = inputs["messages"]
    # Must contain the 4 prior messages + 1 new HumanMessage, in order.
    assert len(msgs) == 5, (
        f"expected 5 messages (4 prior + 1 new), got {len(msgs)}: "
        f"{[type(m).__name__ for m in msgs]}"
    )
    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "first question"
    assert isinstance(msgs[1], AIMessage) and msgs[1].content == "first answer"
    assert isinstance(msgs[2], HumanMessage) and msgs[2].content == "second question"
    assert isinstance(msgs[3], AIMessage) and msgs[3].content == "second answer"
    assert isinstance(msgs[4], HumanMessage) and msgs[4].content == "third question"
    # The new HumanMessage must be appended (last position), not prepended —
    # the runner appends it AFTER copying in prior messages.
    assert msgs[4].content == "third question"


@pytest.mark.asyncio
async def test_stream_agent_propagates_original_query_across_turns(monkeypatch):
    """A multi-turn thread must keep the first user's question as
    ``original_query`` so ``intent_analysis`` can still reference it
    on the 5th turn."""
    from src.agent import runner
    from src.storage import checkpointer

    prior_state: Dict[str, Any] = {
        "messages": [
            HumanMessage(content="the very first ask", additional_kwargs={"created_at": "t0"}),
            AIMessage(content="first answer", additional_kwargs={"created_at": "t1"}),
        ],
        "original_query": "the very first ask",
        "thread_id": "thread-B",
    }

    async def fake_load_latest(thread_id: str):
        return prior_state

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", _fake_run_fsm_capture_inputs)
    monkeypatch.setattr(checkpointer, "load_latest", fake_load_latest)

    agen = runner.stream_agent("thread-B", "follow-up question")
    async for _ in agen:
        pass

    inputs = _fake_run_fsm_capture_inputs.last_inputs  # type: ignore[attr-defined]
    assert inputs["original_query"] == "the very first ask", (
        "original_query must carry forward the first question in the "
        "thread; otherwise a 5th-turn answer can't reference the original ask"
    )


@pytest.mark.asyncio
async def test_stream_agent_handles_missing_prior_state(monkeypatch):
    """A fresh thread (no prior state) must NOT crash and must produce
    a state with exactly the new HumanMessage (no spurious prior entries)."""
    from src.agent import runner
    from src.storage import checkpointer

    async def fake_load_latest(thread_id: str):
        return None  # no prior state

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", _fake_run_fsm_capture_inputs)
    monkeypatch.setattr(checkpointer, "load_latest", fake_load_latest)

    agen = runner.stream_agent("thread-C", "first ever question")
    async for _ in agen:
        pass

    inputs = _fake_run_fsm_capture_inputs.last_inputs  # type: ignore[attr-defined]
    msgs = inputs["messages"]
    assert len(msgs) == 1, f"fresh thread should have 1 message, got {len(msgs)}"
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "first ever question"
    assert inputs["original_query"] == "first ever question", (
        "first turn must seed original_query from the user's message"
    )


@pytest.mark.asyncio
async def test_stream_agent_degrades_when_load_latest_raises(monkeypatch):
    """If ``load_latest`` throws (DB hiccup, corrupt row, transient
    filesystem issue), the runner must NOT bubble the exception —
    it must log a WARNING and start a fresh turn so the user's
    message still gets answered."""
    from src.agent import runner
    from src.storage import checkpointer

    async def fake_load_latest_raises(thread_id: str):
        raise RuntimeError("simulated storage failure")

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", _fake_run_fsm_capture_inputs)
    monkeypatch.setattr(checkpointer, "load_latest", fake_load_latest_raises)

    agen = runner.stream_agent("thread-D", "question despite storage failure")
    # Drain — must complete without raising.
    async for _ in agen:
        pass

    inputs = _fake_run_fsm_capture_inputs.last_inputs  # type: ignore[attr-defined]
    msgs = inputs["messages"]
    assert len(msgs) == 1, (
        "when load_latest raises, runner must fall back to a fresh state "
        f"with exactly 1 message; got {len(msgs)}"
    )
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "question despite storage failure"


@pytest.mark.asyncio
async def test_stream_agent_resets_step_count_per_turn(monkeypatch):
    """``step_count`` is the FSM's per-turn recursion budget; it must
    reset to 0 each turn even if the prior state had a non-zero value.
    Otherwise a 5th turn in a thread would have 0 step budget left."""
    from src.agent import runner
    from src.storage import checkpointer

    prior_state: Dict[str, Any] = {
        "messages": [
            HumanMessage(content="q1", additional_kwargs={"created_at": "t0"}),
            AIMessage(content="a1", additional_kwargs={"created_at": "t1"}),
        ],
        # Pre-fix behavior would have stored a step_count from the prior
        # turn's FSM run. Post-fix: each turn resets to 0.
        "step_count": 9,
        "original_query": "q1",
        "thread_id": "thread-E",
    }

    async def fake_load_latest(thread_id: str):
        return prior_state

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", _fake_run_fsm_capture_inputs)
    monkeypatch.setattr(checkpointer, "load_latest", fake_load_latest)

    agen = runner.stream_agent("thread-E", "q2")
    async for _ in agen:
        pass

    inputs = _fake_run_fsm_capture_inputs.last_inputs  # type: ignore[attr-defined]
    assert inputs["step_count"] == 0, (
        f"step_count must reset to 0 each turn (FSM's per-turn recursion "
        f"budget); got {inputs['step_count']}"
    )


@pytest.mark.asyncio
async def test_stream_agent_preserves_tool_messages_in_history(monkeypatch):
    """ToolMessages from prior ReAct rounds must survive the round-trip —
    the frontend renders them as tool call cards, so dropping them would
    hide prior reasoning steps from the user on replay."""
    from src.agent import runner
    from src.storage import checkpointer

    prior_messages: List[Any] = [
        HumanMessage(content="search for X", additional_kwargs={"created_at": "t0"}),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [{"id": "tc1", "name": "web_search", "args": {"q": "X"}}],
                "created_at": "t1",
            },
        ),
        ToolMessage(content="result A", tool_call_id="tc1"),
        AIMessage(content="based on A, the answer is X", additional_kwargs={"created_at": "t2"}),
    ]
    prior_state: Dict[str, Any] = {
        "messages": prior_messages,
        "original_query": "search for X",
        "thread_id": "thread-F",
    }

    async def fake_load_latest(thread_id: str):
        return prior_state

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", _fake_run_fsm_capture_inputs)
    monkeypatch.setattr(checkpointer, "load_latest", fake_load_latest)

    agen = runner.stream_agent("thread-F", "now search for Y")
    async for _ in agen:
        pass

    inputs = _fake_run_fsm_capture_inputs.last_inputs  # type: ignore[attr-defined]
    msgs = inputs["messages"]
    assert len(msgs) == 5, f"expected 5 messages (4 prior + 1 new), got {len(msgs)}"
    assert isinstance(msgs[2], ToolMessage), (
        f"ToolMessage must survive hydration; got {type(msgs[2]).__name__}"
    )
    assert msgs[2].tool_call_id == "tc1"
