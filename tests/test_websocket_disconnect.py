"""Regression tests for WebSocket disconnect handling.

Bug
---
When the WebSocket client disconnected mid-stream (the most common cause
was the user clicking ``+ New chat`` while waiting for an answer),
``src/api/websocket.py`` used ``aclosing(stream_agent(...))``. On
disconnect, Starlette closed the WebSocket, ``aclosing.__aexit__``
threw ``GeneratorExit`` into the generator, which propagated into
``graph.astream`` and aborted the agent run before its final checkpoint
was written. The user would then reopen that thread and see a stale
snapshot from before the question was asked — the answer appeared to
have "failed".

Fix
---
Hold a strong reference to the ``stream_agent`` generator and, on
``WebSocketDisconnect``, hand it off to ``_drain_generator`` instead of
letting it be torn down. The agent run continues to completion and
writes its checkpoint; the user can switch back to that thread (or
refresh) and ``GET /sessions/{thread_id}/messages`` returns the full
Q+A.

These tests verify the drain mechanism.

v2.0.19 (Phase 3) — the fake ``_FakeGraph.astream(...)`` is replaced
by a fake ``runner.run_fsm`` async generator that yields FSMEvent
dicts. The drain contract (generator runs to natural end, finally
clause releases references, inner exceptions swallowed) is unchanged.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List

import pytest


async def _drive_fake_run_fsm(
    fsm_events: List[Dict[str, Any]],
    inputs,
    *,
    thread_id: str = "",
    max_steps: int | None = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield each test's pre-built FSMEvent list one by one."""
    for evt in fsm_events:
        # Allow asyncio to deliver cancellation between events so
        # the drain can race against an aclose if needed.
        await asyncio.sleep(0)
        yield evt


def _patch_runner_run_fsm(monkeypatch, fsm_events):
    """Wire ``runner.run_fsm`` to a one-shot async generator that
    yields the test's pre-built FSMEvent list. Also bypasses the
    "models still loading" gate.
    """
    from src.agent import runner
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)
    monkeypatch.setattr(
        runner,
        "run_fsm",
        lambda inputs, *, thread_id="", max_steps=None: _drive_fake_run_fsm(
            fsm_events, inputs, thread_id=thread_id, max_steps=max_steps
        ),
    )


@pytest.mark.asyncio
async def test_drain_generator_runs_to_completion(monkeypatch):
    """A stream_agent generator drained by ``_drain_generator`` must run
    to its natural end — yielding every remaining event including the
    final ``done`` that the original WS consumer bailed out on.
    """
    from src.agent import runner
    from src.api import websocket as ws_module

    completion = asyncio.Event()
    fsm_events = [
        {"kind": "token", "text": f"token-{i}"} for i in range(5)
    ] + [
        {
            "kind": "answer_complete",
            "answer": "done",
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
    ]

    async def tracked_fsm(inputs, *, thread_id="", max_steps=None):
        async for evt in _drive_fake_run_fsm(
            fsm_events, inputs, thread_id=thread_id, max_steps=max_steps
        ):
            yield evt
        completion.set()

    monkeypatch.setattr(
        runner,
        "run_fsm",
        tracked_fsm,
    )
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)

    agen = runner.stream_agent("thread-x", "hi")
    ws_module._background_runs.add(agen)
    try:
        received = []
        async for ev in agen:
            received.append(ev)
            if len(received) >= 2:
                break
        # Simulate WebSocketDisconnect mid-stream: spawn the drain.
        drain_task = asyncio.create_task(ws_module._drain_generator(agen))
        await asyncio.wait_for(drain_task, timeout=5)
    finally:
        ws_module._background_runs.discard(agen)

    # The drain must have let the generator reach its natural end.
    assert completion.is_set(), (
        "drain did not run the generator to completion — the agent "
        "run was effectively torn down by the disconnect"
    )


@pytest.mark.asyncio
async def test_drain_releases_background_runs_reference(monkeypatch):
    """After ``_drain_generator`` finishes, ``_background_runs`` must no
    longer hold a reference to the generator.
    """
    from src.agent import runner
    from src.api import websocket as ws_module

    _patch_runner_run_fsm(monkeypatch, [{"kind": "token", "text": "hi"}])

    agen = runner.stream_agent("thread-x", "hi")
    ws_module._background_runs.add(agen)
    assert any(g is agen for g in ws_module._background_runs)

    await asyncio.wait_for(
        asyncio.create_task(ws_module._drain_generator(agen)), timeout=5
    )

    # Give the drain task's finally clause a tick to run.
    await asyncio.sleep(0)

    assert all(g is not agen for g in ws_module._background_runs), (
        "_background_runs still holds the generator after drain "
        "completed — would leak memory across sessions"
    )


@pytest.mark.asyncio
async def test_drain_swallows_inner_exception(monkeypatch):
    """If the FSM raises mid-drain, ``_drain_generator`` must catch it
    and log — never let it propagate into the event loop.
    """
    from src.agent import runner
    from src.api import websocket as ws_module

    async def failing_fsm(inputs, *, thread_id="", max_steps=None):
        yield {"kind": "token", "text": "first "}
        raise RuntimeError("fsm exploded mid-drain")

    monkeypatch.setattr(runner, "run_fsm", failing_fsm)
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)

    agen = runner.stream_agent("thread-x", "hi")
    ws_module._background_runs.add(agen)
    try:
        # No exception should leak — _drain_generator catches Exception.
        await asyncio.wait_for(
            asyncio.create_task(ws_module._drain_generator(agen)), timeout=5
        )
    finally:
        ws_module._background_runs.discard(agen)

    assert all(g is not agen for g in ws_module._background_runs)


@pytest.mark.asyncio
async def test_drain_generator_alone_without_handoff(monkeypatch):
    """Even if ``_drain_generator`` is the ONLY thing keeping the
    generator alive (no entry in ``_background_runs``), the run still
    completes.
    """
    from src.agent import runner
    from src.api import websocket as ws_module

    completion = asyncio.Event()

    async def tracked_fsm(inputs, *, thread_id="", max_steps=None):
        yield {"kind": "token", "text": "a"}
        await asyncio.sleep(0.001)
        yield {
            "kind": "answer_complete",
            "answer": "done",
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        }
        completion.set()

    monkeypatch.setattr(runner, "run_fsm", tracked_fsm)
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker
    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)

    # Deliberately do NOT add to _background_runs.
    agen = runner.stream_agent("thread-x", "hi")
    drain_task = asyncio.create_task(ws_module._drain_generator(agen))
    await asyncio.wait_for(drain_task, timeout=5)
    assert completion.is_set()