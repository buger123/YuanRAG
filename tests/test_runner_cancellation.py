"""Regression tests for ``stream_agent`` WebSocket cancellation safety.

Bug history
-----------
The pre-Phase-3 ``stream_agent`` put ``yield {"type": "done"}`` inside a
``finally`` block. When a WebSocket client disconnected, Starlette called
``aclose()`` on the generator, which threw ``GeneratorExit`` at the
currently suspended ``yield``. The exception propagated up through the
``try`` and ``except Exception`` (GeneratorExit is BaseException, not
Exception — so it's not caught, which is correct), then hit the
``finally`` where the inner ``yield`` ran during GeneratorExit cleanup →
``RuntimeError("async generator ignored GeneratorExit")``. The exception
leaked into the event loop and was logged as ``Task exception was never
retrieved``.

These tests verify that:
1. ``stream_agent`` emits a final ``{"type": "done"}`` event when the FSM
   completes normally.
2. ``aclose()`` mid-stream completes cleanly — no RuntimeError leaks.
3. Exceptions raised inside the FSM are surfaced as ``error`` events
   (then ``done``).

v2.0.19 (Phase 3) — the fake graph used to mock LangGraph's
``astream(inputs, cfg, stream_mode)``; it now mocks
``runner.run_fsm`` directly. ``run_fsm`` is a plain async generator
function (no graph, no factory), so the fake is a one-liner async
generator that yields :class:`FSMEvent` dicts.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

# v1.1.4 — the four tests in this file previously "passed" without ever
# exercising the FSM path, because ``stream_agent`` short-circuits at
# runner.py with ``error`` + ``done`` whenever ``models_loaded()`` /
# ``model_loaded()`` return False (which is always in the test env).
# Importing ``bypass_models_loaded_gate`` from conftest and calling it
# per-test is what makes the assertions below actually mean what they say.
from tests.conftest import bypass_models_loaded_gate


async def _fake_run_fsm_basic(inputs, *, thread_id="", max_steps=None):
    """Tiny happy-path FSM stream: two tokens, one answer_complete."""
    yield {"kind": "token", "text": "Hello"}
    yield {"kind": "token", "text": " world"}
    yield {
        "kind": "answer_complete",
        "answer": "Hello world",
        "sources": [],
        "source_kinds": [],
        "created_at": "2024-01-01T00:00:00+00:00",
        "route_decision": None,
    }


@pytest.mark.asyncio
async def test_stream_agent_emits_done_on_normal_completion(monkeypatch):
    """When the FSM completes normally, the consumer receives 'done'."""
    from src.agent import runner

    async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
        async for evt in _fake_run_fsm_basic(inputs, thread_id=thread_id, max_steps=max_steps):
            yield evt

    # v1.1.4 — bypass the "models still loading" early-return gate so
    # this test actually exercises the FSM path. Without this,
    # ``stream_agent`` returns at runner.py with ``error`` + ``done``
    # BEFORE the fake FSM is consulted.
    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", fake_run_fsm)

    events = []
    agen = runner.stream_agent("thread-x", "hi")
    async for ev in agen:
        events.append(ev)

    types = [e["type"] for e in events]
    assert types[-1] == "done", f"expected last event 'done', got {types}"

    # v1.1.4 — assert the FSM events actually flowed through.
    assert "token" in types, (
        f"fake FSM token events never reached the consumer: {types} "
        f"— gate bypass missing or runner regression"
    )


@pytest.mark.asyncio
async def test_stream_agent_aclose_mid_stream_does_not_raise(monkeypatch):
    """Regression: aclose() mid-stream must complete cleanly.

    Previously: aclose() during a yield inside the ``try`` block caused the
    inner ``yield {"type": "done"}`` in the ``finally`` to run during
    GeneratorExit cleanup, raising
    ``RuntimeError("async generator ignored GeneratorExit")`` into the
    event loop (where it was logged as "Task exception was never retrieved").
    """
    from src.agent import runner

    async def many_events(inputs, *, thread_id="", max_steps=None):
        for i in range(1000):
            yield {"kind": "token", "text": f"token-{i}"}
            await asyncio.sleep(0.001)

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", many_events)

    agen = runner.stream_agent("thread-x", "hi")

    # Read a couple then close.
    received = []
    async for ev in agen:
        received.append(ev)
        if len(received) >= 3:
            break

    # v1.1.4 — assert we ACTUALLY received at least one FSM event
    # before closing.
    assert any(e["type"] == "token" for e in received), (
        f"no token events received before aclose: "
        f"{[e['type'] for e in received]} — gate bypass missing or runner regression"
    )

    # Now close the generator. This is what Starlette does on disconnect.
    # It must complete without raising.
    await agen.aclose()

    # If we got here without RuntimeError leaking, the fix works.


@pytest.mark.asyncio
async def test_stream_agent_aclose_immediately_does_not_raise(monkeypatch):
    """Edge case: aclose() called before any event is consumed."""
    from src.agent import runner

    async def slow_events(inputs, *, thread_id="", max_steps=None):
        await asyncio.sleep(10)
        yield {"kind": "token", "text": "never reached"}

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", slow_events)

    agen = runner.stream_agent("thread-x", "hi")
    await agen.aclose()
    # No exception leaked.


@pytest.mark.asyncio
async def test_stream_agent_emits_error_event_on_exception(monkeypatch):
    """If the FSM raises, the consumer receives an error event + done."""
    from src.agent import runner

    async def failing_events(inputs, *, thread_id="", max_steps=None):
        yield {"kind": "token", "text": "first "}
        raise RuntimeError("fsm exploded")

    bypass_models_loaded_gate(monkeypatch)
    monkeypatch.setattr(runner, "run_fsm", failing_events)

    events = []
    agen = runner.stream_agent("thread-x", "hi")
    async for ev in agen:
        events.append(ev)

    types = [e["type"] for e in events]
    assert "error" in types
    assert types[-1] == "done"

    # v1.1.4 — assert the FSM's ``token`` event actually flowed
    # through before the exception. Without this, the test passes via
    # the gate (which yields ``error`` + ``done`` but is unrelated to
    # the FSM's actual RuntimeError).
    assert "token" in types, (
        f"fake FSM token events never reached the consumer: {types} "
        f"— gate bypass missing or runner regression"
    )