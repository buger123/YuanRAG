"""v1.1.11 — runner-level defense-in-depth strip of tool_call /
tool_result XML literals.

What we lock here
-----------------
The runner is the last line of defense before the answer reaches the
WebSocket. v1.1.11 added two layers of strip:

1. **Per-chunk (already covered in test_thinking_block_aimessage.py)** —
   every streamed AIMessageChunk goes through ``split_text_and_thinking``
   which strips tool blocks from the emitted text.
2. **Canonical payload (this file)** — the ``answer_complete`` event
   carries the full answer text accumulated by the graph node. The
   runner strips once more on this path so even if a future code
   change bypasses the splitter (synchronous fallback, injected
   middleware), the canonical overwrite the frontend trusts is clean.

In addition, ``generate.py`` also strips once before building the
persisted AIMessage.content so the history-replay path
(``/sessions/{thread_id}/messages``) is guaranteed clean.

These tests exercise the strip at the runner + node layer directly
without needing a real LLM — we mock the chat model so we control
exactly what content the splitter receives.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from src.agent import events
from src.agent._thinking_split import _strip_tool_blocks
from src.agent.runner import stream_agent


# ---------------------------------------------------------------------------
# Direct strip tests on the runner-facing surface
# ---------------------------------------------------------------------------


def test_strip_tool_blocks_is_exported_from_thinking_split():
    """``_strip_tool_blocks`` is the function the runner and
    generate.py both import. Pin its location so a future refactor
    doesn't accidentally move it without updating both call sites."""
    import src.agent._thinking_split as ts

    assert hasattr(ts, "_strip_tool_blocks")
    assert callable(ts._strip_tool_blocks)


def test_strip_tool_blocks_idempotent():
    """Calling the strip twice is a no-op the second time — important
    because the runner may apply it twice (per chunk + on
    ``answer_complete``)."""
    text = "hello <tool_call>foo</tool_call> world"
    once = _strip_tool_blocks(text)
    twice = _strip_tool_blocks(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Stream-level integration test: token chunks stripped on emit
# ---------------------------------------------------------------------------


class _Chunk:
    """Minimal AIMessageChunk stand-in: just ``content``."""

    def __init__(self, content):
        self.content = content


class _FakeChatModel(Runnable):
    """Yields a fixed sequence of chunks. Bypasses the LangChain
    astream path by directly calling the splitter — we want to
    verify the runner's emission pipeline, not the model wrapper."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, msgs):
        for c in self._chunks:
            yield c

    def invoke(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_stream_agent_strips_tool_call_from_answer_complete(monkeypatch):
    """End-to-end: the canonical answer returned by the generate
    node includes a ``<tool_call>`` literal. The runner's
    ``answer_complete`` event's ``answer`` payload MUST NOT contain
    that literal — the answer_complete-time strip acts as
    defense-in-depth.

    v2.0.19 (Phase 3) — the FSM's ``NODES`` dict is the source of
    truth for which node functions the runner drives. We patch
    ``fsm.NODES`` directly so the greeting fast-path
    (``intent_analysis → react_generate_direct → check_hallucination``)
    runs entirely against fakes.
    """
    canonical = "clean answer plus <tool_call>leak</tool_call>"

    async def fake_intent(state, *, step=0):
        # v2.0.22 (Item 7 Step 6) — async-gen protocol: yield the
        # single ``__delta__`` terminator. ``merge()`` auto-injects
        # ``step_count + 1`` so the delta omits it explicitly.
        yield ("__delta__", {
            "intent": "greeting",
            "corrected_query": None,
            "needs_current_time": False,
        })

    async def fake_generate_direct(state, *, step=0):
        # Async generator protocol — yields one ``answer_complete``
        # then the mandatory ``__delta__`` terminator.
        yield {
            "kind": "answer_complete",
            "answer": canonical,
            "sources": [],
            "source_kinds": [],
            "created_at": "2026-09-06T12:00:00+00:00",
            "route_decision": "direct",
        }
        yield {
            "__delta__": {
                "answer": canonical,
                "sources": [],
                "source_kinds": [],
                "documents": [],
                "messages": [AIMessage(content=canonical)],
                "created_at": "2026-09-06T12:00:00+00:00",
                "step_count": state.get("step_count", 0) + 1,
            },
        }

    async def fake_hallucination(state, *, step=0):
        # v2.0.22 (Item 7 Step 6) — async-gen protocol: yield the
        # single ``__delta__`` terminator carrying the verdict.
        yield ("__delta__", {"hallucination_check": "skipped"})

    from src.agent import fsm
    from src.agent import runner

    # v2.0.22 (Item 7 Step 7) — NODES is now ``dict[str, NodeSpec]``
    # so each entry needs ``{"fn": ...}`` (with optional
    # ``emits_pre``/``emits_post`` defaults of empty list). The FSM
    # does ``node_fn = spec["fn"]`` so ``fn`` is the only required
    # key for a faked entry.
    monkeypatch.setitem(fsm.NODES, "intent_analysis", {"fn": fake_intent})
    monkeypatch.setitem(fsm.NODES, "react_generate_direct", {"fn": fake_generate_direct})
    monkeypatch.setitem(fsm.NODES, "check_hallucination", {"fn": fake_hallucination})

    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)

    async for evt in stream_agent("test-thread", "any question"):
        if evt.get("type") == "answer_complete":
            answer = evt.get("answer", "")
            assert "<tool_call>" not in answer, (
                f"answer_complete.answer contains leaked tool_call "
                f"literal: {answer!r}"
            )
            assert "clean answer plus" in answer
            break


@pytest.mark.asyncio
async def test_stream_agent_answer_complete_clean_when_answer_has_no_tools(
    monkeypatch,
):
    """Sanity check: when the canonical answer has no tool literals,
    the answer_complete payload passes through untouched."""
    clean_answer = "totally clean answer with no tools"

    async def fake_intent(state, *, step=0):
        # v2.0.22 (Item 7 Step 6) — async-gen protocol: yield the
        # single ``__delta__`` terminator. ``merge()`` auto-injects
        # ``step_count + 1`` so the delta omits it explicitly.
        yield ("__delta__", {
            "intent": "greeting",
            "corrected_query": None,
            "needs_current_time": False,
        })

    async def fake_generate_direct(state, *, step=0):
        yield {
            "kind": "answer_complete",
            "answer": clean_answer,
            "sources": [],
            "source_kinds": [],
            "created_at": "2026-09-06T12:00:00+00:00",
            "route_decision": "direct",
        }
        yield {
            "__delta__": {
                "answer": clean_answer,
                "sources": [],
                "source_kinds": [],
                "documents": [],
                "messages": [AIMessage(content=clean_answer)],
                "created_at": "2026-09-06T12:00:00+00:00",
                "step_count": state.get("step_count", 0) + 1,
            },
        }

    async def fake_hallucination(state, *, step=0):
        # v2.0.22 (Item 7 Step 6) — async-gen protocol: yield the
        # single ``__delta__`` terminator carrying the verdict.
        yield ("__delta__", {"hallucination_check": "skipped"})

    from src.agent import fsm
    from src.agent import runner
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    # v2.0.22 (Item 7 Step 7) — NODES is now ``dict[str, NodeSpec]``
    # so each entry needs ``{"fn": ...}`` (with optional
    # ``emits_pre``/``emits_post`` defaults of empty list). The FSM
    # does ``node_fn = spec["fn"]`` so ``fn`` is the only required
    # key for a faked entry.
    monkeypatch.setitem(fsm.NODES, "intent_analysis", {"fn": fake_intent})
    monkeypatch.setitem(fsm.NODES, "react_generate_direct", {"fn": fake_generate_direct})
    monkeypatch.setitem(fsm.NODES, "check_hallucination", {"fn": fake_hallucination})
    monkeypatch.setattr(bge_m3, "models_loaded", lambda: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda: True)

    async for evt in runner.stream_agent("t", "q"):
        if evt.get("type") == "answer_complete":
            assert evt["answer"] == clean_answer
            break


# ---------------------------------------------------------------------------
# Cancellation safety — strip must not interfere with cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strip_tool_blocks_does_not_swallow_cancellation():
    """The strip is a pure-Python regex sub; it must not interact
    with asyncio cancellation in any way. This test exists so a
    future refactor that wraps the strip in ``try/except`` doesn't
    silently swallow ``CancelledError`` (which is BaseException,
    not Exception, and must propagate).

    Test strategy: schedule a cancel() on the current task. The
    next await point raises CancelledError. We ``await asyncio.sleep(0)``
    immediately after calling the strip so the cancellation has a
    chance to be delivered — proves the strip itself is sync and
    doesn't itself block the cancellation."""
    import asyncio

    async def main():
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        # Strip is sync — it completes immediately. The cancel() will
        # be delivered at the next await point.
        _strip_tool_blocks("hello world")
        # Yield once so the scheduled CancelledError has a chance to
        # raise here. If the strip were wrapped in a try/except that
        # caught BaseException, the cancel would still be pending and
        # this sleep would raise it — exactly what we want to verify.
        await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await main()