"""Regression tests for the runner's streaming dedup guards.

Bug background
--------------
v1.1.5/v1.1.6 fixed two bugs that both produced a doubled live answer:

1. **v1.1.5 (prompt-side)** — ``generate.generate_answer_async``
   appended a duplicate ``HumanMessage(query)`` so the LLM saw the
   user's question twice and answered twice. (v2.0 ReAct rewrite:
   equivalent bug would live in ``react_generate``.)
2. **v1.1.6 (stream-side)** — LangGraph's
   ``astream(stream_mode=["messages", "updates"])`` ``messages``
   channel delivers incremental ``AIMessageChunk``s (each chunk is
   the NEW portion) AND a synthesized final chunk whose content is
   the FULL accumulated message. Without protection, the frontend
   appended both the incremental stream AND the full final chunk,
   producing exactly 2x the canonical answer.

Why the v1.1.6c suffix-match dedup works
----------------------------------------
LangGraph's synthesized final chunk has ``text == accumulated_text``,
i.e. it equals the accumulated tail. Dropping any chunk whose text
equals the accumulated tail correctly removes the synthesized chunk
without false-positives on legitimate incremental deltas (a 1-char
chunk whose character happens to match the last char of accumulated
is not a suffix match of length 1 against accumulated).

v2.0.19 (Phase 3) — what changed
---------------------------------
The FSM emits ONE ``token`` event per LLM chunk, so the
"synthesized final chunk" bug class no longer exists in the wire
format — the FSM does not replay the accumulated message at the
end. Two guards remain:

1. ``cleaned == last_emitted_text[0]`` — drops duplicate consecutive
   chunks (an LLM provider quirk; still observed in practice).
2. ``accumulated_text[0].endswith(cleaned)`` — kept as
   defense-in-depth in case a future LLM provider re-emits the
   accumulated tail.

After ``answer_complete``, ``_emit_fsm_event`` drops any further
``token`` / ``reasoning`` events so a hypothetical re-stream (or
the answer_complete stream-buffer flush token) can't double the
display.

The Phase-3 tests below keep the consecutive-chunk dedup guard
and the post-``answer_complete`` guard, and drop the
"synthesized final chunk" tests (no longer applicable — FSM
never emits a replay). Reasoning dedup is also still tested
(same O(1) consecutive-chunk check).
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

import pytest

from tests.conftest import bypass_models_loaded_gate


async def _drive_fake_run_fsm(
    fsm_events: List[Dict[str, Any]],
    inputs,
    *,
    thread_id: str = "",
    max_steps: int | None = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield each test's pre-built FSMEvent list one by one."""
    for evt in fsm_events:
        yield evt


def _run_stream_agent(fsm_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Invoke ``stream_agent`` against a fake FSM and collect events.

    Bypasses the "models still loading" gate and patches
    ``runner.run_fsm`` with a pre-built event list.
    """
    import asyncio
    from src.agent import runner

    monkeypatch = pytest.MonkeyPatch()
    try:
        bypass_models_loaded_gate(monkeypatch)
        monkeypatch.setattr(
            runner,
            "run_fsm",
            lambda inputs, *, thread_id="", max_steps=None: _drive_fake_run_fsm(
                fsm_events, inputs, thread_id=thread_id, max_steps=max_steps
            ),
        )

        out: List[Dict[str, Any]] = []

        async def _drive():
            async for evt in runner.stream_agent("thread-x", "hi"):
                out.append(evt)

        asyncio.run(_drive())
    finally:
        monkeypatch.undo()

    return out


# ---------------------------------------------------------------------------
# Baseline: no re-delivery, single yield per chunk
# ---------------------------------------------------------------------------


def test_normal_stream_yields_each_chunk_once():
    """Baseline (no FSM re-delivery): each token event becomes
    exactly one ``token`` wire event. Guards against accidental
    over-dedup where the fix strips out legitimate distinct chunks.

    The FSM emits one ``token`` per LLM chunk; consecutive distinct
    chunks must each yield a ``token`` event. With the dedup guards,
    no chunk matches the accumulated tail so none should be dropped.
    """
    final_answer = "你好！我是 Yuan RAG"
    fsm_events = [
        {"kind": "token", "text": "你好"},
        {"kind": "token", "text": "你好！"},
        {"kind": "token", "text": "你好！我是"},
        {
            "kind": "answer_complete",
            "answer": final_answer,
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
    ]
    events = _run_stream_agent(fsm_events)

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["content"] for e in token_events] == [
        "你好", "你好！", "你好！我是"
    ], (
        f"Expected each chunk yielded once, got: "
        f"{[e['content'] for e in token_events]!r}"
    )


# ---------------------------------------------------------------------------
# Consecutive-chunk dedup (LLM-provider quirk, still observed)
# ---------------------------------------------------------------------------


def test_consecutive_duplicate_chunk_is_dropped():
    """The O(1) ``cleaned == last_emitted_text`` guard drops duplicate
    consecutive chunks (some LLM providers re-deliver the same chunk
    on a retry). Distinct consecutive chunks must still both flow.
    """
    final_answer = "你好"
    fsm_events = [
        {"kind": "token", "text": "你"},
        # Duplicate consecutive chunk — must be dropped.
        {"kind": "token", "text": "你"},
        {"kind": "token", "text": "好"},
        {
            "kind": "answer_complete",
            "answer": final_answer,
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
    ]
    events = _run_stream_agent(fsm_events)

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["content"] for e in token_events] == ["你", "好"], (
        f"Duplicate consecutive chunk leaked through, got: "
        f"{[e['content'] for e in token_events]!r}"
    )


def test_suffix_match_chunk_is_dropped():
    """Defense-in-depth: ``accumulated_text.endswith(cleaned)`` drops
    a chunk whose text is a suffix of the accumulated tail. The FSM
    itself doesn't emit a synthesized full-message chunk, but the
    guard is kept so a future provider quirk (or a buggy streaming
    middleware) can't slip through.
    """
    final_answer = "你好！我是"
    fsm_events = [
        {"kind": "token", "text": "你好"},
        {"kind": "token", "text": "你好！"},
        {"kind": "token", "text": "你好！我是"},
        # Synthesized full-message replay — must be dropped by the
        # suffix-match guard (text equals accumulated tail).
        {"kind": "token", "text": "你好！我是"},
        {
            "kind": "answer_complete",
            "answer": final_answer,
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
    ]
    events = _run_stream_agent(fsm_events)

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["content"] for e in token_events] == [
        "你好", "你好！", "你好！我是"
    ], (
        f"Suffix-match chunk leaked through, got: "
        f"{[e['content'] for e in token_events]!r}"
    )


# ---------------------------------------------------------------------------
# Layer-2 guard: events after answer_complete are dropped
# ---------------------------------------------------------------------------


def test_post_answer_complete_tokens_are_dropped():
    """After ``answer_complete`` fires, the FSM (and runner) must drop
    any further ``token`` / ``reasoning`` events. The frontend's
    canonical overwrite (``answer`` field on the AIMessage) is the
    source of truth for the live UI; additional tokens would
    re-double the display.
    """
    final_answer = "你好！我是 Yuan RAG"
    fsm_events = [
        {"kind": "token", "text": "你好"},
        {"kind": "token", "text": "你好！我是"},
        {
            "kind": "answer_complete",
            "answer": final_answer,
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
        # Post-answer_complete re-emission — must be dropped even
        # though the text is not a suffix of accumulated.
        {"kind": "token", "text": "你好"},
        {"kind": "token", "text": "你好！我是"},
    ]
    events = _run_stream_agent(fsm_events)

    token_events = [e for e in events if e["type"] == "token"]
    assert [e["content"] for e in token_events] == [
        "你好", "你好！我是"
    ], (
        f"Post-answer_complete tokens leaked through, got: "
        f"{[e['content'] for e in token_events]!r}"
    )

    # answer_complete still carries the canonical single answer.
    answer_complete = [e for e in events if e["type"] == "answer_complete"]
    assert len(answer_complete) == 1
    assert answer_complete[0]["answer"] == final_answer


# ---------------------------------------------------------------------------
# Reasoning dedup (parallel contract for extended-thinking)
# ---------------------------------------------------------------------------


def test_reasoning_consecutive_duplicate_is_dropped():
    """Same O(1) consecutive-chunk dedup applies to reasoning events.

    Anthropic extended-thinking emits ``chunk.content`` as a LIST of
    structured blocks (``[{"type": "thinking", "thinking": "..."}]``,
    not a single dict). The FSM's ``react_generate`` /
    ``react_generate_direct`` yield one ``("reasoning", {"text": "..."})``
    per thinking block — the runner sees these as ``{"kind":
    "reasoning", "text": "..."}`` and drops duplicate consecutive
    text via the same guard as ``token``.
    """
    fsm_events = [
        {"kind": "reasoning", "text": "用户问"},
        # Duplicate consecutive reasoning chunk — must be dropped.
        {"kind": "reasoning", "text": "用户问"},
        {"kind": "reasoning", "text": "的是..."},
        {
            "kind": "answer_complete",
            "answer": "你好",
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
    ]
    events = _run_stream_agent(fsm_events)

    reasoning_events = [e for e in events if e["type"] == "reasoning"]
    assert [e["content"] for e in reasoning_events] == [
        "用户问", "的是..."
    ], (
        f"Duplicate reasoning chunk leaked through, got: "
        f"{[e['content'] for e in reasoning_events]!r}"
    )


# ---------------------------------------------------------------------------
# Sanity: grounding and done still fire after answer_complete
# ---------------------------------------------------------------------------


def test_grounding_and_done_still_fire_after_answer_complete():
    """Post-answer_complete guard only drops ``token`` / ``reasoning``
    events; ``grounding`` and ``done`` must still fire.
    """
    fsm_events = [
        {"kind": "token", "text": "a"},
        {
            "kind": "answer_complete",
            "answer": "a",
            "sources": [],
            "source_kinds": [],
            "created_at": "2024-01-01T00:00:00+00:00",
            "route_decision": None,
        },
        {"kind": "grounding", "status": "grounded"},
        # Post-answer_complete token — must be dropped.
        {"kind": "token", "text": "a"},
    ]
    events = _run_stream_agent(fsm_events)

    types = [e["type"] for e in events]
    assert types == ["token", "answer_complete", "grounding", "done"], (
        f"Expected [token, answer_complete, grounding, done], got: "
        f"{types!r}. grounding/done must still fire after "
        f"answer_complete; only token/reasoning events are dropped."
    )