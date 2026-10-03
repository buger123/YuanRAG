"""Streaming runner: invoke the FSM and yield WebSocket-friendly events.

See :mod:`src.agent.events` for the wire-payload shape; this module
just consumes :class:`src.agent.fsm.FSMEvent`s from :func:`src.agent.fsm.run_fsm`
and maps each event kind to the corresponding ``events.<kind>(...)``
wire factory.

v2.0.19 (Phase 3) — pure async FSM replaces the LangGraph ``StateGraph``
runtime. The runner no longer multiplexes ``messages`` / ``updates``
channels or decodes LangGraph node meta — each FSMEvent carries the
exact wire fields it needs (the FSM owns the ReAct topology, the
recursion budget, and the per-step event ordering). See
:mod:`src.agent.fsm` for the protocol.

Two streaming dedup guards — preserved from v1.1.6c + v2.0.10.2
---------------------------------------------------------------
1. ``text == last_emitted_text`` strict equality drops duplicate
   consecutive chunks (the O(1) hot path).
2. ``accumulated_text.endswith(cleaned_text)`` drops the synthesized
   full-message replay (only fires once per turn — amortized).

v2.0.10.2 leak-strip buffer (``stream_safe_chunk``) is preserved so a
MiniMax-style ``<invoke>`` leak that arrives split across chunks is
caught by the regex on the buffered slice. The buffer is flushed via
``_strip_tool_blocks`` when the LLM node finalizes the answer.

Cancellation contract — unchanged from v1.1.16
----------------------------------------------
``aclose()`` while suspended inside the try propagates
``GeneratorExit`` cleanly out (it's ``BaseException``, not
``Exception``). We yield ``done`` AFTER the try block — yielding
inside a finally during aclose causes ``RuntimeError("async
generator ignored GeneratorExit")``.
"""
from __future__ import annotations

from contextlib import aclosing
from typing import AsyncIterator, Optional

from langchain_core.messages import HumanMessage

from config.constants import _resolve_max_steps
from src.agent import events
# v2.0.19 (Phase 3) — the FSM owns the recursion limit going forward
# and raises ``MaxStepsExceeded``. We still catch ``GraphRecursionError``
# (LangGraph runtime) until Step 7 deletes the LangGraph runtime. The
# tuple form covers both during the migration window without doubling
# the events.error yield.
from src.agent.errors import MaxStepsExceeded
from src.agent._thinking_split import (
    _strip_tool_blocks,
    stream_safe_chunk,
)
# v2.0.22 (Item 7 Step 1) — now_iso lives in src.agent._time so the
# runner, FSM nodes, and answer builders share one clock. Aliased to
# the existing ``_now_iso`` name so call sites don't change.
from src.agent._time import now_iso as _now_iso
from src.agent.citations import (
    filter_sources_to_cited,
    renumber_citations_and_sources,
)
from src.agent.fsm import run_fsm
from src.core.logging import logger


# ---------------------------------------------------------------------------
# FSM singletons
# ---------------------------------------------------------------------------
#
# The FSM itself is stateless — ``run_fsm`` is a pure function over
# the state dict. The runner only needs the recursion budget, which
# comes from ``_resolve_max_steps()``. We re-resolve on every call
# so tests that override ``RAG_REACT_MAX_STEPS`` mid-session pick up
# the new budget on the next turn.

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
# v2.0.22 (Item 7 Step 1) — ``now_iso`` now lives in
# ``src.agent._time`` (imported above as ``_now_iso``). The alias keeps
# every existing call site (``_now_iso()``) working unchanged.


def _trace_id_for(exc: BaseException) -> str:
    """Pull a stable, no-secret correlation id out of an exception.

    Anthropic exceptions carry ``request_id`` (e.g.
    ``06ec8d6337baf0546e3ae24673f80135``) which is safe to surface
    to operators and end-users — it's the same id their LLM console
    uses to find the failed request. We fall back to a content-based
    hash of the exception repr when the upstream didn't provide one.
    Never include the exception message itself (it commonly embeds
    the user's API key / proxy URL — see P1-4).
    """
    rid = getattr(exc, "request_id", None)
    if rid:
        return str(rid)
    import hashlib

    return "anon-" + hashlib.sha1(
        (type(exc).__name__ + str(getattr(exc, "status_code", ""))).encode()
    ).hexdigest()[:12]


def _initial_state(
    thread_id: str,
    message: str,
    *,
    high_precision: str = "auto",
) -> dict:
    """v2.0 — initial ReAct state.

    Same shape as the pre-Phase-3 runner. ``step_count`` is the FSM's
    budget counter; the runner doesn't increment it (the FSM bumps it
    on every node that returns ``"step_count": N+1``). ``messages``
    starts with the user's HumanMessage; ``intent_analysis`` is the
    FSM's entry node and fills the rest.

    v2.0.29.9 (Phase 8) — ``high_precision`` tristate is forwarded
    from the per-thread toggle (ChatRequest / WS payload). Default
    ``"auto"`` → run hybrid detect (regex + cheap-LLM). The
    ``intent_analysis`` node reads this field and decides whether to
    force-extractive vs allow normal synthesis.
    """
    return {
        "messages": [
            HumanMessage(
                content=message,
                additional_kwargs={"created_at": _now_iso()},
            )
        ],
        "thread_id": thread_id,
        "original_query": message,
        "current_query": message,
        "step_count": 0,
        "intent": None,
        "corrected_query": None,
        "needs_current_time": False,
        "created_at": None,
        # v2.0.29.9 (Phase 8) — verbatim extraction tristate, set
        # BEFORE ``intent_analysis`` runs so the user's choice wins
        # over hybrid auto-detect when set to ``"off"`` (per user
        # decision 2026-09-28).
        "high_precision": high_precision,
    }


# ---------------------------------------------------------------------------
# FSMEvent → WS-event mapping
# ---------------------------------------------------------------------------


def _emit_fsm_event(
    fsm_evt: dict,
    *,
    accumulated_text: list,
    last_emitted_text: list,
    stream_buffer: list,
    answer_complete_sent: list,
) -> Optional[dict]:
    """Map a single :class:`FSMEvent` to its wire frame, or return None.

    Streaming events (``token`` / ``reasoning``) are subject to the
    same dedup + leak-strip guards as the pre-Phase-3 runner. The
    four lists are mutable single-cell holders (``[value]``) so the
    runner can maintain streaming state across yields without
    resorting to a class. ``answer_complete_sent`` is read+written
    so post-answer token events are dropped (matches v2.0 layer-2
    guard).

    Returns the wire dict, or ``None`` to drop the event.
    """
    kind = fsm_evt.get("kind")

    if kind == "token":
        if answer_complete_sent[0]:
            return None
        text = fsm_evt.get("text") or ""
        # v2.0.4 + v2.0.10.2 leak-strip buffer (preserved across
        # Phase 3 — the FSM yields one token per chunk so the
        # leak-detection contract is identical to the pre-Phase-3
        # ``messages`` channel).
        cleaned, stream_buffer[0] = stream_safe_chunk(text, stream_buffer[0])
        if not cleaned:
            return None
        # dedup: (a) duplicate consecutive chunk; (b) full-message
        # replay where text equals the accumulated tail.
        if cleaned == last_emitted_text[0]:
            return None
        if accumulated_text[0].endswith(cleaned):
            return None
        accumulated_text[0] += cleaned
        last_emitted_text[0] = cleaned
        return events.token(cleaned)

    if kind == "reasoning":
        if answer_complete_sent[0]:
            return None
        text = fsm_evt.get("text") or ""
        if not text:
            return None
        # Reasoning doesn't go through ``stream_safe_chunk`` — only
        # visible-text leaks need that; reasoning blocks never
        # contain wire-format syntax. We still drop the duplicate
        # consecutive chunk (same contract as the pre-Phase-3
        # ``messages`` channel).
        if text == last_emitted_text[0]:
            return None
        last_emitted_text[0] = text
        return events.reasoning(text)

    if kind == "intent":
        return events.intent(
            intent=fsm_evt.get("intent_value"),
            corrected_query=fsm_evt.get("corrected_query"),
        )

    if kind == "step_start":
        return events.step_start(fsm_evt.get("step") or 1)

    if kind == "step_end":
        # ``ok=True`` — the FSM only emits step_end after the node
        # completes successfully (errors raise out and propagate
        # to the runner's except block).
        return events.step_end(fsm_evt.get("step") or 1, ok=True)

    if kind == "tool_call_start":
        return events.tool_call_start(
            name=fsm_evt.get("tool_name") or "",
            args=fsm_evt.get("tool_args") or {},
            tool_call_id=fsm_evt.get("tool_call_id") or "",
            step=fsm_evt.get("step") or 1,
        )

    if kind == "tool_call_end":
        return events.tool_call_end(
            tool_call_id=fsm_evt.get("tool_call_id") or "",
            result_summary=fsm_evt.get("tool_result") or "",
            ok=bool(fsm_evt.get("tool_ok", True)),
            step=fsm_evt.get("step") or 1,
            elapsed_ms=int(fsm_evt.get("elapsed_ms") or 0),
        )

    if kind == "web_search":
        return events.web_search(bool(fsm_evt.get("attempted", False)))

    if kind == "answer_complete":
        # Final answer — flush the leak-strip buffer first (matches
        # the pre-Phase-3 ``_handle_final_answer`` flush step).
        if stream_buffer[0]:
            flushed = _strip_tool_blocks(stream_buffer[0])
            stream_buffer[0] = ""
            if (
                flushed
                and not accumulated_text[0].endswith(flushed)
            ):
                accumulated_text[0] += flushed
                last_emitted_text[0] = flushed
                yield_token = events.token(flushed)
            else:
                yield_token = None
        else:
            yield_token = None
        answer = fsm_evt.get("answer") or ""
        answer = _strip_tool_blocks(answer)
        raw_sources = fsm_evt.get("sources") or []
        route_decision = fsm_evt.get("route_decision") or None
        answer, sources = renumber_citations_and_sources(
            answer, raw_sources, route_decision=route_decision,
        )
        sources = filter_sources_to_cited(
            sources, answer, route_decision=route_decision,
        )
        source_kinds = fsm_evt.get("source_kinds")
        if source_kinds is None:
            from src.agent.legacy_helpers.doc_formatting import _derive_source_kinds
            source_kinds = _derive_source_kinds(sources)
        answer_evt = events.answer_complete(
            sources=sources,
            answer=answer,
            source_kinds=source_kinds,
            created_at=fsm_evt.get("created_at"),
        )
        # Mark as sent so any further token/reasoning events (none
        # should appear, but defensive) are dropped. The list-mutator
        # pattern propagates the flag back to the runner.
        answer_complete_sent[0] = True
        # v2.0.29.2 (Phase 2) — observability hook. Emit the
        # metrics snapshot at end of every turn when YuanRAG_METRICS_EMIT
        # is enabled. Off by default (zero runtime cost); flip on
        # for debugging or for the post-Phase-2 metrics dashboard.
        # Emits one JSON line per counter under
        # ``{"event": "metrics", "counters": [...]}``.
        from src.agent.metrics import _emit_metrics_line

        _emit_metrics_line()
        if yield_token is not None:
            # Yield the flushed token first, then the answer_complete.
            # The runner loops over the returned list when this
            # function returns multiple events — see stream_agent.
            return [yield_token, answer_evt]
        return answer_evt

    if kind == "grounding":
        return events.grounding(fsm_evt.get("status") or "skipped")

    if kind == "verification_result":
        # v2.0.29.7 (Phase 6) — forwarder to the wire factory.
        # The FSM yields ``("verification_result", payload_dict)``
        # with all fields already populated; we just pass them
        # through. Defensive ``or False`` / ``or 0`` for fields
        # that may be omitted on the skipped path.
        return events.verification_result(
            consistent=bool(fsm_evt.get("consistent", True)),
            mismatches_count=int(fsm_evt.get("mismatches_count") or 0),
            regenerated=bool(fsm_evt.get("regenerated", False)),
            fallback_refusal=bool(fsm_evt.get("fallback_refusal", False)),
            skipped=bool(fsm_evt.get("skipped", False)),
            reason=fsm_evt.get("reason"),
            mismatches=fsm_evt.get("mismatches") or [],
        )

    # Unknown kind — log and drop. Defensive against future FSM
    # additions that the runner hasn't learned yet.
    logger.warning(f"runner: unknown FSMEvent kind={kind!r}; dropping")
    return None


# ---------------------------------------------------------------------------
# Main streaming entry point
# ---------------------------------------------------------------------------


async def stream_agent(
    thread_id: str,
    user_message: str,
    *,
    high_precision: str = "auto",
) -> AsyncIterator[dict]:
    """Async generator yielding events for the WebSocket layer.

    v2.0.19 (Phase 3) — consumes :class:`FSMEvent`s from
    :func:`src.agent.fsm.run_fsm`. The FSM owns the ReAct topology,
    the recursion budget, and the per-step event ordering. The
    runner is now a thin mapping layer: ``FSMEvent.kind`` →
    ``events.<kind>(...)``.

    Defensive gate first: if the embedding / reranker models haven't
    finished loading yet, emit a single error + done and exit.
    Letting the run proceed would freeze the WS frame writer for
    ~30 s while ``_load_blocking`` does its work.

    Cancellation contract (unchanged from v1.1.16): ``aclose()``
    while suspended inside the try propagates ``GeneratorExit``
    cleanly out. ``done`` is yielded AFTER the try block so aclose
    during a yield doesn't raise ``RuntimeError("async generator
    ignored GeneratorExit")``.
    """
    from src.embeddings.bge_m3 import models_loaded
    from src.reranker.bge_reranker import model_loaded

    if not (models_loaded() and model_loaded()):
        yield events.error("Models are still loading. Please wait a few seconds and retry.")
        yield events.done()
        return

    # v2.0.28.9 — hydrate prior turns from the checkpointer. Without
    # this, every turn starts with a fresh ``state["messages"]``
    # containing only the current HumanMessage, and the end-of-turn
    # save (``INSERT OR REPLACE`` on a ``thread_id`` primary key in
    # ``src/storage/checkpointer.py``) clobbers the previous snapshot
    # — so each thread only ever shows its latest Q&A pair on replay.
    #
    # Why this is the runner's job, not the FSM's: the FSM
    # (``src/agent/fsm.py``) treats the input ``state`` as authoritative
    # and never touches the DB; the runner is the only point that
    # knows the difference between "fresh thread" and "existing
    # thread" because it has the ``thread_id`` at entry.
    #
    # Why we don't propagate ``step_count``: that counter is the
    # FSM's per-turn recursion budget — a new turn always starts at 0
    # (the user gets a fresh step budget for their new question). We
    # also intentionally drop ``intent`` / ``corrected_query`` because
    # those are per-turn FSM scratch state, not durable history.
    # ``original_query`` is propagated because it represents the
    # user's first question in the thread (used by ``intent_analysis``
    # on multi-turn answers).
    #
    # v2.0.29.2 (Phase 2) — hydration boundary contract:
    #
    # The hydration block explicitly allowlists two fields from
    # ``prior_state``: ``messages`` and ``original_query``. EVERYTHING
    # else from prior_state is dropped. This is intentional — the
    # durable / per-turn boundary is:
    #
    #   DURABLE (carries across turns):
    #     - ``messages``           — conversation history
    #     - ``original_query``     — thread's first question
    #
    #   PER-TURN (always fresh from _initial_state):
    #     - ``step_count``         — ReAct recursion budget
    #     - ``intent``             — current-turn classification
    #     - ``corrected_query``    — current-turn typo fix
    #     - ``current_query``      — current-turn rewrite
    #     - ``retrieval_status``   — current-turn retrieval outcome
    #     - ``needs_current_time`` — current-turn intent flag
    #     - ``created_at``         — current-turn assistant ts
    #     - ``documents`` / ``web_documents`` / ``graded_documents``
    #                                 — current-turn retrieval output
    #     - ``answer`` / ``sources`` / ``source_kinds`` /
    #       ``hallucination_check`` — current-turn synthesis output
    #
    # If you add a NEW field to AgentState, decide which list it
    # belongs to. Default = per-turn (most common case). Anything
    # in the durable list MUST be added to the explicit
    # allowlist below (or it silently resets on reload, which
    # v2.0.28.9 fixed for ``messages``).
    prior_messages: list = []
    prior_original_query: str | None = None
    if thread_id:
        try:
            from src.storage.checkpointer import load_latest
            prior_state = await load_latest(thread_id)
        except Exception as exc:
            # Storage hiccup must NOT block the current turn —
            # degrade gracefully and start a fresh thread. Log at
            # WARNING so an operator can spot a pattern of broken
            # hydration in the app log.
            logger.warning(
                f"runner.stream_agent failed to hydrate prior state "
                f"for thread_id={thread_id!r}: {type(exc).__name__}: {exc}"
            )
            prior_state = None
        if prior_state:
            prior_messages = list(prior_state.get("messages") or [])
            prior_original_query = prior_state.get("original_query")

    inputs = _initial_state(thread_id, user_message, high_precision=high_precision)
    if prior_messages:
        inputs["messages"] = prior_messages + inputs["messages"]
    if prior_original_query:
        # Carry forward the very first question the user asked in
        # this thread, so a 5th-turn answer can still reference
        # the original ask.
        inputs["original_query"] = prior_original_query

    # Streaming state — single-cell lists so the mapping helper can
    # mutate them across yields without declaring a class.
    accumulated_text = [""]
    last_emitted_text = [""]
    stream_buffer = [""]
    answer_complete_sent = [False]

    # v2.0.22 (Item 7 Step 9, P2-B3) — ``pending_tool_calls`` is
    # GONE from the runner. The FSM owns the tool-call lifecycle now:
    # ``react_agent`` records the start on ``state["_pending_tool_calls"]``,
    # ``_tools_step`` pops on match, and ``run_fsm`` drains the
    # remaining entries on Exception (synthetic ``tool_call_end``
    # events with ``ok=False``). The runner is now a pure
    # forwarder of FSM events — no bookkeeping.

    max_steps = _resolve_max_steps()

    try:
        async with aclosing(
            run_fsm(inputs, thread_id=thread_id, max_steps=max_steps)
        ) as stream:
            async for fsm_evt in stream:
                # v2.0.22 (Item 7 Step 9) — pure forwarder. The FSM
                # owns pending_tool_calls now (react_agent records
                # start, _tools_step pops on match, run_fsm drains
                # on Exception). We just map the FSM event to its
                # wire frame.
                wire = _emit_fsm_event(
                    fsm_evt,
                    accumulated_text=accumulated_text,
                    last_emitted_text=last_emitted_text,
                    stream_buffer=stream_buffer,
                    answer_complete_sent=answer_complete_sent,
                )
                if wire is None:
                    continue
                # ``answer_complete`` may flush a held-back token in
                # the same step — ``_emit_fsm_event`` returns a list
                # in that case.
                if isinstance(wire, list):
                    for w in wire:
                        yield w
                else:
                    yield wire
    except Exception as exc:
        # GeneratorExit is BaseException, not Exception, so
        # aclose() during a yield inside the try above will NOT be
        # caught here — it propagates cleanly out, which is exactly
        # what we want.
        logger.exception("Agent run failed")
        # v2.0.11 — classify the recursion-budget exception first.
        # The LLM hit the resolved ``max_steps`` budget (configured by
        # ``AGENT_BUDGETS["max_steps"]`` / ``RAG_REACT_MAX_STEPS``) without
        # converging. The generic redact_exception message says
        # "抱歉,生成回答时出现了问题" which doesn't tell the user WHY the
        # agent stopped or what to do. Surface an actionable message:
        # simplify the question or split into multiple turns.
        #
        # v2.0.21 (Phase 3 Step 7) — ``MaxStepsExceeded`` is the only
        # recursion-budget exception now (LangGraph was removed).
        # Previously we also caught ``GraphRecursionError`` for the
        # migration window; that import is gone.
        if isinstance(exc, MaxStepsExceeded):
            try:
                max_steps_val = (
                    getattr(exc, "max_steps", None) or _resolve_max_steps()
                )
            except Exception:
                max_steps_val = "?"
            yield events.error(
                f"Agent 思考步骤超过 max_steps={max_steps_val},已自动停止。"
                "这通常意味着:1) 问题需要拆分为多个步骤;"
                "2) LLM 在某个工具上反复重试。请尝试简化问题或"
                "将复杂任务拆分为多个对话。"
                f"\n\ntrace_id: {_trace_id_for(exc)}"
            )
        else:
            # v2.0.2 — classify the exception so transient Anthropic
            # errors get a clear, actionable user-facing message instead
            # of the generic "model unavailable" fallback.
            try:
                from src.security.prompt_safety import redact_exception

                cls_name = type(exc).__name__
                status_code = getattr(exc, "status_code", None) or 0
                if "AnthropicAPIError" in cls_name and int(status_code or 0) >= 500:
                    yield events.error(
                        "模型服务暂时繁忙(Anthropic "
                        f"{status_code})。请稍后重试。"
                        f"\n\ntrace_id: {_trace_id_for(exc)}"
                    )
                elif "AnthropicInvalidRequestError" in cls_name and int(status_code or 0) == 400:
                    yield events.error(
                        "内部错误:模型工具调用的请求格式与上游不兼容"
                        "(Anthropic 400 / tool contract)。"
                        f"\n\ntrace_id: {_trace_id_for(exc)}"
                    )
                else:
                    yield redact_exception(exc)
            except Exception:
                yield events.error("抱歉,生成回答时出现了问题。请稍后重试。")

    # v2.0.22 (Item 7 Step 9) — synthetic ``tool_call_end`` events
    # for unmatched starts are emitted by the FSM (``run_fsm``'s
    # except-clause drain yields them BEFORE the exception
    # propagates here, so we just forward them). The pre-Step-9
    # function-body drain block is gone. CancelledError /
    # ``aclose()`` still bypass drain (matches pre-Step-9
    # semantics — they are ``BaseException``, not ``Exception``,
    # so neither the FSM's nor the runner's except clause fires).

    # Yield 'done' AFTER the try block (not in finally). If the
    # generator is being closed via aclose() while suspended inside
    # the try, we never reach this line.
    yield events.done()


def reset_for_tests() -> None:
    """Reset runner state for tests.

    Phase 3: the runner has no module-level singleton cache (the
    FSM is stateless). This stub is preserved so the existing
    autouse fixture ``tests.conftest._reset_all_singletons``
    doesn't break. Step 7 may remove the call site if no other
    runner state remains.
    """
    return None


__all__ = [
    "stream_agent",
    "reset_for_tests",
]


# Backward-compat alias: tests still import
# ``from src.agent.runner import _extract_cited_indices``. The function
# lives in :src.agent.citations; we re-export it here so existing
# tests don't break on the ReAct rewrite.
from src.agent.citations import extract_cited_indices as _extract_cited_indices