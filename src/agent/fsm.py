"""Phase 3 (v2.0.19) — pure async FSM replacing LangGraph's StateGraph.

The FSM owns the recursion limit + step counter going forward and
replaces ``graph.astream(inputs, cfg, stream_mode=...)`` with a
single-event async iterator. Phase 3 ships with:

* this file (FSMEvent + merge + predicates + run_fsm loop + tools step)
* :mod:`src.agent.errors` (custom :class:`MaxStepsExceeded`)
* :mod:`src.storage.checkpointer` (own SQLite state store)

Pre-Phase-3 had ``src/agent/graph.py`` compiling a LangGraph
``StateGraph`` with the same topology; the file was deleted in
Step 7 along with all ``from langgraph`` imports across the codebase.

Migration history (Steps 1 → 7)
------------------------------
* Step 1 — :class:`MaxStepsExceeded` introduced; runner classifies it.
* Step 2 — predicates extracted to this file (still imported from ``graph.py``).
* Step 3 — ``run_fsm`` + async-generator node protocol shipped; ``graph.py`` still compiled the LangGraph StateGraph for tests.
* Step 4 — ``add_messages`` reducer dropped from ``AgentState``.
* Step 5 — ``InjectedState`` → ``InjectedToolArg`` (langchain_core, not LangGraph).
* Step 6 — own checkpointer shipped + one-shot migration from LangGraph ``checkpoints`` table.
* Step 7 (this step) — ``graph.py`` + ``history.py`` deleted; zero ``from langgraph`` imports remain.

Node protocol
-------------
Every node is an ``async def`` function that returns an async
generator yielding ``(event_kind, payload_dict)`` tuples. The last
yielded tuple MUST be ``("__delta__", delta_dict)`` carrying the
state delta the FSM merges into the running state. Sync nodes (no
streaming) just yield the ``__delta__`` and nothing else.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Literal, NotRequired, Tuple, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from config.constants import _resolve_max_steps
from src.agent._thinking_split import _strip_tool_blocks, split_text_and_thinking
# v2.0.22 (Item 7 Step 1+3) — shared helpers. ``now_iso`` lives in
# ``src.agent._time`` and the AIMessage content-block fallback lives
# in ``src.agent.nodes._content_blocks`` (kept aligned with
# ``nodes/react_generate.py`` via the helper).
from src.agent._time import now_iso as _now_iso
from src.agent.errors import MaxStepsExceeded
from src.agent.nodes._content_blocks import preserve_content_blocks
# v2.0.22 (Item 7 Step 11) — error-recovery helpers (P2-B4 关掉).
# ``llm_call_failure`` is used by ``react_generate_direct`` below;
# ``log_node_failure`` is used by ``summary_path`` and the
# end-of-turn checkpointer save.
from src.agent.nodes._recovery import llm_call_failure, log_node_failure
from src.agent.nodes.hallucination import check_hallucination_async
# v2.0.22 (Item 7 Step 12) — single source for "what docs does the
# hallucination judge look at?" (P1-B6 关掉). Pre-Step-12 the FSM
# predicate and the judge node used different fallback chains
# (``graded_documents → documents → []`` in the node vs just
# ``documents`` in the predicate). Both now route through the
# helper so the definition lives in one place.
from src.agent.nodes._docs_for_hallucination import docs_for_hallucination
from src.agent.nodes.intent_analysis import intent_analysis as _intent_analysis_fn
from src.agent.nodes.react_agent import react_agent as _react_agent_fn
from src.agent.nodes.react_generate import react_generate as _react_generate_fn
# v2.0.29.9 (Phase 8) — verbatim extraction node. Used INSTEAD of
# ``react_agent`` + ``react_generate`` when ``state["high_precision"]
# == "on"`` (user toggle or hybrid auto-detect). Sits at the same
# FSM position as ``react_generate`` (post-retrieval, pre-grounding)
# and emits ``answer_complete`` with sources[].verbatim=True.
from src.agent.nodes.react_generate_extractive import (
    react_generate_extractive as _react_generate_extractive_fn,
)
from src.agent.nodes.retrieve import _bulk_chunk_to_document, _filter_long_docs
# v2.0.29.7 (Phase 6) — second-pass verification node sits between
# ``react_generate`` and ``check_hallucination``. The predicate
# ``after_react_generate`` (below) routes through it only when
# ``should_trigger_verification`` returns True; the simple cases
# (greeting / direct / summary / no docs) skip straight to the
# grounding judge. See ``nodes/verify_answer.py`` for the node
# contract and Phase 6's fail-closed semantics.
from src.agent.nodes.verify_answer import verify_answer_node as _verify_answer_fn
from src.agent.state import AgentState
# v2.0.29.2 (Phase 2) — observability hook. Direct / refuse_X /
# generate template tags power dashboards verifying Phase 1 refusal
# contract is firing when expected (template=refusal_empty) vs
# bypassed (template=direct).
from src.agent.metrics import SYNTHESIS_PROMPT_TEMPLATE
from src.agent.tools import ALL_TOOLS
# v2.0.22 (Item 7 Step 7) — NodeSpec registry shape
# v2.0.22 (Item 7 Step 10) — RouteDecision Literal lives here so the
# FSMEvent shape and the citation helpers share the same closed
# vocabulary (instead of three modules each declaring ``str | None``).
from src.agent.types import NodeSpec, RouteDecision
from src.core.logging import logger
from src.llm.factory import build_chat_model
from src.llm.prompts import DIRECT_SYSTEM
from src.storage.lancedb_store import list_chunks_by_thread_cached


# Exit sentinel for the FSM. Plain ``None`` so a ``while next_node is
# not None`` loop terminates without depending on LangGraph's ``END``.
FSM_END: None = None


# ============================================================================
# FSMEvent TypedDict
# ============================================================================


class FSMEvent(TypedDict, total=False):
    """Single event yielded by ``run_fsm``. The runner maps each
    ``kind`` to the corresponding ``events.<kind>(...)`` WS frame so
    the frontend sees no change.

    The TypedDict is intentionally permissive (``total=False``) — the
    runner reads ``kind`` first and then accesses the kind-specific
    fields, so an absent field is just ``None`` at access time. This
    mirrors how ``events.<kind>(...)`` factories accept kwargs.
    """

    # ---- All kinds ----
    kind: Literal[
        "step_start",
        "step_end",
        "token",
        "reasoning",
        "intent",
        "tool_call_start",
        "tool_call_end",
        "answer_complete",
        "grounding",
        "web_search",
        # v2.0.29.7 (Phase 6) — second-pass verification outcome.
        # The frontend can ignore this in PR-1 (backward-compat:
        # an unknown ``type`` value lands in
        # ``window.dispatchEvent`` without breaking the existing
        # discriminated union). PR-2 / later will read it for a
        # ✓/✗ icon.
        "verification_result",
    ]

    # ---- Step boundary (step_start / step_end) ----
    node: str
    step: int

    # ---- Streaming (token / reasoning) ----
    text: str

    # ---- Intent classification (intent) ----
    intent_value: str
    corrected_query: str

    # ---- Tool calls (tool_call_start / tool_call_end / web_search) ----
    tool_call_id: str
    tool_name: str
    tool_args: dict
    tool_result: str
    tool_ok: bool
    elapsed_ms: int
    started_at: str

    # ---- Final answer (answer_complete) ----
    answer: str
    sources: list
    source_kinds: list
    created_at: str
    # v2.0.22 (Item 7 Step 10) — closed vocabulary on the wire (was
    # bare ``str`` before; now constrained to ``"direct"`` /
    # ``"retrieve"``). ``NotRequired`` because only ``answer_complete``
    # carries it; runners reading other kinds simply don't access the
    # field. The runner drops it before emitting the wire frame (it's
    # not in :mod:`src.agent.events`), so this declaration is purely
    # for in-process type discipline.
    route_decision: NotRequired[RouteDecision]

    # ---- Grounding (grounding) ----
    status: str

    # ---- v2.0.29.7 (Phase 6) verification_result ----
    consistent: bool
    mismatches_count: int
    regenerated: bool
    fallback_refusal: bool
    skipped: bool
    reason: str
    mismatches: list


# ============================================================================
# State merge
# ============================================================================


def merge(state: AgentState, delta: dict) -> AgentState:
    """Merge a node's delta dict into the running state.

    Plain dict update; ``messages`` is appended (no LangGraph
    reducer — Phase 3 dropped ``Annotated[List[BaseMessage],
    add_messages]``). Returns a new state dict; does not mutate
    ``state`` in place (callers can rely on snapshot semantics for
    debugging / checkpointing).

    v2.0.22 (Item 7 Step 5) — ``step_count`` is auto-incremented
    when the delta doesn't carry it. Previously every node's return
    dict had to remember ``"step_count": state.get("step_count", 0) + 1``
    verbatim; forgetting one would silently disable
    ``MaxStepsExceeded`` enforcement and let the FSM run forever.
    Nodes that need to opt out (e.g. ``_handle_tool_error`` /
    ``_tools_step`` helpers that fire mid-step without bumping the
    budget) can pass an explicit ``step_count`` value to override.

    v2.0.28.10 — ``replace_last_message`` key support. Synthesis
    nodes (``react_generate``) REPLACE the prior AIMessage instead
    of appending a new one. Without this, the FSM saves 2 AIMessages
    per turn when no tool calls are involved (react_agent's
    text-only "draft" + react_generate's "synthesis"); reload shows
    extra assistant bubbles that were never seen live. The
    ``after_react_agent`` predicate contract guarantees
    react_generate only runs when the last AIMessage has no
    ``tool_calls`` — so the replaced entry is always a text-only
    draft, never a tool-call AIMessage (which must stay paired
    with its ToolMessage for the ToolCallCard to render). Falls
    back to ``[v]`` if state["messages"] is empty (defensive —
    shouldn't happen because runner always seeds a HumanMessage).
    """
    if not delta:
        return state
    out = dict(state)
    for k, v in delta.items():
        if k == "messages":
            existing = list(out.get("messages") or [])
            out["messages"] = existing + list(v or [])
        elif k == "replace_last_message":
            # v2.0.28.10 — synthesis nodes REPLACE the prior
            # AIMessage. See module docstring for the rationale and
            # safety analysis. ``messages`` (above) and
            # ``replace_last_message`` are mutually exclusive per
            # node — react_generate uses the latter, every other
            # node uses the former. Read from ``out["messages"]``
            # (the in-flight accumulator), NOT from ``state``
            # (the original snapshot) — so if a future node ever
            # yielded both keys in the same delta, the order of
            # dict iteration would still compose correctly.
            existing = list(out.get("messages") or [])
            if existing:
                existing[-1] = v
            else:
                existing = [v]
            out["messages"] = existing
        else:
            out[k] = v
    # Centralized step budget (v2.0.22). Only inject when the node
    # didn't carry an explicit value — gives helpers like
    # ``_handle_tool_error`` the escape hatch to merge without
    # bumping the counter.
    if "step_count" not in delta:
        out["step_count"] = int(state.get("step_count", 0) or 0) + 1
    return out


# ============================================================================
# Predicates (formerly graph._after_intent / _after_react_agent / _should_check_hallucination)
# ============================================================================


def after_intent(state: AgentState) -> str | None:
    """Map ``state["intent"]`` to the next node.

    Greeting    → ``react_generate_direct``  (no retrieval, no tools)
    simple_fact → ``react_generate_direct``  (no retrieval, no tools —
                  v2.0.5; answers from training knowledge)
    Summary     → ``summary_path``           (bulk chunks → react_generate)
    qa_complex  → ``react_agent``            (ReAct loop begins)

    v2.0.29.9 (Phase 8) — verbatim extraction override. The
    ``high_precision`` state flag is a TRISTATE (per user decision
    2026-09-28):

      - ``"off"`` (user override) → NEVER route to extractive,
        regardless of intent. This is the only literal that wins
        unconditionally — even if hybrid detect flagged the query,
        user override fires.
      - ``"on"`` (user forced OR hybrid detect) → route
        ``qa_complex`` → ``react_generate_extractive`` (verbatim mode).
        ``greeting`` / ``simple_fact`` / ``summary`` skip extractive
        because they don't have retrieval context to extract from.
      - ``"auto"`` (default) → check if hybrid detect set the flag to
        ``"on"`` during intent_analysis; if yes, route to extractive.
        Otherwise fall through to normal routing.

    Cost on the ``"off"`` / ``"auto"`` paths: 1 dict.get() per turn
    (~µs). Cost on the ``"on"`` path: 1 dict.get() + 1 string compare.
    No LLM calls.
    """
    intent = state.get("intent")
    hp = state.get("high_precision") or "auto"

    # Phase 8 — verbatim routing (priority over intent classification
    # for qa_complex; gated by user override semantics).
    if hp == "on" and intent == "qa_complex":
        return "react_generate_extractive"
    # "auto" with detect fired == state.high_precision was set to "on"
    # by intent_analysis (regex match OR cheap-LLM judge returned True).
    if hp == "on" and intent not in ("greeting", "simple_fact", "summary"):
        return "react_generate_extractive"

    # Normal routing — applies when high_precision is "auto" / "off" /
    # unset.
    if intent in ("greeting", "simple_fact"):
        return "react_generate_direct"
    if intent == "summary":
        return "summary_path"
    return "react_agent"


def after_react_agent(state: AgentState) -> str | None:
    """Pick next node after ``react_agent``.

    Mirrors LangChain's prebuilt ``tools_condition`` (looks at the
    last AIMessage's ``tool_calls`` field) — re-implemented inline
    so the FSM doesn't depend on ``langgraph.prebuilt.tools_condition``.
    """
    msgs = state.get("messages") or []
    if not msgs:
        return "react_generate"
    last = msgs[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "react_generate"


def should_check_hallucination(state: AgentState) -> str | None:
    """Skip the judge when there's nothing to judge.

    Returns ``None`` (= FSM end) when the intent is one of
    ``{greeting, summary, simple_fact}`` OR there are no documents
    to verify against. Otherwise routes to ``check_hallucination``.

    v2.0.22 (Item 7 Step 12) — the "what counts as docs" lookup
    now goes through :func:`docs_for_hallucination` (the
    ``graded_documents → documents → []`` chain) so this predicate
    and the ``check_hallucination_async`` node always agree. Pre-
    Step-12 this predicate read ``state["documents"]`` directly
    while the node read ``graded_documents → documents`` — a
    divergence that would matter as soon as a future node writes
    ``graded_documents``.
    """
    intent = state.get("intent")
    if intent in ("greeting", "summary", "simple_fact"):
        return None
    docs = docs_for_hallucination(state)
    if not docs:
        return None
    return "check_hallucination"


def after_react_generate(state: AgentState) -> str | None:
    """v2.0.29.7 (Phase 6) — route react_generate's output through
    second-pass verification when warranted.

    Flow:

      react_generate
        → verify_answer (Phase 6 — when should_trigger)
        → check_hallucination
        → END

    Skips verify_answer (goes straight to check_hallucination) when:

      * there's no answer (refusal path / direct route already
        walked away)
      * no documents to cross-check against
      * ``should_trigger_verification`` returns False — zero-cost
        path; the answer is chitchat or has no extractable fields

    On the regen path, ``verify_answer_node`` stamps the FSM state
    with ``verification_mismatch_directive``; the FSM re-enters
    ``react_generate`` via the explicit edge below.
    """
    if not state.get("answer"):
        return should_check_hallucination(state)
    if not docs_for_hallucination(state):
        return should_check_hallucination(state)
    answer_text = state.get("answer") or ""
    query = (
        state.get("original_query")
        or state.get("current_query")
        or ""
    )
    # Lazy import to avoid a hard dependency on the verification
    # subsystem at module-load time (test collection would fail
    # if anyone had a typo in verifier.py).
    from src.agent.verification.verifier import should_trigger_verification
    if not should_trigger_verification(answer_text, query):
        return should_check_hallucination(state)
    return "verify_answer"


def after_react_generate_extractive(state: AgentState) -> str | None:
    """v2.0.29.9 (Phase 8) — route react_generate_extractive's output
    through the same Phase 6 verification chain as the normal path.

    Mirror of ``after_react_generate`` (line 317). Verbatim output
    is more likely to contain extractable fields (dates / numbers /
    article numbers) — the Phase 6 rule_engine + cheap-LLM still
    applies usefully and the routing decision stays consistent with
    the normal synthesis path.

    Falls through to ``check_hallucination`` (skipping verify_answer)
    when there's no answer (refusal fallback already walked away)
    or no docs to cross-check against.
    """
    if not state.get("answer"):
        return should_check_hallucination(state)
    if not docs_for_hallucination(state):
        return should_check_hallucination(state)
    answer_text = state.get("answer") or ""
    query = (
        state.get("original_query")
        or state.get("current_query")
        or ""
    )
    from src.agent.verification.verifier import should_trigger_verification
    if not should_trigger_verification(answer_text, query):
        return should_check_hallucination(state)
    return "verify_answer"


# Predicate map keyed by the SOURCE node. Each predicate returns the
# NEXT node name to run, or ``None`` to terminate the FSM. Static
# edges (``add_edge`` in LangGraph) are listed alongside the
# conditional ones so the FSM has a complete routing table.
PREDICATES: dict[str, Any] = {
    "intent_analysis": after_intent,
    "react_agent": after_react_agent,
    # v2.0.29.7 (Phase 6) — react_generate routes through
    # verify_answer (when should_trigger) before reaching the
    # grounding judge. Pre-Phase-6 this was a direct edge to
    # check_hallucination (the v2.0.22+ should_check_hallucination
    # predicate was the only filter — it skipped when there were no
    # docs to verify, but didn't catch fabrication cases where
    # the answer cites a doc that doesn't actually contain the
    # claimed fact).
    "react_generate": after_react_generate,
    # v2.0.29.9 (Phase 8) — verbatim extraction node routes through
    # the same Phase 6 verification chain (when should_trigger).
    # Pre-Phase-8 routing didn't include this entry; the FSM's
    # ``run_fsm`` loop returns ``None`` (= FSM_END) when an unknown
    # source node appears in a PREDICATES lookup, which would have
    # silently dropped the bubble on a verbatim turn.
    "react_generate_extractive": after_react_generate_extractive,
    # Explicit unconditional edges that LangGraph's
    # ``g.add_edge("X", "Y")`` set up implicitly. The FSM has no
    # implicit topology, so every routing must be named here.
    "tools": lambda _state: "react_agent",
    "summary_path": lambda _state: "react_generate",
    "react_generate_direct": lambda _state: "check_hallucination",
    # v2.0.29.7 (Phase 6) — verify_answer always routes to
    # check_hallucination. The regen path stamps the FSM state
    # with ``verification_mismatch_directive``; the next predicate
    # below (``after_verify_answer``) catches the regen case and
    # routes back into react_generate.
    "verify_answer": lambda state: (
        "react_generate"
        if state.get("verification_mismatch_directive")
        else "check_hallucination"
    ),
    "check_hallucination": lambda _state: FSM_END,
}


# ============================================================================
# Tool error helper (formerly graph._handle_tool_error)
# ============================================================================


def _handle_tool_error(exception: Exception) -> str:
    """ToolNode ``handle_tool_errors`` callback equivalent.

    Returning a string becomes the ``ToolMessage.content`` (the LLM
    sees it on its next ReAct step). Returning ``True`` would
    swallow the error silently — we never want that.
    """
    msg = f"Tool execution failed: {type(exception).__name__}: {exception}"
    logger.debug(f"react_agent: tool error → {msg}")
    return msg


# ============================================================================
# Fast-path nodes (formerly graph.react_generate_direct / summary_path)
# ============================================================================


async def react_generate_direct(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """Greeting fast-path: stream a direct conversational answer.

    Async generator protocol (Phase 3): yields ``("token", {...})`` /
    ``("reasoning", {...})`` while the LLM streams, then
    ``("answer_complete", {...})`` carrying the direct-path payload
    (no sources, ``route_decision="direct"``). Ends with the
    mandatory ``("__delta__", {...})`` terminator the FSM merges.

    Same v1.1.13 contract as ``generate.generate_answer_async`` on
    the ``route_decision == "direct"`` branch — no documents, no
    citations, no tools.
    """
    # v2.0.29.2 (Phase 2) — observability hook. Bumped BEFORE the
    # LLM stream so dashboards see which template was selected even
    # when the LLM raises mid-stream (counter tells you "direct was
    # chosen but never delivered").
    SYNTHESIS_PROMPT_TEMPLATE.inc(template="direct")
    history = state.get("messages") or []
    model = build_chat_model(enable_thinking=True)
    msgs = [SystemMessage(content=DIRECT_SYSTEM), *history]

    answer_text = ""
    last_chunk = None
    try:
        async for chunk in model.astream(msgs):
            last_chunk = chunk
            text, thinking = split_text_and_thinking(getattr(chunk, "content", ""))
            if text:
                answer_text += text
                yield ("token", {"text": text})
            if thinking:
                # v2.0.28.15 — DO NOT yield reasoning events for the
                # synthesis LLM. Synthesis extended-thinking text is
                # meta-commentary ("I'm about to write the answer")
                # that actively MISLEADS users (probe Run 1 saw "I
                # already provided the answer in my previous response"
                # — false). Planning-step reasoning from ``react_agent``
                # still streams for transparency. Pre-fix path yielded
                # ("reasoning", ...) → reasoningHandler.ts →
                # <ThinkingDrawer text={m.reasoning}/> default-open →
                # user saw meta-commentary above the direct answer.
                # Drop it. See also ``nodes/react_generate.py`` for
                # the synthesis-LLM side of the same fix.
                pass
        if not answer_text:
            answer_text = "(no answer text generated)"
    except Exception as exc:
        # v2.0.22 (Item 7 Step 11) — centralized LLM-failure recovery
        # (P2-B4 关掉). Mirrors ``react_generate`` byte-for-byte;
        # both used to inline the same apology text + token yield.
        answer_text, token_event = llm_call_failure("react_generate_direct", exc)
        last_chunk = None
        # Emit the fallback message as a single token so the user
        # still sees the answer stream live (rare path).
        yield token_event

    answer_text = _strip_tool_blocks(answer_text)
    created_at = _now_iso()

    additional_kwargs: dict = {}
    # v2.0.28.15 — DO NOT stamp reasoning on synthesis AIMessage.
    # Same rationale as the streaming-loop pass above. See
    # ``nodes/react_generate.py`` for the full rationale + pre-fix
    # probe evidence.
    additional_kwargs["source_kinds"] = []  # direct path: no banner
    additional_kwargs["created_at"] = created_at

    # v2.0.22 (Item 7 Step 3) — block-preservation logic moved to
    # ``src.agent.nodes._content_blocks.preserve_content_blocks`` so
    # this file and ``nodes/react_generate.py`` stay in sync when
    # Anthropic adds a new block type.
    final_content = preserve_content_blocks(last_chunk, answer_text)

    response = AIMessage(content=final_content, additional_kwargs=additional_kwargs)

    yield ("answer_complete", {
        "answer": answer_text,
        "sources": [],
        "source_kinds": [],
        "created_at": created_at,
        "route_decision": "direct",
    })

    yield ("__delta__", {
        "answer": answer_text,
        "sources": [],
        "source_kinds": [],
        "documents": [],
        "messages": [response],
        "created_at": created_at,
        # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
        # ``merge()`` since this delta doesn't carry it explicitly.
    })


async def summary_path(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """Summary fast-path: dump every chunk of the thread's documents.

    Bypasses semantic retrieval because meta-questions ("summarize
    this doc" / "what's in this PDF") share no embedding space
    with the doc body. The LLM reads everything and writes a
    summary in ``react_generate``.

    v2.0.22 (Item 7 Step 6) — async-generator protocol: yields
    exactly one ``("__delta__", {"documents": ...})`` terminator.
    Falls through to react_generate with empty docs if the thread
    has no documents — the LLM will see "<documents source=none/>"
    and respond with "this thread has no uploaded documents".
    """
    thread_id = state.get("thread_id")
    if not thread_id:
        yield ("__delta__", {"documents": []})
        return

    try:
        # v2.0.7 perf-3: summary path uses the 30 s TTL cache to
        # avoid a 200-500 ms full-table scan on every turn.
        chunks = await asyncio.to_thread(list_chunks_by_thread_cached, thread_id)
    except Exception as exc:
        # v2.0.22 (Item 7 Step 11) — uniform logging via the
        # centralized helper (was ``"summary_path: list_chunks_by_thread
        # failed: ..."``). Recovery decision (empty chunks → empty
        # docs) stays inline because it's storage-specific.
        log_node_failure("summary_path", exc)
        chunks = []

    docs = [_bulk_chunk_to_document(c) for c in (chunks or [])]
    # v2.0.29.8 (Phase 7) — sliding window for per-doc overflow in
    # the bulk summary path. This is the most common "long doc"
    # scenario (a 200-page PDF dumped wholesale). The helper is
    # defined in ``src/agent/nodes/retrieve.py`` so both the FSM
    # summary_path and the retrieve.py summary-intent branch share
    # the same contract.
    docs = _filter_long_docs(docs, thread_id=thread_id)
    # v2.0.29.1 (Phase 1) — summary-intent path distinguishes
    # "thread has docs to summarize" from "thread has no docs". The
    # latter must walk the REFUSAL_EMPTY_BULK_TEMPLATE so the user
    # sees "请先上传文档" instead of a fabricated summary.
    retrieval_status = "success" if docs else "empty_bulk"
    yield ("__delta__", {
        "documents": docs,
        "retrieval_status": retrieval_status,
        # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
        # ``merge()`` since this delta doesn't carry it explicitly.
    })


# ============================================================================
# Tools step (replaces langgraph.prebuilt.ToolNode)
# ============================================================================


_TOOL_RESULT_PREVIEW_MAX = 800  # matches react_generate's persisted tool log


async def _tools_step(
    state: AgentState,
    *,
    step: int = 0,
) -> AsyncIterator[Tuple[str, Any]]:
    """Hand-rolled replacement for :class:`langgraph.prebuilt.ToolNode`.

    Reads the LAST AIMessage from ``state["messages"]``, iterates its
    ``tool_calls`` list, and for each call:

    1. Looks up the tool from :data:`src.agent.tools.ALL_TOOLS` (by
       ``tool.name``) and awaits ``tool.ainvoke(args)``. Exceptions
       are caught and routed through :func:`_handle_tool_error` so
       the graph sees an honest error string in the matching
       ``ToolMessage`` (matches the pre-Phase-3 ToolNode contract).
    2. Yields ``("tool_call_end", {...})`` with ``ok``, elapsed_ms,
       step, and the (truncated) result. ``tool_call_start`` is
       emitted by the FSM (in :func:`run_fsm`) before this step
       runs so the frontend can show "calling tool" BEFORE the tool
       starts — the pre-Phase-3 ``_handle_react_agent`` fired
       tool_call_start before the tool ran; we preserve that
       ordering. ``step`` is the 1-based step number the
       ``tool_call_start`` event for this tool was emitted on, so
       the frontend pairs them by id and labels both with the same
       step number.
    3. Appends a ``ToolMessage`` to the messages list.

    Finally yields ``("__delta__", {"messages": [...], "step_count": N+1})``
    so the FSM merges the tool results into state and routes back to
    ``react_agent``.
    """
    history = list(state.get("messages") or [])
    if not history:
        yield ("__delta__", {"messages": []})
        return
    last_msg = history[-1]
    if not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
        yield ("__delta__", {"messages": []})
        return

    tool_messages: list[ToolMessage] = []
    for tool_call in last_msg.tool_calls:
        # Tool calls come either as dicts (the LangChain stable shape
        # since 0.1) or as ToolCall objects (the legacy shape). Both
        # formats are tolerated so this step works regardless of the
        # LangChain version pinned.
        if isinstance(tool_call, dict):
            name = tool_call.get("name", "")
            args = tool_call.get("args", {}) or {}
            call_id = tool_call.get("id", "")
        else:
            name = getattr(tool_call, "name", "")
            args = getattr(tool_call, "args", {}) or {}
            call_id = getattr(tool_call, "id", "")

        tool = next((t for t in ALL_TOOLS if getattr(t, "name", "") == name), None)
        start_time = time.monotonic()
        ok = True
        if tool is None:
            result = f"Tool execution failed: unknown tool {name!r}"
            ok = False
        else:
            try:
                # Phase 3 — v2.0.20. Inject ``thread_id`` from the FSM
                # state into the tool call's args dict. The tool
                # function declares ``thread_id: Annotated[str,
                # InjectedToolArg] = ""`` (langchain_core, NOT
                # langgraph) which keeps it out of the JSON schema the
                # LLM sees, so the model never hallucinates thread
                # ids. LangChain's ``_to_args_and_kwargs`` turns the
                # args dict into function kwargs, so the injected
                # value lands on the tool's ``thread_id`` parameter
                # without us needing a LangGraph ``ToolNode``.
                result = await tool.ainvoke({
                    **args,
                    "thread_id": state.get("thread_id", ""),
                })
            except Exception as exc:
                result = _handle_tool_error(exc)
                ok = False
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        truncated = str(result)[:_TOOL_RESULT_PREVIEW_MAX]

        # v2.0.22 (Item 7 Step 9) — pop from FSM pending tracker
        # (P2-B3 — lifecycle moves from runner.py to FSM). Mirrors
        # the pre-Step-9 runner.py:354 bookkeeping. Defensive
        # ``.get(...)`` because react_agent may not have recorded
        # the start (e.g. tool_call_id mismatch — never observed
        # but cheap insurance).
        state.get("_pending_tool_calls", {}).pop(call_id, None)

        yield ("tool_call_end", {
            "tool_call_id": call_id,
            "tool_ok": ok,
            "elapsed_ms": elapsed_ms,
            "step": step,
            "tool_result": truncated,
        })

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call_id, name=name)
        )

    yield ("__delta__", {
        "messages": tool_messages,
        # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
        # ``merge()`` since this delta doesn't carry it explicitly.
    })


# ============================================================================
# Node registry
# ============================================================================


# Each entry maps a node name to a :class:`NodeSpec` declaring the
# async-generator ``fn`` plus any FSM-managed wire events emitted
# before / after the node runs. ``run_fsm`` (below) uses this dict
# as the single source of truth for "what does this node do?".
#
# v2.0.22 (Item 7 Step 7) — :class:`NodeSpec` shape replaces the
# pre-Step-7 ``dict[str, Callable]`` (which forced the FSM main loop
# to hardcode ``if current == "react_agent"`` branches for
# ``step_start`` / ``step_end`` / ``intent`` / ``grounding`` wire
# events — the P0-B1 audit finding).
#
# Per-call DATA events (``intent`` / ``grounding`` /
# ``tool_call_start`` / ``web_search``) are now yielded by the node
# function itself, BEFORE ``__delta__``. The node knows the data;
# the FSM only knows when (before / after merge). ``emits_pre`` /
# ``emits_post`` carry the FSM-managed events that are the same on
# every call (``step_start`` / ``step_end`` for ``react_agent``).
NODES: dict[str, NodeSpec] = {
    "intent_analysis": NodeSpec(
        fn=_intent_analysis_fn,
        emits_pre=[],
        emits_post=[],
    ),
    "react_generate_direct": NodeSpec(
        fn=react_generate_direct,
        emits_pre=[],
        emits_post=[],
    ),
    "summary_path": NodeSpec(
        fn=summary_path,
        emits_pre=[],
        emits_post=[],
    ),
    "react_agent": NodeSpec(
        fn=_react_agent_fn,
        emits_pre=["step_start"],
        emits_post=["step_end"],
    ),
    "tools": NodeSpec(
        fn=_tools_step,
        emits_pre=[],
        emits_post=[],
    ),
    "react_generate": NodeSpec(
        fn=_react_generate_fn,
        emits_pre=[],
        emits_post=[],
    ),
    "check_hallucination": NodeSpec(
        fn=check_hallucination_async,
        emits_pre=[],
        emits_post=[],
    ),
    # v2.0.29.7 (Phase 6) — second-pass verification node. Sits
    # between react_generate and check_hallucination. Emits a
    # ``verification_result`` wire event (forwarded by the FSM
    # via the standard events-list path). No step_start/step_end
    # because it's a synchronous-looking node (single LLM call,
    # ~2 s).
    "verify_answer": NodeSpec(
        fn=_verify_answer_fn,
        emits_pre=[],
        emits_post=[],
    ),
    # v2.0.29.9 (Phase 8) — verbatim extraction node. Sits between
    # intent_analysis and the verification chain (replaces the
    # react_agent + react_generate pair when state.high_precision ==
    # "on"). Emits ``answer_complete`` with sources[].verbatim=True
    # (forwarded by the FSM via the standard events-list path). No
    # step_start/step_end because it's a single LLM call (~2 s) and
    # mirrors react_generate's event shape.
    "react_generate_extractive": NodeSpec(
        fn=_react_generate_extractive_fn,
        emits_pre=[],
        emits_post=[],
    ),
}


# ============================================================================
# FSM core loop
# ============================================================================


async def _run_node(
    node_fn, state: AgentState, *, step: int = 0
) -> Tuple[list, dict | None]:
    """Run a node function, returning ``(events_list, delta_dict)``.

    v2.0.22 (Item 7 Step 6) — single protocol: every node is
    ``async def (state, *, step) -> AsyncIterator[(event_kind, payload)]``.
    It yields zero or more ``(event_kind, payload)`` events, then
    MUST yield exactly one ``("__delta__", delta_dict)`` tuple
    terminator. The previous sync ``async def (state) -> dict`` shape
    was retired along with the ``inspect.isasyncgenfunction`` sniff +
    ``_accepts_step`` helper — all production nodes now match the
    streaming shape, so ``_run_node`` is straight-line async-gen
    dispatch.

    ``step`` is the 1-based step number for the current FSM
    iteration (matches the pre-Phase-3 runner's tool-call tracker
    ``step`` arg). Streaming nodes that emit per-tool events include
    it so the frontend pairs ``tool_call_start`` and
    ``tool_call_end`` on the same step number; sync-shape nodes
    accept and ignore it.

    A node that violates the protocol (no ``__delta__`` yielded) raises
    :class:`RuntimeError` so the FSM surfaces a clear error instead of
    silently looping forever.
    """
    events: list = []
    delta = None
    async for evt in node_fn(state, step=step):
        kind, payload = evt
        if kind == "__delta__":
            if delta is not None:
                raise RuntimeError(
                    f"{node_fn.__name__}: yielded __delta__ twice"
                )
            delta = payload
        else:
            events.append((kind, payload))
    if delta is None:
        raise RuntimeError(
            f"{node_fn.__name__} exhausted without yielding __delta__ — "
            f"all streaming nodes must end with ('__delta__', delta_dict)"
        )
    return events, delta


async def run_fsm(
    state: AgentState,
    *,
    thread_id: str = "",
    max_steps: int | None = None,
) -> AsyncIterator[FSMEvent]:
    """Drive the ReAct topology as a plain ``while`` loop.

    Topology (replaces ``StateGraph``):

      ``intent_analysis``
        ├ greeting/simple_fact → ``react_generate_direct``
        │                        → ``check_hallucination``
        ├ summary               → ``summary_path``
        │                        → ``react_generate``
        │                        → ``check_hallucination``
        └ qa_complex            → ``react_agent``
                                 ↔ ``tools`` (loop)
                                 → ``react_generate``
                                 → ``check_hallucination``

    Yields :class:`FSMEvent` dicts in execution order; the runner
    maps each kind to the corresponding ``events.<kind>(...)`` WS
    frame so the frontend sees no change.

    Raises :class:`src.agent.errors.MaxStepsExceeded` when
    ``step_count`` hits the resolved ``max_steps`` budget.
    """
    if max_steps is None:
        max_steps = _resolve_max_steps()

    # v2.0.22 (Item 7 Step 9, P2-B3) — pending tool-call lifecycle
    # moves from runner.py to FSM. The tracker is keyed by
    # ``tool_call_id`` and is mutated by:
    #
    # * ``react_agent`` on ``tool_call_start`` (record step / name /
    #   started_at so the FSM can drain on shutdown with the same
    #   metadata the runner used to track)
    # * ``_tools_step`` on ``tool_call_end`` (pop, so the FSM's
    #   drain only catches truly unmatched starts — same end was
    #   matched, no synthetic close needed)
    #
    # The tracker lives on ``state`` (not module-level) so concurrent
    # runs don't share state. ``merge``'s shallow-copy keeps the
    # nested dict identity stable across node calls — as long as we
    # never put ``_pending_tool_calls`` in a node's ``__delta__``
    # payload (we don't), mutations via ``state["_pending_tool_calls"]``
    # persist through every ``state = merge(state, delta)`` rebind.
    state["_pending_tool_calls"] = {}

    current = "intent_analysis"

    try:
        while current is not None:
            # Budget guard — checked BEFORE the node so the user gets an
            # actionable error rather than the generic redact fallback.
            # ``step_count`` reflects the number of nodes already run on
            # this turn (intent + N react_agent iterations + tools +
            # react_generate + check_hallucination).
            step_so_far = int(state.get("step_count", 0) or 0)
            if step_so_far >= max_steps:
                raise MaxStepsExceeded(max_steps)

            spec = NODES.get(current)
            if spec is None:
                raise RuntimeError(f"Unknown FSM node: {current!r}")
            node_fn = spec["fn"]

            # 1-based step number for THIS iteration. The first
            # ``react_agent`` call after intent_analysis is step 1; each
            # ``react_agent → tools`` loop bump advances it by 1. We pass
            # it to streaming nodes (so they can stamp it on per-tool
            # events) and to the step_start/step_end boundary events.
            current_step = step_so_far + 1

            # v2.0.22 (Item 7 Step 7) — declarative pre-step wire events
            # (``step_start`` for ``react_agent`` so the frontend's
            # "thinking..." spinner lights up at the right moment). All
            # other streaming nodes carry their own progress via token
            # events so they don't need a pre-step signal.
            for kind in spec.get("emits_pre", []) or ():
                yield FSMEvent(kind=kind, node=current, step=current_step)

            events, delta = await _run_node(node_fn, state, step=current_step)

            # Forward per-node events to the caller. Per-call DATA
            # events (``intent`` / ``grounding`` / ``tool_call_start``
            # / ``web_search``) are yielded by the node itself BEFORE
            # ``__delta__`` — the node knows the data, the FSM only
            # knows the dispatch order.
            for kind, payload in events:
                yield FSMEvent(kind=kind, **payload)

            if delta is None:
                # Defensive — _run_node already raises if delta is None
                # for an async-gen node. A plain async node returning
                # ``None`` would land here; treat it as empty.
                delta = {}

            # Merge delta into the running state.
            state = merge(state, delta)

            # Declarative post-step wire events (``step_end`` for
            # ``react_agent`` so the per-step border renders after the
            # tool calls settle). Per-node data events have already been
            # yielded in the ``events`` loop above.
            for kind in spec.get("emits_post", []) or ():
                yield FSMEvent(kind=kind, node=current, step=current_step)

            # Determine the next node.
            next_predicate = PREDICATES.get(current, lambda s: FSM_END)
            current = next_predicate(state)

        # ----- End of turn: persist the final snapshot -----
        # Phase 3 (v2.0.21) — own checkpointer replaces LangGraph's
        # auto-save hook. We persist the latest ``state`` (which now
        # contains the full accumulated message list including this
        # turn's H + A + T) so /sessions + /sessions/{id}/messages can
        # replay the history on page refresh. Failures are logged but
        # do NOT abort the run — the user already got their answer via
        # the events emitted above; a failed save just means the next
        # /sessions list won't show this turn until it succeeds.
        if thread_id:
            try:
                from src.storage.checkpointer import save as _save_state

                await _save_state(
                    thread_id,
                    state,
                    step=int(state.get("step_count") or 0),
                )
            except Exception as exc:
                # v2.0.22 (Item 7 Step 11) — uniform logging via the
                # centralized helper (was ``"FSM: failed to persist
                # end-of-turn snapshot (thread_id={...!r})"``).
                # Dropped the explicit thread_id from the message
                # because the exception traceback already names
                # the failing call site, and ``state.get("thread_id")``
                # is still recoverable from the surrounding scope.
                log_node_failure("fsm.save_state", exc)
    # ----- End of FSM try/except/finally (Item 7 Step 9) -----
    except Exception:
        # Drain pending tool calls — yield synthetic ``tool_call_end``
        # events with ``ok=False`` so the frontend closes open pills.
        # Mirrors the pre-Step-9 runner.py:430-441 behavior exactly
        # (same payload shape: ``(cancelled before tool result)`` /
        # ``ok=False`` / ``elapsed_ms=0``). CancelledError /
        # GeneratorExit (``aclose()``) bypass this because they are
        # ``BaseException``, not ``Exception`` — same semantics as
        # the runner's pre-Step-9 drain block (function-body level,
        # not in a finally).
        for tool_call_id, meta in list(
            state.get("_pending_tool_calls", {}).items()
        ):
            yield FSMEvent(
                kind="tool_call_end",
                tool_call_id=tool_call_id,
                tool_ok=False,
                tool_result="(cancelled before tool result)",
                step=int(meta.get("step", 1) or 1),
                elapsed_ms=0,
            )
        raise  # re-raise so the runner's except catches the original
    finally:
        # Cleanup only — no yield here. ``runner.py` cancellation
        # contract (Item 7 Step 9 lineage, runner.py:30-34) says
        # yielding inside finally during aclose causes
        # ``RuntimeError("async generator ignored GeneratorExit")``.
        # The drain (with yield) lives in the ``except`` block above.
        state.pop("_pending_tool_calls", None)


__all__ = [
    "FSMEvent",
    "FSM_END",
    "merge",
    "PREDICATES",
    "NODES",
    "after_intent",
    "after_react_agent",
    # v2.0.29.7 (Phase 6) — react_generate's predicate now routes
    # through verify_answer when should_trigger; kept separate from
    # should_check_hallucination so the existing grounding predicate
    # stays a single-purpose call.
    "after_react_generate",
    "should_check_hallucination",
    "react_generate_direct",
    "summary_path",
    "_tools_step",
    "_handle_tool_error",
    "_run_node",
    "run_fsm",
]