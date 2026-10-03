"""react_agent node — the ReAct "think + tool-call" half of the loop.

This is the LLM step. We:

1. Take the current ``state["messages"]`` (which by now contains the
   user's HumanMessage, any prior AI/Tool exchanges, and the optional
   time-tool hint injected when ``state["needs_current_time"]``).
2. Build the main chat model with ``bind_tools(ALL_TOOLS)``.
3. ``ainvoke`` the model — it returns an ``AIMessage`` that either
   carries ``tool_calls`` (continue loop) or a final assistant text
   (proceed to ``react_generate``).
4. Append the AIMessage to ``state["messages"]`` and bump
   ``step_count`` by 1.

LangGraph's prebuilt ``ToolNode`` + ``tools_condition`` handle the
"if tool_calls → tools, else → react_generate" branch in the graph
wiring (see :mod:`src.agent.graph`). This module only does the LLM
call.

Why we use the main model here (not the cheap one)
--------------------------------------------------
The ReAct agent needs the same reasoning quality that produces the
final answer. If we used the cheap model for the planning step and
the main model only for the answer, the LLM would plan one thing
and write another — and the planning-quality gap shows up as "the
LLM called the wrong tool" (e.g. calling ``web_search`` for a pure
doc question, or ``retrieve_docs`` for a current-events query). The
user's whole reason for the ReAct rewrite was to put a capable
planner in charge; using the cheap model here would defeat the
point.

Why extended thinking on this call
----------------------------------
The thinking block gives the LLM time to reason about which tool
to call. Without thinking, the cheap-router pre-step would have
already done the planning — but we just argued against that. So:
keep thinking on, inherit the same Anthropic temperature=1.0
requirement, let the LLM budget 1500 tokens of reasoning before
emitting tool calls.

Why we inject a time-tool hint when ``needs_current_time``
----------------------------------------------------------
The cheap-model pre-step already detected recency markers. We
surface that as a system-reminder inside the LLM's prompt so it
considers the time tool first. This is a HINT — the LLM is free
to ignore it (e.g. "yesterday's stock price" doesn't need the
current time to answer). The point is to nudge the LLM, not
mandate it.

Tool-call log accumulation
--------------------------
Every time this node returns, ``state["messages"][-1]`` is an
AIMessage. The runner reads ``last.tool_calls`` and emits a
``tool_call_start`` event for each (before the ``ToolNode``
executes them). The matching ``tool_call_end`` fires when the
ToolNode produces its ToolMessage. See
:mod:`src.agent.runner._accumulate_tool_calls`.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Tuple

from langchain_core.messages import AIMessage, SystemMessage

from src.agent._time import now_iso as _now_iso  # v2.0.22 (Item 7 Step 7) — stamp tool_call_start.started_at
from src.agent.nodes._history_frame_hint import _HISTORY_FRAME_HINT
# v2.0.22 (Item 7 Step 11) — error-recovery helpers (P2-B4 关掉).
# ``log_node_failure`` replaces the bare ``except Exception`` that
# previously swallowed ``list_documents_cached`` failures silently.
from src.agent.nodes._recovery import log_node_failure
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS, TOOL_METADATA  # v2.0.22 (Item 7 Step 8) — TOOL_METADATA drives per-tool wire events
from src.core.logging import logger
from src.llm.factory import build_chat_model


# Injected when the cheap-model pre-step flagged recency markers.
# Plain Chinese hint; deliberately short so it doesn't dominate the
# LLM's context window.
#
# v2.0.28.12 — strengthened from "建议先调用" (recommend) to
# "必须调用" (must call) when the cheap-model flagged the query
# as time-sensitive. Pre-v2.0.28.12 the LLM was free to ignore the
# hint and fall back to its training knowledge (which hallucinates
# cutoff dates) or to DIRECT_SYSTEM-style denial ("I don't have a
# time tool"). Post-v2.0.28.12 the LLM is told the time tool is
# the only sanctioned source of truth for "now" — the cheaper
# fallback (saying "我不知道" or quoting a training cutoff) is
# explicitly disallowed. The directive phrasing still leaves the
# LLM room to call additional tools alongside get_current_time
# (e.g. retrieve_docs + get_current_time for "今天的会议纪要
# 几点提到的"); the only behavior that's disallowed is answering
# time-sensitive questions WITHOUT calling get_current_time first.
_TIME_TOOL_HINT = (
    "提示:用户的问题涉及「今天 / 现在 / 最近」之类的相对时间,"
    "你必须先调用 get_current_time 工具获取准确的当前时间,再回答。"
    "不允许根据训练数据猜测时间,也不允许告诉用户「我不知道」或「无"
    "法获取实时信息」——只要 needs_current_time 标记为 True,这一步是必"
    "须的。"
)

# v2.0.4 — retrieve-first nudge. When the active thread has at
# least one uploaded document, we inject this SystemMessage at the
# FRONT of the messages list so the LLM prefers ``retrieve_docs``
# before ``web_search``. Without it, the LLM often jumps straight
# to web_search (e.g. on a "告诉我腾讯音乐 10 亿美元债券" query),
# missing the user's own upload entirely. The hint is a nudge, not
# a hard rule: the LLM is still free to call web_search if the
# retrieved docs don't address the question (the time-tool and
# complementary web search fallbacks still work).
#
# Concurrency: ``list_documents_cached`` is the existing in-process
# cached version of ``list_documents`` (see lancedb_store.py:689).
# The cache lives per-process so even cold-path queries complete in
# ~0 ms after the first call.
_RETRIEVE_FIRST_HINT = (
    "提示:当前对话已上传文档,本轮回答请优先调用 retrieve_docs "
    "检索本地文档。只有在本地文档确实无法回答(返回空 / 主题不符)"
    "时,再考虑调用 web_search 补充。"
)

# v2.0.4 — explicit frame for the LLM about what the messages
# represent. Without this, the model can confuse intermediate
# AIMessage(tool_calls=...) entries with prior user turns (the
# tool-call JSON looks conversational in shape). Same hint also
# gets injected in react_generate so both phases agree on the
# role mapping. The constant itself moved to
# ``src.agent.nodes._history_frame_hint`` (v2.0.22 Item 7 Step 2)
# so the two copies can't drift.

# Loop breaker — after the LLM has called the SAME tool N times in
# one turn, force a final answer by unbinding tools and injecting a
# stop hint. Empirically, ``web_search`` is the most common culprit:
# when Bing/DDG returns irrelevant hits (e.g. a US law database for
# "双鱼座"), the LLM tends to keep rephrasing the query instead of
# admitting "I couldn't find anything relevant". Without this brake,
# the graph eats the full ``recursion_limit=25`` budget on one
# dead-end tool and the user gets a generic error.
_MAX_SAME_TOOL_CALLS = 3

# v2.0.11 — also break on TOTAL tool calls (not just same-tool). The
# same-tool breaker misses the "alternating" pattern: LLM calls
# ``retrieve_docs`` → ``web_search`` → ``retrieve_docs`` →
# ``web_search`` → ... which racks up 6+ tool calls without any
# single tool hitting 3. The result: legitimate multi-tool queries
# still hit ``GraphRecursionError`` even though the LLM is making
# *progress*, just slowly. The total cap (6) covers multi-corpus
# retrieval (2-3 retrieve) + 1-2 web searches + 1 time check
# without false-positive breaks on normal 1-2 tool turns.
_MAX_TOTAL_TOOL_CALLS = 6
_LOOP_BREAK_HINT = (
    "提示:你已对同一个工具调用了多次且仍未获得理想结果。请基于目前已"
    "收集到的信息直接给出最终回答,不要再调用工具了。"
)


def _count_tool_calls_since_last_human(messages: list) -> dict[str, int]:
    """Count tool-call occurrences from the LAST HumanMessage onward.

    Counts AIMessage.tool_calls + the matching ToolMessage presence
    (so we don't double-count). Returns ``{tool_name: count}``. Used
    by the loop breaker to detect "the LLM keeps hammering the same
    tool" within a single turn.

    We start counting from the last HumanMessage to avoid stale
    tool-call counts leaking across turns (e.g. a 5-turn conversation
    where each turn uses ``retrieve_docs`` once would otherwise look
    like 5 calls of the same tool).
    """
    # Find the last HumanMessage — start counting from there.
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if getattr(m, "type", None) == "human":
            last_human_idx = i
            break
    relevant = messages[last_human_idx + 1:] if last_human_idx >= 0 else messages

    counts: dict[str, int] = {}
    for m in relevant:
        if not isinstance(m, AIMessage):
            continue
        for tc in (getattr(m, "tool_calls", None) or []):
            name = (tc.get("name") if isinstance(tc, dict) else None) or ""
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


async def react_agent(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """One step of the ReAct loop: think → tool-call OR final answer.

    v2.0.22 (Item 7 Step 6) — async-generator protocol: yields
    exactly one ``("__delta__", {"messages": [response]})``
    terminator.

    v2.0.22 (Item 7 Step 7) — node is now the wire event's single
    source of truth. Before ``__delta__`` we yield one
    ``("tool_call_start", {...})`` per AIMessage.tool_call (stamped
    with ``step`` for the runner's start/end pairing) and a
    ``("web_search", {"attempted": True})`` event when the LLM chose
    ``web_search`` (matches v1.1.8 wire contract — frontend uses the
    boolean as "yes, the LLM decided to search"). The FSM's old
    ``if current == "react_agent"`` post-step branch that read state
    after merge is gone.

    ``step`` is the 1-based step number the FSM passes in; we forward
    it on each ``tool_call_start`` so the frontend pairs it with the
    matching ``tool_call_end`` on the same step.

    Loop breaker
    ------------
    If the LLM has called the SAME tool >= ``_MAX_SAME_TOOL_CALLS``
    times in the current turn, we unbind tools and inject a stop
    hint, forcing the LLM to produce a final answer on the next pass
    (which routes to ``react_generate`` because no tool_calls are
    emitted). The user gets a coherent answer built from whatever
    docs/snippets we already gathered — vastly better than the
    "Recursion limit of 10 reached" error path.

    We also preserve the existing tool-call IDs across loop iterations
    by NOT minting new AIMessages here — every call to ``react_agent``
    APPENDS one AIMessage to ``state["messages"]``, so the runner's
    accumulator can match ``tool_call_start`` events to their
    ``tool_call_end`` events by ``tool_call_id`` (which LangChain
    stamps once and reuses across loop iterations).
    """
    msgs = list(state.get("messages") or [])

    # v2.0.4 — history frame hint. The react_agent ALSO sees the
    # full messages list (with tool_calls + tool results). A model
    # that mistakes tool calls for user turns here will call the
    # wrong tool (e.g. treat its own web_search call as "the user
    # asked me to search"). The hint makes the role mapping
    # explicit; react_generate also gets a copy of this hint when
    # synthesizing the final answer.
    msgs = [SystemMessage(content=_HISTORY_FRAME_HINT)] + msgs

    # Inject the time-tool hint at the FRONT of the messages list so
    # it acts like a system-level nudge. We don't insert before
    # HumanMessages (the LLM should still see the user query in its
    # original position); a SystemMessage at the front is the
    # canonical way to add per-turn guidance without disturbing the
    # chat history.
    if state.get("needs_current_time"):
        msgs = [SystemMessage(content=_TIME_TOOL_HINT)] + msgs

    # v2.0.4 — retrieve-first nudge. Only when the thread has at
    # least one uploaded document. We check the doc_registry /
    # lancedb cache (synchronous, ~0 ms after the first call) so
    # the nudge is only injected for the relevant turns — a
    # greeting or a current-events query on an empty thread sees
    # no extra prompt pollution.
    #
    # Loop-breaker decision still wins: if the LLM has been
    # hammering the same tool, we override below with the
    # loop-break hint regardless of the retrieve-first nudge.
    thread_id = state.get("thread_id")
    if thread_id:
        try:
            from src.storage.lancedb_store import list_documents_cached

            thread_docs = list_documents_cached(thread_id) or []
        except Exception as exc:
            # v2.0.22 (Item 7 Step 11) — pre-Step-11 this was a
            # bare ``except Exception`` with **no log at all**
            # (the silent-failure bug). Now logs uniformly with
            # the other recovery sites so a broken LanceDB cache
            # surfaces in production logs instead of the user
            # silently not getting the retrieve-first hint.
            log_node_failure("react_agent.list_documents_cached", exc)
            thread_docs = []
        if thread_docs:
            msgs = [SystemMessage(content=_RETRIEVE_FIRST_HINT)] + msgs

    # Loop-breaker decision: count tool calls in the current turn.
    # If any tool has been called too many times, OR the total
    # tool-call count exceeds the safe budget, force the LLM to
    # finalize by stripping the tool bindings. Two thresholds:
    #   - same-tool hammer (e.g. web_search × 3) → catches the
    #     "LLM keeps rephrasing an unprofitable query" pattern
    #   - total tool budget (e.g. 6 different mixed calls) → catches
    #     the "LLM alternates retrieve/web without converging" pattern
    #     that the same-tool breaker misses
    tool_counts = _count_tool_calls_since_last_human(state.get("messages") or [])
    total_tool_calls = sum(tool_counts.values())
    loop_breaker_active = any(
        c >= _MAX_SAME_TOOL_CALLS for c in tool_counts.values()
    ) or total_tool_calls >= _MAX_TOTAL_TOOL_CALLS

    base_model = build_chat_model(enable_thinking=True)
    if loop_breaker_active:
        # No tools → LLM can't call them. It must produce prose.
        # The hint explains WHY (so it doesn't try to be clever and
        # sneak in a tool call via direct JSON or some other path).
        msgs = [SystemMessage(content=_LOOP_BREAK_HINT)] + msgs
        model = base_model
    else:
        model = base_model.bind_tools(ALL_TOOLS)

    # v2.0.29.3 (Phase 3) — propagate the loop-breaker state flag
    # so react_generate can decide whether to insert the loop-break
    # synthesis hint. The FSM's ``merge()`` (``src/agent/fsm.py:163``)
    # persists arbitrary state keys verbatim (``out[k] = v``), so
    # this delta lands on the next node's ``state["loop_breaker_active"]``.
    # Reset to False on the non-break path: only the round that
    # actually tripped the breaker should fire the synthesis hint;
    # subsequent react_generate calls (e.g. retry-after-thought) read
    # a clean False. Default-on-write preserves the FSM contract
    # (state keys are server-internal, no wire event).
    loop_breaker_delta = (
        {"loop_breaker_active": True}
        if loop_breaker_active
        else {"loop_breaker_active": False}
    )

    response: AIMessage = await model.ainvoke(msgs)

    # v2.0.28.12 — enforce get_current_time call when needs_current_time
    # is True. Anthropic Claude stochasticity sometimes ignores the
    # "必须先调用" hint and produces a text-only AIMessage instead —
    # leaving react_generate with no ToolMessage to feed the synthesis
    # LLM, which then hallucinates a greeting-style or denial-style
    # response. Force a single retry with a much more directive
    # message; if the LLM still refuses, fall back to yielding a
    # synthetic get_current_time tool_call so the downstream synthesis
    # has the data it needs. The retry is bounded by ``step`` (max 3
    # retries across a single turn) to prevent infinite loops.
    _needs_time = bool(state.get("needs_current_time"))
    _has_time_call = any(
        (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) == "get_current_time"
        for tc in (getattr(response, "tool_calls", None) or [])
    )
    if (
        _needs_time
        and not _has_time_call
        and not loop_breaker_active
        and step < 3
    ):
        logger.debug(
            "react_agent: needs_current_time=True but LLM didn't call "
            "get_current_time; re-invoking with strict directive"
        )
        strict_msgs = [
            SystemMessage(content=(
                "严格提醒:用户的问题需要当前时间,你必须立刻调用 "
                "get_current_time 工具(不要写任何解释文字、不要先回答"
                "用户),这是唯一被允许的下一步动作。"
            )),
            *msgs,
        ]
        response = await model.ainvoke(strict_msgs)
        _has_time_call = any(
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) == "get_current_time"
            for tc in (getattr(response, "tool_calls", None) or [])
        )
        # If still no get_current_time call, synthesize one so the
        # downstream synthesis has data. The ToolNode will execute the
        # synthetic tool_call when it sees it on the next pass.
        if not _has_time_call:
            logger.debug(
                "react_agent: LLM still didn't call get_current_time on "
                "retry; synthesizing a fallback tool_call so the user "
                "still gets a time answer"
            )
            from src.agent.tools import get_current_time as _get_time_tool
            response = AIMessage(
                # Plain-Chinese trigger text mirrors what a non-stubborn
                # LLM would have written for its tool-call round. The
                # downstream synthesis LLM sees this via
                # ``_HISTORY_FRAME_HINT`` and correctly classifies it
                # as a planning step (the AIM has tool_calls), not as
                # a prior reply. Empty content would have read as a
                # blank assistant bubble and confused the synthesis
                # LLM into producing greeting / refusal language.
                content="好的,我先查询一下当前的准确时间。",
                tool_calls=[{
                    "id": f"synthetic-time-{step}",
                    "name": "get_current_time",
                    "args": {},
                }],
            )

    # v2.0.22 (Item 7 Step 7+8) — yield per-tool wire events BEFORE
    # ``__delta__``. The FSM forwards them to the runner which maps
    # them to the v1.1.8 WS frame contract. ``started_at`` is shared
    # across all tool_calls of this AIMessage so the frontend renders
    # a coherent "calling N tools at HH:MM:SS" pill.
    #
    # Step 8 (P0-B3 fix) — per-tool EXTRA events are metadata-driven
    # via :data:`src.agent.tools.TOOL_METADATA`. Pre-Step-8 the
    # ``web_search`` event was hardcoded here as
    # ``if tc_name == "web_search": yield ...`` — a leak of tool
    # names into the FSM that forced node edits to add a new
    # search-class tool. Now ``TOOL_METADATA[tc_name]["emits"]``
    # returns the extra event kinds for that tool. Adding a new
    # search tool (e.g. ``web_search_tier2``) is a one-line entry
    # in ``TOOL_METADATA``, no node code change needed.
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        started_at = _now_iso()
        for tc in tool_calls:
            if isinstance(tc, dict):
                tc_name = tc.get("name", "")
                tc_id = tc.get("id", "")
                tc_args = tc.get("args", {}) or {}
            else:
                tc_name = getattr(tc, "name", "")
                tc_id = getattr(tc, "id", "")
                tc_args = getattr(tc, "args", {}) or {}
            # v2.0.22 (Item 7 Step 9, P2-B3) — record in the FSM
            # pending tracker so the FSM (not the runner) can drain
            # unmatched starts on shutdown. ``setdefault`` is
            # defensive — run_fsm initializes the tracker before the
            # main loop, but tests that drive this node directly
            # (without run_fsm) still work. Mirrors the pre-Step-9
            # runner.py:347-352 bookkeeping exactly: same
            # ``step`` / ``name`` / ``started_at`` fields.
            state.setdefault("_pending_tool_calls", {})[tc_id] = {
                "step": step,
                "name": tc_name,
                "started_at": started_at,
            }
            yield (
                "tool_call_start",
                {
                    "tool_call_id": tc_id,
                    "tool_name": tc_name,
                    "tool_args": tc_args,
                    "started_at": started_at,
                    "step": step,
                },
            )
            # Lookup metadata for extra wire events. Default to
            # empty list when the tool isn't registered (LLM
            # shouldn't call unregistered tools but defensively
            # skip rather than KeyError). ``logger.debug`` keeps
            # the path observable without polluting INFO logs.
            meta = TOOL_METADATA.get(tc_name) or {}
            for evt_kind in meta.get("emits", []) or ():
                if evt_kind == "web_search":
                    # v1.1.8 wire contract — frontend uses the
                    # boolean as a "yes, the LLM decided to
                    # search" signal so the pill renders even if
                    # the call later fails.
                    yield ("web_search", {"attempted": True})
                else:
                    # Forward-compatibility: future ``emits``
                    # kinds without a per-kind builder below
                    # silently drop with a debug log instead of
                    # crashing the FSM. Tools that need a richer
                    # payload than ``{"attempted": True}`` should
                    # add a dedicated branch here.
                    logger.debug(
                        f"react_agent: tool {tc_name!r} metadata "
                        f"declares emits={evt_kind!r} but no "
                        f"per-event builder exists; dropping"
                    )

    yield ("__delta__", {
        "messages": [response],
        **loop_breaker_delta,
        # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
        # ``merge()`` since this delta doesn't carry it explicitly.
    })


__all__ = ["react_agent", "_count_tool_calls_since_last_human"]