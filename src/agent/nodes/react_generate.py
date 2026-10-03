"""react_generate node — synthesize the final answer from accumulated context.

By the time this node runs, ``react_agent`` has either:
  (a) decided the answer is ready (no tool calls), OR
  (b) gone through N tool rounds and the LLM is satisfied.

We:
1. Read every Document accumulated across the tool rounds — they live
   in ``state["messages"]`` as ``ToolMessage.content`` (lists of
   Documents). We extract those Documents back into a flat list and
   deposit them in ``state["documents"]`` for citation building.
2. Also merge in any ``state["web_documents"]`` from the tool results
   so the v1.1.13 complementary-path contract (local + web in the same
   prompt) is honored.
3. Build ``sources`` from the merged list (filtered to cited indices),
   ``source_kinds`` from the kinds present, and stamp them on the new
   AIMessage via ``additional_kwargs`` (history replay) and on the
   state delta (live wire).
4. Stream the answer through ``model.astream(msgs)`` exactly like the
   legacy ``generate.py`` did — preserving all v1.1.6c / v1.1.7 / v1.1.9
   invariants (synthesized-chunk dedup, thinking-block preservation,
   thinking-only synthesis fallback).

Why we re-extract Documents from ToolMessages
--------------------------------------------
LangChain's ``ToolNode`` serializes Document lists to JSON strings
inside ``ToolMessage.content``. We can't read the original Document
objects off the state without deserializing — so we do that
here. The tool itself returned a ``list[Document]``; ``ToolNode``
stuffed them through ``langchain_core``'s standard encoder; we
reverse it via ``json.loads`` + Document reconstruction.

Why we share ``_format_documents`` / ``_derive_source_kinds`` /
``_to_sources`` with the legacy ``generate.py``
----------------------------------------------------
The answer-prompt contract is unchanged: local docs go first (user's
own uploads are higher trust), web docs second. The fence wrapping
(P1-4 untrusted-doc boundary) and the v1.1.14 banner-kind logic
must apply identically. We import those helpers directly from the
existing module to avoid drift — if v1.1.18 changes the banner
semantics, react_generate follows automatically.

v2.0.19 (Phase 3) — async generator protocol. Yields ``("token", ...)``
and ``("reasoning", ...)`` events while the LLM streams the answer,
then ``("answer_complete", ...)`` carrying the raw answer / sources /
source_kinds / created_at / route_decision. The runner applies the
``filter_sources_to_cited`` + ``renumber_citations_and_sources``
transformations before emitting the final WS frame. Ends with the
mandatory ``("__delta__", delta_dict)`` terminator that the FSM
expects from every streaming node.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# v2.0.22 (Item 7 Step 1+2+3) — shared helpers. The ``_now_iso`` and
# ``_HISTORY_FRAME_HINT`` constants and the AIMessage content-block
# fallback used to be copy-pasted into this file; they now live in
# ``src.agent._time`` and ``src.agent.nodes._history_frame_hint`` /
# ``_content_blocks`` so future prompt / format changes ship in one place.
from src.agent._thinking_split import _strip_tool_blocks, split_text_and_thinking
from src.agent._time import now_iso as _now_iso
from src.agent.legacy_helpers.doc_formatting import (
    _derive_source_kinds,
    _format_documents,
    _to_sources,
)
from src.agent.nodes._content_blocks import preserve_content_blocks
from src.agent.nodes._history_frame_hint import _HISTORY_FRAME_HINT
# v2.0.29.3 (Phase 3) — defensive-override registry. The directive
# builder, fingerprint extractor, and fallback builder for each
# tool that returns scalar data the synthesis LLM MUST incorporate
# (get_current_time today) now live in a 3-table registry keyed by
# tool name. Adding a new tool is one line per registry; the
# hardcoded helpers below are gone. See
# ``src/agent/nodes/_defensive_override_registry.py``.
from src.agent.nodes._defensive_override_registry import (
    DIRECTIVE_BUILDERS,
    apply_defensive_overrides,
)
# v2.0.22 (Item 7 Step 11) — error-recovery helpers (P2-B4 关掉).
# ``llm_call_failure`` replaces the inline apology + token-yield
# block that used to live below (``fsm.react_generate_direct``
# has the matching call site).
from src.agent.nodes._recovery import llm_call_failure
from src.agent.state import AgentState
# v2.0.22 (Item 7 Step 10) — closed vocabulary for the
# ``_intent_to_route_decision`` return type (was bare ``str``).
from src.agent.types import RouteDecision
from src.core.logging import logger
from src.llm.factory import build_chat_model
from src.llm.prompts import DIRECT_SYSTEM, GENERATE_SYSTEM, REFUSAL_TEMPLATES


# v2.0.3 — how many characters of a tool's result content we record in
# the persisted AIMessage.additional_kwargs["tool_calls"] entry. The
# frontend's ToolCallCard renders the first 800 chars (see
# ToolCallCard.tsx:78-86), and we want the persisted record to match
# what the user saw live — so we cap at 800 here. Anything beyond is
# already on the ToolMessage in the checkpointer; the frontend never
# reads it.
_TOOL_RESULT_PREVIEW_MAX = 800


def _build_tool_call_log(messages: list) -> list[dict]:
    """Walk ``messages`` and pair every AIMessage ``tool_calls`` with
    its matching ``ToolMessage`` so we can stamp a per-call log onto
    the final AIMessage.

    v2.0.3 — the runner emits ``tool_call_start`` / ``tool_call_end``
    live events but never persists the log onto the AIMessage itself,
    so a page refresh after a ReAct turn loses them. This helper
    reconstructs the equivalent log by reading the messages list
    directly (which IS persisted by LangGraph's checkpointer).

    Each entry is shaped like the live wire event (so the frontend's
    ToolCallCard renders identically live + on reload):

      {
        "id":           "<tool_call_id>",
        "name":         "<tool name>",
        "args":         {...}              # from AIMessage.tool_calls[i]['args']
        "result":       "<truncated str>", # from matching ToolMessage.content
        "ok":           True,              # we set True; failed tools
                                           # are caught by ToolNode and
                                           # returned as a ToolMessage
                                           # with error content — the
                                           # frontend already renders
                                           # those via its own flow.
        "step":         <int>,             # the AIMessage's position
                                           # in the messages list
        "started_at":   "<iso>",           # the AIMessage's server stamp
        "ended_at":     "<iso>",           # now-ish
        "elapsed_ms":   <int>,             # ended_at - started_at
      }

    Defensive: malformed AIMessages without ``tool_calls``, AIMessages
    whose matching ToolMessage has been pruned by the checkpointer, and
    ToolMessages with no preceding AIMessage are all silently skipped —
    we never want a stale log entry to crash ``react_generate``.
    """
    log: list[dict] = []
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    for i, m in enumerate(messages or []):
        if not isinstance(m, AIMessage):
            continue
        tool_calls = getattr(m, "tool_calls", None) or []
        if not tool_calls:
            continue
        # The AIMessage's server-side created_at (stamped in
        # ``_initial_state`` / per-AIMessage write paths). Fall back
        # to "now" if the field is missing so the elapsed_ms is still
        # computable.
        started_at = (getattr(m, "additional_kwargs", None) or {}).get(
            "created_at"
        )
        try:
            started_dt = (
                datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                if started_at
                else now
            )
        except Exception:
            started_dt = now

        # Build a tool_call_id -> ToolMessage map for this position
        # so we can look up the matching result. ToolMessages follow
        # the AIMessage that produced them in the messages list.
        following_results: dict[str, ToolMessage] = {}
        for j in range(i + 1, len(messages)):
            tm = messages[j]
            if isinstance(tm, ToolMessage):
                following_results[tm.tool_call_id] = tm
            else:
                # Stop scanning after we hit a non-ToolMessage
                # (e.g. the next AIMessage); results from the
                # previous round are irrelevant.
                break

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = str(tc.get("id") or "")
            tc_name = str(tc.get("name") or "")
            tc_args = tc.get("args") or {}
            if not isinstance(tc_args, dict):
                tc_args = {}
            tm = following_results.get(tc_id)
            result_str = ""
            ok = True
            if tm is not None:
                raw_content = getattr(tm, "content", "")
                if isinstance(raw_content, str):
                    result_str = raw_content
                elif isinstance(raw_content, list):
                    # Anthropic-style content blocks — concatenate text
                    result_str = "".join(
                        str(b.get("text", ""))
                        for b in raw_content
                        if isinstance(b, dict)
                    )
                else:
                    result_str = str(raw_content)
                # ToolNode returns ok=False via tool_call_end when
                # the tool raised (Pydantic validation, etc.). The
                # ToolMessage itself doesn't carry that signal —
                # we approximate via the standard "Tool execution
                # failed: " prefix that ToolNode produces on error.
                if "Tool execution failed" in result_str:
                    ok = False
            ended_dt = datetime.now(timezone.utc)
            elapsed_ms = max(
                0,
                int((ended_dt - started_dt).total_seconds() * 1000),
            )
            log.append(
                {
                    "id": tc_id,
                    "name": tc_name,
                    "args": tc_args,
                    "result": (
                        result_str[:_TOOL_RESULT_PREVIEW_MAX]
                        + ("…" if len(result_str) > _TOOL_RESULT_PREVIEW_MAX else "")
                    ),
                    "ok": ok,
                    "step": i,
                    "started_at": started_at,
                    "ended_at": ended_dt.isoformat(),
                    "elapsed_ms": elapsed_ms,
                }
            )
    return log


# v2.0.22 (Item 7 Step 1) — ``_now_iso`` now lives in
# ``src.agent._time`` (imported at the top of this file as
# ``_now_iso``). Removing the local copy keeps the two call sites
# in this file working unchanged.


def _extract_docs_from_messages(messages: list) -> List[Document]:
    """Pull every Document out of every ToolMessage in ``messages``.

    ToolMessages carry the tool's return value in ``content``. For
    our two retrieval tools (retrieve_docs / web_search) v2.0.2
    emits a JSON-serialized list of Documents like
    ``[{"page_content": ..., "metadata": {...}}, ...]``. For
    ``get_current_time`` it's a plain string (we ignore it — time
    isn't a Document).

    Defensive: malformed JSON, non-list payloads, and entries that
    aren't Document-shaped are silently dropped. We never want a
    bad ToolMessage to crash the final-answer generation.

    v2.0.18 — dead-code cleanup
    ---------------------------
    The previous version had a "Branch 2" fallback for legacy
    v2.0 / v2.0.1 Python-repr ToolMessage content. That branch
    has been dead since v2.0.2 (2026-09-06) — every ToolMessage
    now stores a JSON string, and SqliteSaver has had ~10 days
    to trim pre-v2.0.2 checkpoints. Removed 42 LOC; users on a
    v2.0.2-pre thread who try to resume see no usable docs
    (graceful degradation — same as any malformed payload today).
    """
    docs: List[Document] = []
    for m in messages or []:
        if not isinstance(m, ToolMessage):
            continue
        content = getattr(m, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue

        # Branch 1: v2.0.2+ JSON format
        parsed_json = None
        try:
            parsed_json = json.loads(content)
        except Exception:
            parsed_json = None
        if isinstance(parsed_json, list):
            for entry in parsed_json:
                if not isinstance(entry, dict):
                    continue
                page_content = entry.get("page_content") or ""
                meta = entry.get("metadata") or {}
                if not isinstance(meta, dict):
                    meta = {}
                docs.append(Document(page_content=page_content, metadata=meta))
            continue

        # Anything else: malformed JSON / non-list payload /
        # non-Document plain string. Silently dropped — same
        # graceful-degradation behavior as the old Branch 2 minus
        # the brittle regex reconstruction.
    return docs


# v2.0.9 — tools whose return value is a plain string (NOT a
# Document). For these, the ToolMessage content IS the data the LLM
# needs to synthesize the final answer — there is no Documents fence
# to fall back to. We MUST preserve these ToolMessages when building
# the final prompt, otherwise the LLM in ``react_generate`` is asked
# to re-synthesize the answer with NO source material and either
# hallucinates or truthfully apologizes ("I can't determine the
# current time" even though the tool just returned it).
#
# Concretely: ``get_current_time`` returns a string like
# ``"2026-09-11T14:23:11+08:00 (Asia/Shanghai)\n2026年9月11日 周五 14:23"``.
# The pre-v2.0.9 sanitizer dropped the ToolMessage because the
# comment said "Tool result — already represented in DOCUMENTS:".
# That's true for retrieve_docs / web_search (Documents go into
# state.documents → DOCUMENTS fence), NOT true for get_current_time.
#
# To extend safely: add a tool name here when its result is
# a non-Document scalar that the final-answer LLM must see.
_STRING_RETURNING_TOOLS = frozenset({"get_current_time"})


# v2.0.28.16 — synthesis LLM tool-result utilization helpers.
# Background: user reported (2026-09-23 23:44) that asking
# "现在是几点" in a multi-turn thread (after warmup turns ending with
# "请问你想聊些什么?") sometimes (40% per 5-run stochastic probe) gave
# chat replies ("嗯?还想问点什么吗?", "好的,那就早点休息吧,晚安",
# or repeated previous answer) instead of the time data the
# get_current_time TM had returned. The TM is in the sanitized
# history (v2.0.10.1 + v2.0.9 preserve), and the _HISTORY_FRAME_HINT
# explicitly says "use TM data to answer the latest HM" — but the
# LLM's extended-thinking sometimes biases it toward continuing
# the prior turn's conversational mode rather than answering
# factually.
#
# v2.0.29.3 (Phase 3) — the two helpers used to live here as
# hardcoded ``_build_get_current_time_directive`` /
# ``_extract_time_marker_from_history`` / ``_answer_has_marker``.
# They now live in
# ``src/agent/nodes/_defensive_override_registry.py`` as a 3-table
# registry keyed by tool name (``DIRECTIVE_BUILDERS`` /
# ``FINGERPRINT_EXTRACTORS`` / ``FALLBACK_BUILDERS``); this node
# iterates the registry instead of touching the helper directly.
# Adding a new tool that returns scalar data the synthesis LLM must
# use is one line per registry — zero call-site change in
# ``react_generate``.


# v2.0.29.3 (Phase 3) — loop-break synthesis hint. When the
# ReAct loop breaker fires (same tool × ≥3 OR total ≥6 in one
# turn), ``react_agent`` strips the LLM's tool bindings and forces
# it to produce prose. Pre-fix the synthesis LLM still sometimes
# emitted lines like "我需要继续搜索更多资料才能给出答案..." — which
# reads as a refusal to answer, not as "I've been told to stop
# researching, answer from what I have". The synthesis LLM has no
# signal of WHY tools were unbound; without an explanation it falls
# back to its conversational-mode bias. This hint makes the reason
# explicit and tells the LLM to either (a) answer from what's
# already gathered or (b) walk the refusal template if the data
# doesn't support an answer. v2.0.28.17 consecutiveness: INSERT at
# index 2 (same convention as the time directive + refusal
# directive) so all SystemMessages stay consecutive.
_LOOP_BREAK_SYNTHESIS_HINT = (
    "[CRITICAL — 研究已达上限] 上一轮 ReAct 已触发 loop breaker "
    "(同一工具多次调用或总工具调用超出预算)。请你**仅基于已检索到"
    "的资料**回答,**不要再暗示可以继续 retrieve 或 web_search**。"
    "如果资料不足,请按 REFUSAL_TEMPLATES 拒绝模板回应。"
)


def _dedupe_documents(docs: list[Document]) -> list[Document]:
    """Dedupe a list of Documents by the 3-tuple
    ``(doc_id, chunk_id, page_content[:200])``.

    v2.0.29.4 (Phase 4 PR-1) — promoted from an inline block inside
    ``react_generate`` so the dedupe contract is independently
    unit-testable. The 3-tuple key is necessary because:

      * ``(doc_id, chunk_id)`` alone is the OLD (insufficient) key.
        A re-uploaded doc with same ``doc_id`` + ``chunk_id`` but
        different content silently collapses to the stale version
        — the LLM reads the wrong chunk for "current" queries.
      * ``page_content[:200]`` as the third element disambiguates
        without false-positive collisions deeper in the chunk
        (chunks with the same prefix are by definition the same
        chunk from a retrieval perspective).

    Order-preserving: the FIRST occurrence of any key wins (so the
    ``tool_docs`` side of the union beats ``state.documents`` when
    they collide — important because tool results are fresher than
    any state hydration).
    """
    seen: set[tuple[str, str, str]] = set()
    merged: list[Document] = []
    for d in docs:
        meta = d.metadata or {}
        key = (
            str(meta.get("doc_id") or ""),
            str(meta.get("chunk_id") or ""),
            (d.page_content or "")[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged


def _sanitize_history_for_generate(messages: list) -> list:
    """Strip tool-calls-only AIMessages and ToolMessages from history.

    v2.0.4 — closes the "LLM hallucinates prior user turns" bug.

    Before this filter, ``react_generate`` built ``msgs = [System, *history]``
    where ``history`` was the entire ``state["messages"]``. That
    history contains the agent's own execution trace:

      [HumanMessage, AIMessage(tool_calls=[...]), ToolMessage,
       AIMessage(tool_calls=[...]), ToolMessage, ...]

    The LLM tended to read the intermediate ``AIMessage``s (with
    their ``tool_calls`` field, which looks like "the AI said
    something with structured args") as "previous user turns". On
    a fresh first-turn question, the LLM would then respond with
    things like "你刚才说 X" or "you sent me a thumbs up" — there
    was no such previous turn, the LLM was reading its own
    tool-call trail as conversation.

    v2.0.9 — preserve ToolMessages for ``_STRING_RETURNING_TOOLS``.

    Pre-v2.0.9 the sanitizer dropped EVERY ToolMessage with the
    comment "Tool result — already represented in DOCUMENTS:". That
    is correct for ``retrieve_docs`` / ``web_search`` (Documents go
    into state.documents → DOCUMENTS fence) but WRONG for
    ``get_current_time`` (the result is a plain string the LLM
    needs to see). Without this fix, the LLM in ``react_generate``
    sees no source material for a time query and hallucinates
    "抱歉,我无法确定现在..." (apologizes for not having a time tool
    — even though it just got the time). Pin the exception set
    above; add new entries when a non-Document-returning tool ships.

    v2.0.10.1 — preserve the parent AIMessage(tool_calls=[...])
    of any whitelisted ToolMessage. v2.0.9 stage 2 preserved the
    ToolMessage for ``get_current_time`` but the AIMessage that
    PROMPTED the tool call is a tool-calls-only AIMessage, which
    the filter below drops at the "no visible text + has tool_calls"
    branch. Result: the surviving ``out`` list contains a
    ToolMessage whose parent AIMessage(tool_calls=[...]) is gone,
    and Anthropic rejects with ``400 invalid_request_error: tool
    call result does not follow tool call (2013)``. The fix is a
    two-pass scan: pass 1 collects the set of ``tool_call_id``s
    whose ToolMessages we keep, pass 2 walks the messages and
    KEEPs any AIMessage that has at least one tool_call whose id is
    in that set — even if the AIMessage has no visible text.

    What we keep vs what we drop:

      * **HumanMessage** → keep (real user input)
      * **AIMessage with non-empty text content** → keep (real
        prior assistant answer)
      * **AIMessage with only ``tool_calls`` (no visible text)** →
        drop UNLESS one of its ``tool_calls.id`` matches a
        preserved ToolMessage's ``tool_call_id`` (then KEEP — it
        IS the parent the preserved ToolMessage needs in order to
        be a valid conversation turn; v2.0.9 dropped this branch,
        breaking Anthropic's tool_use ↔ tool_result pairing)
      * **ToolMessage** → drop, UNLESS its ``name`` is in
        ``_STRING_RETURNING_TOOLS`` (then KEEP — the string IS
        the data the LLM needs)

    Returns the filtered list. Order is preserved. Empty list → []
    (the caller's ``[System] + []`` is then ``[System]`` only,
    which is the desired behavior on a brand-new thread).
    """
    if not messages:
        return []

    # v2.0.10.1 — Pass 1: collect the set of tool_call_ids whose
    # ToolMessages we will keep. The set is what pass 2 uses to
    # decide whether a tool-calls-only AIMessage is a parent we
    # must also keep. Doing this as a separate pass keeps the logic
    # linear and avoids the "should we keep this AIMessage before
    # we've seen the ToolMessage that follows it?" ordering hazard
    # — without this scan the only safe answer was to keep ALL
    # tool-calls-only AIMessages (which reintroduces the v2.0.4
    # hallucinate-prior-user-turns bug). The whitelist is small
    # (only ``get_current_time`` today) so the cross-reference is
    # cheap; future tools that return plain strings join by adding
    # their name to ``_STRING_RETURNING_TOOLS`` above.
    preserved_tool_call_ids: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        tool_name = (getattr(m, "name", "") or "")
        if tool_name in _STRING_RETURNING_TOOLS:
            tcid = (getattr(m, "tool_call_id", "") or "")
            if tcid:
                preserved_tool_call_ids.add(tcid)

    out: list = []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_name = (getattr(m, "name", "") or "")
            tcid = (getattr(m, "tool_call_id", "") or "")
            if tool_name in _STRING_RETURNING_TOOLS and tcid in preserved_tool_call_ids:
                # The result is a plain string the LLM needs. Keep
                # the ToolMessage as-is. Its parent AIMessage(tool_calls=[...])
                # will be preserved by the AIMessage branch below —
                # both ends of the tool_use ↔ tool_result pair are
                # kept together so Anthropic's messages API accepts
                # the conversation (was 400/2013 before this fix).
                out.append(m)
            # else: drop — the result is in state.documents → DOCUMENTS fence
            continue
        if isinstance(m, AIMessage):
            # Tool-calls-only AIMessages (the ReAct planning steps)
            # are NOT user-facing turns — drop them by default. But
            # if any of this AIMessage's tool_call ids is in the
            # preserved set, we MUST keep it: the matching ToolMessage
            # is staying in ``out``, and Anthropic rejects messages
            # where a ToolMessage isn't directly preceded by its
            # AIMessage(tool_calls=[...]) parent. So the "drop tool-
            # calls-only AIMessage" rule has an exception: keep
            # when its tool_calls feed a preserved ToolMessage.
            content = getattr(m, "content", "")
            if isinstance(content, str):
                has_text = bool(content.strip())
            elif isinstance(content, list):
                has_text = any(
                    isinstance(b, dict)
                    and b.get("type") == "text"
                    and (b.get("text") or "").strip()
                    for b in content
                )
            else:
                has_text = False
            tool_calls = getattr(m, "tool_calls", None) or []
            tool_call_ids = {
                tc.get("id")
                for tc in tool_calls
                if isinstance(tc, dict) and tc.get("id")
            }
            is_parent_of_preserved = bool(tool_call_ids & preserved_tool_call_ids)
            if not has_text and tool_calls and not is_parent_of_preserved:
                # Tool-calls-only AIMessage whose tool_calls don't
                # feed a preserved ToolMessage. Drop — pure ReAct
                # planner noise that confuses the LLM (v2.0.4 bug).
                continue
            # Keep: real prior answer (visible text), OR parent of
            # a preserved ToolMessage (tool-calls-only but needed
            # for tool_use ↔ tool_result pairing), OR both.
            out.append(m)
            continue
        if isinstance(m, HumanMessage):
            out.append(m)
            continue
        # SystemMessage / function messages / anything else: keep
        # verbatim. They're part of the existing prompt scaffolding
        # and we don't want to silently drop them.
        out.append(m)

    # v2.0.28.11 — drop the trailing text-only AIMessage. The
    # ``after_react_agent`` predicate (src/agent/fsm.py:247-260)
    # guarantees the last AIMessage at react_generate entry has no
    # ``tool_calls``. That AIMessage is react_agent's final-round
    # text-only "draft" — it has been streamed live, and v2.0.28.10
    # will overwrite it via ``replace_last_message`` when
    # react_generate yields its synthesis AIMessage. Without this
    # strip, the synthesis LLM reads the draft via
    # ``_HISTORY_FRAME_HINT`` ("[AIMessage] 是你之前的回答") and
    # misinterprets the turn as a request to summarize its own
    # previous reply — producing meta-text like "上一轮的回复简要
    # 总结:..." instead of a direct answer. Pin the contract via
    # the regression tests in tests/test_v2_0_28_11_bugfixes.py
    # (TestV2_0_28_11DraftStrippedFromSynthesis).
    #
    # Safety / scope:
    # - ``isinstance(out[-1], AIMessage)``: if trailing is a
    #   HumanMessage (multi-turn where user just spoke) we don't
    #   touch it.
    # - ``not getattr(out[-1], "tool_calls", None)``: if trailing is
    #   a tool-calls AIMessage (defensive — ``after_react_agent``
    #   would never route here today, but pin for future FSM
    #   changes) we don't touch it — tool_use ↔ tool_result pairing.
    # - ``isinstance(content, str)``: ONLY strip when the AIMessage
    #   content is a string (the "draft" form). Thinking-only
    # AIMessages have ``content`` as a list of thinking blocks per
    #   v1.1.7/v1.1.9 — they are NOT drafts, they are intermediate
    #   reasoning that the next ``ainvoke`` must carry forward via
    #   Anthropic's extended thinking. Pin: tests/test_v2_0_4_bugfixes.py
    #   ``TestP04SanitizeHistory::test_keeps_thinking_only_ai_messages``.
    if out and isinstance(out[-1], AIMessage):
        if not getattr(out[-1], "tool_calls", None):
            content = getattr(out[-1], "content", None)
            if isinstance(content, str):
                out = out[:-1]

    return out


# v2.0.4 — explicit frame for the LLM about what the messages
# represent. The history sanitizer above removes the tool-call
# trace from the conversation, but the LLM can still confuse
# adjacent turns ("你刚才说 X") if the message types aren't labeled.
# This hint makes the role-mapping explicit so a model that
# confuses Human vs AI can't claim "the user said X" out of thin
# air. The constant itself lives in
# ``src.agent.nodes._history_frame_hint`` (v2.0.22 Item 7 Step 2)
# so ``react_agent.py`` and this file stay in sync automatically.


def _intent_to_route_decision(intent: str | None) -> RouteDecision:
    """Map v2.0 intent to the v1.1.3-vintage ``route_decision``.

    The legacy ``generate._format_documents`` /
    ``generate._to_sources`` branch on ``route_decision == "direct"``
    to suppress citations. v2.0 keeps that branch — but we no
    longer have a ``route_decision`` field on state. Translate
    ``intent`` here so the same gating works:

      - "greeting" / "simple_fact" / "direct" → "direct" (no docs,
        no citations)
      - "summary" / "qa_complex" / None → "retrieve" (docs in
        prompt, citations active)

    v2.0.5 — ``simple_fact`` joins the direct set so the prompt
    falls back to ``DIRECT_SYSTEM`` (no documents slot, no [n]
    requirement) and the citation filter at the runner boundary
    strips any spurious sources that survived.

    Why "summary" maps to "retrieve" not "direct"
    --------------------------------------------
    The summary fast-path (``summary_path``) dumps every chunk of the
    thread's documents into ``state["documents"]`` *before* this node
    runs. We want the LLM to actually read those chunks — feeding the
    main model ``DIRECT_SYSTEM`` (which has no ``{documents}`` slot)
    would tell it "no docs were attached", and the LLM responds with
    "I can't see any documents". But ``_derive_source_kinds`` would
    still report ``["local"]`` from ``merged``, so the banner would
    say "本回答参考了文档内容" while the body denies it — a confusing
    contradiction. Fix: summary uses the same prompt as qa_complex,
    so the LLM sees the chunks it just summarized.
    """
    if intent in ("greeting", "simple_fact", "direct"):
        return "direct"
    return "retrieve"


async def react_generate(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """Synthesize the final answer from the ReAct round's tool results.

    Async generator protocol (Phase 3): yields ``("token", {...})`` /
    ``("reasoning", {...})`` while the LLM streams, then
    ``("answer_complete", {...})`` carrying the raw answer +
    sources + source_kinds + created_at + route_decision (the
    runner applies ``filter_sources_to_cited`` and
    ``renumber_citations_and_sources`` before emitting the wire
    frame). Ends with the mandatory ``("__delta__", delta_dict)``
    terminator the FSM merges into state.

    Branches on intent:
      - "greeting" / "simple_fact" → ``DIRECT_SYSTEM`` (no docs, no [n]
        citations). v2.0.5 — simple_fact joins greeting on the direct
        branch so definitional lookups get a clean knowledge-only
        answer without document-injection overhead.
      - "summary" / "qa_complex" / None → ``GENERATE_SYSTEM`` with
        accumulated documents formatted into a single fenced block.
    """
    # Pull docs out of ToolMessages in the chat history. These
    # represent the LLM's chosen retrieval round — possibly multiple
    # tool calls in the same turn (LLM is free to call multiple tools
    # in parallel before deciding).
    tool_docs = _extract_docs_from_messages(state.get("messages") or [])

    # Union with anything already on the state (some paths populate
    # ``state["documents"]`` directly via early nodes). Dedupe via the
    # ``_dedupe_documents`` helper — the dedupe key is now
    # ``(doc_id, chunk_id, page_content[:200])`` (Phase 4 PR-1,
    # pre-PR-1 the key was 2-tuple and a re-uploaded doc with
    # different content silently collapsed to the stale version).
    merged = _dedupe_documents(tool_docs + list(state.get("documents") or []))

    intent = state.get("intent")
    route_decision = _intent_to_route_decision(intent)

    # v2.0.29.2 (Phase 2) — observability hook. Which prompt
    # template did react_generate walk? ``direct`` for simple
    # conversational turns (no retrieval), ``refusal_empty`` /
    # ``refusal_empty_bulk`` for the Phase 1 refusal contract
    # (empty retrieval status), ``generate`` for the normal
    # synthesize-from-context path. Dashboards use this to monitor
    # refusal-template usage growth over time.
    from src.agent.metrics import SYNTHESIS_PROMPT_TEMPLATE

    if route_decision == "direct":
        sys_prompt = DIRECT_SYSTEM
        SYNTHESIS_PROMPT_TEMPLATE.inc(template="direct")
    else:
        # v1.1.13 contract: merge local + web. With the ReAct rewrite,
        # every Document the LLM saw is in ``merged`` (the runner's
        # tool accumulator writes web docs with ``source_kind="web"``
        # and local docs with ``source_kind="local"``). Same merge
        # ordering as before: local first, web second.
        local_docs = [d for d in merged if (d.metadata or {}).get("source_kind", "local") == "local"]
        web_docs = [d for d in merged if (d.metadata or {}).get("source_kind") == "web"]
        if local_docs and web_docs:
            docs = list(local_docs) + list(web_docs)
        elif web_docs:
            docs = list(web_docs)
        else:
            docs = list(local_docs)

        # v1.1.11 web-search status hint. Only when the LLM actually
        # tried web_search and got something back; otherwise the
        # prompt stays clean.
        #
        # v2.0.4 — rewrote the "have been deemed relevant" branch.
        # That string was a LIE: there is NO relevance grader in the
        # ReAct path (``search_web.py`` is dead code, and the
        # ``web_search`` tool has no grader of its own). Telling the
        # LLM "the results below have been deemed relevant" actively
        # encouraged fabrication when Bing/DDG returned irrelevant
        # hits (e.g. "腾讯音乐 10 亿美元债券" → unrelated US court
        # cases). The new contract:
        #
        #   * 0 docs  → explicit refusal ("not found")
        #   * docs but ALL are below a minimum content-length floor
        #     (likely auto-generated snippet / anti-bot body) →
        #     same explicit refusal
        #   * docs with non-trivial content → honest "results
        #     present, but you MUST verify they answer the user's
        #     question before citing them"
        #
        # The minimum-content floor (100 chars on page_content) is
        # deliberately conservative — anything shorter than that is
        # almost certainly a search-engine stub or a fetch failure
        # body, not a usable source.
        web_search_attempted = bool(
            any(
                isinstance(m, ToolMessage)
                and (getattr(m, "name", "") or "") == "web_search"
                for m in state.get("messages") or []
            )
        )
        web_docs_usable = (
            bool(web_docs)
            and any(
                len((d.page_content or "").strip()) >= 100
                for d in web_docs
            )
        )
        if web_search_attempted and not web_docs_usable:
            ws_status = (
                "A web search was attempted for this question but "
                "returned no usable results. Tell the user that you "
                "searched the web but couldn't find anything matching "
                "their query — do NOT invent excuses about lacking "
                "internet access or knowledge cutoffs, and do NOT "
                "fabricate plausible-sounding headlines / numbers / "
                "URLs that weren't in the search results."
            )
        elif web_docs_usable:
            ws_status = (
                "A web search was attempted and returned the snippets "
                "below. Treat each one as a search-engine snapshot — "
                "the page title and snippet are NOT independently "
                "verified. Verify the snippet actually addresses the "
                "user's question before citing it; if a snippet is "
                "off-topic (different entity, different time period, "
                "different language) do NOT cite it as evidence."
            )
        else:
            ws_status = ""

        formatted = _format_documents(docs)
        sys_prompt = GENERATE_SYSTEM.format(
            documents=formatted,
            web_search_status=ws_status,
        )

    history = state.get("messages") or []
    # v2.0.4 — sanitize history before passing it to the LLM so the
    # tool-call trace doesn't get re-read as user conversation.
    sanitized_history = _sanitize_history_for_generate(history)
    msgs = [
        SystemMessage(content=sys_prompt),
        SystemMessage(content=_HISTORY_FRAME_HINT),
        *sanitized_history,
    ]
    # v2.0.28.16 — post-script directive when get_current_time was just
    # called. Pre-fix the synthesis LLM was given the TM data in the
    # sanitized history but ~40% of runs emitted chat replies instead
    # of using the time data (probe debug-now-bug3-probe.js: 2/5 runs
    # gave "嗯?还想问点什么吗?" / "好问题!我是 源 RAG..." / "早点休息
    # 吧,晚安" instead of stating the time). The TM is in the
    # conversation but Anthropic extended-thinking sometimes biases
    # the LLM toward continuation of prior turn's conversational
    # mode (especially when the previous turn ended with a chat
    # wrap-up like "请问你想聊些什么?"). This post-script is a STRONG
    # directive right before the LLM responds — empirically biases
    # the LLM toward using the tool data for factual answers. If
    # no get_current_time TM is in history, ``time_directive`` is
    # None and we don't append anything (no noise on unrelated
    # prompts). Belt-and-suspenders: even when this directive fires,
    # the defensive fingerprint fallback below still catches runs
    # where the LLM ignored both the directive and the TM data.
    #
    # v2.0.28.17 — CRITICAL: Anthropic's API requires ALL system
    # messages to be CONSECUTIVE at the START of the messages list
    # (raises ``ValueError: Received multiple non-consecutive system
    # messages`` otherwise — confirmed via backend log
    # ``langchain_anthropic/chat_models.py:527``). Pre-fix this
    # directive was APPENDED at the end (after sanitized history),
    # violating consecutiveness whenever history contained any
    # HumanMessage/AIMessage/ToolMessage. The v2.0.28.16 defensive
    # fingerprint fallback MASKED the LLM failure by overriding
    # with the constructed fallback — user saw correct answer but
    # the LLM itself never ran (caught by ``llm_call_failure`` →
    # apology string → defensive override). Fix: INSERT the
    # directive at index 2 (BEFORE sanitized history), keeping
    # msgs[0..2] all SystemMessage. System messages stay
    # consecutive; sanitized history still follows.
    #
    # v2.0.29.1 (Phase 1) — same consecutiveness rule applies to the
    # refusal directive. When ``retrieval_status`` is one of the
    # refusal values, we INSERT the matching REFUSAL_TEMPLATES entry
    # at index 2 — BEFORE the time directive (which would land at
    # index 3 if it also fires). This ordering is intentional: the
    # refusal directive is a STRONG contract (the LLM MUST walk the
    # template); the time directive is a SOFT nudge (the LLM should
    # use the TM data but the refusal override catches misses). When
    # refusal fires, the LLM is forbidden from synthesizing any
    # factual claim, so the time directive is moot anyway.
    #
    # v2.0.29.3 (Phase 3) — two changes:
    #   (a) the time directive is no longer hardcoded here; it comes
    #       from ``DIRECTIVE_BUILDERS["get_current_time"](messages)``
    #       in the defensive-override registry. The iteration
    #       pattern is identical (insert at index 2, only on non-
    #       direct route, only if the builder returned a string), so
    #       the v2.0.28.16 / v2.0.28.17 invariants are preserved.
    #       Adding a new tool that returns scalar data the LLM must
    #       use is a one-line registry entry; zero call-site change.
    #   (b) NEW: loop-break synthesis hint. ``react_agent`` sets
    #       ``state["loop_breaker_active"] = True`` in the delta
    #       when it un-binds tools (same tool × ≥3 OR total ≥6 in
    #       one turn). When that flag is set, we INSERT a synthesis
    #       hint telling the LLM that research has been capped and
    #       it should answer from what's gathered (or walk the
    #       refusal template if the data is insufficient). Same
    #       ``insert(2, ...)`` rule keeps msgs[0..2] consecutive.
    #       Ordering: refusal → loop-break → tool-directive. Refusal
    #       is the strongest contract (no factual claims allowed);
    #       loop-break tells the LLM it can't continue research;
    #       tool-directive is the softest nudge (use the data you
    #       have). They compose cleanly when all three fire.
    if route_decision != "direct":
        retrieval_status = state.get("retrieval_status")
        refusal_directive = REFUSAL_TEMPLATES.get(retrieval_status or "")
        if refusal_directive:
            msgs.insert(2, SystemMessage(content=refusal_directive))
            # v2.0.29.2 (Phase 2) — observability hook. Record which
            # refusal template was walked so dashboards can monitor
            # refusal-template usage growth. The 3 statuses are
            # pre-registered as REFUSAL_TEMPLATES keys (empty /
            # empty_bulk / low_relevance).
            SYNTHESIS_PROMPT_TEMPLATE.inc(template=f"refusal_{retrieval_status}")
        # v2.0.29.3 (Phase 3) — loop-break hint (NEW). Fires when
        # ``react_agent`` un-bound tools on the previous round.
        # Inert when False (the default on the non-break path —
        # ``react_agent`` always stamps the flag, default-False on
        # success). The FSM persists arbitrary keys via
        # ``merge()`` so the state key crosses node boundaries.
        if state.get("loop_breaker_active"):
            msgs.insert(2, SystemMessage(content=_LOOP_BREAK_SYNTHESIS_HINT))
        # v2.0.29.3 (Phase 3) — registry-driven tool directive.
        # Iterates every registered directive builder; each one is
        # responsible for returning ``None`` when its tool wasn't
        # called this turn (so we don't insert empty/noise
        # SystemMessages on unrelated prompts). Order within the
        # loop is dict-insertion-order, which today is just
        # ``get_current_time``; future tools keep the same shape.
        for _tool_name, _builder in DIRECTIVE_BUILDERS.items():
            _directive = _builder(sanitized_history)
            if _directive:
                msgs.insert(2, SystemMessage(content=_directive))
    if route_decision != "direct":
        # The "generate" template is the default. We bump it here
        # AFTER the refusal branch so a refusal walk doesn't double-
        # count as both refusal + generate. The conditional mirrors
        # the build path: only synthesize-from-context runs the
        # GENERATE_SYSTEM template.
        SYNTHESIS_PROMPT_TEMPLATE.inc(template="generate")
    model = build_chat_model(enable_thinking=True)

    # ----- Stream the final answer -----
    answer_text = ""
    last_chunk = None
    error: Exception | None = None
    try:
        async for chunk in model.astream(msgs):
            last_chunk = chunk
            text, thinking = split_text_and_thinking(getattr(chunk, "content", ""))
            if text:
                answer_text += text
                # v2.0.19 — yield each chunk to the FSM so the
                # frontend streams in real time. The runner forwards
                # as ``events.token(text)`` on the WS.
                yield ("token", {"text": text})
            if thinking:
                # v2.0.28.15 — DO NOT yield reasoning events for the
                # synthesis LLM. Synthesis extended-thinking text is
                # meta-commentary ("I'm about to write the answer")
                # that actively MISLEADS users by hallucinating prior-
                # turn context (probe Run 1 saw "I already provided
                # the answer in my previous response" — false).
                # Planning-step reasoning from ``react_agent`` still
                # streams for transparency (action-oriented tool-
                # selection rationale). Pre-fix path yielded
                # ("reasoning", ...) → reasoningHandler.ts →
                # <ThinkingDrawer text={m.reasoning}/> default-open →
                # user saw meta-commentary above the direct answer.
                # Drop it.
                pass
        if not answer_text:
            # v2.0.28.12.1 — DO NOT set the "(no answer text generated)"
            # placeholder here. Leave ``answer_text`` empty so the
            # get_current_time-TM-data fallback below can fire when
            # the synthesis LLM emits thinking-only. Pre-fix this
            # assignment short-circuited the fallback (the fallback
            # checks ``if not answer_text``, which is now FALSE
            # because the placeholder string is truthy), leaving the
            # user to see the placeholder string even though the
            # TM had the correct time data. Confirmed via
            # ``tests/e2e/debug-now-bug2.py`` (TURN 2 saved message
            # content was the placeholder string, not the time).
            # The placeholder is set AFTER the fallback below.
            pass
    except Exception as exc:
        # v2.0.22 (Item 7 Step 11) — centralized LLM-failure recovery
        # (P2-B4 关掉). Mirrors ``fsm.react_generate_direct``
        # byte-for-byte; both used to inline the same apology text
        # + token yield (the inline ``error = exc`` local is gone —
        # ``exc`` is already in scope). Returns ``(answer_text,
        # token_event)`` so the downstream ``__delta__`` path can
        # carry the apology as the answer.
        answer_text, token_event = llm_call_failure("react_generate", exc)
        last_chunk = None
        # Emit the fallback message as a single token so the user
        # still sees the answer stream live (rare path).
        yield token_event

    answer_text = _strip_tool_blocks(answer_text)

    # v2.0.28.12 — fallback for the synthesis LLM emitting thinking-
    # only. Anthropic Claude occasionally streams a content-block list
    # whose ``text`` parts stay empty across the whole stream — every
    # chunk carries a ``thinking`` block with the actual answer text
    # embedded inside it, but no visible ``text`` block. Without the
    # fallback, ``answer_text`` stays empty, the wire answer_complete
    # fires with the "(no answer text generated)" placeholder, and the
    # user sees a blank bubble. When the conversation history contains
    # a get_current_time ToolMessage (string-returning whitelist),
    # parse the time string out of it and synthesize a direct answer
    # — the user explicitly asked for the time, we have the data, we
    # should not let the LLM's mood decide whether to show it.
    #
    # Why only fall back when get_current_time is in the history:
    # - The fallback is a hard-coded string format, not a generic
    #   "answer-from-TM" helper. Other string-returning tools would
    #   need their own format string (e.g. ``get_weather`` would want
    #   "今天 {city} 的天气是 {cond}"). Keep the fallback narrow to
    #   the tool we know about (added in v2.0.10.1) until a second
    #   tool needs the same treatment.
    # - Pre-fix cost: the user saw "(no answer text generated)" even
    #   though the LLM had the correct time in its thinking. That's
    #   a UX regression the synthesis LLM has no signal to recover
    #   from (the post-Layer-5-refine hint update nudges it toward
    #   visible text, but stochasticity remains).
    # - Defensive: the fallback only fires when ``not answer_text`` —
    #   if the synthesis LLM DID produce visible text, we keep its
    #   answer verbatim. The fallback is a safety net, not a rewrite.
    if not answer_text:
        for _m in history:
            if (
                isinstance(_m, ToolMessage)
                and (getattr(_m, "name", "") or "") == "get_current_time"
            ):
                _raw = (getattr(_m, "content", "") or "").strip()
                # Tool output is e.g.
                # ``2026-09-23T13:07:59.900947+08:00 (Asia/Shanghai)
                #   2026年9月23日 周三 13:07``
                # Build a human-friendly line. If parsing fails, fall
                # back to the raw content (still better than
                # "(no answer text generated)").
                _answer = (
                    f"根据刚刚查询,当前时间是 {_raw.splitlines()[0] if _raw else '未知'}。"
                    + (f"\n详细信息:{_raw}" if "\n" in _raw else "")
                )
                answer_text = _answer
                logger.debug(
                    "react_generate: synthesis LLM emitted thinking-only; "
                    "constructed fallback answer from get_current_time TM"
                )
                break

    # v2.0.28.16 — defensive: even with the post-script directive,
    # the LLM can still emit text that IGNORES the get_current_time
    # TM data (e.g. "好的,那就早点休息吧,晚安" — chat reply instead of
    # the time). Per the 5-run stochastic probe
    # (``tests/e2e/debug-now-bug3-probe.js``) this happens ~40% of
    # the time when the prior turn ended with a chat wrap-up. The
    # fix is fingerprint-matching: extract the most-distinctive
    # time marker from the registered tool's TM (ISO date preferred
    # for ``get_current_time`` today), and if the synthesis answer
    # does NOT contain that marker verbatim, OVERRIDE the LLM's
    # answer with the registered fallback builder's output.
    # Belt-and-suspenders with the post-script directive above:
    # directive tries to PREVENT the wrong-text emission; this
    # check CATCHES the runs that slip through.
    #
    # v2.0.29.3 (Phase 3) — the fingerprint + override logic now
    # lives in the defensive-override registry as
    # ``apply_defensive_overrides``. The registry iterates
    # ``FINGERPRINT_EXTRACTORS`` + ``FALLBACK_BUILDERS`` for every
    # tool that registered both (intersection = aligned set).
    # Adding a new tool is one line per registry; this node never
    # needs to change. The guards (``route_decision != "direct"`` +
    # ``answer_text`` non-empty) live inside the helper so the
    # call site stays one line.
    answer_text = apply_defensive_overrides(answer_text, history, route_decision)

    # v2.0.28.12.1 — only NOW, AFTER the TM-data fallback had a chance
    # to fill ``answer_text``, set the placeholder for the case where
    # the synthesis LLM emitted thinking-only AND there was no
    # get_current_time TM in history (or some other unexpected
    # thinking-only failure mode). Pre-v2.0.28.12.1 this assignment
    # ran BEFORE the fallback, short-circuiting it.
    if not answer_text:
        answer_text = "(no answer text generated)"

    # ----- Build the AIMessage for history + the state delta -----
    sources = _to_sources(merged) if route_decision != "direct" else []
    additional_kwargs: dict = {}
    if sources:
        additional_kwargs["sources"] = sources
    # v2.0.28.15 — DO NOT stamp reasoning on synthesis AIMessage.
    # Same rationale as the streaming-loop pass above. The save-side
    # ``additional_kwargs["reasoning"]`` is what
    # ``_serialize_messages:375-377`` lifts onto the wire as
    # ``MessageRecord.reasoning``. Dropping here means NEW DB rows
    # are clean. (Pre-fix DB rows still corrupt — see frontend fix
    # in ``chat-utils.mergeAdjacentAssistantTurns`` for belt-and-
    # suspenders; reload rows will lose synthesis reasoning via
    # the no-fallback rule regardless.)
    additional_kwargs["source_kinds"] = _derive_source_kinds(merged)
    # v2.0 — created_at stamp. The frontend renders this as a
    # per-message timestamp; history replay re-derives it from
    # ``additional_kwargs`` so it survives a page refresh.
    created_at = _now_iso()
    additional_kwargs["created_at"] = created_at

    # v2.0.3 — persist tool_calls on the AIMessage so history replay
    # can re-render ToolCallCards. Without this, ``MessageRecord.
    # tool_calls`` is always None on reload (the runner emits the
    # live ``tool_call_start`` / ``tool_call_end`` events but never
    # stamps the AIMessage itself). The helper walks the full
    # messages list (which IS persisted by LangGraph's checkpointer)
    # and pairs AIMessage.tool_calls entries with their matching
    # ToolMessage results.
    tool_log = _build_tool_call_log(history)
    if tool_log:
        additional_kwargs["tool_calls"] = tool_log

    # v1.1.7 + v1.1.9 — preserve thinking-block content list when the
    # last chunk carried one (so Anthropic's extended thinking
    # continues on the next turn). Logic moved to
    # ``src.agent.nodes._content_blocks.preserve_content_blocks``
    # (v2.0.22 Item 7 Step 3) so this file and ``fsm.react_generate_direct``
    # stay in sync automatically when Anthropic adds a new block type.
    final_content = preserve_content_blocks(last_chunk, answer_text)

    response = AIMessage(content=final_content, additional_kwargs=additional_kwargs)

    # v2.0.19 — emit the canonical answer_complete event the runner
    # forwards as ``events.answer_complete`` on the WS. Sources are
    # RAW here; the runner applies ``filter_sources_to_cited`` and
    # ``renumber_citations_and_sources`` (matches the pre-Phase-3
    # ``_handle_final_answer`` contract). ``route_decision`` drives
    # the source-empty short-circuit ("direct" → no citations).
    yield ("answer_complete", {
        "answer": answer_text,
        "sources": sources,
        "source_kinds": additional_kwargs["source_kinds"],
        "created_at": created_at,
        "route_decision": route_decision,
    })

    yield ("__delta__", {
        "answer": answer_text,
        "sources": sources,
        "source_kinds": additional_kwargs["source_kinds"],
        "documents": merged,
        # v2.0.28.10 — REPLACE the prior AIMessage (react_agent's
        # text-only "draft") instead of appending a new one. Without
        # this, the FSM saves 2 AIMessages per turn when no tool
        # calls are involved; reload shows the draft that the user
        # never saw live. The ``after_react_agent`` predicate
        # guarantees the last AIMessage at this point is text-only
        # (no tool_calls) — so we never clobber a tool-call
        # AIMessage that needs its matching ToolMessage to render.
        # ``merge()`` (src/agent/fsm.py) handles this key.
        "replace_last_message": response,
        "created_at": created_at,
        # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
        # ``merge()`` since this delta doesn't carry it explicitly.
    })


__all__ = ["react_generate", "_build_tool_call_log"]