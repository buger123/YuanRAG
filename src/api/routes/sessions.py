"""Session listing + clearing endpoints + per-thread message replay."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Response
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.api.schemas import MessageRecord, SessionSummary
from src.agent.citations import renumber_citations_and_sources
from src.api.i18n import error_message, parse_accept_language
from src.core.logging import logger
from src.storage.checkpointer import delete_thread, list_threads
from src.storage.lancedb_store import (
    count_documents_by_threads,
    latest_ingested_at_by_threads,
    list_threads_with_documents,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


# Hard cap on the number of sessions returned. Phase 3 — the new
# checkpointer already enforces ``_MAX_THREADS`` via the SQL ``LIMIT``
# clause, but we keep the same 500 cap at the route layer so a
# caller that grows it later (or a test that overrides it) gets the
# pre-Phase-3 behaviour.
_MAX_SESSIONS = 500


def _extract_msg_text(m: BaseMessage) -> str:
    """Extract plain text from a LangChain message.

    ``AIMessageChunk.content`` is normally a string but Anthropic tool-use
    responses arrive as ``[{"text": "...", "type": "text"}, ...]`` blocks;
    we mirror the same defensive normalization done in the streaming
    runner so historical replays don't crash anything that tries to
    render the assistant turn verbatim.
    """
    c = getattr(m, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for b in c:
            if isinstance(b, dict):
                btype = b.get("type")
                if btype in ("text", None) and "text" in b:
                    parts.append(str(b["text"]))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return ""


def _msg_role(m: BaseMessage) -> str:
    """Map a LangChain message class to a UI role string.

    v2.0.3 — ``ToolMessage`` is now classified as ``"tool"`` (a distinct
    role), NOT ``"assistant"``. Pre-v2.0.3 the class-name fallback in
    this function returned ``"assistant"`` for any ``tool``-named class
    so ToolMessages were tagged as assistant messages and serialized
    into ``MessageRecord`` with role=assistant. The result: every turn
    that fired a tool call produced a giant JSON-serialized blob (the
    tool result) as a user-visible assistant message. Refreshing the
    page after a ReAct turn showed ``[{"page_content": ...`` as the
    assistant's "answer".

    The serializer below also filters on the explicit ``role``
    return value: ``"tool"`` is dropped before ``MessageRecord``
    construction (ToolMessages are intentionally invisible to the UI;
    their content lives on the AIMessage's ``additional_kwargs`` via
    the v2.0.3 tool_calls persistence fix).
    """
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, AIMessage):
        return "assistant"
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, ToolMessage):
        return "tool"
    # Fall back to class-name sniffing for any custom message subclasses
    # the graph might inject (defensive — covers any BaseMessage subtype
    # that doesn't import one of the well-known classes above).
    cls = type(m).__name__.lower()
    if "human" in cls or "user" in cls:
        return "user"
    if "ai" in cls or "assistant" in cls:
        return "assistant"
    if "system" in cls:
        return "system"
    if "tool" in cls:
        return "tool"
    # Truly unknown role — log it once and treat as assistant so a
    # novel message subclass doesn't silently disappear from history.
    logger.warning(
        f"_msg_role: unknown message class {type(m).__name__}; defaulting to assistant"
    )
    return "assistant"


def _safe_channel_values(cp: Any) -> dict:
    """DEPRECATED in Phase 3 — kept as a stub so any in-flight
    monkeypatched test that still sets it doesn't blow up with
    AttributeError. Phase 3's :func:`src.storage.checkpointer.list_threads`
    already returns plain dicts (each with a ``messages`` key), so the
    listing loop reads ``entry["messages"]`` directly — no
    channel_values unwrapping needed.

    Returns ``{}`` to keep the (now-defunct) callers' guard
    ``channel_values.get("messages")`` syntax working in tests.
    """
    if isinstance(cp, dict):
        return cp
    try:
        return dict(cp) if cp else {}
    except Exception:
        return {}


def _title_from_messages(messages: list[BaseMessage]) -> str:
    """Pick a meaningful session title.

    Prefer the most recent user message (it tells the user what the
    thread is actually about). Fall back to the first non-empty message
    of any role. Final fallback is the literal ``"New chat"``.
    """
    # Newest first; messages are appended chronologically by add_messages.
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            text = _extract_msg_text(m).strip()
            if text:
                return text[:50]
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        text = _extract_msg_text(m).strip()
        if text:
            return text[:50]
    return "New chat"


def _count_messages(messages: list[BaseMessage]) -> int:
    """Number of UI-visible turns in the checkpoint's message list.

    System messages and empty assistant turns (e.g. tool-use stubs that
    produced no text) are excluded so the badge matches what the user
    sees in the chat pane.
    """
    n = 0
    for m in messages or []:
        if isinstance(m, SystemMessage):
            continue
        if _extract_msg_text(m).strip():
            n += 1
    return n


def _extract_thinking_text(m: BaseMessage) -> str:
    """Pull Anthropic ``thinking`` blocks out of an AIMessage's content
    list and join them with blank lines.

    v2.0.24 — newly added (Item 8 P0-F4, "historical conversations
    only show most recent Q&A"). Intermediate ReAct AIMessages
    persist their reasoning INSIDE ``content`` as
    ``{"type": "thinking", "thinking": "..."}`` blocks (Anthropic
    extended-thinking feature) — they have no ``text`` block and no
    ``additional_kwargs["reasoning"]`` entry. Without this helper the
    line 301 filter in :func:`_serialize_messages` would drop every
    intermediate round and the user would only see the final answer.

    Returns ``""`` for messages without a content list (str content,
    HumanMessage, ToolMessage, SystemMessage). Multiple thinking
    blocks are joined with ``\\n\\n`` — the ThinkingDrawer renders the
    string verbatim, so the user sees each round's reasoning as its
    own paragraph.
    """
    content = getattr(m, "content", "")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "thinking":
            text = b.get("thinking")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n\n".join(parts)


def _find_following_tool_result(
    messages: list[BaseMessage],
    start_idx: int,
    tool_call_id: str,
) -> Optional[str]:
    """Look forward in ``messages`` from ``start_idx`` for the
    ToolMessage whose ``tool_call_id`` matches.

    v2.0.28.13 — helper for the wire serializer to enrich
    synthesized tool-call cards (built from the native AIMessage
    ``tool_calls`` attr) with the matching ToolMessage's content +
    success status. Without this, intermediate planning-step
    AIMessages render as ⏳ pending forever on history reload
    because the native attr only carries ``{id, name, args}`` — no
    completion data.

    Scans forward until the next non-ToolMessage (typically the
    synthesis AIMessage); ToolMessages from prior rounds are not
    considered (Anthropic's tool_use ↔ tool_result pairing is
    position-based). Returns ``None`` if no match (tool was killed
    mid-flight or never ran — caller should mark ``ok=False``).

    Content normalization mirrors ``react_generate._build_tool_call_log``
    lines 188-199: string content is used as-is; Anthropic-style
    content-block lists are flattened by joining the ``text`` of each
    block; anything else is coerced via ``str(...)``. The truncation
    cap is applied by the caller to keep the truncation policy in one
    place.
    """
    for j in range(start_idx + 1, len(messages)):
        tm = messages[j]
        if not isinstance(tm, ToolMessage):
            break
        if tm.tool_call_id == tool_call_id:
            raw = getattr(tm, "content", "")
            if isinstance(raw, str):
                return raw
            if isinstance(raw, list):
                return "".join(
                    str(b.get("text", ""))
                    for b in raw
                    if isinstance(b, dict)
                )
            return str(raw)
    return None


def _serialize_messages(messages: list[BaseMessage]) -> list[MessageRecord]:
    """Flatten a checkpoint's message list to the wire format.

    Sources attached to an AIMessage via ``additional_kwargs["sources"]``
    and reasoning attached via ``additional_kwargs["reasoning"]`` (both
    written by ``src.agent.nodes.generate``) are extracted here and
    returned in the wire record so the frontend can re-render citation
    chips AND the thinking drawer when the user reopens a conversation.
    Old checkpoints predating either field simply have no
    ``additional_kwargs`` entry and come back with ``None`` — graceful
    degradation, no crash.

    v2.0 — also extract ``tool_calls`` (per-tool-call log, written
    by the runner on the AIMessage) and ``created_at`` (ISO 8601, on
    both HumanMessage and AIMessage). Both are stamped by the runner;
    we rehydrate the wire record here so the per-message timestamp
    and the per-tool-call cards survive a page refresh / thread
    switch.

    v2.0.3 — ToolMessages are now classified as role="tool" and SKIPPED
    entirely (not emitted as MessageRecord). Pre-v2.0.3 the
    class-name fallback in ``_msg_role`` returned ``"assistant"`` for
    any ``tool``-named class, so the JSON-serialized Document lists
    (the tool results) appeared as user-visible assistant messages on
    reload. Tool result content is reconstructed by react_generate via
    ``_extract_docs_from_messages`` so no information is lost.

    v2.0.24 (Item 8 P0-F4) — intermediate ReAct AIMessages now
    survive the line 301 filter. Previously every AIMessage with
    content = ``[thinking, tool_use*]`` and empty ``additional_kwargs``
    was silently dropped, so reopening a multi-round thread only
    showed the user's question and the FINAL answer — all the
    in-between "agent decided to call retrieve_docs, then web_search"
    rounds vanished. The fix has three pieces (see each branch below):

    * **Reasoning** — :func:`_extract_thinking_text` pulls Anthropic
      ``thinking`` blocks out of ``AIMessage.content`` and we emit
      them via the ``reasoning`` field. The frontend's
      ThinkingDrawer renders them verbatim.
    * **Tool calls** — for AIMessages that have native ``tool_calls``
      (the langchain built-in attribute, set by every ``llm.invoke``
      that produced tool calls) but NO ``additional_kwargs["tool_calls"]``
      (the per-round log that ``react_generate._build_tool_call_log``
      only stamps on the FINAL answer), we synthesize a minimal card
      list from ``{id, name, args}``. The frontend's ToolCallCard
      gracefully handles missing ``result`` / ``elapsed_ms`` / ``step``.
    * **Filter** — the ``if not text and not reasoning and not
      tool_calls: continue`` guard now ALSO accepts native
      ``AIMessage.tool_calls`` so a tool-use-only stub without any
      extracted thinking still survives (defense in depth — covers
      legacy content shapes where thinking isn't in a ``thinking``
      block).
    """
    out: list[MessageRecord] = []
    for idx, m in enumerate(messages or []):
        role = _msg_role(m)
        if role == "system":
            continue
        # v2.0.3 — ToolMessages (the LLM's tool call results) are NOT
        # serialized as user-visible assistant messages. Their content
        # is reconstructed into Document objects by react_generate on
        # history replay, so there's no information lost by skipping
        # them here. Before v2.0.3 the role fallback returned
        # ``"assistant"`` for any ``tool``-named class, producing giant
        # JSON blobs as assistant messages on reload.
        if role == "tool":
            continue
        text = _extract_msg_text(m).strip()
        # v2.0.24 — pull Anthropic thinking blocks out of content list.
        # Falls back to "" for non-AIMessages or string content.
        thinking_text = _extract_thinking_text(m)
        # Pull sources / reasoning / source_kinds off additional_kwargs
        # if present. Defensive: any non-list / non-string value (None,
        # dict, etc.) is dropped — we never want a malformed checkpoint
        # to 500 the listing endpoint.
        sources: list | None = None
        reasoning: str | None = None
        # v1.1.14 — banner kinds for history replay. Forwarded from
        # the AIMessage's ``additional_kwargs["source_kinds"]`` (set
        # by ``generate._derive_source_kinds``). ``None`` for old
        # checkpoints predating this field — the frontend should
        # render no banner (graceful degradation, same as missing
        # sources). We DON'T derive from ``sources`` here so old
        # behavior doesn't leak through: if the user reloaded the
        # page mid-flight before we stamped ``source_kinds``, the
        # banner should stay blank, not "guess" based on sources.
        source_kinds: list | None = None
        # v2.0 — per-tool-call log (AIMessage) and ISO 8601
        # timestamp (both HumanMessage and AIMessage). The runner
        # stamps ``created_at`` on both message types so the
        # frontend can show a per-turn timestamp on both sides.
        tool_calls: list | None = None
        created_at: str | None = None
        kwargs = getattr(m, "additional_kwargs", None) or {}
        if isinstance(kwargs, dict):
            raw_tc = kwargs.get("tool_calls")
            if isinstance(raw_tc, list):
                # Validate: each entry must be a dict (the runner
                # emits ToolCallRecord TypedDicts). A malformed
                # checkpoint with non-dict entries (e.g. accidentally
                # stamped a list of strings) would otherwise crash the
                # frontend's ToolCallCard render.
                tool_calls = [
                    tc for tc in raw_tc if isinstance(tc, dict)
                ]
                # An empty list still gets emitted — the frontend's
                # ``Array.isArray(r.tool_calls)`` guard handles it
                # as "no cards". Setting ``None`` on empty list
                # would force the frontend through the legacy
                # no-cards branch, which is also correct but loses
                # the "this turn DID run through ReAct" signal.
                # We keep the empty list here.
            raw_ts = kwargs.get("created_at")
            if isinstance(raw_ts, str) and raw_ts:
                created_at = raw_ts
            if isinstance(m, AIMessage):
                raw_sources = kwargs.get("sources")
                if isinstance(raw_sources, list):
                    # v2.0.7 SoT-2: history replay now goes through
                    # the same renumber pass as the live stream. The
                    # runner emits post-renumber sources for the
                    # live ``answer_complete`` event, but on history
                    # reload the SqliteSaver round-trips whatever
                    # was stamped on the AIMessage — which for
                    # messages generated before v2.0.5 P1-3 may
                    # carry dangling ``[n]`` markers that don't
                    # match the surviving sources. Re-applying the
                    # renumber here guarantees the persisted
                    # sources always match the answer text the
                    # frontend re-renders.
                    sources = renumber_citations_and_sources(
                        answer=text,
                        sources=[
                            dict(s) for s in raw_sources if isinstance(s, dict)
                        ],
                    )[1]
                raw_reasoning = kwargs.get("reasoning")
                if isinstance(raw_reasoning, str) and raw_reasoning:
                    reasoning = raw_reasoning
                raw_kinds = kwargs.get("source_kinds")
                if isinstance(raw_kinds, list):
                    # Validate: each entry must be a string. A
                    # malformed checkpoint with non-string entries
                    # (e.g. accidentally stamped a list of dicts)
                    # would otherwise explode the frontend's
                    # ``source_kinds.includes("web")`` check.
                    source_kinds = [
                        k for k in raw_kinds if isinstance(k, str)
                    ]
        # v2.0.28.15 — DO NOT populate wire `reasoning` from content-
        # block thinking text. Pre-fix the `_extract_thinking_text`
        # fallback lifted the planning-step AIMessage's raw Anthropic
        # thinking blocks onto the wire as `MessageRecord.reasoning`
        # → restoreMessages → mergeAdjacentAssistantTurns carried it
        # forward → <ThinkingDrawer> rendered the LLM's meta-commentary
        # ("The user is asking ... I need to call get_current_time").
        # The synthesis-side stamp was already dropped in
        # react_generate.py / fsm.py:react_generate_direct; this line
        # was the LAST source still feeding the drawer. Drop it:
        # every AIMessage now has wire `reasoning=None` regardless of
        # how the LLM held its reasoning (additional_kwargs or content
        # blocks). NOTE: `_sanitize_history_for_generate` reads
        # `AIMessage.content` directly (not the wire), so this does
        # NOT affect Anthropic extended-thinking continuity — planning
        # AIMessages still carry their thinking blocks into the next
        # turn's prompt.
        # (v2.0.24 fallback removed in v2.0.28.15.)
        # v2.0.24 — if the AIMessage has native ``tool_calls``
        # (langchain built-in attr, set on every LLM response that
        # produced tool calls) but no ``additional_kwargs["tool_calls"]``
        # log (which the runner only stamps on the final answer),
        # synthesize a minimal card list from the native attr. The
        # final AIMessage's rich log already wins because the
        # ``if tool_calls is None`` check below is satisfied first.
        if isinstance(m, AIMessage) and tool_calls is None:
            native_tc = getattr(m, "tool_calls", None)
            if isinstance(native_tc, list) and native_tc:
                cards: list[dict] = []
                for tc in native_tc:
                    if not isinstance(tc, dict):
                        continue
                    card: dict = {}
                    tc_id = tc.get("id")
                    if tc_id is not None:
                        card["id"] = str(tc_id)
                    tc_name = tc.get("name")
                    if tc_name is not None:
                        card["name"] = str(tc_name)
                    tc_args = tc.get("args")
                    if isinstance(tc_args, dict):
                        card["args"] = tc_args
                    # v2.0.28.13 — the frontend's ToolCallCard renders
                    # ``running`` when ``ok === undefined`` (line 43 of
                    # ToolCallCard.tsx). For an intermediate planning-
                    # step AIMessage (the one produced by react_agent
                    # BEFORE the tool ran), the native ``tool_calls``
                    # attr only carries ``{id, name, args}`` — no
                    # ``result``, no ``ok``. Synthesizing the card
                    # without these fields made the reload view show
                    # the planning-step card as ⏳ pending forever
                    # (above the synthesis card which has the rich log
                    # with ok:true). The user sees TWO cards per turn:
                    # one pending, one completed — looks like the tool
                    # call is "long-running" / stuck.
                    #
                    # Fix: scan forward in ``messages`` for the
                    # matching ToolMessage. If we find one, we know the
                    # tool ran (ok=True) AND we can pull the truncated
                    # result so the card shows the same content live +
                    # reload. If we DON'T find one, the tool was
                    # killed mid-flight or never ran — mark ok=False
                    # so the frontend shows the ✗ error state instead
                    # of an infinite spinner.
                    if card and tc_id is not None:
                        tc_result = _find_following_tool_result(
                            messages, idx, str(tc_id)
                        )
                        if tc_result is not None:
                            card["ok"] = True
                            # Truncate to match react_generate's
                            # _build_tool_call_log preview cap
                            # (800 chars + ellipsis) so the wire
                            # shape is consistent between the two
                            # card sources.
                            preview_max = 800
                            card["result"] = (
                                tc_result[:preview_max]
                                + ("…" if len(tc_result) > preview_max else "")
                            )
                        else:
                            # No matching ToolMessage → tool never
                            # completed. Mark as error so the
                            # frontend shows ✗ instead of an infinite
                            # ⏳ spinner.
                            card["ok"] = False
                    if card:
                        cards.append(card)
                if cards:
                    tool_calls = cards
        # v1.1.9 — defense in depth against the v1.1.7 regression.
        # If the message has no extractable visible text BUT has reasoning
        # (i.e. an AIMessage with content = list of only thinking blocks),
        # still emit the MessageRecord with empty content + reasoning so
        # the user sees "the model thought, but emitted no visible text"
        # rather than a silent hole in the chat history. Without this
        # guard, a single empty-text AIMessage would drop the entire
        # turn from the user's view (the user reports "某些对话的回答
        # 消失了, 用户的问题还在" — exactly this code path firing on
        # every turn whose last streaming chunk lacked a text block).
        #
        # v2.0 — extended the guard: a tool-calls-only AIMessage
        # (LLM emitted a tool call but the run was killed mid-flight
        # before any text/think reply arrived) should also survive
        # the serializer. Without this, a half-finished ReAct turn
        # would silently disappear from history on reload.
        #
        # v2.0.24 — extended the guard AGAIN: an intermediate ReAct
        # AIMessage whose reasoning lives in ``content[0]["thinking"]``
        # (NOT in additional_kwargs) should also survive. Without this,
        # every multi-round thread on history reload only shows the
        # user's question and the final answer — all the in-between
        # "agent decided to call retrieve_docs, then web_search"
        # rounds silently disappear (P0-F4 user-reported bug).
        #
        # We still skip AIMessages with NEITHER text NOR reasoning
        # NOR tool_calls — those are genuinely empty (e.g. tool-use
        # stubs the LLM emitted but then didn't follow up with
        # anything) and would just clutter the chat pane.
        if not text and not reasoning and not tool_calls:
            continue
        out.append(
            MessageRecord(
                role=role,
                content=text,  # may be empty when only reasoning survived
                sources=sources,
                reasoning=reasoning,
                source_kinds=source_kinds,
                tool_calls=tool_calls,
                created_at=created_at,
            )
        )
    return out


@router.get("", response_model=list[SessionSummary])
async def list_sessions(response: Response) -> list[SessionSummary]:
    """List all chat threads persisted in the own SQLite ``states`` table.

    v2.0.21 (Phase 3) — the new :func:`checkpointer.list_threads`
    returns one summary per thread, newest-first, with the
    deserialized ``messages`` already attached. No more ``alist``
    walk + manual dedup; the SQL ``ORDER BY created_at DESC LIMIT
    500`` does the work in one round trip.

    v2.0.27.2 P1-P5 — :func:`checkpointer.list_threads` now returns
    ``(valid_threads, corrupt_count)`` instead of silently swallowing
    corrupt rows. We unpack the tuple and forward the corrupt count
    as an ``X-Corrupt-Thread-Count`` response header so the operator
    sees the data-loss signal in dev tools + server logs. Future
    PR can promote this to a frontend banner (currently the
    frontend is unchanged — wire shape preserved).
    """
    # thread_id -> {updated_at, messages}
    aggregated: dict[str, dict] = {}
    corrupt_count = 0
    try:
        rows, corrupt_count = await list_threads(limit=_MAX_SESSIONS)
        for row in rows:
            tid = row.get("thread_id", "")
            if not tid:
                continue
            aggregated[tid] = {
                "updated_at": row.get("updated_at") or "",
                "messages": row.get("messages") or [],
            }
        if corrupt_count:
            # Set AFTER iterating rows so we don't emit the header
            # when no corruption actually occurred (avoids spurious
            # ``X-Corrupt-Thread-Count: 0`` on healthy responses).
            response.headers["X-Corrupt-Thread-Count"] = str(corrupt_count)
            logger.warning(
                f"list_sessions: skipped {corrupt_count} corrupt thread(s) "
                "— see checkpointer logs for trace_ids"
            )
    except Exception:
        logger.exception("list sessions failed")

    # Threads that have documents but no checkpoint yet (user uploaded
    # but never sent a message) still need a sidebar entry. Pull distinct
    # thread_ids from LanceDB and merge into ``aggregated`` with an empty
    # message list. ``updated_at`` is sourced from the most recent
    # ``ingested_at`` for that thread — the upload's real timestamp —
    # so doc-only threads sort correctly in the sidebar (v2.0.6 P2-1
    # closes the "updated_at always empty for doc-only threads" bug).
    try:
        doc_only_tids = [
            tid for tid in list_threads_with_documents()
            if tid and tid not in aggregated
        ]
        if doc_only_tids:
            doc_only_ts = latest_ingested_at_by_threads(doc_only_tids)
            for tid in doc_only_tids:
                aggregated[tid] = {
                    "updated_at": doc_only_ts.get(tid, ""),
                    "messages": [],
                }
    except Exception:
        logger.exception("list_threads_with_documents failed")

    out: list[SessionSummary] = []
    # Aggregate doc_count + first_filename for EVERY thread in ONE
    # call. The previous implementation called
    # ``list_documents(thread_id=tid)`` inside the for-loop below — one
    # full ``table.to_arrow()`` scan + Python group-by per thread, so
    # 50 sessions cost ~50 scans (~5 s end-to-end on a populated
    # corpus). ``count_documents_by_threads`` uses an in-memory
    # reverse index that's rebuilt once and maintained incrementally
    # on every add/delete, so the typical cost is a dict lookup.
    thread_ids = list(aggregated.keys())
    try:
        doc_info = count_documents_by_threads(thread_ids)
    except Exception:
        logger.exception("count_documents_by_threads failed")
        doc_info = {tid: (0, "") for tid in thread_ids}

    # Sort newest first so the sidebar shows the most recent thread on top.
    for tid, info in sorted(
        aggregated.items(),
        key=lambda kv: kv[1]["updated_at"],
        reverse=True,
    ):
        msgs = info["messages"]
        title = _title_from_messages(msgs)
        message_count = _count_messages(msgs)
        doc_count, first_doc_filename = doc_info.get(tid, (0, ""))
        # Upload-only threads have no checkpoint, so ``message_count``
        # is 0. Without a doc_count check the previous "skip empty"
        # rule would drop them — the whole point of the LanceDB merge
        # above is to keep them. Title falls back to the first doc's
        # filename (e.g. "report.pdf") instead of the generic
        # "New chat" stub.
        if message_count == 0 and doc_count == 0:
            continue
        if not title or title == "New chat":
            if first_doc_filename:
                title = first_doc_filename[:50]
            else:
                title = "New chat"
        out.append(
            SessionSummary(
                thread_id=tid,
                title=title,
                updated_at=info["updated_at"],
                message_count=message_count,
                doc_count=doc_count,
            )
        )
    return out


@router.get("/{thread_id}/messages", response_model=list[MessageRecord])
async def get_messages(
    thread_id: str,
    # PR-4 (v2.0.26.1): threads ``Accept-Language`` so the user sees
    # the same locale for session errors as for everything else.
    accept_language: str | None = Header(None, alias="accept-language"),
) -> list[MessageRecord]:
    """Replay the persisted message history for a thread.

    Returns ``[]`` for unknown threads (rather than 404) so the frontend
    can always render an empty chat pane after switching to a brand-new
    thread. The checkpointer's :func:`load_latest` is the authoritative
    source — it returns the latest snapshot of the full state, so we
    get every prior turn, not just the last one.
    """
    locale = parse_accept_language(accept_language)
    if not thread_id:
        raise HTTPException(
            status_code=400,
            detail=error_message("session.thread_id_required", locale),
        )
    from src.storage.checkpointer import load_latest

    try:
        state = await load_latest(thread_id)
    except Exception as exc:
        logger.exception(f"load_latest failed for thread {thread_id!r}")
        # PR-4: was hardcoded English "load history failed: {exc}".
        # Now localized; the exception detail stays in the log only.
        raise HTTPException(
            status_code=500,
            detail=error_message("session.load_history_failed", locale),
        ) from exc
    if state is None:
        return []
    messages = state.get("messages") or []
    return _serialize_messages(messages)


@router.delete("/{thread_id}")
async def clear_session(
    thread_id: str,
    # PR-4: locale threading for the error path (delete can fail on
    # a DB lock, a missing row, etc — we want the user to see a
    # localized message rather than English regardless of UI).
    accept_language: str | None = Header(None, alias="accept-language"),
) -> dict:
    """Remove all checkpoints for a thread."""
    locale = parse_accept_language(accept_language)
    try:
        await delete_thread(thread_id)
    except Exception as exc:
        # PR-4: was hardcoded English "clear_session failed: {exc}".
        # Now localized; the exception detail stays in the log only.
        raise HTTPException(
            status_code=500,
            detail=error_message("session.clear_failed", locale),
        ) from exc
    return {"ok": True, "thread_id": thread_id}