"""AgentState — the shared typed dict across all LangGraph nodes.

v2.0 — ReAct rewrite. Compared to the v1.1.13 state:

Added:
  * ``intent``              — pre-ReAct classification result.
  * ``corrected_query``     — cheap-model typo correction (optional).
  * ``needs_current_time``  — recency marker, drives get_current_time hint.
  * ``step_count``          — ReAct loop counter (replaces iteration_count).
  * ``created_at``          — server-side ISO 8601 timestamp on assistant turn.

Note: tool-call records live on the AIMessage (``additional_kwargs["tool_calls"]``,
written by ``react_generate._build_tool_call_log``). The legacy
``tool_call_log`` state field was a dead write-only stub (declared
+ initialized to ``[]`` but never written) and was removed in
v2.0.22 Item 7 Step 12 (P1-B5).

Removed (ReAct subsumes their function):
  * ``iteration_count``     — step_count replaces it.
  * ``retrieval_count``     — ReAct decides.
  * ``rewrite_count``       — ReAct decides.
  * ``max_retrieval_rounds``— unused (tool cap is in graph config).
  * ``max_rewrites``        — unused.
  * ``summary_intent_used`` — ReAct route picks the bulk path explicitly.
  * ``web_intent``          — ReAct lets the LLM decide when to web-search.
  * ``web_search_attempted``— ``web_search`` tool is idempotent per-call;
                              the LLM may call it multiple times.
  * ``route_decision``      — ``intent`` is the canonical branch driver now.

Kept (the tool layer + final-answer layer both still touch these):
  * ``messages``            — Plain ``List[BaseMessage]``; Phase 3 FSM
                              merges via ``merge()`` (plain append, no
                              LangChain reducer).
  * ``thread_id``           — thread scoping for hybrid search.
  * ``original_query``      — never mutated; canonical user query.
  * ``current_query``       — ReAct-rewritten query (currently unused
                              but kept for back-compat / future use).
  * ``documents``           — accumulated docs (tool layer writes here).
  * ``graded_documents``    — currently unused in the ReAct topology
                              (the LLM itself decides relevance);
                              kept so any future post-tool grader can
                              re-use the field without schema migration.
  * ``web_documents``       — v1.1.13 complementary path. ReAct may
                              still produce web docs via the
                              ``web_search`` tool; they end up here.
  * ``sources``             — final-answer-layer writes here for the
                              wire (filter_sources_to_cited uses it).
  * ``source_kinds``        — banner kinds, see v1.1.14.
  * ``answer``              — final prose (legacy field name).
  * ``hallucination_check`` — written by ``check_hallucination`` (still
                              in the graph, still cheap-model-based).

Backwards compatibility
-----------------------
``total=False`` on every TypedDict means missing keys are read as
None / empty. Old checkpoints persisted under v1.1.16 still load —
fields they don't carry (e.g. ``intent``) fall through to the
graph's defaults (``None`` → ``intent_analysis` is the START node
so the missing-field case only happens on a hot reload).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class Source(TypedDict, total=False):
    """A single citation surfaced to the user.

    Mirrors v1.1.16 — preserved as-is so the frontend's Source
    contract is unchanged. ``source_kind`` distinguishes local from
    web. ``url`` / ``domain`` only populated for web sources.

    v2.0.29.9 (Phase 8) — ``verbatim`` flag marks sources used in
    verbatim extraction mode. Frontend reads this to render a 🔒
    icon on the bubble (additive field, default False — old clients
    that don't know about it simply render normally).
    """

    index: int
    chunk_id: str
    doc_id: str
    filename: str
    page: Optional[int]
    section: Optional[str]
    sheet: Optional[str]
    text: str
    score: float
    url: Optional[str]
    domain: Optional[str]
    source_kind: Literal["local", "web"]
    verbatim: bool


class ToolCallRecord(TypedDict, total=False):
    """Per-call log entry, written by the runner as ``tool_call_end``
    events arrive. Used by the frontend to render ``ToolCallCard``
    components in the chat pane (each card renders one record).

    Fields
    ------
    id           LangChain-issued tool call id (stable across loop iterations).
    name         Tool name ("retrieve_docs" / "web_search" / "get_current_time").
    args         Tool arguments dict (validated by Pydantic ``args_schema``).
    result       Truncated ToolMessage content (first 800 chars).
    ok           True if the tool succeeded, False on exception.
    step         1-based ReAct step counter when this call ran.
    started_at   ISO 8601 server timestamp when the call started.
    ended_at     ISO 8601 server timestamp when the call finished.
    elapsed_ms   Round-trip time in milliseconds.
    """

    id: str
    name: str
    args: Dict[str, Any]
    result: str
    ok: bool
    step: int
    started_at: str
    ended_at: str
    elapsed_ms: int


class AgentState(TypedDict, total=False):
    """v2.0 — ReAct rewrite state.

    See module docstring for the field-by-field rationale.
    """

    # ---- Chat history (Phase 3 — plain List, FSM merges via merge()) ----
    messages: List[BaseMessage]

    # ---- Thread / query identity ----
    thread_id: str
    original_query: str
    current_query: str  # mirrored to original_query unless LLM rewrites

    # ---- v2.0 NEW: pre-ReAct classification ----
    intent: Literal["greeting", "summary", "qa_complex"]
    corrected_query: Optional[str]
    needs_current_time: bool

    # ---- v2.0 NEW: ReAct loop counter ----
    step_count: int

    # ---- v2.0 NEW: server-side timestamp on the assistant turn ----
    created_at: Optional[str]

    # ---- Document accumulation (tool layer writes here) ----
    documents: List[Document]
    graded_documents: List[Document]  # currently unused in ReAct topology
    web_documents: List[Document]    # populated by web_search tool

    # ---- v2.0.29.1 (Phase 1) — Refusal contract signal ----
    # Set by ``retrieve`` / ``summary_path`` based on the quality and
    # presence of the retrieved document set. Read by ``react_generate``
    # to decide whether to inject a refusal SystemMessage at msgs[2]
    # (forcing the synthesis LLM to walk the matching template in
    # ``src.llm.prompts.REFUSAL_TEMPLATES``).
    #
    # Values:
    #   - "empty"           → 0 documents retrieved from hybrid search
    #                         AND summary path was not attempted
    #   - "empty_bulk"      → summary-intent path returned 0 chunks
    #                         (thread has no uploaded documents)
    #   - "low_relevance"   → retrieval returned docs but top
    #                         rerank_score < config threshold
    #                         (Phase 4 will introduce this)
    #   - "success"         → retrieval returned relevant docs;
    #                         synthesis LLM has ground truth to work with
    #   - None / missing    → backward-compat with pre-Phase-1 state
    #                         (treated as "success" by react_generate —
    #                         no behavioral change for legacy runs)
    retrieval_status: Literal["empty", "empty_bulk", "low_relevance", "success"]

    # ---- v2.0.29.9 (Phase 8) — Verbatim extraction mode flag ----
    # When truthy ("on"), the FSM routes qa_complex intent to
    # ``react_generate_extractive`` instead of ``react_agent`` —
    # bypassing the ReAct tool-use loop and forcing the synthesis
    # LLM into verbatim-quote-only mode (no rewriting / no
    # summary / no inference).
    #
    # Tri-state literal (NOT bool) — the tristate is load-bearing
    # for user-override semantics (user decision 2026-09-28):
    #
    #   - "auto" (default) → run hybrid detect (regex + cheap-LLM
    #     fallback in intent_analysis); may or may not trigger.
    #   - "on"            → force verbatim mode (skip detection,
    #     skip react_agent tool-use loop).
    #   - "off"           → force normal mode (skip extractive node
    #     EVEN if regex/cheap-LLM detect flagged the query).
    #
    # Persistence: per-thread in ThreadState (App.tsx); forwarded
    # to backend via ChatRequest.high_precision (default "auto").
    # Pre-Phase-8 sessions and unset state fall through to "auto".
    high_precision: Literal["auto", "on", "off"]

    # ---- Final answer outputs ----
    # ``answer`` is the canonical final-answer prose. The
    # ``react_generate`` / ``react_generate_direct`` nodes write
    # this; the runner reads it back off the FSM delta to emit
    # ``answer_complete`` events. It is kept on the state so the
    # FSM merge carries it through to the next node. Fields NOT
    # declared here are silently dropped (TypedDict filter).
    answer: str
    sources: List[Source]
    source_kinds: List[str]
    hallucination_check: Literal["grounded", "ungrounded", "skipped"]


__all__ = ["AgentState", "Source", "ToolCallRecord"]