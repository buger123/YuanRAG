"""retrieve_docs tool — hybrid retrieval over the active thread's documents.

The tool wraps the existing :func:`src.agent.nodes.retrieve.retrieve_hybrid_async`
implementation, which already does parallel BM25 + dense BGE-M3 + rerank
+ thread-scoped filtering. Returns a JSON-serialized list of
Documents — see v2.0.2 note below.

Why we surface this as a tool (vs the legacy graph node)
-------------------------------------------------------
* ReAct is the new control loop. The LLM decides whether to call
  ``retrieve_docs`` — for a simple greeting "hi" it won't, for a
  factual question "what's in Q3?" it will, for a current-events
  question "today's weather" it won't (it'll call ``web_search``).
* Tool arguments carry a ``top_k`` knob and a free-form ``query``
  string. Pydantic ``args_schema`` enforces ``1 ≤ top_k ≤ 20`` so a
  rogue LLM can't ask for 200 chunks and OOM the embedder.

v2.0.2 — Return type changed from ``list[Document]`` to ``str`` (JSON)
-------------------------------------------------------------------
Mirrors the web_search fix. Previously this tool returned a raw
``list[Document]`` and relied on LangChain's ``ToolNode._stringify``
fallback (``str(content)`` → Python repr), which is not valid JSON.
That broke ``react_generate._extract_docs_from_messages`` (its
``json.loads`` silently failed) and produced ugly tool_result payloads
on the wire. Now we serialize to a JSON string ourselves via
``_docs_to_json`` so the format is consistent across both retrieval
tools.

v2.0.4 — thread_id injection via ``InjectedState`` → ``InjectedToolArg`` (Phase 3)
----------------------------------------------------------------------------------
Pre-v2.0.4 the tool built a state slice with a hardcoded thread-id placeholder
(a placeholder that was never replaced at runtime), which meant
``retrieve_docs`` was unscoped — it searched the ENTIRE ``chunks``
table across every thread, returning chunks from documents uploaded
to OTHER conversations. This is the silent-data-leak root cause of
the "LLM gets unrelated chunks from someone else's docs" symptom
and a compliance concern (a user could see content from another
user's thread if they share an indexing cluster).

Fix (Phase 3 — v2.0.20): LangGraph's ``InjectedState`` was removed
in favour of LangChain's native ``InjectedToolArg`` marker (from
``langchain_core.tools.base``, **not** LangGraph). The semantics
are identical: LangChain filters the field out of the JSON schema
sent to the LLM, so the model can never see — or hallucinate — a
``thread_id`` argument. The runtime value is supplied by the FSM's
``_tools_step`` (see ``src/agent/fsm.py``) which passes
``thread_id=state["thread_id"]`` at tool-call time. No LangGraph
``ToolNode`` involved.

State side-effect: every returned Document is ALSO appended to
``state["documents"]`` (the canonical accumulated doc set) so the
final ``react_generate`` node can read it for citation building. The
accumulation happens via the ``ToolMessage``-→-state delta pipeline
in the runner (see :func:`src.agent.runner._accumulate_tool_docs`).

Failure policy: this tool NEVER raises to the caller. It returns the
JSON string ``[]`` so the ReAct loop continues gracefully — a flaky
embedder doesn't kill the agent run. The error is logged at WARNING
and surfaced as ``ToolMessage.content`` (which the LLM sees on its
next decision step) when appropriate.
"""
from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, List, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
# Phase 3 — InjectedToolArg is langchain_core (NOT langgraph). Replaces
# the old ``from langgraph.prebuilt import InjectedState`` — the FSM
# now passes thread_id via ``tool.ainvoke(args, thread_id=...)`` so
# the LangGraph ToolNode is no longer needed.
from langchain_core.tools.base import InjectedToolArg
from pydantic import BaseModel, Field

from src.agent.nodes.retrieve import retrieve_hybrid_async
from src.core.logging import logger
# v2.0.18 — _docs_to_json moved out of src/agent/tools/web_search.py
# (now deleted) into src/web_search/_wire.py. Same wire format
# contract: ``[{"page_content": ..., "metadata": {...}}, ...]``.
from src.web_search._wire import _docs_to_json


class RetrieveDocsArgs(BaseModel):
    """Pydantic schema for ``retrieve_docs`` arguments.

    Pydantic validation runs BEFORE the tool body executes — a
    malformed tool call (e.g. ``top_k=99``) raises ``ValidationError``
    which LangChain's ``ToolNode`` converts to a ``ToolMessage`` with
    error content, NOT an unhandled exception. Cleaner than letting
    bad args reach the embedder.

    v2.0.4 — ``thread_id`` is intentionally NOT in this schema. The
    LLM would happily hallucinate any string it liked, which would
    either (a) silently scope the search to a non-existent thread
    (empty results), or (b) worse, route through a hostile filter
    expression (mitigated by ``hybrid_search._thread_filter`` but
    still wasteful). The real thread_id is supplied at tool-call
    time by the FSM (``src/agent/fsm.py::_tools_step``).
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Search query optimized for semantic + keyword match. "
            "Rephrase technical terms, expand abbreviations, and "
            "include likely document keywords. Don't include greetings "
            "or filler like 'please help me' — those dilute the match."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of top reranked chunks to return. Default 5; "
            "raise to 8-12 for broad questions ('summarize this "
            "section'), drop to 2-3 for pinpoint lookups."
        ),
    )


@tool("retrieve_docs", args_schema=RetrieveDocsArgs)
async def retrieve_docs(
    query: str,
    top_k: int = 5,
    # Phase 3 — v2.0.20. LangChain's ``InjectedToolArg`` marker from
    # ``langchain_core.tools.base`` (no LangGraph dependency) tells
    # the schema builder to hide this param from the LLM's tool
    # binding. The FSM's ``_tools_step`` supplies the real value at
    # call time via ``tool.ainvoke(args, thread_id=...)``.
    thread_id: Annotated[str, InjectedToolArg] = "",
) -> str:
    """Search the user's uploaded documents for content relevant to the question.

    Use this when the question likely needs project context, uploaded
    files, or specific factual recall from the thread's document set.

    Returns a JSON-encoded list of the top ``top_k`` reranked chunks.
    Each chunk carries the full Document (text + metadata:
    filename, chunk_id, page/section, rerank score). Empty ``[]``
    means no relevant chunks were found; the LLM should fall back to
    ``web_search`` or answer from its own knowledge with an explicit
    caveat.

    v2.0.29.6 (Phase 5) — ``top_k`` is now passed through as an
    explicit kwarg to :func:`src.agent.nodes.retrieve.retrieve_hybrid_async`.
    Pre-Phase-5 used a context-manager override of the module-level
    ``RETRIEVAL_DEFAULTS`` dict — that approach was a thread-safety
    hazard because two concurrent ``retrieve_docs`` calls on different
    threads would race on the shared mutation (one call's ``finally``
    would restore the dict, clobbering the other call's override).
    Per-call kwargs keep ``top_k`` as a local variable — no shared
    state to race on.
    """
    # Phase 3 — v2.0.20. ``thread_id`` is now a real function kwarg
    # supplied by the FSM's ``_tools_step`` (passed via
    # ``tool.ainvoke(args, thread_id=state["thread_id"])``). The
    # pre-Phase-3 hardcoded ``None`` placeholder is gone — see module
    # docstring for the data-leak root cause that fix addressed.

    # Build a minimal AgentState slice that ``retrieve_hybrid_async``
    # expects. We pass ``current_query`` (not ``original_query``) so
    # the rewrite layer's output is honored if a previous tool call
    # rewrote the query. ``thread_id`` carries the active thread so
    # hybrid search applies the right LanceDB filter — see
    # ``src.retrieval.hybrid_search._thread_filter``.
    state_slice = {
        "current_query": query,
        "original_query": query,
        "thread_id": thread_id,
    }

    # v2.0.29.6 (Phase 5) — pass top_k as an explicit kwarg; no
    # module-level dict mutation. See docstring above for the
    # thread-safety rationale (pre-Phase-5 race between concurrent
    # calls clobbered each other's top_k).
    try:
        delta = await retrieve_hybrid_async(state_slice, top_k=top_k)
    except Exception as exc:
        logger.exception(f"retrieve_docs tool failed: {exc}")
        return "[]"

    docs = delta.get("documents") or []
    # Mark docs with explicit ``source_kind="local"`` so the runner
    # can derive banner kinds uniformly with web-search docs.
    for d in docs:
        meta = d.metadata or {}
        meta.setdefault("source_kind", "local")
        d.metadata = meta

    # v2.0.2 — see module docstring + web_search.py for the wire-format
    # rationale. Round-tripping through ``_docs_to_json`` produces the
    # same JSON shape web_search uses, so ``react_generate`` only has
    # one extraction path.
    return _docs_to_json(docs)


# Eagerly bind the tool's ``invoke`` / ``ainvoke`` coroutines so
# LangChain's ``bind_tools`` recognises it without further config.
# ``@tool`` already does this on import; nothing else needed here.


__all__ = ["retrieve_docs", "RetrieveDocsArgs"]