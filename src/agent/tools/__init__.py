"""Tool registry for the ReAct agent (v2.0).

Three tools are exposed to the LLM via LangChain's ``bind_tools``:

* ``retrieve_docs`` — hybrid retrieval over the thread's uploaded
  documents (BM25 + BGE-M3 dense + rerank). Internal handler calls
  the existing :func:`src.agent.nodes.retrieve.retrieve_hybrid_async`
  pipeline.
* ``web_search`` — DuckDuckGo / Bing fallback with v1.1.11 page-fetch
  enrichment. Returns Documents with ``source_kind="web"``.
* ``get_current_time`` — returns the current wall-clock in the
  requested IANA timezone (ISO 8601 + human-readable). Closes the
  "hallucinated date" failure mode for "今天" / "昨天" / "now" /
  "latest" prompts.

Why we use LangChain ``@tool`` (with Pydantic ``args_schema``)
instead of the deprecated ``create_react_agent`` factory:

* LangGraph >=0.2 dropped ``create_react_agent``; the official
  replacement is manual ``StateGraph`` + ``ToolNode`` + ``tools_condition``.
  We get full control over the loop, the messages we inject into the
  ReAct LLM, and the deduplication of tool calls in the streaming
  layer.
* ``args_schema`` (Pydantic) gives us runtime validation — a tool call
  with ``top_k=42`` fails the Pydantic schema (max=20) before any
  retrieval cost. Better than the alternative of waiting until the LLM
  sends bad arguments and then surfacing a generic exception to the
  graph.
* Returning ``list[Document]`` (or ``str`` for time) is what
  ``ToolNode`` natively serializes into ``ToolMessage.content`` — no
  custom encoder, no stringification layer.

Why we ship a tool-call event log on the wire (not just the bare
``ToolMessage`` content):

* Users want to see "🔍 called retrieve_docs('Q3 营收') → got 5 docs"
  in the UI as a foldable card, not just the final answer. The runner
  emits ``tool_call_start`` / ``tool_call_end`` events by reading
  ``ToolMessage`` updates off the LangGraph stream and the
  ``step_start`` / ``step_end`` markers emitted by ``react_agent``.
* The tool itself is decoupled from the wire — it returns Documents,
  not events. The wire shaping lives in
  :mod:`src.agent.runner` / :mod:`src.agent.events`, matching the
  precedent set by the existing token / reasoning / retrieval event
  pipeline.

v2.0.18 — ``web_search`` moved out of ``src/agent/tools/web_search.py``
into ``src/web_search/tool.py`` (Phase 2 Web search simplification).
The import below is the bridge: LangGraph's ``ALL_TOOLS`` still
resolves ``web_search`` here, but the implementation lives next to
the dispatcher it wraps.

v2.0.22 (Item 7 Step 8) — :data:`TOOL_METADATA` declares per-tool
wire events the FSM should yield AFTER each ``tool_call_start``.
This is the ``P0-B3`` fix: pre-Step-8 ``react_agent`` had a
hardcoded ``if tc_name == "web_search": yield web_search event``
that violated the "FSM is tool-name-agnostic" invariant. Adding a
new search-class tool (e.g. ``web_search_tier2``) used to require
editing ``react_agent.py``; now it's a one-line ``TOOL_METADATA``
entry. ``emits: []`` means "no extra wire events beyond the
default ``tool_call_start`` / ``tool_call_end`` (which every tool
gets automatically)".
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.tools import BaseTool

from .get_current_time import get_current_time
from .retrieve_docs import retrieve_docs
# v2.0.18 — web_search tool moved to src/web_search/tool.py (the only
# LangChain-coupled piece in the Web search package lives at the edge,
# next to the dispatcher + wire serializer it wraps).
from src.web_search.tool import web_search


ALL_TOOLS: list[BaseTool] = [
    retrieve_docs,
    web_search,
    get_current_time,
]


# v2.0.22 (Item 7 Step 8) — tool metadata drives wire-event emission.
# ``emits`` is a list of extra wire event kinds the ``react_agent``
# node yields after the default ``tool_call_start``. The default
# ``tool_call_start`` / ``tool_call_end`` events fire for EVERY tool
# call (those are not in ``emits`` — they're unconditional FSM
# output, not metadata-driven).
#
# Why ``web_search`` emits ``"web_search"``: v1.1.8 wire contract
# — the frontend uses the ``attempted=True`` boolean as a "yes, the
# LLM decided to search" signal so the search pill renders even if
# the call later fails. ``retrieve_docs`` and ``get_current_time``
# don't have such a pill — they only show up via ``tool_call_start``
# / ``tool_call_end``.
class ToolMetadata(TypedDict, total=False):
    """Per-tool wire-event metadata. See :data:`TOOL_METADATA`."""

    emits: list[str]


TOOL_METADATA: dict[str, ToolMetadata] = {
    # Search-class tool — emits ``"web_search"`` so the UI shows
    # the pill immediately.
    "web_search": ToolMetadata(emits=["web_search"]),
    # Retrieval — no extra event beyond the default
    # ``tool_call_start`` / ``tool_call_end``.
    "retrieve_docs": ToolMetadata(emits=[]),
    # Time tool — same; the actual answer surfaces via ``tool_call_end``.
    "get_current_time": ToolMetadata(emits=[]),
}


__all__ = [
    "ALL_TOOLS",
    "TOOL_METADATA",
    "ToolMetadata",
    "retrieve_docs",
    "web_search",
    "get_current_time",
]