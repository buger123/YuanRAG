"""``web_search`` LangChain ``@tool`` wrapper — the thin edge code
that adapts the dispatcher to the LangGraph ToolNode contract.

v2.0.18 — extracted from ``src/agent/tools/web_search.py`` where
``web_search`` previously lived alongside 200 LOC of dispatch +
fetch-enrich logic. Now:

* The dispatcher + enrichment lives in ``src/web_search/__init__.py``
  (``search_web(query, max_results, *, enrich, preferred_domains)``
  returns ``list[Document]``).
* The wire-format serializer lives in ``src/web_search/_wire.py``
  (``_docs_to_json``).
* This file: just the ``@tool`` decorator + Pydantic ``args_schema``
  + a 12-LOC body that calls ``search_web`` and returns JSON.

Why the move
------------
``web_search`` is now the only LangChain-coupled piece in the
package. By isolating it here we make the Phase 3 migration (drop
LangGraph, replace with a pure-async state machine) easier: Phase 3
will replace the LangChain ``@tool`` decorator with a direct
``ToolRegistry`` call into ``search_web()``, and the rest of the
package doesn't need to change.

Why we return JSON ``str`` (not ``list[Document]``)
---------------------------------------------------
v2.0.2 contract — ``ToolMessage.content`` MUST be a JSON-serializable
string. LangChain's ``ToolNode._stringify`` falls back to
``str(content)`` which produces Python ``repr`` for non-JSON objects;
``Document`` is one such object. Returning ``str`` ourselves keeps the
wire format stable and survives Anthropic's wire validator
(code 2013 "tool call result does not follow tool call").

The 12-LOC body is deliberate
-----------------------------
This file's only job is to wire ``@tool`` + ``args_schema`` to
``search_web``. Anything more (enrichment logic, domain hints
extraction, page-fetch fan-out) lives in ``search_web`` itself —
adding it here would duplicate ``search_web``'s contract.
"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.web_search import search_web
from src.web_search._wire import _docs_to_json


class WebSearchArgs(BaseModel):
    """Pydantic schema for ``web_search`` arguments."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Web search query. Be specific: include names, dates, "
            "publisher names, keywords. Example: '腾讯新闻 2026-09-06 "
            "今天 头条'. Don't include filler like 'please help me'."
        ),
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Maximum number of search results to fetch. Default 5 "
            "covers most questions; raise to 8-10 for broad surveys "
            "('latest news on X this week')."
        ),
    )


@tool("web_search", args_schema=WebSearchArgs)
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web for current events, news, prices, weather, recent info.

    Use this when the question asks for today's news, latest
    information, stock prices, weather, scores, or anything that
    needs real-time data the uploaded documents can't supply.

    Returns a JSON-encoded list of search results. Each result has
    ``page_content`` (title + page body / snippet) and ``metadata``
    (``source_kind="web"``, ``url``, ``domain``, ``filename``).
    Empty ``[]`` means nothing usable was found. The frontend renders
    result entries as clickable external links rather than static
    citation chips.

    For best results:
      * include a publisher name if the user asked about one
        ('腾讯新闻', 'BBC', 'Reuters');
      * include a date when the question is time-anchored
        ('2026-09-06', '今天', 'this week');
      * include the language the source is in (Chinese sources
        surface better when the query is in Chinese).

    Limitations:
      * Search engines throttle — 30 req/min/IP is the rough ceiling.
        ``RAG_WEB_SEARCH_ENGINES=bing,ddg`` lets you chain engines.
      * Snippet-only results (when page-fetch fails) may not contain
        enough detail for the LLM to answer precisely — the tool
        warns when enrichment fails.

    v2.0.18 — body slimmed: dispatcher + enrichment now live in
    ``src.web_search.search_web``. This wrapper is a thin adapter.
    """
    try:
        docs = await search_web(query, max_results=max_results, enrich=True)
    except Exception as exc:
        logger.exception(f"web_search tool failed: {exc}")
        return "[]"

    if not docs:
        return "[]"
    return _docs_to_json(docs)


__all__ = ["web_search", "WebSearchArgs"]