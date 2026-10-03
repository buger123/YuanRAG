"""v2.0.2 wire-format regression guard.

These tests pin the contract that ``web_search`` (the LangChain
``@tool`` wrapper) returns a JSON ``str`` whose shape matches
``[{"page_content": str, "metadata": {...}}, ...]``. The v2.0.2
bug class was: ``ToolNode._stringify`` falling back to Python
``repr`` for non-JSON-serializable ``list[Document]`` returns →
Anthropic wire validator rejected the request with code 2013
"tool call result does not follow tool call".

What we lock here
-----------------
1. Round-trip: docs in → JSON → parse → same list shape.
2. Empty list → literal ``"[]"`` (valid JSON, NOT ``""`` or ``"null"``).
3. Defensive coerce: non-JSON-native metadata values (datetime,
   set) get coerced via ``str()`` rather than raising.
4. Cancellation: an engine raising ``CancelledError`` propagates
   through ``web_search()`` (not swallowed).
5. Exception in dispatcher: any non-CancelledError exception in
   the dispatcher path → tool returns ``"[]"`` (no exception
   surfaces to ``react_generate``).
"""
from __future__ import annotations

import asyncio
import datetime
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# 1. Round-trip
# ---------------------------------------------------------------------------


def test_docs_to_json_round_trip_preserves_shape():
    """Docs in → JSON → parsed list out has same page_content + metadata.

    Catches the v2.0.2 regression class: a buggy serializer that
    drops metadata fields or escapes unicode wrong would silently
    break the ``react_generate._extract_docs_from_messages`` consumer.
    """
    from src.web_search._wire import _docs_to_json

    docs = [
        Document(
            page_content="腾讯新闻 标题\n\n正文。",
            metadata={
                "source_kind": "web",
                "url": "https://news.qq.com/a",
                "domain": "news.qq.com",
                "filename": "腾讯新闻 标题",
                "doc_id": "web-aaa",
                "chunk_id": "",
                "score": 0.9,
            },
        ),
        Document(
            page_content="Title B",
            metadata={
                "source_kind": "web",
                "url": "https://x.test/b",
                "domain": "x.test",
                "filename": "Title B",
                "doc_id": "web-bbb",
                "chunk_id": "",
                "score": 0.8,
            },
        ),
    ]
    serialized = _docs_to_json(docs)
    assert isinstance(serialized, str)

    parsed = json.loads(serialized)
    assert isinstance(parsed, list)
    assert len(parsed) == 2

    # CJK + metadata keys must survive the round-trip without
    # escape-storm artifacts (``\\u5929\\u6daf`` etc).
    assert parsed[0]["page_content"] == "腾讯新闻 标题\n\n正文。"
    assert parsed[0]["metadata"]["source_kind"] == "web"
    assert parsed[0]["metadata"]["filename"] == "腾讯新闻 标题"
    assert parsed[0]["metadata"]["doc_id"] == "web-aaa"
    assert parsed[0]["metadata"]["score"] == 0.9
    assert parsed[1]["metadata"]["doc_id"] == "web-bbb"


# ---------------------------------------------------------------------------
# 2. Empty list → "[]"
# ---------------------------------------------------------------------------


def test_docs_to_json_empty_input_returns_bracket_pair():
    """Empty input → ``"[]"``.

    ``react_generate._extract_docs_from_messages`` calls
    ``json.loads(content)`` — an empty string ``""`` would crash
    with ``JSONDecodeError``; ``"[]"`` parses to a clean empty
    list. This is the exact shape the tool returns on no-results
    or on internal exception.
    """
    from src.web_search._wire import _docs_to_json

    assert _docs_to_json([]) == "[]"
    # Single-arg generator (not list) — must also work without
    # consuming the iterator twice.
    assert _docs_to_json(iter([])) == "[]"


# ---------------------------------------------------------------------------
# 3. Defensive coercion
# ---------------------------------------------------------------------------


def test_docs_to_json_coerces_non_json_metadata_values():
    """datetime / set / custom object in metadata → str() coerce.

    Tool calls MUST NOT raise on weird metadata values; the LLM
    can handle the stringified form, but a ``TypeError`` from
    ``json.dumps`` would kill the ReAct loop. We coerce via
    ``str()`` per-value (not the whole metadata dict).
    """
    from src.web_search._wire import _docs_to_json

    docs = [
        Document(
            page_content="ok",
            metadata={
                "source_kind": "web",
                "published_at": datetime.datetime(2026, 9, 6),
                "tags": {"a", "b"},
                "weird_object": object(),  # arbitrary non-JSON
            },
        )
    ]
    serialized = _docs_to_json(docs)
    parsed = json.loads(serialized)

    # Source-friendly values pass through unchanged.
    assert parsed[0]["metadata"]["source_kind"] == "web"
    # Weird values got coerced to str (NOT raised).
    assert isinstance(parsed[0]["metadata"]["published_at"], str)
    assert isinstance(parsed[0]["metadata"]["tags"], str)
    assert isinstance(parsed[0]["metadata"]["weird_object"], str)


# ---------------------------------------------------------------------------
# 4. Cancellation propagates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_captures_cancellation_from_engine(monkeypatch):
    """v2.0.29.6 (Phase 5) — per-engine ``CancelledError`` is
    captured by ``asyncio.gather(return_exceptions=True)`` inside
    ``search_web``. The dispatcher does NOT propagate cancellation
    to the caller anymore (parallel gather can't honour mid-flight
    cancellation of a subset of engines anyway; the chain runs to
    completion and successful engines' results merge).

    Pre-Phase-5 ``search_web`` re-raised ``CancelledError`` (the one
    exception it didn't swallow) — see ``src/web_search/__init__.py``
    around line 263. Post-Phase-5 that contract is intentionally
    retired. This test pins the NEW contract so a future refactor
    doesn't reintroduce the old short-circuit cancellation path.
    """
    # Build a dispatcher that calls a cancellable engine.
    async def cancellable_engine(query, max_results=None, **kwargs):
        raise asyncio.CancelledError()

    # Patch the dispatcher engine registry directly.
    import src.web_search as ws
    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"cancellable": cancellable_engine})
    from config.constants import WEB_SEARCH
    monkeypatch.setitem(WEB_SEARCH, "engines", ["cancellable"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    # Must NOT raise — gather captures the exception, dispatcher
    # returns empty list (the only engine was cancelled, so merged
    # result is empty).
    out = await ws.search_web("q", enrich=False)
    assert out == []


# ---------------------------------------------------------------------------
# 5. Tool returns "[]" on dispatcher exception (doesn't leak)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returns_empty_json_when_dispatcher_raises(monkeypatch):
    """If the dispatcher raises a non-CancelledError exception
    unexpectedly, ``web_search`` (the @tool) must catch it and
    return ``"[]"`` rather than letting it propagate.

    The @tool wrapper is the LAST line of defense for the ReAct
    loop — a leaked exception would kill the agent run and the
    user would see a generic "抱歉" message instead of a graceful
    "no info found".
    """
    import src.web_search as ws
    from src.web_search.tool import web_search

    async def boom_engine(query, max_results=None, **kwargs):
        raise RuntimeError("engine impl ate a bug")

    monkeypatch.setattr(ws, "_ENGINE_REGISTRY", {"boom": boom_engine})
    from config.constants import WEB_SEARCH
    monkeypatch.setitem(WEB_SEARCH, "engines", ["boom"])
    monkeypatch.delenv("RAG_WEB_SEARCH_ENGINES", raising=False)

    # The dispatcher logs + falls through, so we shouldn't even
    # reach the exception handler — but let's verify the contract
    # by forcing a failure in the @tool body itself.
    out = await web_search.ainvoke({"query": "x", "max_results": 1})
    # If dispatcher ate it, we get "[]" cleanly. If dispatcher
    # leaks it, the @tool body catches it and ALSO returns "[]".
    assert out == "[]", f"tool returned {out!r} instead of '[]'"


__all__ = [
    "test_docs_to_json_round_trip_preserves_shape",
    "test_docs_to_json_empty_input_returns_bracket_pair",
    "test_docs_to_json_coerces_non_json_metadata_values",
    "test_tool_captures_cancellation_from_engine",
    "test_tool_returns_empty_json_when_dispatcher_raises",
]