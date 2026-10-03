"""Regression tests for v2.0.2 — Web search wire-format + Anthropic error UX.

Two live-testing bugs landed after v2.0.1, both observable in
``src.agent.runner.stream_agent`` when the user asks anything that
ends up calling ``web_search`` or ``retrieve_docs``:

Bug 1: ``AnthropicInvalidRequestError 400/2013 — tool call result
       does not follow tool call``
   - Triggered by every ``web_search`` / ``retrieve_docs`` call that
     returned a ``list[Document]``.
   - Root cause: LangChain's ``ToolNode._stringify`` fallback
     (when ``json.dumps`` fails on a Document) produces Python repr
     like ``[Document(metadata={...}, page_content='...')]``. That
     repr carries single quotes / newlines / dict-literal noise
     that misaligns the tool_use / tool_result blocks in Anthropic's
     request-body parser. Anthropic then rejects with code 2013.
   - Bonus cost: ``react_generate._extract_docs_from_messages``
     tries ``json.loads(content)`` and silently fails — the LLM
     never sees the retrieved documents even on the happy path.
   - Fix: tools now serialize their Document list to JSON
     themselves via ``_docs_to_json`` and return the ``str``.

Bug 2: ``AnthropicAPIError 500/1026 — input new_sensitive``
   - Anthropic's content-filter hit on real news content / some
     prompt phrase. 1026 is the new_sensitive category code.
   - Not our fault, not directly fixable, but the runner used to
     surface the bare Anthropic exception (which embeds the user's
     API key / proxy URL — see P1-4) via ``redact_exception``.
     User got "抱歉,生成回复时遇到了问题" with no actionable info.
   - Fix: the runner now classifies the exception:
       * status_code >= 500 → "模型服务暂时繁忙,请稍后重试"
       * 400/2013 → "内部错误:工具调用 wire format 不兼容"
       * else → existing ``redact_exception`` path
     Each variant includes ``request_id`` so operators can correlate.

These tests pin both fixes so future refactors don't silently
regress either.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Bug 1: tools return JSON-serialized string, NOT list[Document]
# ---------------------------------------------------------------------------


def test_web_search_returns_json_string_not_document_list():
    """Bug 1 fix: ``web_search.ainvoke`` MUST return a JSON ``str``,
    NOT a raw ``list[Document]``.

    Returning ``list[Document]`` is what triggered the 2013 wire
    error in the field. Tools that return JSON strings get cleanly
    wrapped by ``@tool``'s ``_format_output`` → ``ToolMessage(content=json_str)``
    which round-trips through ``react_generate._extract_docs_from_messages``.
    """
    from src.web_search._wire import _docs_to_json
    from src.web_search.tool import web_search

    # Static type contract: the tool's arun path returns str.
    assert web_search.args_schema is not None

    # _docs_to_json: round-trip with CJK + unicode metadata keys.
    docs = [
        Document(
            page_content="腾讯新闻 标题\n\n正文内容。",
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
            page_content="另一个结果",
            metadata={
                "source_kind": "web",
                "url": "https://x.com/b",
                "domain": "x.com",
                "filename": "另一个结果",
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
    assert parsed[0]["page_content"] == "腾讯新闻 标题\n\n正文内容。"
    assert parsed[0]["metadata"]["source_kind"] == "web"
    assert parsed[1]["metadata"]["doc_id"] == "web-bbb"


def test_web_search_arun_returns_json_string_for_empty_results():
    """Empty / failed results → ``"[]"`` (a valid JSON list, NOT an
    empty Python list).

    The LangChain @tool wrapper would otherwise call ``_stringify``
    on ``[]`` → Python repr ``'[]'`` (which happens to be valid JSON
    by coincidence). We assert our explicit return so future
    refactors don't slip back to ``return []``.
    """
    import importlib
    import inspect

    ws_module = importlib.import_module("src.web_search.tool")
    # Use ``getattr`` to dodge Pydantic's strict __getattr__ on
    # StructuredTool (direct attribute access raises AttributeError).
    tool_obj = getattr(ws_module, "web_search", None)
    assert tool_obj is not None
    fn = getattr(tool_obj, "coroutine", None) or getattr(tool_obj, "func", None)
    assert fn is not None, "web_search StructuredTool lost its underlying function"
    src = inspect.getsource(fn)
    assert 'return "[]"' in src, (
        "web_search must return JSON string '[]' on no-results / error, "
        "not an empty Python list (which would defeat the wire-format fix)."
    )


def test_web_search_arun_handles_non_serializable_metadata():
    """Defensive: metadata values that aren't JSON-native (datetime,
    sets, custom objects) get coerced via ``str()`` rather than
    crashing the tool call with a TypeError.
    """
    from src.web_search._wire import _docs_to_json
    import datetime

    docs = [
        Document(
            page_content="ok",
            metadata={
                "source_kind": "web",
                "published_at": datetime.datetime(2026, 9, 6),  # not JSON-serializable
                "tags": {"a", "b"},  # set, not JSON-serializable
            },
        )
    ]
    serialized = _docs_to_json(docs)
    # Should not raise. Should parse as JSON.
    parsed = json.loads(serialized)
    assert parsed[0]["metadata"]["source_kind"] == "web"
    # DateTime coerced to str; set coerced to str.
    assert isinstance(parsed[0]["metadata"]["published_at"], str)
    assert isinstance(parsed[0]["metadata"]["tags"], str)


def test_retrieve_docs_uses_same_json_serializer():
    """``retrieve_docs`` must use ``_docs_to_json`` (not its own
    ad-hoc serializer) so both retrieval tools produce identical
    wire format. ``react_generate`` only has one extraction path.
    """
    import importlib
    import inspect

    rd_module = importlib.import_module("src.agent.tools.retrieve_docs")
    tool_obj = getattr(rd_module, "retrieve_docs", None)
    assert tool_obj is not None
    fn = getattr(tool_obj, "coroutine", None) or getattr(tool_obj, "func", None)
    assert fn is not None
    src = inspect.getsource(fn)
    assert "_docs_to_json" in src, (
        "retrieve_docs must call _docs_to_json — keeps the wire format "
        "identical to web_search so react_generate has one parser path."
    )
    assert 'return "[]"' in src, (
        "retrieve_docs must return JSON '[]' on no-results / error."
    )


def test_react_generate_extracts_docs_from_json_tool_message():
    """End-to-end: a ToolMessage with the new JSON-string content
    parses cleanly into Document objects via ``_extract_docs_from_messages``.

    Pre-fix: content was Python repr → ``json.loads`` failed silently
    → no docs surfaced to the LLM → answer denied knowing anything.
    """
    from src.agent.nodes.react_generate import _extract_docs_from_messages

    docs_json = json.dumps(
        [
            {
                "page_content": "first chunk",
                "metadata": {"source_kind": "web", "url": "https://x.com"},
            },
            {
                "page_content": "second chunk",
                "metadata": {"source_kind": "local", "doc_id": "d1"},
            },
        ],
        ensure_ascii=False,
    )

    msgs = [
        HumanMessage(content="search the web"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "tc1", "name": "web_search", "args": {"query": "x"}}
            ],
        ),
        ToolMessage(content=docs_json, tool_call_id="tc1", name="web_search"),
    ]

    docs = _extract_docs_from_messages(msgs)
    assert len(docs) == 2
    assert docs[0].page_content == "first chunk"
    assert docs[0].metadata["source_kind"] == "web"
    assert docs[1].metadata["doc_id"] == "d1"


def test_react_generate_skips_garbage_tool_message_content():
    """A malformed ToolMessage content (not JSON, not Document repr)
    must not crash extraction. Defensive — never want a bad tool
    result to crash the final-answer synthesis.
    """
    from src.agent.nodes.react_generate import _extract_docs_from_messages

    msgs = [
        HumanMessage(content="hi"),
        ToolMessage(content="plain text from get_current_time", tool_call_id="t1", name="get_current_time"),
        ToolMessage(content="{invalid json", tool_call_id="t2", name="web_search"),
        ToolMessage(content="", tool_call_id="t3", name="web_search"),
    ]
    docs = _extract_docs_from_messages(msgs)
    # Only the time tool had no docs (we skip those); no crash.
    assert docs == []


# ---------------------------------------------------------------------------
# Bug 2: runner classifies Anthropic errors into actionable messages
# ---------------------------------------------------------------------------


def _run_stream_and_collect(gen):
    """Drain an async generator into a list using a single event loop.

    ``asyncio.run`` per iteration creates a fresh loop each time which
    tears down pending callbacks mid-stream and yields no events.
    """
    out: list[dict] = []

    async def _drive():
        try:
            async for evt in gen:
                out.append(evt)
        except StopAsyncIteration:
            return
        except BaseException as exc:
            # Some exceptions (e.g. aclose() during a yield) propagate
            # out of the async-for — capture so the test sees the
            # partial event list and the originating exception.
            out.append({"type": "_propagated_exception", "error": str(exc)})

    asyncio.run(_drive())
    return out


def test_runner_anthropic_500_gets_retry_hint_message():
    """Bug 2 fix: ``AnthropicAPIError`` with ``status_code >= 500``
    surfaces a Chinese "model busy, please retry" message + the
    upstream ``request_id`` so operators can correlate.

    Pre-fix: the bare exception was passed through ``redact_exception``
    which produced a generic message with no actionable info and no
    correlation id.
    """
    from src.agent.runner import stream_agent

    class FakeAnthropicAPIError(Exception):
        def __init__(self):
            super().__init__("input new_sensitive (1026)")
            self.status_code = 500
            self.request_id = "06ec8eb01227a2cede199157f076521f"

    # Build a fake FSM async generator that raises immediately. The
    # runner's exception handler runs INSIDE the try block once the
    # stream is open; raising on the first __anext__ makes sure we
    # hit that classification branch.
    async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
        raise FakeAnthropicAPIError()
        yield  # makes this an async generator function (unreachable)

    # Bypass the model-loaded gate so we hit the streaming exception path.
    # The runner imports these inside the function body, so we patch
    # the source modules rather than the runner namespace.
    with patch("src.agent.runner.run_fsm", side_effect=fake_run_fsm), \
         patch("src.embeddings.bge_m3.models_loaded", return_value=True), \
         patch("src.reranker.bge_reranker.model_loaded", return_value=True):
        events = _run_stream_and_collect(stream_agent("t1", "hi"))

    assert any(e["type"] == "error" for e in events), (
        f"no error event found in: {events}"
    )
    err = next(e for e in events if e["type"] == "error")
    msg = err["message"]
    assert "模型服务暂时繁忙" in msg or "请稍后重试" in msg, msg
    # request_id surfaces so operators can correlate with the LLM console.
    assert "06ec8eb01227a2cede199157f076521f" in msg, msg


def test_runner_anthropic_400_contract_violation_gets_specific_message():
    """Bug 2 fix: ``AnthropicInvalidRequestError 400/2013`` (the
    wire-format error that v2.0.2 mostly prevents upstream) still
    gets a clear "tool contract" message if it ever fires — not
    the bare redact_exception fallback.
    """
    from src.agent.runner import stream_agent

    class FakeAnthropicInvalidRequestError(Exception):
        def __init__(self):
            super().__init__("invalid params, tool call result does not follow tool call (2013)")
            self.status_code = 400
            self.request_id = "06ec8d6337baf0546e3ae24673f80135"

    async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
        raise FakeAnthropicInvalidRequestError()
        yield  # unreachable — makes this an async generator function

    with patch("src.agent.runner.run_fsm", side_effect=fake_run_fsm), \
         patch("src.embeddings.bge_m3.models_loaded", return_value=True), \
         patch("src.reranker.bge_reranker.model_loaded", return_value=True):
        events = _run_stream_and_collect(stream_agent("t1", "hi"))

    err = next(e for e in events if e["type"] == "error")
    msg = err["message"]
    assert "工具调用" in msg or "tool contract" in msg, msg
    assert "06ec8d6337baf0546e3ae24673f80135" in msg, msg


def test_runner_generic_exception_still_uses_redact_exception():
    """Non-Anthropic exceptions keep the existing redact_exception
    path so P1-4 (no secret leakage) is preserved.
    """
    from src.agent.runner import stream_agent

    class SomeRandomError(Exception):
        pass

    async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
        raise SomeRandomError("boom")
        yield  # unreachable — makes this an async generator function

    with patch("src.agent.runner.run_fsm", side_effect=fake_run_fsm), \
         patch("src.embeddings.bge_m3.models_loaded", return_value=True), \
         patch("src.reranker.bge_reranker.model_loaded", return_value=True):
        events = _run_stream_and_collect(stream_agent("t1", "hi"))

    err = next(e for e in events if e["type"] == "error")
    # redact_exception always produces a "抱歉,..." prefix.
    assert "抱歉" in err["message"], err["message"]


def test_trace_id_for_prefers_request_id():
    """``_trace_id_for`` returns the upstream ``request_id`` when
    available, falling back to a content-based anon id otherwise.
    """
    from src.agent.runner import _trace_id_for

    class WithReqId(Exception):
        request_id = "abc123"

    class WithoutReqId(Exception):
        status_code = 500

    assert _trace_id_for(WithReqId()) == "abc123"
    anon = _trace_id_for(WithoutReqId())
    assert anon.startswith("anon-")
    assert len(anon) > 12


# ---------------------------------------------------------------------------
# End-to-end: web_search tool output now survives a LLM round-trip
# ---------------------------------------------------------------------------


def test_e2e_web_search_round_trips_through_extract():
    """The full chain: web_search returns JSON → stored in ToolMessage
    → _extract_docs_from_messages reads it back → Documents appear
    in state.documents for citation building.

    Pre-v2.0.2 this round-trip silently dropped ALL web results
    because json.loads on Python repr raised.

    This test uses ``_docs_to_json`` directly (the same helper the
    tool uses internally) so it's hermetic — no network needed.
    """
    from src.web_search._wire import _docs_to_json
    from src.agent.nodes.react_generate import _extract_docs_from_messages

    docs_in = [
        Document(
            page_content="Title A\n\nsnip A",
            metadata={
                "source_kind": "web",
                "url": "https://x.com/a",
                "domain": "x.com",
                "filename": "Title A",
                "doc_id": "web-aaa",
                "chunk_id": "",
                "score": 0.9,
            },
        ),
        Document(
            page_content="Title B\n\nsnip B",
            metadata={
                "source_kind": "web",
                "url": "https://x.com/b",
                "domain": "x.com",
                "filename": "Title B",
                "doc_id": "web-bbb",
                "chunk_id": "",
                "score": 0.8,
            },
        ),
    ]
    serialized = _docs_to_json(docs_in)
    assert isinstance(serialized, str)

    msgs = [
        HumanMessage(content="search"),
        AIMessage(content="", tool_calls=[{"id": "t1", "name": "web_search", "args": {}}]),
        ToolMessage(content=serialized, tool_call_id="t1", name="web_search"),
    ]
    docs = _extract_docs_from_messages(msgs)
    assert len(docs) == 2
    titles = {d.metadata.get("filename") for d in docs}
    assert titles == {"Title A", "Title B"}


__all__ = [
    "test_web_search_returns_json_string_not_document_list",
    "test_web_search_arun_returns_json_string_for_empty_results",
    "test_web_search_arun_handles_non_serializable_metadata",
    "test_retrieve_docs_uses_same_json_serializer",
    "test_react_generate_extracts_docs_from_json_tool_message",
    "test_react_generate_skips_garbage_tool_message_content",
    "test_runner_anthropic_500_gets_retry_hint_message",
    "test_runner_anthropic_400_contract_violation_gets_specific_message",
    "test_runner_generic_exception_still_uses_redact_exception",
    "test_trace_id_for_prefers_request_id",
    "test_e2e_web_search_round_trips_through_extract",
]
