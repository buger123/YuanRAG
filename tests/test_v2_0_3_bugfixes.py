"""Regression tests for v2.0.3 — Three chat-rendering fixes after v2.0.2.

Three live-testing bugs were observed after the v2.0.2 wire-format fix,
all of them front-end / persistence gaps the v2.0.2 unit suite didn't
cover:

Bug 1: ``ToolMessage → role=assistant`` mislabel on history reload.
   - Triggered every time a ReAct turn fired a tool call (which is
     basically every doc + web hybrid question).
   - Symptom: after a ReAct turn, refreshing the page or reopening
     the conversation showed the tool's raw JSON output as a giant
     user-visible assistant message:
         [{"page_content": "..."}]
   - Root cause: ``src.api.routes.sessions._msg_role`` had a
     class-name fallback that returned ``"assistant"`` for any
     ``tool``-named class. ``ToolMessage`` matched. ``ToolMessage``
     content was serialized as ``MessageRecord(role="assistant",
     content=<JSON blob>)``.
   - Fix:
       * ``_msg_role`` returns ``"tool"`` for ``ToolMessage``
         (via isinstance check, before the fallback).
       * The serializer skips any record whose role is ``"tool"`` —
         no ``MessageRecord`` is emitted for tool messages.
   - Tool content is reconstructed into Document objects on the next
     turn by ``react_generate._extract_docs_from_messages`` (already
     wired in v2.0.2), so no information is lost.

Bug 2: Tool-call events were never persisted on the AIMessage.
   - The runner emits ``tool_call_start`` / ``tool_call_end`` events
     live so the front-end can render ``ToolCallCard`` components
     during streaming, but it never stamps those calls onto the
     ``AIMessage.additional_kwargs["tool_calls"]`` field. So a page
     refresh after a ReAct turn lost every tool call — the chat pane
     showed only the final assistant prose, with no "retrieve_docs"
     / "web_search" cards above it.
   - Fix: ``src.agent.nodes.react_generate._build_tool_call_log``
     walks the persisted ``state["messages"]`` list, pairs each
     ``AIMessage.tool_calls`` with the following ``ToolMessage``,
     and stamps the resulting log onto the final AIMessage's
     ``additional_kwargs["tool_calls"]`` before it lands in the
     checkpointer.

Bug 3: User-side timestamp invisible on the blue chat bubble.
   - The ``.message-timestamp`` rule used
     ``color: var(--text-dim, #888); opacity: 0.75`` — that's roughly
     ``#a8b5c8`` (very dim gray-blue) on a ``#2563eb`` gradient
     background. Contrast ratio well below WCAG AA. Completely
     illegible.
   - Fix: ``src.frontend/src/styles.css`` adds a
     ``.message.user .message-timestamp { color: rgba(255,255,255,0.85); }``
     override so the timestamp is white at 85% opacity on the blue
     user bubble.

These tests pin the three fixes so a future refactor of the
session-serialization layer, the runner tool-call pipeline, or the
chat-pane CSS can't silently regress any of them.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ---------------------------------------------------------------------------
# Bug 1: ToolMessage skip in _serialize_messages
# ---------------------------------------------------------------------------


def test_msg_role_tool_message_returns_tool_string():
    """Bug 1 fix: ``_msg_role(ToolMessage(...))`` returns ``"tool"``,
    not the pre-v2.0.3 ``"assistant"`` fallback."""
    from src.api.routes.sessions import _msg_role

    m = ToolMessage(content="some tool result", tool_call_id="t1", name="web_search")
    role = _msg_role(m)
    assert role == "tool", (
        f"ToolMessage must classify as 'tool' so the serializer skips it; "
        f"got {role!r} (pre-v2.0.3 returned 'assistant' for any tool-named class)"
    )


def test_msg_role_distinguishes_all_message_classes():
    """Bug 1 fix: every standard LangChain message class maps to the
    expected UI role. A regression that re-introduces the class-name
    fallback for ToolMessage would fail here."""
    from src.api.routes.sessions import _msg_role

    h = HumanMessage(content="hi")
    a = AIMessage(content="hello")
    s = HumanMessage(content="system-hidden-test")  # HumanMessage w/ system-like content
    t = ToolMessage(content="x", tool_call_id="t1", name="web_search")
    # Replace the HumanMessage above with a real SystemMessage
    from langchain_core.messages import SystemMessage
    s = SystemMessage(content="you are a helpful assistant")

    assert _msg_role(h) == "user"
    assert _msg_role(a) == "assistant"
    assert _msg_role(s) == "system"
    assert _msg_role(t) == "tool"


def test_serialize_messages_skips_tool_messages():
    """Bug 1 fix: ``_serialize_messages`` MUST NOT emit a MessageRecord
    for any ToolMessage. Pre-v2.0.3 the role was 'assistant' so a JSON
    blob (the tool result) was rendered as an assistant message on
    reload."""
    from src.api.routes.sessions import _serialize_messages

    msgs = [
        HumanMessage(content="find me docs"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "retrieve_docs", "args": {"query": "q"}}
            ],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(
            content='[{"page_content": "secret doc text", "metadata": {"source_kind": "local"}}]',
            tool_call_id="call_1",
            name="retrieve_docs",
        ),
        AIMessage(
            content="based on the docs I found...",
            additional_kwargs={
                "sources": [
                    {
                        "index": 1,
                        "chunk_id": "abc",
                        "doc_id": "d1",
                        "filename": "x.txt",
                        "source_kind": "local",
                        "text": "secret doc text",
                        "score": 0.9,
                    }
                ],
                "source_kinds": ["local"],
                "tool_calls": [
                    {"id": "call_1", "name": "retrieve_docs", "args": {"query": "q"}}
                ],
                "created_at": "2026-09-07T00:00:01+00:00",
            },
        ),
    ]
    records = _serialize_messages(msgs)
    # v2.0.24 (Item 8 P0-F4) — the contract change: intermediate
    # ReAct AIMessages now SURVIVE the serializer (tool_calls cards
    # synthesized from the native ``AIMessage.tool_calls`` attr), so
    # the user sees the full ReAct trace on history reload, not just
    # the final answer. The v2.0.3 ToolMessage skip contract is
    # preserved (ToolMessages still don't become assistant messages).
    # Expected: human (1) + intermediate assistant with tool_calls (1)
    # + final assistant with the rich additional_kwargs log (1) = 3.
    assert len(records) == 3, (
        f"expected 3 records (human + intermediate + final assistant), "
        f"got {len(records)}: {[r.role for r in records]}"
    )
    roles = [r.role for r in records]
    assert "tool" not in roles, (
        f"Serializer leaked role=tool — frontend would render the tool "
        f"result as a chat turn. Got roles: {roles}"
    )
    # CRITICAL: no message record's content starts with "[{" — the JSON
    # blob from the ToolMessage must not leak through as an assistant.
    for r in records:
        c = (r.content or "").lstrip()
        assert not c.startswith("[{"), (
            f"Found a JSON-array assistant message (the ToolMessage leak): "
            f"role={r.role!r} content[:60]={c[:60]!r}"
        )
    # v2.0.24 — the intermediate AIMessage should expose its tool call
    # (from the native ``tool_calls`` attr, since its additional_kwargs
    # didn't carry one). The final AIMessage keeps its richer log.
    intermediate, final = records[1], records[2]
    assert intermediate.role == "assistant"
    assert intermediate.tool_calls is not None
    assert len(intermediate.tool_calls) == 1
    assert intermediate.tool_calls[0]["name"] == "retrieve_docs"
    # Final AIMessage's log is the rich additional_kwargs entry (with
    # sources + created_at stamps that the intermediate round lacks).
    assert final.tool_calls is not None
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0]["name"] == "retrieve_docs"
    # Final has sources attached (from its additional_kwargs), the
    # intermediate doesn't.
    assert final.sources is not None and len(final.sources) >= 1
    assert intermediate.sources is None


def test_serialize_messages_multiple_tool_messages_all_skipped():
    """Bug 1 fix: multi-tool-call turns also drop every ToolMessage.
    A real ReAct turn fires 2-5 tool calls; the serializer must skip
    every one."""
    from src.api.routes.sessions import _serialize_messages

    msgs = [
        HumanMessage(content="hybrid q"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "retrieve_docs", "args": {}},
                {"id": "c2", "name": "web_search", "args": {}},
            ],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(content='[{"page_content":"a"}]', tool_call_id="c1", name="retrieve_docs"),
        ToolMessage(content='[{"page_content":"b"}]', tool_call_id="c2", name="web_search"),
        AIMessage(
            content="synthesized answer",
            additional_kwargs={
                "tool_calls": [
                    {"id": "c1", "name": "retrieve_docs", "args": {}, "ok": True, "step": 1},
                    {"id": "c2", "name": "web_search", "args": {}, "ok": True, "step": 1},
                ],
                "created_at": "2026-09-07T00:00:01+00:00",
            },
        ),
    ]
    records = _serialize_messages(msgs)
    # v2.0.24 (Item 8 P0-F4) — intermediate AIMessage now SURVIVES
    # with tool_calls cards synthesized from the native attr. The two
    # ToolMessages still get dropped (v2.0.3 contract preserved).
    # Expected: human (1) + intermediate assistant with 2 tool cards (1)
    # + final assistant with the rich additional_kwargs log (1) = 3.
    assert len(records) == 3, (
        f"expected 3 records (human + intermediate + final assistant), "
        f"got {len(records)}: {[r.role for r in records]}"
    )
    assert all(r.role != "tool" for r in records)
    # Intermediate AIMessage has both tool cards synthesized.
    intermediate = records[1]
    assert intermediate.role == "assistant"
    assert intermediate.tool_calls is not None and len(intermediate.tool_calls) == 2
    # Final assistant has the rebuilt tool_calls (rich log wins).
    final = records[-1]
    assert final.tool_calls is not None and len(final.tool_calls) == 2


def test_serialize_messages_handles_tool_without_tool_calls_block():
    """Bug 1 defensive: even if someone constructs a malformed AIMessage
    with empty tool_calls attribute, the serializer still works. Just
    makes sure the role/skip logic doesn't accidentally crash on a
    legacy AIMessage that didn't carry tool_calls metadata."""
    from src.api.routes.sessions import _serialize_messages

    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="ok"),  # plain final answer, no tool calls
    ]
    records = _serialize_messages(msgs)
    assert len(records) == 2
    assert records[0].role == "user"
    assert records[1].role == "assistant"


# ---------------------------------------------------------------------------
# Bug 2: AIMessage.additional_kwargs["tool_calls"] persistence
# ---------------------------------------------------------------------------


def test_build_tool_call_log_pairs_ai_with_tool_messages():
    """Bug 2 fix: ``_build_tool_call_log`` walks the messages list and
    pairs every AIMessage.tool_calls with its matching ToolMessage to
    produce the wire-shape dict the frontend ToolCallCard expects."""
    from src.agent.nodes.react_generate import _build_tool_call_log

    msgs = [
        HumanMessage(content="find docs"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "retrieve_docs",
                    "args": {"query": "revenue target", "top_k": 5},
                }
            ],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(
            content='[{"page_content": "Q3 revenue = 1.2 亿", "metadata": {"source_kind": "local"}}]',
            tool_call_id="call_1",
            name="retrieve_docs",
        ),
    ]
    log = _build_tool_call_log(msgs)
    assert isinstance(log, list)
    assert len(log) == 1
    entry = log[0]
    # Required keys (frontend ToolCallCard reads all of these):
    assert entry["id"] == "call_1"
    assert entry["name"] == "retrieve_docs"
    assert entry["args"] == {"query": "revenue target", "top_k": 5}
    assert entry["ok"] is True
    assert isinstance(entry["step"], int)
    assert "started_at" in entry
    assert "ended_at" in entry
    assert isinstance(entry["elapsed_ms"], int)
    assert entry["elapsed_ms"] >= 0
    # result preview must be a string (might be truncated to 800 chars
    # by _TOOL_RESULT_PREVIEW_MAX; we only assert non-empty)
    assert isinstance(entry["result"], str)
    assert "Q3 revenue" in entry["result"]


def test_build_tool_call_log_handles_multiple_calls_in_one_ai_message():
    """Bug 2 fix: one AIMessage can carry multiple tool_calls
    (Anthropic parallel-tool-use). Each must produce its own log entry."""
    from src.agent.nodes.react_generate import _build_tool_call_log

    msgs = [
        HumanMessage(content="hybrid q"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "retrieve_docs", "args": {"query": "a"}},
                {"id": "c2", "name": "web_search", "args": {"query": "b"}},
            ],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(content="[]", tool_call_id="c1", name="retrieve_docs"),
        ToolMessage(content="[]", tool_call_id="c2", name="web_search"),
    ]
    log = _build_tool_call_log(msgs)
    names = sorted(e["name"] for e in log)
    assert names == ["retrieve_docs", "web_search"], (
        f"Parallel tool_calls must produce 2 log entries, got {names}"
    )
    ids = sorted(e["id"] for e in log)
    assert ids == ["c1", "c2"]


def test_build_tool_call_log_skips_ai_messages_without_tool_calls():
    """Bug 2 fix: an AIMessage with no tool_calls (e.g. a final
    synthesis message) must NOT produce log entries."""
    from src.agent.nodes.react_generate import _build_tool_call_log

    msgs = [
        HumanMessage(content="q"),
        AIMessage(content="just a plain answer, no tools used"),
    ]
    log = _build_tool_call_log(msgs)
    assert log == [], (
        f"AIMessage without tool_calls must produce empty log; got {log}"
    )


def test_build_tool_call_log_truncates_long_results():
    """Bug 2 fix: long tool results are truncated to fit ToolCallCard.
    The exact limit is _TOOL_RESULT_PREVIEW_MAX (800 chars by default).
    A 5000-char JSON result must be cut to <= 801 chars (with trailing
    '…'). Matches the front-end truncation in ToolCallCard.tsx so
    live + reload render identically."""
    from src.agent.nodes.react_generate import (
        _TOOL_RESULT_PREVIEW_MAX,
        _build_tool_call_log,
    )

    huge = "x" * 5000
    msgs = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "retrieve_docs", "args": {}}],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(content=huge, tool_call_id="c1", name="retrieve_docs"),
    ]
    log = _build_tool_call_log(msgs)
    assert len(log) == 1
    result = log[0]["result"]
    assert len(result) <= _TOOL_RESULT_PREVIEW_MAX + 1  # +1 for "…"
    assert result.endswith("…"), (
        f"Truncated result should end with ellipsis; got {result[-3:]!r}"
    )


def test_build_tool_call_log_marks_failed_tool_calls():
    """Bug 2 fix: a ToolMessage whose content indicates a tool
    execution failure (LangChain's ToolNode convention: content starts
    with 'Tool execution failed') is marked ok=False in the log."""
    from src.agent.nodes.react_generate import _build_tool_call_log

    msgs = [
        HumanMessage(content="q"),
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "web_search", "args": {}}],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(
            content="Tool execution failed: connection timeout",
            tool_call_id="c1",
            name="web_search",
        ),
    ]
    log = _build_tool_call_log(msgs)
    assert len(log) == 1
    assert log[0]["ok"] is False, (
        f"Tool execution failure should produce ok=False; got {log[0]['ok']!r}"
    )


def test_serialize_messages_extracts_tool_calls_from_additional_kwargs():
    """Bug 2 fix (end-to-end): the session serializer pulls
    ``additional_kwargs["tool_calls"]`` off the AIMessage and exposes
    it as a list of dicts on the MessageRecord — this is what the
    front-end reads to render ToolCallCards on reload."""
    from src.api.routes.sessions import _serialize_messages

    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_x", "name": "retrieve_docs", "args": {"query": "q"}}
            ],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(content="[]", tool_call_id="call_x", name="retrieve_docs"),
        AIMessage(
            content="answer",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_x",
                        "name": "retrieve_docs",
                        "args": {"query": "q"},
                        "result": "[]",
                        "ok": True,
                        "step": 1,
                        "started_at": "2026-09-07T00:00:00+00:00",
                        "ended_at": "2026-09-07T00:00:01+00:00",
                        "elapsed_ms": 1000,
                    }
                ],
                "source_kinds": ["local"],
                "created_at": "2026-09-07T00:00:02+00:00",
            },
        ),
    ]
    records = _serialize_messages(msgs)
    # Skip the tool-calls-only intermediate assistant (no text/reasoning).
    final = [r for r in records if (r.content or "").startswith("answer")]
    assert len(final) == 1
    assert final[0].tool_calls is not None
    assert len(final[0].tool_calls) == 1
    tc = final[0].tool_calls[0]
    assert tc["id"] == "call_x"
    assert tc["name"] == "retrieve_docs"
    assert tc["args"] == {"query": "q"}
    assert tc["ok"] is True
    assert tc["elapsed_ms"] == 1000
    assert final[0].source_kinds == ["local"]


def test_serialize_messages_handles_malformed_tool_calls_list():
    """Bug 2 defensive: a malformed ``tool_calls`` field (None, dict
    instead of list, list of non-dicts) must NOT crash the serializer.
    Falls back to None / empty list gracefully."""
    from src.api.routes.sessions import _serialize_messages

    msgs = [
        HumanMessage(content="q"),
        AIMessage(
            content="answer",
            additional_kwargs={"tool_calls": "not a list"},
        ),
        HumanMessage(content="q2"),
        AIMessage(
            content="answer2",
            additional_kwargs={
                "tool_calls": [
                    {"id": "ok", "name": "x"},
                    "stray string",  # not a dict — must be skipped
                    {"name": "no-id"},  # missing id — must still pass
                ]
            },
        ),
    ]
    records = _serialize_messages(msgs)
    # First answer: tool_calls="not a list" → must NOT crash; falls back
    # to None (since no list found).
    r0 = [r for r in records if (r.content or "") == "answer"][0]
    assert r0.tool_calls is None or r0.tool_calls == []
    # Second answer: each dict entry must survive validation regardless
    # of which keys it carries. The contract is "list of dicts" — we
    # don't drop dicts just because they lack an id.
    r1 = [r for r in records if (r.content or "") == "answer2"][0]
    assert isinstance(r1.tool_calls, list)
    dicts = [tc for tc in r1.tool_calls if isinstance(tc, dict)]
    assert len(dicts) == 2, (
        f"Stray non-dict entries must be filtered out; got {r1.tool_calls}"
    )


# ---------------------------------------------------------------------------
# Bug 3: timestamp CSS on the blue user bubble
# ---------------------------------------------------------------------------


def test_css_user_timestamp_uses_white_color():
    """Bug 3 fix: ``.message.user .message-timestamp`` overrides the
    dim default (``var(--text-dim, #888)`` + ``opacity: 0.75``) with
    a near-white text color AND ``opacity: 1`` so the timestamp is
    legible on the blue gradient bubble. This pins the literal CSS
    rule in styles.css.

    Important: the override must include ``opacity: 1`` (or higher),
    because CSS element-level opacity multiplies with the color's
    alpha channel. Without it, ``rgba(255,255,255,0.85)`` at the
    inherited ``opacity: 0.75`` → effective alpha 0.6375 — still
    too dim on bright blue. We also lock ``font-size`` and
    ``font-weight`` so the timestamp has enough visual weight on the
    blue bubble without touching the assistant bubble's styling."""
    from pathlib import Path

    css_path = Path("src/frontend/src/styles.css")
    assert css_path.exists(), (
        "styles.css must exist at the conventional path for this regression "
        "test to be meaningful"
    )
    css = css_path.read_text(encoding="utf-8")
    # The override MUST exist with white color.
    assert ".message.user .message-timestamp" in css, (
        "User-bubble timestamp override is missing from styles.css — "
        "the timestamp is illegible on the blue gradient again"
    )
    import re

    match = re.search(
        r"\.message\.user\s+\.message-timestamp\s*\{([^}]+)\}",
        css,
        flags=re.DOTALL,
    )
    assert match, (
        "Couldn't find a CSS block scoped to `.message.user .message-timestamp`. "
        "The override must be specifically scoped to the user bubble (not the "
        "default `.message-timestamp`) so the assistant bubble keeps its "
        "subtle dim gray."
    )
    block = match.group(1)
    # Must contain a near-white color at >= 90% alpha (rgb(255,...) / rgba(255,...)/#fff).
    # Round 1 was rgba(...,0.85) — still too faint after opacity
    # multiplication on bright blue, so user still couldn't read it.
    # Round 2 is rgba(...,0.95) — must not regress below that.
    color_patterns = [
        r"rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*0\.[9]\d?\s*\)",  # rgba white 90-99%
        r"rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*1(?:\.0+)?\s*\)",  # rgba white 100%
        r"rgb\(\s*255\s*,\s*255\s*,\s*255\s*\)",                       # rgb white
        r"#fff(?:fff)?",                                                # hex white
    ]
    assert any(re.search(p, block) for p in color_patterns), (
        f"`.message.user .message-timestamp` block must use white at "
        f"high opacity (>= 90%); got block:\n{block}"
    )
    # CRITICAL: must also override opacity to >= 1 (or remove the base
    # rule's 0.75). Otherwise the base opacity multiplies with the
    # color alpha and the timestamp still looks faint.
    assert re.search(r"opacity\s*:\s*1(?:\.0+)?\b", block), (
        f"`.message.user .message-timestamp` must override `opacity: 1` "
        f"to undo the base `.message-timestamp`'s `opacity: 0.75`. "
        f"Without it, the color's alpha is multiplied by 0.75 and the "
        f"timestamp remains invisible on the blue gradient. Got block:\n{block}"
    )
    # font-size must be raised above the base 11px (otherwise the
    # timestamp is too small to read at a glance on a busy bubble).
    size_match = re.search(r"font-size\s*:\s*(\d+)px", block)
    assert size_match and int(size_match.group(1)) >= 12, (
        f"`.message.user .message-timestamp` must override font-size "
        f"to >= 12px so the timestamp is readable; got block:\n{block}"
    )
    # font-weight must be set to >= 500 so the timestamp has visual
    # weight on the blue bubble. The base rule omits font-weight
    # entirely (inheriting body's 400), which reads as washed-out.
    weight_match = re.search(r"font-weight\s*:\s*(\d{3})", block)
    assert weight_match and int(weight_match.group(1)) >= 500, (
        f"`.message.user .message-timestamp` must set font-weight "
        f">= 500 so the timestamp stands out on the blue bubble; got block:\n{block}"
    )


# ---------------------------------------------------------------------------
# Bug 2 + 3 integration: the persisted AIMessage carries both
# tool_calls AND source_kinds (the "show tools + show banner" combo).
# ---------------------------------------------------------------------------


def test_final_ai_message_carries_both_tool_calls_and_source_kinds():
    """Integration: after a hybrid (local+web) ReAct turn, the FINAL
    assistant AIMessage in history must carry BOTH:
        * ``tool_calls``: list of dicts (for ToolCallCard re-render)
        * ``source_kinds``: list[str] (for the banner / chip)
    This pins the full v2.0.3 fix bundle together so the front-end
    rebuilds the entire ReAct view (tool cards + answer + banner)
    from a single AIMessage."""
    from src.api.routes.sessions import _serialize_messages

    msgs = [
        HumanMessage(content="hybrid q"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "retrieve_docs", "args": {"query": "q"}},
                {"id": "c2", "name": "web_search", "args": {"query": "q"}},
            ],
            additional_kwargs={"created_at": "2026-09-07T00:00:00+00:00"},
        ),
        ToolMessage(content="[]", tool_call_id="c1", name="retrieve_docs"),
        ToolMessage(content="[]", tool_call_id="c2", name="web_search"),
        AIMessage(
            content="hybrid answer",
            additional_kwargs={
                "sources": [
                    {
                        "index": 1,
                        "chunk_id": "a",
                        "doc_id": "d",
                        "filename": "f.txt",
                        "source_kind": "local",
                        "text": "x",
                        "score": 0.9,
                    },
                    {
                        "index": 2,
                        "chunk_id": "",
                        "doc_id": "web-1",
                        "filename": "site",
                        "url": "https://x",
                        "domain": "x.com",
                        "source_kind": "web",
                        "text": "y",
                        "score": 0.0,
                    },
                ],
                "source_kinds": ["local", "web"],
                "tool_calls": [
                    {"id": "c1", "name": "retrieve_docs", "args": {}, "ok": True, "step": 1},
                    {"id": "c2", "name": "web_search", "args": {}, "ok": True, "step": 1},
                ],
                "created_at": "2026-09-07T00:00:01+00:00",
            },
        ),
    ]
    records = _serialize_messages(msgs)
    final = [r for r in records if (r.content or "") == "hybrid answer"]
    assert len(final) == 1
    r = final[0]
    assert r.source_kinds == ["local", "web"]
    assert r.tool_calls is not None and len(r.tool_calls) == 2
    assert r.created_at == "2026-09-07T00:00:01+00:00"
    # Sources are also present (used by the citation chip renderer).
    assert r.sources is not None and len(r.sources) == 2
    # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
    source_kinds_set = {s.source_kind for s in r.sources}
    assert source_kinds_set == {"local", "web"}
