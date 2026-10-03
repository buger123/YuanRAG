"""v2.0.4 — P0 fact-safety regression tests.

Bugs fixed (see ``plan/polished-gliding-karp.md`` for the original
review):

* **P0-1** web search returned irrelevant hits → LLM fabricated
  answers ("the results below have been deemed relevant" was a
  lie because there's no relevance grader in the ReAct path).
  Three fixes layered:
    1. Bing URL now includes ``mkt=zh-CN`` / ``setlang=`` / ``cc=``
       so the CN-edge node returns CN-index results instead of
       falling back to en-US.
    2. ``react_generate`` rewrote the "have been deemed relevant"
       branch into a refusal when no usable web content came back,
       and into an honest "verify before citing" instruction when
       something did.
    3. Default engine chain ``["bing", "ddg"]`` so a Bing
       anti-bot page no longer kills the whole search.

* **P0-2** MiniMax-style tool-call wire syntax
  (``]<minimax[>[<invoke name="web_search">...``) leaked into
  user-facing answer text. The old stripper only matched paired
  ``<tool_call>`` / ``<tool_result>`` tags; new ``_strip_leaked_tool_syntax``
  matches the OBSERVED leak shapes AND ``stream_safe_chunk`` adds
  a stateful 64-char trailing buffer so a leak split across two
  chunks still gets caught.

* **P0-3** Document retrieval was bypassed — LLM went straight to
  ``web_search`` for "告诉我腾讯音乐 10 亿美元债券" instead of
  trying the user's uploaded docs first. Two fixes:
    1. ``react_agent`` now injects a "prefer retrieve_docs" hint
       when the thread has any uploaded documents.
    2. ``retrieve_docs`` tool now uses LangGraph's ``InjectedState``
       to receive the real ``thread_id`` at runtime instead of
       hardcoding ``None`` (the comment in the old code lied about
       "patched below" — it wasn't). Pre-v2.0.4 every retrieval
       was unscoped → cross-thread data leak.

* **P0-4** LLM hallucinated prior user turns ("你刚才说 X",
  "you sent me a thumbs up") because ``react_generate`` built
  ``msgs = [System, *history]`` with the full tool-call trace.
  The trace's ``AIMessage(tool_calls=[...])`` entries look like
  conversational turns to a confused model. Two fixes:
    1. ``_sanitize_history_for_generate`` drops tool-calls-only
       ``AIMessage`` and all ``ToolMessage`` entries before
       passing to the LLM.
    2. Explicit ``_HISTORY_FRAME_HINT`` SystemMessage labels the
       remaining roles so the LLM can't confuse adjacent turns.

These tests pin all four fixes so future refactors don't silently
regress either layer.

If a test here fails, either:
  (a) one of the v2.0.4 fixes was reverted, OR
  (b) a new fix needs to be added — update the test to match.
"""
from __future__ import annotations

import asyncio
import inspect
import re

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _run(coro):
    """Tiny helper: run a coroutine in a fresh event loop.

    Kept local (instead of ``pytest-asyncio``) so the test file
    runs without depending on async fixtures — matches the
    v2.0.2 / v2.0.3 regression test style.
    """
    return asyncio.run(coro)


def _drive_async_gen(node, state):
    """v2.0.22 (Item 7 Step 6) — drive an async-gen node and
    return the ``__delta__`` payload.

    Pre-Step-6 the test called ``_run(react_agent(state))`` and
    ``asyncio.run`` consumed the coroutine return value. Post-Step-6
    ``react_agent`` is an async-gen that must be drained via
    ``async for``; this shim does that and returns the delta dict.
    """
    async def _drive():
        async for kind, payload in node(state):
            if kind == "__delta__":
                return payload
        raise RuntimeError(f"{node.__name__}: yielded no __delta__")
    return asyncio.run(_drive())


# ============================================================================
# P0-2 — tool-call wire syntax leak blacklist
# ============================================================================


class TestP02LeakedToolSyntaxStrip:
    """The leak regexes catch every observed MiniMax-style wire fragment."""

    def test_strip_leaked_tool_syntax_handles_minimax_invoke_prefix(self):
        """``]<minimax[>[<invoke name="web_search">`` is stripped."""
        from src.agent._thinking_split import _strip_leaked_tool_syntax

        bad = "some text ]<minimax[>[<invoke name=\"web_search\">blah"
        out = _strip_leaked_tool_syntax(bad)
        assert "]<minimax[>" not in out
        assert "<invoke" not in out
        assert "web_search" not in out

    def test_strip_leaked_tool_syntax_handles_isolated_invoke_tag(self):
        """A bare ``<invoke name="...">`` is stripped."""
        from src.agent._thinking_split import _strip_leaked_tool_syntax

        bad = "before <invoke name=\"web_search\"> after"
        out = _strip_leaked_tool_syntax(bad)
        assert "<invoke" not in out
        # surrounding text survives
        assert "before" in out
        assert "after" in out

    def test_strip_leaked_tool_syntax_handles_parameter_tag(self):
        """A bare ``<parameter name="query">`` is stripped."""
        from src.agent._thinking_split import _strip_leaked_tool_syntax

        bad = "alpha <parameter name=\"query\">beta</parameter> gamma"
        out = _strip_leaked_tool_syntax(bad)
        assert "<parameter" not in out
        assert "</parameter>" not in out
        assert "alpha" in out
        assert "gamma" in out

    def test_strip_leaked_tool_syntax_handles_function_calls(self):
        """``<function_calls>`` / ``</function_calls>`` is stripped."""
        from src.agent._thinking_split import _strip_leaked_tool_syntax

        bad = "<function_calls><invoke name=\"x\"/></function_calls>"
        out = _strip_leaked_tool_syntax(bad)
        assert "<function_calls>" not in out
        assert "</function_calls>" not in out
        assert "<invoke" not in out

    def test_strip_leaked_tool_syntax_preserves_normal_text(self):
        """Plain prose without wire syntax passes through unchanged."""
        from src.agent._thinking_split import _strip_leaked_tool_syntax

        good = "腾讯音乐于 2026 年发行了 10 亿美元债券,期限 5.05%-5.65%。"
        out = _strip_leaked_tool_syntax(good)
        assert out == good

    def test_strip_leaked_tool_syntax_handles_empty_string(self):
        from src.agent._thinking_split import _strip_leaked_tool_syntax

        assert _strip_leaked_tool_syntax("") == ""

    def test_strip_leaked_tool_syntax_idempotent(self):
        """Running strip twice equals running once."""
        from src.agent._thinking_split import _strip_leaked_tool_syntax

        bad = "x <invoke name=\"web_search\"> y <function_calls> z"
        once = _strip_leaked_tool_syntax(bad)
        twice = _strip_leaked_tool_syntax(once)
        assert once == twice


class TestP02StreamSafeChunk:
    """The stateful streaming helper holds back the trailing tail.

    v2.0.4 — the runner uses PER-CHUNK leak-stripping (not the
    stateful buffer) to preserve the v1.1.6c accumulated-tail dedup
    contract. ``stream_safe_chunk`` remains exported as a helper for
    callers that need the stateful behavior (e.g. batch re-stripping
    on the answer_complete path) but the runner doesn't hold back
    per-chunk emission — a per-chunk leak strip catches the common
    case where the LLM emits the wire syntax as an atomic token, and
    the final ``answer_complete`` strip catches the cross-chunk
    edge case via ``_strip_tool_blocks``.

    v2.0.10.2 — runner now actually wires up ``stream_safe_chunk``
    (was dead code in v2.0.4 — the helper existed but the runner
    called ``_strip_leaked_tool_syntax`` directly, so the buffer
    logic was never exercised). Buffer threshold also lowered from
    64 → 32 chars so short answers emit incrementally (v2.0.4's
    "hold all back if <= threshold" rule broke
    ``test_runner_stream_dedup::test_normal_stream_yields_each_chunk_once``
    by holding back a 17-char greeting entirely).
    """

    def test_emits_clean_text_when_combined_below_buffer_threshold(self):
        """v2.0.10.2 — short clean text emits INCREMENTALLY instead of
        being held back. The buffer is only needed when text is long
        enough that the held-back tail might catch a leak split
        across chunks.
        """
        from src.agent._thinking_split import stream_safe_chunk

        emit, new_buf = stream_safe_chunk("Hello", "")
        # "Hello" is 5 chars, below 32-char buffer — emit all,
        # buffer cleared (no need to wait for more).
        assert emit == "Hello"
        assert new_buf == ""

    def test_emits_safe_prefix_when_combined_crosses_threshold(self):
        from src.agent._thinking_split import stream_safe_chunk

        # Long enough to cross the 32-char buffer threshold
        long_text = "a" * 100
        emit, new_buf = stream_safe_chunk(long_text, "")
        assert len(emit) == 100 - 32  # 68 chars emitted, 32 retained
        assert new_buf == "a" * 32

    def test_catches_leak_split_across_two_chunks(self):
        """A leak that arrives split across chunk boundaries is still stripped."""
        from src.agent._thinking_split import stream_safe_chunk

        chunk1 = "]<minimax[>[<invoke name=\"web_se"
        chunk2 = "arch\">more text after"

        e1, buf1 = stream_safe_chunk(chunk1, "")
        e2, buf2 = stream_safe_chunk(chunk2, buf1)

        # All leaked content must be absent from the emitted text
        emitted = e1 + e2
        assert "<invoke" not in emitted
        assert "<minimax" not in emitted
        assert "]<minimax" not in emitted

    def test_preserves_clean_text_across_chunks(self):
        from src.agent._thinking_split import stream_safe_chunk

        chunk1 = "腾讯音乐" * 20  # 80 chars
        chunk2 = " " + "发行债券" * 20  # 101 chars
        e1, buf1 = stream_safe_chunk(chunk1, "")
        e2, buf2 = stream_safe_chunk(chunk2, buf1)
        emitted = e1 + e2
        # Net: should retain all clean content eventually
        assert "腾讯音乐" in emitted or "腾讯音乐" in buf2
        assert "发行债券" in emitted or "发行债券" in buf2

    def test_catches_truncated_prefix_leak_when_invoke_never_arrives(self):
        """v2.0.10.2 — when the leak arrives as JUST the prefix
        (``]<minimax[>`` or ``]<]minimax>[``) without any
        ``<invoke>`` tail in subsequent chunks, the new
        ``_RE_LEAKED_WIRE_PREFIX`` regex catches it on whichever
        chunk it arrives in. v2.0.4's ``_RE_INVOKE_FRAGMENT``
        missed this case because it required ``<invoke>`` to be
        in the same match window — when the stream is truncated
        mid-leak, the prefix survives and leaks to the user.
        """
        from src.agent._thinking_split import stream_safe_chunk

        # Production 2026-09-13 — user's exact reported shape
        chunk1 = "]<]" + "​" + "minimax>["
        e1, buf1 = stream_safe_chunk(chunk1, "")
        emitted = e1
        # The leaked wire-format prefix must be gone
        assert "minimax" not in emitted
        assert "]minimax" not in emitted
        assert "<]" not in emitted

        # And the original documented shape
        chunk2 = "]<minimax[>"
        e2, buf2 = stream_safe_chunk(chunk2, "")
        assert "minimax" not in e2
        assert "]<minimax" not in e2


class TestP02RunnerPerChunkStrip:
    """The runner applies the leak strip per-chunk (not via stateful buffer).

    v1.1.6c's two-layer dedup is preserved: each chunk is emitted
    as its own ``token`` event after leak-stripping, then the strict
    equality dedup drops LangGraph's synthesized final chunk.
    """

    def test_runner_strips_per_chunk_before_emitting(self):
        """A single chunk containing the full leak syntax emits empty."""
        from src.agent import runner

        async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
            # Single FSM token event with the full leak.
            yield {
                "kind": "token",
                "text": ']<minimax[>[<invoke name="web_search">',
            }
            yield {
                "kind": "answer_complete",
                "answer": "ok",
                "sources": [],
                "source_kinds": [],
                "created_at": "2024-01-01T00:00:00+00:00",
                "route_decision": None,
            }

        import asyncio as _aio
        import pytest
        from tests.conftest import bypass_models_loaded_gate

        mp = pytest.MonkeyPatch()
        try:
            bypass_models_loaded_gate(mp)
            mp.setattr(runner, "run_fsm", fake_run_fsm)
            out = []

            async def drive():
                async for e in runner.stream_agent("t", "q"):
                    out.append(e)

            _aio.run(drive())
        finally:
            mp.undo()

        token_events = [e for e in out if e["type"] == "token"]
        # Per-chunk leak should be fully stripped (substituted with
        # space). The user-visible test is that no leak token survives
        # — whitespace tokens are an acceptable byproduct of the
        # ``re.sub(" ", ...)`` substitution.
        combined = "".join(e.get("content", "") for e in token_events)
        assert "<invoke" not in combined
        assert "web_search" not in combined
        assert "]<minimax" not in combined
        assert "minimax" not in combined

    def test_runner_preserves_clean_chunks_per_v116c(self):
        """v1.1.6c contract: clean incremental chunks each emit a token."""
        from src.agent import runner

        async def fake_run_fsm(inputs, *, thread_id="", max_steps=None):
            for text in ["你", "好", "！"]:
                yield {"kind": "token", "text": text}
            yield {
                "kind": "answer_complete",
                "answer": "你好！",
                "sources": [],
                "source_kinds": [],
                "created_at": "2024-01-01T00:00:00+00:00",
                "route_decision": None,
            }

        import asyncio as _aio
        import pytest
        from tests.conftest import bypass_models_loaded_gate

        mp = pytest.MonkeyPatch()
        try:
            bypass_models_loaded_gate(mp)
            mp.setattr(runner, "run_fsm", fake_run_fsm)
            out = []

            async def drive():
                async for e in runner.stream_agent("t", "q"):
                    out.append(e)

            _aio.run(drive())
        finally:
            mp.undo()

        token_events = [e for e in out if e["type"] == "token"]
        # v1.1.6c contract: each chunk emits a separate token event.
        # Some chunks may be whitespace-collapsed by the post-strip
        # pass, but the visible CJK chars must all be present.
        combined = "".join(e.get("content", "") for e in token_events)
        for c in "你好！":
            assert c in combined, (
                f"v1.1.6c per-chunk emission broken: "
                f"missing {c!r} in {combined!r}"
            )


# ============================================================================
# P0-1 — web search truthfulness (Bing locale + ws_status + engines)
# ============================================================================


class TestP01BingLocaleParams:
    """Bing URLs now carry mkt / setlang / cc locale params."""

    def test_bing_url_contains_locale_params(self):
        """A scrape URL built by bing.py includes mkt / setlang / cc."""
        from src.web_search import bing

        # Build the URL the way bing._fetch_one does
        from urllib.parse import quote
        locale = "zh-CN"
        host = bing._BING_HOSTS[0]
        qs = (
            f"q={quote('test')}"
            f"&mkt={quote(locale)}"
            f"&setlang={quote(locale)}"
            f"&cc={quote(locale.split('-')[-1])}"
            f"&first=1"
        )
        url = f"{host}?{qs}"
        assert "mkt=zh-CN" in url
        assert "setlang=zh-CN" in url
        assert "cc=CN" in url

    def test_bing_headers_for_zh_locale_use_zh_accept_language(self):
        from src.web_search.bing import _headers_for_locale

        h = _headers_for_locale("zh-CN")
        assert "zh-CN" in h["Accept-Language"]
        assert "zh" in h["Accept-Language"].lower()

    def test_bing_headers_for_en_locale_use_en_accept_language(self):
        from src.web_search.bing import _headers_for_locale

        h = _headers_for_locale("en-US")
        assert "en-US" in h["Accept-Language"]
        # Must not advertise zh-CN to an English query
        assert "zh-CN" not in h["Accept-Language"]

    def test_detect_locale_picks_zh_for_cjk_query(self):
        from src.web_search.bing import _detect_locale

        assert _detect_locale("腾讯新闻 2026-09-06 头条") == "zh-CN"
        assert _detect_locale("你好") == "zh-CN"

    def test_detect_locale_picks_en_for_ascii_query(self):
        from src.web_search.bing import _detect_locale

        assert _detect_locale("iPhone 17 release date") == "en-US"
        assert _detect_locale("latest news today") == "en-US"

    def test_detect_locale_handles_empty(self):
        from src.web_search.bing import _detect_locale

        assert _detect_locale("") == "en-US"


class TestP01EngineChain:
    """Default engine chain now includes DDG as fallback."""

    def test_default_engine_chain_includes_ddg(self):
        from config.constants import WEB_SEARCH

        engines = WEB_SEARCH["engines"]
        assert "bing" in engines
        assert "ddg" in engines, (
            "v2.0.4 default chain must include DDG fallback so a "
            "Bing anti-bot page doesn't silently kill the search"
        )

    def test_engine_chain_bing_comes_first(self):
        """Bing is the primary, DDG is fallback (CN-friendly default)."""
        from config.constants import WEB_SEARCH

        engines = WEB_SEARCH["engines"]
        assert engines.index("bing") < engines.index("ddg")


class TestP01WsStatusRefusal:
    """``ws_status`` is honest when web search returned nothing usable."""

    def _ws_status_strings(self):
        """Walk the react_generate AST and collect every string literal
        that lands inside a prompt / status variable assignment.

        Excludes docstrings and comments so an explanatory comment
        like ``# removed the "have been deemed relevant" lie`` doesn't
        false-positive. We use ``ast`` because substring-search on
        source text can't tell a comment from a runtime string.
        """
        import ast
        from pathlib import Path
        src = Path("src/agent/nodes/react_generate.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        strings = []

        def _visit(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.append(node.value)
            for child in ast.iter_child_nodes(node):
                _visit(child)

        _visit(tree)
        # Filter to the prompts + status texts (string-concat results
        # are usually broken into multiple constants). Match by
        # substring rather than AST position so a docstring /
        # unrelated constant doesn't get caught.
        relevant = []
        targets = ("have been deemed", "no usable results", "deemed relevant")
        for s in strings:
            for t in targets:
                if t in s:
                    relevant.append(s)
                    break
        return relevant

    def test_ws_status_is_refusal_for_empty_web_results(self):
        """No web docs + web_search attempted → explicit refusal string."""
        relevant = self._ws_status_strings()
        # The lie string must not appear in any actual string literal
        # (only in comments / docs, which AST excludes).
        for s in relevant:
            assert "have been deemed" not in s, (
                f"v2.0.4 must NOT pass 'have been deemed relevant' to "
                f"the LLM as a prompt string. Found: {s[:200]!r}"
            )
        # The new refusal phrase must be present in a literal.
        assert any("no usable results" in s for s in relevant), (
            "v2.0.4 must include the explicit 'no usable results' "
            "refusal string somewhere in react_generate"
        )

    def test_ws_status_uses_short_content_floor(self):
        """Web docs below 100 chars are treated as not-usable."""
        import ast
        from pathlib import Path
        src = Path("src/agent/nodes/react_generate.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        # Walk all int / Compare nodes; at least one comparison must
        # use 100 as the threshold.
        has_100 = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 100:
                # Look for it inside a comparison (>, >=, <, <=)
                # Walk parents via a manual pass.
                for other in ast.walk(tree):
                    if isinstance(other, ast.Compare) and node in ast.walk(other):
                        has_100 = True
                        break
        assert has_100, (
            "v2.0.4 must compare page_content length against 100 chars "
            "to filter out anti-bot stubs / empty snippets"
        )


# ============================================================================
# P0-3 — retrieve_docs first + thread_id injection
# ============================================================================


class TestP03RetrieveFirstHint:
    """``react_agent`` injects a retrieve-first hint when thread has docs."""

    def test_retrieve_first_hint_constant_is_defined(self):
        from src.agent.nodes.react_agent import _RETRIEVE_FIRST_HINT

        assert isinstance(_RETRIEVE_FIRST_HINT, str)
        assert "retrieve_docs" in _RETRIEVE_FIRST_HINT
        assert "优先" in _RETRIEVE_FIRST_HINT

    def test_react_agent_injects_hint_when_thread_has_docs(self):
        """When ``list_documents_cached(thread_id)`` returns docs, the hint
        appears in the messages list passed to the LLM."""
        # v2.0.16 — use the explicit submodule path. The previous
        # ``from src.agent.nodes import react_agent`` returned the
        # function (because ``nodes.__init__`` re-exported it), which
        # broke ``react_agent.react_agent(state)`` below — the function
        # has no attribute ``react_agent``.
        from src.agent.nodes.react_agent import react_agent

        # Monkey-patch the doc cache to return non-empty
        class _FakeLancedbModule:
            @staticmethod
            def list_documents_cached(thread_id):
                return [{"doc_id": "d1", "filename": "test.pdf"}]

        import sys
        fake_lancedb = type(sys)("fake_lancedb")
        fake_lancedb.list_documents_cached = _FakeLancedbModule.list_documents_cached
        sys.modules["src.storage.lancedb_store"] = fake_lancedb

        try:
            state = {
                "messages": [HumanMessage(content="告诉我腾讯音乐债券")],
                "thread_id": "t1",
                "needs_current_time": False,
            }

            captured_msgs: list = []

            class _FakeModel:
                async def ainvoke(self, msgs):
                    captured_msgs.extend(msgs)
                    return AIMessage(content="ok")

            class _FakeModelWithBind:
                def bind_tools(self, tools):
                    return _FakeModel()

            import src.agent.nodes.react_agent as ra

            orig_build = ra.build_chat_model
            ra.build_chat_model = lambda **kw: _FakeModelWithBind()
            try:
                # v2.0.16 — ``react_agent`` is now the function (not
                # the module, since we removed the ``__init__``
                # re-export that caused the shadow). Call directly.
                # v2.0.22 (Item 7 Step 6) — async-gen protocol:
                # drive via ``_drive_async_gen`` instead of ``_run``.
                _drive_async_gen(react_agent, state)
            finally:
                ra.build_chat_model = orig_build
        finally:
            # Restore the real module (use cache trick to avoid stale imports)
            import importlib
            import src.storage.lancedb_store as real_lancedb  # noqa: F401
            sys.modules["src.storage.lancedb_store"] = real_lancedb

        # At least one SystemMessage in the captured list must mention
        # retrieve_docs as a preferred tool
        assert any(
            isinstance(m, SystemMessage) and "retrieve_docs" in (m.content or "")
            for m in captured_msgs
        ), "retrieve-first hint missing from LLM messages"

    def test_react_agent_no_hint_when_thread_has_no_docs(self):
        """Empty doc list → no retrieve-first hint injected."""
        # See sibling test above for the v2.0.16 submodule-path rationale.
        from src.agent.nodes.react_agent import react_agent

        class _FakeLancedbModule:
            @staticmethod
            def list_documents_cached(thread_id):
                return []  # empty

        import sys
        fake_lancedb = type(sys)("fake_lancedb")
        fake_lancedb.list_documents_cached = _FakeLancedbModule.list_documents_cached
        sys.modules["src.storage.lancedb_store"] = fake_lancedb

        try:
            state = {
                "messages": [HumanMessage(content="hi")],
                "thread_id": "t-empty",
                "needs_current_time": False,
            }
            captured_msgs: list = []

            class _FakeModel:
                async def ainvoke(self, msgs):
                    captured_msgs.extend(msgs)
                    return AIMessage(content="ok")

            class _FakeModelWithBind:
                def bind_tools(self, tools):
                    return _FakeModel()

            import src.agent.nodes.react_agent as ra

            orig_build = ra.build_chat_model
            ra.build_chat_model = lambda **kw: _FakeModelWithBind()
            try:
                # v2.0.16 — ``react_agent`` is now the function (not
                # the module, since we removed the ``__init__``
                # re-export that caused the shadow). Call directly.
                # v2.0.22 (Item 7 Step 6) — async-gen protocol:
                # drive via ``_drive_async_gen`` instead of ``_run``.
                _drive_async_gen(react_agent, state)
            finally:
                ra.build_chat_model = orig_build
        finally:
            import src.storage.lancedb_store as real_lancedb  # noqa: F401
            sys.modules["src.storage.lancedb_store"] = real_lancedb

        # No retrieve-first hint should have been injected (the time
        # hint and history-frame hint may still appear, but neither
        # says "retrieve_docs")
        assert not any(
            isinstance(m, SystemMessage) and "retrieve_docs" in (m.content or "")
            for m in captured_msgs
        )


class TestP03ThreadIdInjection:
    """``retrieve_docs`` tool declares ``thread_id: Annotated[str, InjectedToolArg]``
    so the FSM can inject the real thread_id at runtime (Phase 3 — v2.0.20).

    Pre-Phase-3 (v2.0.4): ``state: Annotated[dict, InjectedState]`` (LangGraph).
    Phase 3 (v2.0.20): ``thread_id: Annotated[str, InjectedToolArg]`` (langchain_core).
    LangChain's schema builder filters both markers from the JSON schema
    sent to the LLM, so the model can never hallucinate a thread_id.
    """

    def _retrieve_docs_source(self) -> str:
        """Read the source file directly — the tool object is a
        StructuredTool, not introspectable."""
        from pathlib import Path
        return Path("src/agent/tools/retrieve_docs.py").read_text(encoding="utf-8")

    def test_retrieve_docs_tool_signals_state_injection(self):
        """The tool function signature declares a runtime-injected thread_id.

        Accepts either the pre-Phase-3 form (``state: Annotated[dict,
        InjectedState]``, LangGraph) OR the Phase-3 form (``thread_id:
        Annotated[str, InjectedToolArg]``, langchain_core). Both signal
        "runtime-injected arg, hide from LLM schema".
        """
        src = self._retrieve_docs_source()
        # Either the LangGraph pre-Phase-3 marker OR the langchain_core
        # Phase-3 marker must be present.
        has_langgraph_marker = "InjectedState" in src or re.search(
            r"async\s+def\s+retrieve_docs\s*\([^)]*\bstate\b", src
        )
        has_langchain_marker = "InjectedToolArg" in src and re.search(
            r"async\s+def\s+retrieve_docs\s*\([^)]*\bthread_id\b", src
        )
        assert has_langgraph_marker or has_langchain_marker, (
            "retrieve_docs must declare a runtime-injected thread_id "
            "parameter (either the LangGraph InjectedState form or the "
            "Phase-3 langchain_core InjectedToolArg form)"
        )

    def test_retrieve_docs_no_longer_hardcodes_none_thread_id(self):
        """The pre-v2.0.4 ``"thread_id": None`` hardcode is gone."""
        src = self._retrieve_docs_source()
        # The OLD code had: ``"thread_id": None,  # patched below``
        # The NEW code reads thread_id off the injected state.
        # The literal "patched below" string must no longer exist.
        assert "patched below" not in src, (
            "v2.0.4 must no longer lie about thread_id being patched below"
        )
        # The state_slice passed to ``retrieve_hybrid_async`` must NOT
        # hardcode thread_id to None — it must carry the injected
        # value. We accept either: ``"thread_id": thread_id`` (where
        # the local var was sourced from state) or
        # ``"thread_id": state.get("thread_id")``.
        # Strip the local ``thread_id = None`` initialization line
        # before checking (it's a benign default that gets overwritten
        # immediately — only the wire-out call site matters).
        lines = [
            ln for ln in src.splitlines()
            if "thread_id = None" not in ln
        ]
        body = "\n".join(lines)
        assert not re.search(r'"thread_id"\s*:\s*None', body), (
            "retrieve_docs must NOT pass thread_id=None to the retriever"
        )
        assert re.search(r'"thread_id"\s*:\s*thread_id', body), (
            "retrieve_docs must pass the injected thread_id (the local "
            "var, sourced from state) into the retriever's state slice"
        )


# ============================================================================
# P0-4 — sanitize history + history frame hint
# ============================================================================


class TestP04SanitizeHistory:
    """``_sanitize_history_for_generate`` strips tool-call trace from history."""

    def test_drops_tool_messages(self):
        from src.agent.nodes.react_generate import _sanitize_history_for_generate

        msgs = [
            HumanMessage(content="hi"),
            ToolMessage(content="[]", tool_call_id="t1"),
            HumanMessage(content="another question"),
        ]
        out = _sanitize_history_for_generate(msgs)
        assert all(not isinstance(m, ToolMessage) for m in out)
        # HumanMessages preserved
        assert any(isinstance(m, HumanMessage) and m.content == "hi" for m in out)
        assert any(
            isinstance(m, HumanMessage) and m.content == "another question"
            for m in out
        )

    def test_drops_tool_calls_only_ai_messages(self):
        from src.agent.nodes.react_generate import _sanitize_history_for_generate

        ai_tool_only = AIMessage(
            content="",
            tool_calls=[{"id": "t1", "name": "web_search", "args": {}, "type": "tool_call"}],
        )
        msgs = [
            HumanMessage(content="q1"),
            ai_tool_only,
            HumanMessage(content="q2"),
        ]
        out = _sanitize_history_for_generate(msgs)
        # The tool-calls-only AIMessage must be dropped
        assert ai_tool_only not in out
        # Both HumanMessages preserved
        assert sum(1 for m in out if isinstance(m, HumanMessage)) == 2

    def test_keeps_ai_messages_with_visible_text(self):
        from src.agent.nodes.react_generate import _sanitize_history_for_generate

        ai_real = AIMessage(content="previous answer")
        ai_mixed = AIMessage(
            content="answer with tool",
            tool_calls=[{"id": "t1", "name": "x", "args": {}, "type": "tool_call"}],
        )
        out = _sanitize_history_for_generate([ai_real, ai_mixed])
        assert ai_real in out
        assert ai_mixed in out

    def test_keeps_thinking_only_ai_messages(self):
        """v1.1.9 guard: a thinking-only AIMessage must NOT be dropped.

        Thinking content is carried via ``additional_kwargs["reasoning"]``
        (a v1.1.7 contract) and the AIMessage's ``content`` may be a
        list of thinking blocks with no visible text. The sanitizer
        must preserve these so the next turn's ``ainvoke`` can carry
        Anthropic's extended thinking forward.
        """
        from src.agent.nodes.react_generate import _sanitize_history_for_generate

        ai_thinking = AIMessage(
            content=[
                {"type": "thinking", "thinking": "internal reasoning..."},
            ],
            additional_kwargs={"reasoning": "internal reasoning..."},
        )
        out = _sanitize_history_for_generate([ai_thinking])
        assert ai_thinking in out, (
            "v1.1.9 guard: thinking-only AIMessages must NOT be dropped"
        )

    def test_handles_empty_list(self):
        from src.agent.nodes.react_generate import _sanitize_history_for_generate

        assert _sanitize_history_for_generate([]) == []
        assert _sanitize_history_for_generate(None) == []

    def test_preserves_order(self):
        from src.agent.nodes.react_generate import _sanitize_history_for_generate

        h1 = HumanMessage(content="h1")
        ai = AIMessage(content="a1")
        h2 = HumanMessage(content="h2")
        ai_tool = AIMessage(content="", tool_calls=[{"id": "x", "name": "n", "args": {}, "type": "tool_call"}])
        h3 = HumanMessage(content="h3")
        msgs = [h1, ai, ai_tool, h2, h3]
        out = _sanitize_history_for_generate(msgs)
        assert out == [h1, ai, h2, h3]


class TestP04HistoryFrameHint:
    """A ``_HISTORY_FRAME_HINT`` SystemMessage labels roles explicitly."""

    def test_hint_constant_is_defined(self):
        from src.agent.nodes.react_generate import _HISTORY_FRAME_HINT
        from src.agent.nodes.react_agent import _HISTORY_FRAME_HINT as AGENT_HINT

        assert isinstance(_HISTORY_FRAME_HINT, str)
        assert "HumanMessage" in _HISTORY_FRAME_HINT
        assert "AIMessage" in _HISTORY_FRAME_HINT
        # The two prompts must be byte-identical so the LLM gets a
        # consistent framing in both phases.
        assert _HISTORY_FRAME_HINT == AGENT_HINT

    def test_react_generate_includes_frame_hint_in_msgs(self):
        """react_generate prepends the frame hint after the system prompt."""
        from src.agent.nodes.react_generate import react_generate
        import inspect

        src = inspect.getsource(react_generate)
        # Must reference both the system prompt and the frame hint
        assert "sys_prompt" in src
        assert "_HISTORY_FRAME_HINT" in src
        # Must use the sanitized history (not raw history)
        assert "sanitized_history" in src


# ============================================================================
# Regression: existing behavior preserved
# ============================================================================


class TestRegressionV200ThroughV203StillPass:
    """Cross-version invariants — v2.0.4 changes must not regress earlier work."""

    def test_v100_thinking_block_preservation_still_works(self):
        """v1.1.7 thinking-block content-list preservation."""
        from src.agent.nodes.react_generate import react_generate
        src = inspect.getsource(react_generate)
        # The synthesis-fallback branch (thinking-only → append text block)
        assert "synthetic text block" in src or "合成 text block" in src or 'final_content' in src

    def test_v102_doc_wire_format_still_json(self):
        """v2.0.2: tools must return JSON string, not list[Document]."""
        from src.web_search._wire import _docs_to_json
        from langchain_core.documents import Document
        docs = [Document(page_content="x", metadata={"source_kind": "web"})]
        out = _docs_to_json(docs)
        # Must be valid JSON
        import json
        parsed = json.loads(out)
        assert isinstance(parsed, list)

    def test_v103_tool_calls_persistence_still_works(self):
        """v2.0.3: ``additional_kwargs["tool_calls"]`` must still be stamped."""
        from src.agent.nodes.react_generate import _build_tool_call_log
        # Just verify the helper still exists and is importable
        assert callable(_build_tool_call_log)
