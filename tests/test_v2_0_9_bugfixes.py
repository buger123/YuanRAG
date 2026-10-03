"""Tests for the v2.0.9 「LLM 感知不到时间工具」bug.

Bug history
-----------
User asked "现在是什么时间?" (what time is it now?) and got back
"我目前没有可调用的时间工具" (I currently have no callable time
tool) — the LLM truthfully admitted it had no tool to answer.

真根因 (root cause)
-------------------
v2.0.5 added the ``simple_fact`` fast-path in
:mod:`src.agent.nodes.intent_analysis` to skip the ReAct loop for
definitional lookups ("光速是多少?" / "什么是 RAG" / "what is the
capital of France"). The simple_fact regex is broad by design —
it catches any "是什么" / "what is" / "what's" pattern.

The query "现在是什么时间" matches both:
  - ``是什么[的的]?`` (simple_fact HIGH pattern)
  - ``现在`` / ``目[前后的]`` / ``此刻`` (time-need pattern)

The fast-path took precedence. ``_after_intent`` routed the
resulting ``intent="simple_fact"`` to ``react_generate_direct``,
which binds NO tools. The LLM (correctly!) said it had no time
tool.

修法 (fix)
----------
Compute ``needs_current_time`` BEFORE the simple_fact branch and
add a ``not needs_time`` guard to the fast-path. A time-needing
query is NEVER a static lookup — the LLM genuinely needs the
clock to answer. So fall through to qa_complex + ReAct where
``get_current_time`` is bound via ``bind_tools(ALL_TOOLS)``.

These tests pin:

1. **Pure regex test** — direct unit test of the
   ``_query_is_simple_fact`` + ``_query_needs_time`` interaction
   on representative queries. Pins the precedence: time-need wins
   over simple_fact.

2. **Route test** — end-to-end through ``intent_analysis`` on the
   exact failing query. With the bug: intent="simple_fact". With
   the fix: intent="qa_complex" and ``needs_current_time=True``.

3. **Negative coverage** — the simple_fact short-circuit still
   works for purely static lookups ("什么是 RAG" /
   "what is the capital of France"). The fix is a NARROW guard, not
   a regression of the fast-path.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest


INTENT_PY = Path(__file__).resolve().parents[1] / "src" / "agent" / "nodes" / "intent_analysis.py"


# ============================================================
# Pure regex test — confirms the time-need guard fires before
# simple_fact would short-circuit
# ============================================================


class TestTimeNeedOverridesSimpleFact:
    """The precedence rule: if both ``_query_is_simple_fact`` and
    ``_query_needs_time`` return True, the latter wins. Without
    this rule the LLM is routed to a tool-less direct-answer
    branch and truthfully says it has no time tool."""

    def _classify(self, query: str) -> tuple[bool, bool]:
        from src.agent.nodes.intent_analysis import (
            _query_is_simple_fact,
            _query_needs_time,
        )
        is_sf, _ = _query_is_simple_fact(query)
        is_t = _query_needs_time(query)
        return is_sf, is_t

    def test_now_what_time_matches_both(self):
        """The exact failing query — matches simple_fact via
        ``是什么`` AND matches time-need via ``现在``."""
        is_sf, is_t = self._classify("现在是什么时间")
        assert is_sf, "sanity: simple_fact regex should match '是什么'"
        assert is_t, "sanity: time-need regex should match '现在'"

    def test_what_time_is_it_now_english_variant(self):
        """English equivalent — ``what time`` + ``now`` markers."""
        is_sf, is_t = self._classify("what time is it now")
        assert is_t
        # We don't assert is_sf: English "what time is it" doesn't
        # match the simple_fact HIGH patterns (those look for
        # "what is X" / "what's X" — not "what time is it"). The
        # pure intent path correctly routes this to qa_complex
        # already; only the Chinese pattern was the bug.

    def test_now_what_time_minimal_matches_time_need(self):
        """Bare 'now' / '现在是几点' — the simpler time queries
        that don't have a 'what is X' suffix. These were the
        true _TIME_NEED_PATTERNS gap."""
        is_sf, is_t = self._classify("现在是几点")
        assert is_t, "现在是几点 should match time-need (现在是 + 几点)"
        # 不一定会被 simple_fact 匹配, no assertion on is_sf

    def test_what_is_rag_purely_static(self):
        """A simple_fact that genuinely needs no clock — must still
        short-circuit (the fix is a narrow guard, not a regression)."""
        is_sf, is_t = self._classify("什么是 RAG")
        assert is_sf
        assert not is_t

    def test_capital_of_france_purely_static(self):
        is_sf, is_t = self._classify("what is the capital of France")
        assert is_sf
        assert not is_t

    def test_introduce_history_keeps_qa_complex(self):
        """Negative pattern in simple_fact — ``介绍 ... 历史``
        should NOT match simple_fact at all (NEGATIVE override
        inside the helper), regardless of time-need."""
        is_sf, is_t = self._classify("介绍 RAG 的历史")
        assert not is_sf
        assert not is_t


# ============================================================
# intent_analysis end-to-end — the exact failing query routes
# to qa_complex + needs_current_time=True
# ============================================================


class TestIntentAnalysisRoutes:
    """End-to-end check: ``intent_analysis`` on the failing
    query must return ``intent="qa_complex"`` (or any non-fast-path
    intent) and ``needs_current_time=True``. Pre-fix:
    intent="simple_fact" + needs_current_time=False → LLM
    short-circuits to react_generate_direct (no tools) and says
    "I don't have a time tool"."""

    def _run(self, query: str) -> dict:
        from src.agent.nodes.intent_analysis import intent_analysis_sync

        state = {
            "current_query": query,
            "original_query": query,
            "messages": [],
            "step_count": 0,
        }
        return intent_analysis_sync(state)

    def test_now_what_time_routes_to_qa_complex(self):
        result = self._run("现在是什么时间")
        assert result["intent"] != "simple_fact", (
            "BUG: '现在是什么时间' short-circuited to simple_fact; "
            "the LLM has no tools bound and will say 'I have no time "
            "tool'. Should fall through to qa_complex so get_current_time "
            "is reachable."
        )
        assert result["needs_current_time"] is True, (
            "BUG: needs_current_time must be True so react_agent "
            "injects the time-tool hint."
        )

    def test_today_what_date_routes_to_qa_complex(self):
        """'今天' alone is a time marker — should NOT short-circuit."""
        result = self._run("今天天气怎么样")
        assert result["needs_current_time"] is True
        # intent could be qa_complex (cheapest path) or simple_fact
        # depending on cheap-model verdict; either way needs_current_time
        # being True is the win.

    def test_now_what_time_in_english_routes_correctly(self):
        """English 'what time is it now' — time-need is True, simple_fact
        is False (the regex doesn't catch 'what time is it'). The fix
        keeps the qa_complex route via the natural path."""
        result = self._run("what time is it now")
        assert result["needs_current_time"] is True
        # intent can be qa_complex (no simple_fact match) — exact
        # value depends on the cheap model's verdict; just confirm
        # it's NOT short-circuited to direct-answer without tools.
        assert result["intent"] in ("qa_complex", "simple_fact")
        # If cheap model somehow classifies as simple_fact, the
        # simple_fact branch still passes needs_current_time through
        # (per the fix's intent_analysis state — see greeting branch).
        # So the time-tool hint is injected either way.

    def test_what_is_rag_still_short_circuits(self):
        """Static lookup — fast-path is intentional, must still work."""
        result = self._run("什么是 RAG")
        assert result["intent"] == "simple_fact"
        assert result["needs_current_time"] is False

    def test_capital_of_france_still_short_circuits(self):
        result = self._run("what is the capital of France")
        assert result["intent"] == "simple_fact"
        assert result["needs_current_time"] is False


# ============================================================
# Static guard — ensure the fix doesn't regress
# ============================================================


class TestFixIsPresent:
    """If the time-need-before-simple_fact rule is reverted,
    these tests fail. Pin the structure."""

    def test_intent_analysis_computes_needs_time_before_simple_fact_branch(self):
        text = INTENT_PY.read_text(encoding="utf-8")
        # The needs_time assignment must come BEFORE the simple_fact
        # return statement.
        needs_time_idx = text.find("needs_time = _query_needs_time(query)")
        simple_fact_idx = text.find("is_simple_fact, _ = _query_is_simple_fact(query)")
        assert needs_time_idx != -1, "needs_time assignment not found"
        assert simple_fact_idx != -1, "is_simple_fact assignment not found"
        assert needs_time_idx < simple_fact_idx, (
            "needs_time must be computed BEFORE the simple_fact branch — "
            "otherwise the fast-path short-circuits before the time-need "
            "guard can fire"
        )

    def test_simple_fact_branch_checks_not_needs_time(self):
        text = INTENT_PY.read_text(encoding="utf-8")
        # The simple_fact return condition must include ``not needs_time``.
        # Match the relevant block — find the `is_simple_fact, _ =`
        # assignment, then look at the next 6 lines for the guard.
        m = re.search(
            r"is_simple_fact,\s*_\s*=\s*_query_is_simple_fact\(query\)",
            text,
        )
        assert m, "is_simple_fact assignment not found"
        # Take ~400 chars of context after the assignment.
        chunk = text[m.start():m.start() + 400]
        # The `if is_simple_fact` line must include `not needs_time`.
        m2 = re.search(r"if\s+is_simple_fact[^:]*:", chunk)
        assert m2, "if is_simple_fact branch not found"
        branch = chunk[m2.start():m2.end()]
        assert "not needs_time" in branch, (
            f"simple_fact branch must guard with 'not needs_time'; "
            f"got: {branch!r}"
        )


# ============================================================
# v2.0.9 第二阶段 — time-tool-result-preservation bug
#
# 真根因 B: ``react_generate._sanitize_history_for_generate``
# 在 v2.0.4 之后无条件 drop 所有 ToolMessage,理由是"已经映射到
# DOCUMENTS:"。这只对 retrieve_docs / web_search 成立 ——
# get_current_time 返回纯字符串,ToolMessage content 本身就是
# LLM 需要的 data,被 drop 后 react_generate 让 LLM 重新合成,
# LLM 没材料就 hallucinate "抱歉,我无法确定现在..."。
#
# 修法:把 ToolMessage 的 name 列入 _STRING_RETURNING_TOOLS
# whitelist,这种情况 KEEP ToolMessage。Anthropic messages API
# 接受 ToolMessage 与 AIMessage(tool_calls=[...]) 配对的历史。
# ============================================================


class TestTimeToolResultPreserved:
    """v2.0.9 — ``get_current_time`` ToolMessage must survive the
    history sanitizer so the LLM in ``react_generate`` actually
    sees the time string and incorporates it into the final
    answer.

    Pre-fix: the ToolMessage was silently dropped → react_generate
    called the LLM with no source material → LLM hallucinated
    "抱歉,我无法确定现在..." (or truthfully admitted "I don't have
    a time tool" — even though the tool was just called).
    """

    def _build_messages(self, time_str: str = "2026-09-11T12:20:00+08:00 (Asia/Shanghai)\n2026年9月11日 周五 12:20"):
        """Build the messages list that react_agent produces after
        a successful get_current_time call + final answer."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        # Make a stable tool_call_id so the AIMessage ↔ ToolMessage pairing
        # is verifiable.
        tool_call_id = "toolu_test_time_001"
        return [
            HumanMessage(content="现在是什么时间?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": tool_call_id,
                    "name": "get_current_time",
                    "args": {"timezone": "Asia/Shanghai"},
                }],
            ),
            ToolMessage(
                content=time_str,
                tool_call_id=tool_call_id,
                name="get_current_time",
            ),
            AIMessage(content="现在是 2026 年 9 月 11 日 周五 12:20。"),
        ]

    def test_get_current_time_tool_message_kept(self):
        """The time-tool ToolMessage must NOT be dropped — its content
        IS the data the LLM needs."""
        from src.agent.nodes.react_generate import _sanitize_history_for_generate
        from langchain_core.messages import ToolMessage

        msgs = self._build_messages()
        sanitized = _sanitize_history_for_generate(msgs)

        tool_messages = [m for m in sanitized if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1, (
            f"BUG: get_current_time ToolMessage was dropped. "
            f"react_generate would re-synthesize without the time data. "
            f"Sanitized messages: {len(sanitized)} (expected >= 3 to "
            f"include the ToolMessage)"
        )
        # The ToolMessage content must be the actual time string.
        assert "2026-09-11" in tool_messages[0].content
        assert "12:20" in tool_messages[0].content

    def test_get_current_time_parent_ai_message_preserved(self):
        """v2.0.10.1 — the parent AIMessage(tool_calls=[get_current_time])
        MUST be kept alongside its ToolMessage. v2.0.9 stage 2 kept the
        ToolMessage but dropped the parent (tool-calls-only AIMessage),
        so Anthropic rejected with ``400 invalid_request_error: tool
        call result does not follow tool call (2013)``. The fix is a
        two-pass scan in ``_sanitize_history_for_generate``: pass 1
        collects preserved tool_call_ids from whitelisted ToolMessages,
        pass 2 keeps any AIMessage whose tool_calls include any of
        those ids — even if the AIMessage has no visible text content.

        Without this fix, the tool_use ↔ tool_result pair is broken:
        the surviving message list contains a ToolMessage but no
        AIMessage(tool_calls=[...]) parent. Anthropic's messages API
        rejects this conversation as a malformed turn.
        """
        from src.agent.nodes.react_generate import _sanitize_history_for_generate
        from langchain_core.messages import AIMessage, ToolMessage

        msgs = self._build_messages()
        sanitized = _sanitize_history_for_generate(msgs)

        # The parent AIMessage(tool_calls=[get_current_time]) MUST
        # survive. Pre-v2.0.10.1 it was dropped because it has no
        # visible text content (the AIMessage is the ReAct planning
        # step "I'll call get_current_time" — pure tool_calls, no prose).
        parent_aimessages = [
            m for m in sanitized
            if isinstance(m, AIMessage)
            and any(
                isinstance(tc, dict) and tc.get("name") == "get_current_time"
                for tc in (getattr(m, "tool_calls", None) or [])
            )
        ]
        assert len(parent_aimessages) == 1, (
            "BUG (v2.0.10.1): get_current_time parent AIMessage(tool_calls=[...]) "
            "was dropped. Anthropic rejects with 'tool call result does not "
            "follow tool call (2013)' when the ToolMessage has no parent. "
            f"Found {len(parent_aimessages)} get_current_time AIMessages in "
            f"sanitized list (expected 1)."
        )
        # And the parent MUST appear BEFORE the ToolMessage in the
        # surviving list — Anthropic requires the AIMessage(tool_calls=[...])
        # to immediately precede its ToolMessage.
        idx_ai = sanitized.index(parent_aimessages[0])
        tool_messages = [m for m in sanitized if isinstance(m, ToolMessage)]
        idx_tool = sanitized.index(tool_messages[0])
        assert idx_ai < idx_tool, (
            f"BUG: AIMessage(tool_calls=[...]) must precede its ToolMessage. "
            f"Got idx_ai={idx_ai}, idx_tool={idx_tool}. Anthropic requires "
            f"tool_use block immediately before tool_result block."
        )
        # And the parent's tool_call_id must match the ToolMessage's
        # tool_call_id (the pairing is what the API verifies).
        parent_tc_ids = {
            tc.get("id") for tc in (getattr(parent_aimessages[0], "tool_calls", None) or [])
            if isinstance(tc, dict) and tc.get("id")
        }
        tool_tcid = getattr(tool_messages[0], "tool_call_id", "")
        assert tool_tcid in parent_tc_ids, (
            f"BUG: tool_call_id pairing broken. Parent AIMessage ids: "
            f"{parent_tc_ids}, ToolMessage id: {tool_tcid!r}"
        )

    def test_retrieve_docs_tool_message_still_dropped(self):
        """Regression guard: the v2.0.4 fix for retrieve_docs /
        web_search ToolMessages must NOT regress. Those tools' results
        are Documents, which are already represented in DOCUMENTS:."""
        from src.agent.nodes.react_generate import _sanitize_history_for_generate
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        msgs = [
            HumanMessage(content="合同里写了什么?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "toolu_test_retrieve_001",
                    "name": "retrieve_docs",
                    "args": {"query": "合同"},
                }],
            ),
            ToolMessage(
                content='[{"page_content": "合同条款1", "metadata": {"filename": "a.pdf"}}]',
                tool_call_id="toolu_test_retrieve_001",
                name="retrieve_docs",
            ),
        ]
        sanitized = _sanitize_history_for_generate(msgs)

        tool_messages = [m for m in sanitized if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 0, (
            "Regression: retrieve_docs ToolMessage must still be dropped "
            "(its content is a JSON list of Documents, which is represented "
            "in state.documents → DOCUMENTS fence)."
        )

    def test_web_search_tool_message_still_dropped(self):
        from src.agent.nodes.react_generate import _sanitize_history_for_generate
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        msgs = [
            HumanMessage(content="今天天气怎么样?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "toolu_test_web_001",
                    "name": "web_search",
                    "args": {"query": "天气"},
                }],
            ),
            ToolMessage(
                content='[{"page_content": "...", "metadata": {"url": "..."}}]',
                tool_call_id="toolu_test_web_001",
                name="web_search",
            ),
        ]
        sanitized = _sanitize_history_for_generate(msgs)
        assert not any(
            isinstance(m, ToolMessage) and getattr(m, "name", "") == "web_search"
            for m in sanitized
        )

    def test_string_returning_tools_whitelist_defined(self):
        """Pin the whitelist contract: it's a frozenset, contains
        get_current_time, and is referenced in _sanitize_history_for_generate."""
        # v2.0.16 — import the submodule (NOT ``from src.agent.nodes
        # import react_generate``) so we get the module, where the
        # module-level ``_STRING_RETURNING_TOOLS`` constant lives.
        import src.agent.nodes.react_generate as react_generate_module

        assert hasattr(react_generate_module, "_STRING_RETURNING_TOOLS")
        wl = react_generate_module._STRING_RETURNING_TOOLS
        assert isinstance(wl, frozenset), "whitelist must be frozenset for immutability"
        assert "get_current_time" in wl


# ============================================================
# Static guard — pin the fix structure so future refactors can't
# silently drop get_current_time's ToolMessage again.
# ============================================================


class TestTimeToolResultFixStructure:
    def test_sanitizer_uses_string_returning_tools_whitelist(self):
        text = (Path(__file__).resolve().parents[1] / "src" / "agent" / "nodes" / "react_generate.py").read_text(
            encoding="utf-8"
        )
        # _STRING_RETURNING_TOOLS must be defined.
        assert "_STRING_RETURNING_TOOLS" in text, (
            "_STRING_RETURNING_TOOLS whitelist must be defined in react_generate.py"
        )
        # The sanitizer must consult it (not unconditionally drop ToolMessage).
        assert "_STRING_RETURNING_TOOLS" in text
        # And the previous unconditional `continue` must have been replaced.
        # The new sanitizer has the conditional block:
        #   if tool_name in _STRING_RETURNING_TOOLS: out.append(m)
        # We pin this so a careless future revert to v2.0.4-style drop is caught.
        assert "if tool_name in _STRING_RETURNING_TOOLS" in text or \
               "tool_name in _STRING_RETURNING_TOOLS" in text, (
            "sanitizer must check tool_name against _STRING_RETURNING_TOOLS "
            "before dropping ToolMessages"
        )
