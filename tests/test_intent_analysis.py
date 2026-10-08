"""Regression tests for the ``intent_analysis`` node.

v2.0 — ReAct rewrite's pre-flight classifier. Three jobs:

1. Classify intent into one of {greeting, summary, qa_complex}.
2. Cheap typo-correction (``corrected_query``).
3. Time-need flag (``needs_current_time``) so the ReAct prompt
   injects a get_current_time hint for "今天天气"-style queries.

These tests pin the three fast-path regexes + the time-need
detector. The LLM structured-output path is exercised end-to-end
in verify_v2.0.py (no LLM in the test env).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Greeting fast-path
# ---------------------------------------------------------------------------


def test_greeting_fast_path_for_hi():
    from src.agent.legacy_helpers.greeting import _is_obvious_greeting
    from src.agent.nodes.intent_analysis import _query_needs_time

    assert _is_obvious_greeting("hi") is True
    assert _is_obvious_greeting("hello") is True
    assert _is_obvious_greeting("你好") is True


def test_greeting_fast_path_for_identity_question():
    """v1.1.3 — "你是谁" / "who are you" must short-circuit (no retrieval)."""
    from src.agent.legacy_helpers.greeting import _is_obvious_greeting

    assert _is_obvious_greeting("你是谁") is True
    assert _is_obvious_greeting("who are you") is True


def test_greeting_fast_path_for_thanks():
    from src.agent.legacy_helpers.greeting import _is_obvious_greeting

    assert _is_obvious_greeting("谢谢") is True
    assert _is_obvious_greeting("thanks") is True


def test_greeting_fast_path_does_not_match_complex():
    from src.agent.legacy_helpers.greeting import _is_obvious_greeting

    assert _is_obvious_greeting("今天天气怎么样") is False
    assert _is_obvious_greeting("总结一下这份文档") is False
    assert _is_obvious_greeting("what's the Q3 revenue") is False


# ---------------------------------------------------------------------------
# Summary fast-path
# ---------------------------------------------------------------------------


def test_summary_fast_path_for_summarize_chinese():
    """v1.1.x — summary regex still hits '总结一下这个文档' / '概括全文'."""
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    assert _looks_like_summary_intent("总结一下这个文档") is True
    assert _looks_like_summary_intent("概括全文") is True


def test_summary_fast_path_for_summarize_english():
    """v2.0.29.4 (Phase 4 PR-1) — English summary patterns were
    intentionally REMOVED from the regex.

    Pre-PR-1: ``summarize this document`` matched the unanchored
    English patterns (``summari[sz]e|overview|gist|tl;?dr|...``)
    and fired the summary bulk path.

    Post-PR-1: the regex is Chinese-only AND start-anchored
    (``^(?:总结|摘要|概括|归纳|概述|总览|全文|整篇|介绍|讲讲|
    讲一下|说说|说一下|讲讲内容|主要内容|列一下|列出来|
    分析一下|解释一下|解释下)``). English summary requests like
    ``summarize this document`` fall through to the normal
    embed → hybrid → rerank path, which is the documented intent
    of PR-1 #4 (test ``test_summary_intent_re_ignores_definitional_qa``
    pins the Chinese-only rule with explicit English cases).

    The rationale is in the PR-1 plan: removing English patterns
    reduces false positives (the OLD regex would fire on mid-sentence
    mentions like ``"what is the summary of this chapter"`` because
    it was unanchored). A future PR can add English patterns back
    via a separate, intentionally designed matcher (e.g.
    spaCy-based intent detection) — but Phase 4 PR-1 deliberately
    ships without them.
    """
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    # English summary requests now fall through to hybrid+rerank.
    assert _looks_like_summary_intent("summarize this document") is False
    assert _looks_like_summary_intent("give me an overview") is False
    # Mid-sentence English summary mentions also no longer fire
    # (the OLD unanchored regex would have matched these).
    assert _looks_like_summary_intent("what is the summary of this") is False


def test_summary_fast_path_does_not_match_retrieval():
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    assert _looks_like_summary_intent("what is the Q3 revenue") is False
    assert _looks_like_summary_intent("Q3营收是多少") is False


# ---------------------------------------------------------------------------
# Time-need detection (cheap regex, conservative bias True)
# ---------------------------------------------------------------------------


def test_time_need_matches_chinese_recency():
    from src.agent.nodes.intent_analysis import _query_needs_time

    assert _query_needs_time("今天天气怎么样") is True
    assert _query_needs_time("昨天的新闻") is True
    assert _query_needs_time("最新的股价") is True
    assert _query_needs_time("最近有什么新闻") is True


def test_time_need_matches_english_recency():
    from src.agent.nodes.intent_analysis import _query_needs_time

    assert _query_needs_time("what's the latest news") is True
    assert _query_needs_time("today's weather") is True
    assert _query_needs_time("stock price yesterday") is True


def test_time_need_matches_time_of_day():
    """'几点' / '几号' must trigger even without an explicit
    recency word.

    v2.0.32.4 (2026-10-05) — i18n parity: English time-of-day
    markers now also trigger. Pre-fix the regex missed "what time
    is it" / "current time" → simple_fact fast-path caught the
    "what is X" prefix → route_decision=direct (no tool bound) →
    LLM answered "I don't have access to a real-time clock". Added
    ``current time/date/day`` + ``the time`` + ``what time is it``
    patterns so EN time queries get the same treatment as CN.
    """
    from src.agent.nodes.intent_analysis import _query_needs_time

    assert _query_needs_time("现在几点了") is True
    assert _query_needs_time("今天是几号") is True
    # v2.0.32.4 — EN time-of-day now matches too.
    assert _query_needs_time("What time is it?") is True
    assert _query_needs_time("What time is it in Beijing?") is True
    assert _query_needs_time("What is the current time in Beijing?") is True
    assert _query_needs_time("Tell me the time") is True
    assert _query_needs_time("What is the current date?") is True


def test_time_need_does_not_match_eternal_questions():
    """Pure-knowledge questions (no recency marker) should NOT
    trigger the time hint — saves tool calls + avoids polluting
    the prompt with irrelevant clock data."""
    from src.agent.nodes.intent_analysis import _query_needs_time

    assert _query_needs_time("什么是相对论") is False
    assert _query_needs_time("project Q3 revenue") is False
    assert _query_needs_time("Python怎么用") is False


def test_time_need_handles_empty_query():
    """Defensive — empty / None-ish queries never match."""
    from src.agent.nodes.intent_analysis import _query_needs_time

    assert _query_needs_time("") is False


# ---------------------------------------------------------------------------
# intent_analysis node — fast-path returns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_analysis_greeting_no_llm():
    """Greeting fast-path must NOT invoke the LLM (cheapest path).

    The function yields immediately with intent='greeting' when
    _is_obvious_greeting hits. If a future refactor accidentally
    always invokes the LLM, this test fails (no model in test env).

    v2.0.22 (Item 7 Step 5) — ``step_count`` is auto-injected by
    ``merge()`` rather than returned by the node. We assert that
    path here so a regression in merge() is caught.

    v2.0.22 (Item 7 Step 6) — node is async-gen: drive with
    ``async for`` and collect the ``__delta__`` payload.
    """
    from src.agent.fsm import merge
    from src.agent.nodes.intent_analysis import intent_analysis
    from unittest.mock import patch

    state = {"step_count": 0, "current_query": "hi"}
    delta: dict = {}
    with patch("src.agent.nodes.intent_analysis.build_cheap_model") as mock_build:
        async for kind, payload in intent_analysis(state):
            if kind == "__delta__":
                delta = payload
                break

    assert mock_build.call_count == 0
    assert delta["intent"] == "greeting"
    assert delta["needs_current_time"] is False
    # step_count isn't in the delta anymore; merge() adds it.
    assert "step_count" not in delta
    merged = merge(state, delta)
    assert merged["step_count"] == 1


@pytest.mark.asyncio
async def test_intent_analysis_summary_no_llm():
    """Summary fast-path also skips the LLM.

    v2.0.22 (Item 7 Step 6) — drive the async-gen via ``async for``.
    """
    from src.agent.nodes.intent_analysis import intent_analysis
    from unittest.mock import patch

    state = {"current_query": "总结一下这个文档"}
    delta: dict = {}
    with patch("src.agent.nodes.intent_analysis.build_cheap_model") as mock_build:
        async for kind, payload in intent_analysis(state):
            if kind == "__delta__":
                delta = payload
                break

    assert mock_build.call_count == 0
    assert delta["intent"] == "summary"
    assert delta["needs_current_time"] is False


# ---------------------------------------------------------------------------
# v2.0.32.8 (Stage 5.8 follow-up, 2026-10-07) — eval-found doc-query
# misroutes. The simple_fact fast-path was too eager: 3 eval cases
# (golden-004-docx / golden-007-xlsx / golden-009 multi-turn turn 2)
# contain a definitional surface ("是什么" / "多少") but require doc
# retrieval. Extending `_SIMPLE_FACT_NEGATIVE_PATTERNS` + multi-turn
# guard routes these to qa_complex (ReAct + tools bound).
# ---------------------------------------------------------------------------


def test_simple_fact_negative_blocks_author_stance():
    """golden-004-docx-zh — "作者对 X 持什么态度" must NOT be simple_fact."""
    from src.agent.nodes.intent_analysis import _query_is_simple_fact

    for q in (
        "这篇作文讨论的核心概念是什么?作者对'通话膨胀'持什么态度?",
        "作者对气候变化的看法是什么?",
        "你的立场是什么?",  # NEW: also catches "立场"
        "这篇论文的核心观点是什么?",
    ):
        ok, _ = _query_is_simple_fact(q)
        assert not ok, f"expected NOT simple_fact for {q!r}"


def test_simple_fact_negative_blocks_personal_finance_doc():
    """golden-007-xlsx-budget-zh — personal budget / spending queries
    must NOT be simple_fact (training knowledge can't answer
    "what's MY August food budget")."""
    from src.agent.nodes.intent_analysis import _query_is_simple_fact

    for q in (
        "我8月份餐饮预算是多少?实际花了多少?",
        "这个月预算还剩多少?",
        "本月开支总共多少?",
        "实际花费和预算差多少?",
    ):
        ok, _ = _query_is_simple_fact(q)
        assert not ok, f"expected NOT simple_fact for {q!r}"


def test_simple_fact_negative_blocks_explicit_doc_reference():
    """Single-turn doc-reference ("这篇 / 本文 / 这份文档") must NOT
    be simple_fact (the answer is in the user's uploaded doc, not
    training knowledge)."""
    from src.agent.nodes.intent_analysis import _query_is_simple_fact

    for q in (
        "这篇文档讨论了什么?",
        "本文的核心观点是什么?",
        "这份文件的作者是谁?",
    ):
        ok, _ = _query_is_simple_fact(q)
        assert not ok, f"expected NOT simple_fact for {q!r}"


def test_simple_fact_negative_no_regression_on_existing_positives():
    """Guard: the new NEGATIVE patterns must not flip existing
    simple_fact positives. Per test_v2_0_5_bugfixes.py pin list."""
    from src.agent.nodes.intent_analysis import _query_is_simple_fact

    for q in (
        "光速是多少",  # 多少 without personal-finance keyword
        "什么是 RAG",  # 什么 without stance/concept/doc-ref
        "HTTP 是什么缩写",
        "水的化学式是什么",
        "what is the capital of France",
        "how many planets are there",
        "what's HTTP",
    ):
        ok, _ = _query_is_simple_fact(q)
        assert ok, f"regression: expected simple_fact for {q!r}"


@pytest.mark.asyncio
async def test_intent_analysis_multi_turn_guard_skips_simple_fact():
    """golden-009-multi-turn-zh turn 2 — "那腾讯音乐发债多少钱?"
    must NOT route to simple_fact in multi-turn context, even though
    the regex matches "多少钱" as HIGH. The follow-up needs retrieval
    context from the prior turn (which retrieved 1.txt chunks)."""
    from langchain_core.messages import AIMessage, HumanMessage
    from src.agent.nodes.intent_analysis import intent_analysis
    from unittest.mock import patch

    # Multi-turn state: turn 1 = HumanMessage + AIMessage, turn 2 = HumanMessage
    state = {
        "current_query": "那腾讯音乐发债多少钱?",
        "messages": [
            HumanMessage(content="王者荣耀发生了什么事?"),
            AIMessage(content="王者荣耀 ... 高校认证 ..."),
            HumanMessage(content="那腾讯音乐发债多少钱?"),
        ],
    }
    delta: dict = {}
    # Cheap-model mock — should NOT be called because multi-turn guard
    # forces fall-through to qa_complex (which IS the cheap-model LLM
    # call). We patch it to capture the call; if the guard works the
    # call IS made (qa_complex path) and `intent` ends up qa_complex.
    with patch("src.agent.nodes.intent_analysis.build_cheap_model") as mock_build:
        from src.llm.schemas import IntentDecision
        from unittest.mock import AsyncMock

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=IntentDecision(
            intent="qa_complex",
            corrected_query=None,
            needs_current_time=False,
        ))
        mock_build.return_value = mock_model
        async for kind, payload in intent_analysis(state):
            if kind == "__delta__":
                delta = payload
                break

    # qa_complex reached (not simple_fact), confirming multi-turn guard fired
    assert delta["intent"] == "qa_complex", (
        f"multi-turn guard should force qa_complex; got {delta['intent']!r}"
    )


@pytest.mark.asyncio
async def test_intent_analysis_single_turn_still_simple_facts():
    """Guard: multi-turn guard must NOT block first-turn simple_fact
    routing. "光速是多少" with only one HumanMessage → still simple_fact."""
    from langchain_core.messages import HumanMessage
    from src.agent.nodes.intent_analysis import intent_analysis
    from unittest.mock import patch

    state = {
        "current_query": "光速是多少",
        "messages": [HumanMessage(content="光速是多少")],
    }
    delta: dict = {}
    with patch("src.agent.nodes.intent_analysis.build_cheap_model") as mock_build:
        async for kind, payload in intent_analysis(state):
            if kind == "__delta__":
                delta = payload
                break

    # Simple_fact fast-path: NO LLM call (mock_build.call_count == 0)
    assert mock_build.call_count == 0
    assert delta["intent"] == "simple_fact"
