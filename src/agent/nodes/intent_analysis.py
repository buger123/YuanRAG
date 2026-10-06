"""intent_analysis node — cheap pre-flight classification before ReAct.

Three jobs:

1. **Classify intent** into one of:
   - ``"greeting"`` — pure conversational, no retrieval / search needed.
     Goes to a direct-answer fast-path (no LLM tool-call budget spent).
   - ``"summary"`` — the user is asking to operate on the WHOLE document
     ("summarize this PDF" / "what's in this report"). Goes to a
     bulk-chunks retrieval path (no embedding) then directly to generate.
   - ``"simple_fact"`` (v2.0.5) — definitional / single-fact lookup
     ("光速是多少?" / "水的沸点?" / "what is the capital of France?").
     Goes to a direct-answer branch (LLM answers from training
     knowledge without retrieval, the same way greeting works).
   - ``"qa_complex"`` — anything else. Routes to the ReAct loop where
     the LLM autonomously decides which tool(s) to call.

2. **Light typo correction** — if the user's query has obvious typos or
   ambiguous phrasing (e.g. "帮我整理一哈昨天会议纪要"), the cheap model
   rewrites it to a cleaner form. Stored in ``corrected_query`` so the
   ReAct LLM and the tool args see the cleaner version.

3. **Time-need detection** — pre-compute ``needs_current_time`` so the
   ReAct prompt can inject a hint ("此类问题涉及当前时间,先用
   get_current_time 工具获取准确时间"). Cheap-model regex-free heuristic
   (matches the v1.1.13 ``_WEB_INTENT_PATTERNS`` style). Falls back to
   letting the LLM discover the time need itself.

Why cheap-model + structured output, not the LLM router model
------------------------------------------------------------
Pre-v2.0, the router model was the same as the main model. That's
overkill for what is now a 3-way classification + typo-correction
pass. Using ``build_cheap_model`` (gpt-4o-mini / haiku-class) makes
this ~50-100x faster than using the main model — important because
greetings now skip the ReAct loop entirely and the user expects
~0.5 s response time on a "hi".

Why two-strategy parsing (with_structured_output + plain-JSON)
-------------------------------------------------------------
Same precedent as :mod:`src.agent.nodes.route` — some Anthropic-
compatible proxies (e.g. the MiniMax endpoint this project ships
against) silently return ``None`` from ``with_structured_output``
even when the model emitted valid JSON. The plain-JSON fallback
recovers those cases. See
:func:`src.llm.structured.ainvoke_structured_with_fallback`.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.legacy_helpers.greeting import _is_obvious_greeting
from src.agent.nodes.retrieve import _looks_like_summary_intent
from src.agent.state import AgentState
from src.core.logging import logger
from src.llm.factory import build_cheap_model
from src.llm.prompts import INTENT_SYSTEM, INTENT_SYSTEM_PLAIN
from src.llm.schemas import IntentDecision
from src.llm.structured import ainvoke_structured_with_fallback


# Cheap regex for time-need detection. Mirrors v1.1.13's
# _WEB_INTENT_PATTERNS recency-marker subset. Conservative: bias True
# when in doubt, since the cost of "the LLM sees a time hint" is far
# less than the cost of "the LLM hallucinates 'today' as a date".
_TIME_NEED_PATTERNS = (
    # Chinese: explicit recency
    r"今天|今早|今晚|今夜|今[天日]?早[上晨]?",
    r"昨天|今[天日]?晚[上]?|昨[日天晚]",
    r"明天|明[日天]|明早|明晚",
    r"本周|这周|这星期|本星期|本月|这个月|今年|本年",
    r"上周|上星期|上月|上个月|去年",
    r"下周|下星期|下个月|明年",
    r"最新[的的]?|最近[的的]?|近期[的的]?|近来",
    r"刚刚|刚才|之前",
    # v2.0.9 — ``现在`` standalone is a time marker.
    # The pre-v2.0.9 pattern required ``现在[的情况|状态|如何|...]*``
    # which missed "现在是什么时间" — exactly the failing query.
    # We still want specificity for downstream reasoning, so
    # keep the longer form too: a "现在的" + noun matches both
    # the standalone ``现在`` and the long form.
    r"现在[的的]?(?:情况|状态|如何|怎样|多少|价格|股价)?",
    r"此刻|目[前后的]|此刻[的的]?",
    # Chinese: "now" / "current"
    r"当前[的的]?|现状[的的]?|当下[的的]?",
    # Chinese: time-of-day
    r"几点|几号|几月|星期几|周几|礼拜几",
    # English: time-of-day
    # v2.0.32.4 — eval Stage 5.5 (2026-10-05) i18n parity fix. Pre-fix
    # "What is the current time in Beijing?" matched simple_fact
    # ("what is X") and routed to ``react_generate_direct`` (no
    # tools bound) → LLM answered "I don't have access to a
    # real-time clock" instead of calling ``get_current_time``.
    # Added the missing "current time/date" + "what time is it"
    # patterns so EN time queries get the same fast-path treatment
    # as CN ("现在几点").
    r"\bcurrent\s+(time|date|day|moment|hour|minute|second)\b",
    r"\bthe\s+(current\s+)?time\b",
    r"\bwhat\s+time\s+(is\s+it|is\s+there|now)\b",
    r"\bwhat\s+time\b",
    # English
    r"\btoday\b|\btonight\b|\byesterday\b|\btomorrow\b",
    r"\bjust\s+now\b|\bright\s+now\b|\bcurrently\b",
    r"\blatest\b|\bmost\s+recent\b|\brecently\b",
    r"\bthis\s+(week|month|year|morning|afternoon|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bnext\s+(week|month|year)\b",
    r"\blast\s+(week|month|year)\b",
    r"\bnow\b",
)

_TIME_NEED_RE = re.compile(
    "|".join(_TIME_NEED_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)


def _query_needs_time(query: str) -> bool:
    """Cheap, no-LLM check: does ``query`` likely need current time?"""
    if not query:
        return False
    return bool(_TIME_NEED_RE.search(query))


# v2.0.5 — simple-fact fast-path regex.
#
# Closes the "光速是多少?" → went to ReAct loop → agent wasted 2-3
# iterations thinking about whether to call retrieve_docs on a
# query that's clearly a static lookup. The fast-path detects
# definitional / single-fact lookups ("X 是多少" / "X 的首都" /
# "what is X") and routes them to a direct-answer branch that
# answers from training knowledge without retrieval.
#
# Two-tier design:
#   * HIGH — patterns that strongly indicate a static lookup.
#     Conservative: must match at least one.
#   * NEGATIVE — patterns that turn a HIGH match back into a
#     qa_complex. E.g. "光速是多少 这个研究领域的历史" — the HIGH
#     matches "是多少" but the question is really about the
#     history of the field, not the speed value.
#
# Why a positive+negative gate instead of just positives:
# a pure-positive list would either be huge (catching every
# phrasing) or miss common variants; the negative list lets us
# err on the side of qa_complex when the question has any
# explanatory / historical flavor.
_SIMPLE_FACT_HIGH_PATTERNS = (
    # Chinese: "X 是多少" quantity / measurement questions
    r"是[多几]少|有[多几]少|多[大长远高重宽深]|几[个名条块只张双对层]|几[十百千万亿]|多大|多长|多远",
    # Chinese: physical constants
    r"光速[是为]?多少|声速[是为]?多少|重力[是为]?多少|引力常量|普朗克|阿伏伽德罗|玻尔兹曼",
    # Chinese: definition / naming / abbreviation
    r"[的之]?[全正式]?称[是为叫]?(?:什么|啥)|英文[是为]|缩写[是为]|简写[是为]|全称[是为]",
    # Match 什/什么 in either order — "什么是 RAG" / "什么叫 HTTP".
    # Both orderings are common in Chinese.
    r"什[么啥][是为叫]?|叫[作是]?什[么啥]|什[么啥]叫",
    # Chinese: capitals / current holders / official answer
    r"首[都城][是为]?(?:什么|哪个|哪儿)|的(?:首[都城])|国[名是之]|国家[是之]",
    r"现任[的的]?(?:总[统理]|主席|首相)|在位[的的]?",
    # Chinese: standard / formula / theory lookup
    r"[的之]?定义[是为]?|是什么[的的]?",
    # English: "what is X" / "what's X"
    r"\bwhat\s+is\s+(?:the\s+)?(?:\w+\s+){0,4}\??",
    r"\bwhat'?s\s+(?:the\s+)?(?:\w+\s+){0,4}\??",
    r"\bhow\s+(?:many|much|old|tall|big|long|far|fast|heavy|deep|wide|high)\b",
    r"\bwhen\s+(?:was|is|did)\b",
    r"\bwhere\s+(?:is|are|was|were)\b",
    r"\bwho\s+(?:is|was|are|were)\b",
    r"\bdefine\b|\bdefinition\s+of\b|\bmeaning\s+of\b",
    r"\bcapital\s+of\b|\babbreviation\s+(?:for|of)\b",
)

_SIMPLE_FACT_NEGATIVE_PATTERNS = (
    # Chinese: explanation / analysis / history (turns HIGH off)
    r"为什么|怎么[么办做来]|如何|解释|介绍|分析|比较|区别|历史|原理|演变|由来|原因|影响|意义",
    r"谈谈|说说|讲讲|聊聊|描述",
    # English
    r"\bwhy\b|\bhow\s+to\b|\bexplain\b|\banalyze\b|\banalyse\b|\bcompare\b|\bdifference\b|\bhistory\s+of\b",
)

_SIMPLE_FACT_HIGH_RE = re.compile(
    "|".join(_SIMPLE_FACT_HIGH_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)
_SIMPLE_FACT_NEGATIVE_RE = re.compile(
    "|".join(_SIMPLE_FACT_NEGATIVE_PATTERNS),
    re.IGNORECASE | re.UNICODE,
)


def _query_is_simple_fact(query: str) -> tuple[bool, float]:
    """Return ``(is_simple_fact, confidence)``.

    ``confidence`` is a quick "how sure are we" number used by tests
    to rank borderline matches. HIGH match → ``1.0``; no NEGATIVE
    match → ``1.0``; HIGH + NEGATIVE → ``0.0`` (override).

    Used as fast-path 3 in :func:`intent_analysis`. The cheap LLM
    classifier handles the "definitely" qa_complex cases; this
    regex pass catches the "obviously static lookup" cases so they
    skip the LLM entirely.
    """
    if not query:
        return False, 0.0
    high = _SIMPLE_FACT_HIGH_RE.search(query)
    if not high:
        return False, 0.0
    if _SIMPLE_FACT_NEGATIVE_RE.search(query):
        return False, 0.0
    return True, 1.0


# v2.0.29.9 (Phase 8) — hybrid verbatim trigger detection.
#
# Layer 1 (zero-cost regex) catches obvious legal / medical / financial
# / contractual / quoted-source queries in CN + EN. Layer 2 (cheap-LLM
# via ``IntentDecision.verbatim_needed``) catches paraphrased cases
# ("according to the contract, how does clause 12 read?").
#
# Why both layers:
#   - Pure regex would miss paraphrases; pure LLM would add latency +
#     cost on every query that doesn't need it.
#   - The regex pre-filter is ~µs; only missed queries hit the LLM.
#   - Layer 2 piggy-backs on the existing cheap-LLM intent classifier
#     (single call already happening), so the marginal cost is zero —
#     just one extra boolean in the structured output.
#
# CN triggers: 法律规定 / 合同条款 / 诊断标准 / 财务数字 / 原话 /
# 原文是 / verbatim (EN) / 法条 / 原文
# EN triggers: verbatim / exact wording / what does the contract say /
# according to the agreement / per the agreement / what does the
# regulation say / as stated in
#
# Conservative: false-positive cost = LLM walks extractive prompt
# (no retrieval quality loss). false-negative cost = LLM still walks
# normal synthesis on a verbatim query → rephrases original text →
# Phase 6 verifier rule_engine may catch it but is post-hoc. Bias
# toward recall (catch more verbatim candidates, accept some noise).
_HIGH_PRECISION_TRIGGERS_CN = (
    "法律规定", "合同条款", "诊断标准", "财务数字",
    "原话", "原文是", "verbatim", "法条", "原文",
)
_HIGH_PRECISION_TRIGGERS_EN = (
    "verbatim", "exact wording", "what does the contract say",
    "according to the agreement", "per the agreement",
    "what does the regulation say", "as stated in",
)
_HIGH_PRECISION_TRIGGERS_RE = re.compile(
    r"(?:" + "|".join(map(re.escape, _HIGH_PRECISION_TRIGGERS_CN)) + r")"
    r"|(?:" + "|".join(map(re.escape, _HIGH_PRECISION_TRIGGERS_EN)) + r")",
    re.IGNORECASE | re.UNICODE,
)


def _query_is_high_precision(query: str) -> bool:
    """v2.0.29.9 (Phase 8) — Layer 1 verbatim trigger (zero-cost regex).

    Returns True iff the query strongly suggests verbatim extraction
    from a referenced source. Layer 2 (cheap-LLM ``verbatim_needed``
    field in :class:`IntentDecision`) handles paraphrased cases
    that the regex misses; both layers are OR'd in
    :func:`intent_analysis` to produce the final ``high_precision``
    state value.

    Short-circuits on empty / whitespace queries → False.
    """
    if not query or not query.strip():
        return False
    return bool(_HIGH_PRECISION_TRIGGERS_RE.search(query))


def _resolve_high_precision(
    user_value: Optional[str],
    *,
    regex_hit: bool,
    llm_verbatim: bool = False,
) -> str:
    """v2.0.29.9 (Phase 8) — tristate resolution with user-override priority.

    User override semantics (per user decision 2026-09-28):
      - user_value == "off" → always "off", no override (even if detect fires).
      - user_value == "on"  → always "on", no detection needed.
      - user_value == "auto" (or None / unset) → run hybrid detect;
        return "on" if regex OR cheap-LLM fired, else "auto".

    The returned string is what gets stamped into the ``__delta__``
    payload so the FSM's :func:`src.agent.fsm.after_intent` predicate
    sees a tristate (not bool).
    """
    if user_value == "off":
        return "off"
    if user_value == "on":
        return "on"
    # auto / None → hybrid detect
    if regex_hit or llm_verbatim:
        return "on"
    return "auto"


async def intent_analysis(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """Classify intent + (cheap-model) typo-correct + time-need flag.

    v2.0.22 (Item 7 Step 6) — async-generator protocol: yields
    exactly one ``("__delta__", delta_dict)`` terminator carrying the
    state delta (intent / corrected_query / needs_current_time fields).
    The FSM (``src.agent.fsm._run_node``) collects it and merges into
    the running state. ``step`` is accepted-and-ignored because this
    node never emits per-step wire events of its own.

    Decision tree:

      1. ``_is_obvious_greeting(query)`` → ``intent="greeting"``,
         ``needs_current_time=False``, no LLM call. Fastest path.
      2. ``_looks_like_summary_intent(query)`` → ``intent="summary"``,
         ``needs_current_time=False``, no LLM call. Bulk-chunks path.
      3. (v2.0.5) ``_query_is_simple_fact(query)`` AND no time-need
         → ``intent="simple_fact"``, ``needs_current_time=False``,
         no LLM call. Skips ReAct for definitional lookups.
      4. Otherwise → invoke the cheap model with ``IntentDecision``
         schema. Returns ``intent="qa_complex"`` (default) plus
         ``corrected_query`` (or original if no typo) and
         ``needs_current_time`` (LLM-decided).
    """
    query = (
        state.get("current_query")
        or state.get("original_query")
        or ""
    )

    # v2.0.9 — time-need computed upfront so the simple_fact
    # fast-path can defer to the ReAct loop when the query has
    # recency markers. The simple_fact regex is broad by design
    # (``是什么`` matches "现在是什么时间"); without this override
    # the LLM is short-circuited to ``react_generate_direct`` which
    # has no tools bound — it then says "I don't have a time tool"
    # because the get_current_time binding is exclusive to the
    # ReAct agent. Cost of the regex on the non-time path is ~1 µs
    # (negligible).
    needs_time = _query_needs_time(query)

    # Fast-path 1: greeting. No LLM cost — regex match only.
    # Mirrors v1.1.3 logic; the greeting regex still hits "hi" /
    # "你是谁" / "thanks" / etc.
    #
    # v2.0.28.12 — time-need guard. The cheap-model LLM in the
    # slow-path below sometimes classifies time-sensitive queries
    # ("现在几点了") as ``"greeting"`` (the LLM has no "time"
    # category and rounds to the closest fit). Pre-v2.0.28.12 the
    # hard override below only forced qa_complex for simple_fact /
    # summary, leaving the greeting path open. Routing a time query
    # through greeting hits ``react_generate_direct`` (no tools
    # bound) and the LLM truthfully says "I have no callable time
    # tool" + suggests the user re-send — verbatim from DIRECT_SYSTEM
    # line 234-237. Cheap regex guard; doesn't affect non-time
    # queries.
    if _is_obvious_greeting(query) and not needs_time:
        # v2.0.22 (Item 7 Step 7) — the node is now the wire event's
        # single source of truth. Yield the ``intent`` event BEFORE
        # ``__delta__`` so the FSM's per-node ``for kind, payload in
        # events: yield FSMEvent(...)`` loop forwards it. The FSM
        # used to read ``state["intent"]`` after merge in a hardcoded
        # ``if current == "intent_analysis"`` branch — gone now.
        yield (
            "intent",
            {"intent_value": "greeting", "corrected_query": None},
        )
        yield ("__delta__", {
            "intent": "greeting",
            "corrected_query": None,
            "needs_current_time": needs_time,
            # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
            # ``merge()`` since this delta doesn't carry it explicitly.
            # v2.0.29.9 (Phase 8) — verbatim-extraction tristate.
            # Greeting path doesn't reach the extractive node (after_intent
            # only routes qa_complex → extractive), but we still stamp
            # ``high_precision`` so the state field has a known shape.
            # User override respected verbatim.
            "high_precision": _resolve_high_precision(
                state.get("high_precision"),
                regex_hit=False,
            ),
        })
        return

    # Fast-path 2: summary intent. Same regex as ``retrieve._looks_like_summary_intent``
    # — no embedding cost for "what's in this doc" questions.
    #
    # v2.0.9 — time-need override: the summary regex matches ``是什么``
    # (broadly), which catches "现在是什么时间" — exactly the failing
    # time-tool query. The summary path also binds no tools
    # (it dumps every chunk of the thread's docs into ``documents``
    # and routes to ``react_generate``), so a time query routed here
    # would also see "I don't have a time tool". Same precedence rule
    # as simple_fact: time-need wins.
    if _looks_like_summary_intent(query) and not needs_time:
        yield (
            "intent",
            {"intent_value": "summary", "corrected_query": None},
        )
        yield ("__delta__", {
            "intent": "summary",
            "corrected_query": None,
            "needs_current_time": False,
            # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
            # ``merge()`` since this delta doesn't carry it explicitly.
            # v2.0.29.9 (Phase 8) — verbatim-extraction tristate.
            # Summary path doesn't reach extractive (after_intent
            # blocks summary → extractive). User override respected.
            "high_precision": _resolve_high_precision(
                state.get("high_precision"),
                regex_hit=False,
            ),
        })
        return

    # Fast-path 3 (v2.0.5): simple-fact lookup. No retrieval, no
    # tool calls — the LLM answers from training knowledge. The
    # graph's _after_intent branch routes this to the same direct-
    # answer node as "greeting" (see ``graph.py:_after_intent``).
    #
    # v2.0.9 — time-need override: if the query has recency markers,
    # DO NOT short-circuit. ``react_generate_direct`` binds NO tools,
    # so the LLM would be forced to either hallucinate the date or
    # truthfully answer "I don't have a callable time tool". Both
    # are worse than routing through ReAct where ``get_current_time``
    # is bound. Cheap regex; doesn't affect non-time queries.
    #
    # v2.0.29.9 (Phase 8) — verbatim trigger override. The verbatim
    # regex is checked BEFORE simple_fact because legal / contractual
    # queries often contain ``什么`` (e.g. "法律规定第 12 条是什么",
    # "合同条款是怎么说的") which would otherwise match the simple_fact
    # regex and route to react_generate_direct (no retrieval, no
    # tools). Verbatim queries STRONGLY need retrieval (to quote from
    # the source), so the verbatim signal wins over the simple_fact
    # signal when both fire. Order matters; this comment block belongs
    # to the verbatim fast-path above.
    user_hp_value = state.get("high_precision")
    if user_hp_value != "off":
        regex_hit = _query_is_high_precision(query)
        if regex_hit or user_hp_value == "on":
            logger.debug(
                "intent_analysis: verbatim trigger (regex=%s, user=%s); "
                "forcing qa_complex + high_precision=on",
                regex_hit, user_hp_value,
            )
            yield (
                "intent",
                {"intent_value": "qa_complex", "corrected_query": None},
            )
            yield ("__delta__", {
                "intent": "qa_complex",
                "corrected_query": None,
                "needs_current_time": needs_time,
                "high_precision": "on",
            })
            return

    is_simple_fact, _ = _query_is_simple_fact(query)
    if is_simple_fact and not needs_time:
        yield (
            "intent",
            {"intent_value": "simple_fact", "corrected_query": None},
        )
        yield ("__delta__", {
            "intent": "simple_fact",
            "corrected_query": None,
            "needs_current_time": False,
            # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
            # ``merge()`` since this delta doesn't carry it explicitly.
            # v2.0.29.9 (Phase 8) — verbatim-extraction tristate.
            # Default "auto" preserved here: simple_fact fast-path
            # doesn't reach the extractive node regardless (after_intent
            # checks ``intent not in ("greeting", "simple_fact", ...)``
            # before routing to react_generate_extractive). Stamping
            # "auto" keeps the state field known-shape for downstream
            # consumers (runner reads it for ChatRequest hydration).
            "high_precision": _resolve_high_precision(
                state.get("high_precision"),
                regex_hit=False,
            ),
        })
        return

    # Default: complex question → qa_complex via cheap-model classification.
    # Time-need detection is a regex (cheap). Then the LLM call only does
    # typo correction + intent confirm.
    model = build_cheap_model(temperature=0.0)

    user_msg = f"Query: {query}"
    decision = await ainvoke_structured_with_fallback(
        model=model,
        schema=IntentDecision,
        structured_messages=[
            SystemMessage(content=INTENT_SYSTEM),
            HumanMessage(content=user_msg),
        ],
        plain_messages=[
            SystemMessage(content=INTENT_SYSTEM_PLAIN),
            HumanMessage(content=user_msg),
        ],
        op_name="intent_analysis",
    )

    # Fallback: cheap model returned nothing. Conservative default =
    # complex (ReAct loop), preserve original query, accept the regex
    # time-need signal so a "今天天气" still gets the time hint even
    # when the LLM structured output failed.
    if decision is None:
        logger.warning(
            "intent_analysis: both strategies failed; defaulting to qa_complex"
        )
        yield (
            "intent",
            {"intent_value": "qa_complex", "corrected_query": query},
        )
        yield ("__delta__", {
            "intent": "qa_complex",
            "corrected_query": query,
            "needs_current_time": needs_time,
            # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
            # ``merge()`` since this delta doesn't carry it explicitly.
            # v2.0.29.9 (Phase 8) — verbatim tristate. LLM didn't run
            # so no verbatim detection; respect user override only.
            "high_precision": _resolve_high_precision(
                state.get("high_precision"),
                regex_hit=False,
                llm_verbatim=False,
            ),
        })
        return

    # LLM-confirmed intent; trust the LLM's intent over the regex
    # unless it's a non-standard value (defensive — schema validator
    # already constrained this to {"greeting","summary","simple_fact",
    # "qa_complex"}).
    intent = decision.intent
    if intent not in ("greeting", "summary", "simple_fact", "qa_complex"):
        logger.debug(
            f"intent_analysis: LLM returned unexpected intent {intent!r}; "
            "falling back to qa_complex"
        )
        intent = "qa_complex"

    # v2.0.9 — HARD time-need override (post-LLM). Even with the
    # regex fast-path guards above, the cheap-model classifier is
    # non-deterministic: it sometimes returns ``intent="simple_fact"``
    # or ``"summary"`` for "现在是什么时间" directly (because the LLM
    # itself sees ``是什么`` and matches its own internal "definitional
    # question" heuristic). When that happens the verdict wins over
    # our regex guard, the query lands in ``react_generate_direct``,
    # and the LLM truthfully says "I have no time tool".
    #
    # v2.0.28.12 — extend the override to cover ``"greeting"`` too.
    # Observed: the cheap model sometimes rounds "现在几点了" to
    # ``"greeting"`` (the LLM has no "time" category, so it picks
    # the closest conversational shape). Routing greeting → direct
    # also hits react_generate_direct and the same DIRECT_SYSTEM
    # "re-send to trigger retrieval" text surfaces. Cost of extending
    # the override: a tiny fraction of borderline "what time is it"
    # queries now go through ReAct — they pay one extra LLM call
    # (route_query) but get the right tool binding.
    #
    # The fix: when the regex says ``needs_time=True``, force the
    # route to qa_complex regardless of what the cheap model decided.
    # The ReAct loop will inject the time-tool hint via the existing
    # ``needs_current_time`` flag, and the LLM will reach
    # ``get_current_time`` through ``bind_tools(ALL_TOOLS)``.
    # Cost: a tiny fraction of queries that the cheap model would
    # have classified as simple_fact/summary now go through ReAct —
    # they pay one extra LLM call (route_query) but get the right
    # tool binding.
    if needs_time and intent in ("greeting", "simple_fact", "summary"):
        logger.debug(
            f"intent_analysis: regex needs_time=True but LLM returned "
            f"intent={intent!r}; forcing qa_complex so get_current_time "
            f"is reachable"
        )
        intent = "qa_complex"

    yield (
        "intent",
        {
            "intent_value": intent,
            # ``corrected_query`` overrides ``current_query`` only if
            # the LLM detected a real typo. Otherwise None → downstream
            # keeps the original.
            "corrected_query": (decision.corrected_query or None),
            # v2.0.29.9 (Phase 8) — surface the verbatim verdict on
            # the wire so future frontend PRs (PR-2+) can render a
            # "verbatim mode" badge on the user message immediately
            # after intent classification (currently the wire event is
            # ignored; FS is forward-compat).
            "verbatim_needed": bool(getattr(decision, "verbatim_needed", False)),
        },
    )
    yield ("__delta__", {
        "intent": intent,
        # ``corrected_query`` overrides ``current_query`` only if the
        # LLM detected a real typo. Otherwise None → downstream keeps
        # the original.
        "corrected_query": (decision.corrected_query or None),
        # Combine the LLM's verdict with the regex pre-filter via OR —
        # if either says "needs time", the ReAct prompt injects the
        # time-tool hint. Conservative bias.
        "needs_current_time": bool(decision.needs_current_time) or needs_time,
        # v2.0.22 (Item 7 Step 5) — step_count auto-injected by
        # ``merge()`` since this delta doesn't carry it explicitly.
        # v2.0.29.9 (Phase 8) — verbatim extraction tristate. The LLM
        # Layer-2 verdict (``verbatim_needed``) is OR'd with the
        # Layer-1 regex hit (already short-circuited above) — but
        # when we reach here the regex missed, so only the LLM verdict
        # matters. User override wins over both (handled by helper).
        "high_precision": _resolve_high_precision(
            state.get("high_precision"),
            regex_hit=False,
            llm_verbatim=bool(getattr(decision, "verbatim_needed", False)),
        ),
    })


def intent_analysis_sync(state: AgentState) -> dict:
    """Sync wrapper for tests; the graph calls the async-gen entry directly.

    v2.0.22 (Item 7 Step 6) — drives the async-gen via
    :func:`asyncio.run` and returns the ``__delta__`` payload. Keeps
    the legacy synchronous test API (``test_v2_0_9_bugfixes.py``)
    working without rewriting every callsite to ``async for``.
    """
    async def _drive() -> dict:
        async for kind, payload in intent_analysis(state):
            if kind == "__delta__":
                return payload
        raise RuntimeError(
            "intent_analysis exhausted without yielding __delta__"
        )
    return asyncio.run(_drive())


__all__ = ["intent_analysis", "intent_analysis_sync"]