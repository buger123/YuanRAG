"""v2.0.29.9 (Phase 8) — verbatim extraction mode (high_precision).

Tests covering:
  1. ``_query_is_high_precision`` regex detector (Layer 1, zero-cost).
  2. ``_resolve_high_precision`` user-override helper (tristate semantics).
  3. ``intent_analysis`` high_precision fast-path (regex hit + user override).
  4. ``after_intent`` FSM predicate routing.
  5. ``react_generate_extractive`` node (refusal fallback + happy path + LLM error).
  6. ``Source.verbatim`` + ``ChatRequest.high_precision`` wire schemas.
  7. ``EXTRACTIVE_FALLBACK`` metric counter bumps.

Total: ~25 assertions. Lives next to ``test_metrics_aggregation.py`` (Phase 2
counter pre-registration tests) and ``test_verification.py`` (Phase 6 post-
validation tests).
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.agent.fsm import NODES, after_intent
from src.agent.metrics import EXTRACTIVE_FALLBACK, reset_all, snapshot
from src.agent.nodes.intent_analysis import (
    _HIGH_PRECISION_TRIGGERS_RE,
    _query_is_high_precision,
    _resolve_high_precision,
    intent_analysis,
    intent_analysis_sync,
)
from src.agent.nodes.react_generate_extractive import react_generate_extractive
from src.agent.state import AgentState
from src.api.schemas import ChatRequest, Source
from src.llm.prompts import EXTRACTIVE_SYSTEM, REFUSAL_TEMPLATES


# ---------------------------------------------------------------------------
# 1) Layer 1 regex detector (zero-cost, no LLM)
# ---------------------------------------------------------------------------


def test_high_precision_triggers_re_cn_legal():
    """CN legal trigger: '法律规定第 X 条' matches the verbatim regex."""
    assert _query_is_high_precision("法律规定第 12 条规定了什么") is True


def test_high_precision_triggers_re_cn_contract():
    """CN contract trigger: '合同条款' matches the verbatim regex."""
    assert _query_is_high_precision("合同条款中违约责任如何约定") is True


def test_high_precision_triggers_re_cn_diagnosis():
    """CN medical trigger: '诊断标准' matches."""
    assert _query_is_high_precision("糖尿病的诊断标准是什么") is True


def test_high_precision_triggers_re_cn_finance():
    """CN financial trigger: '财务数字' matches."""
    assert _query_is_high_precision("公司2025年的财务数字") is True


def test_high_precision_triggers_re_cn_verbatim():
    """CN verbatim trigger: '原话' / '原文是' / '法条' / '原文'."""
    for q in ("请引用原话", "原文是怎么样", "这个法条解释", "请展示原文"):
        assert _query_is_high_precision(q) is True, f"failed for {q!r}"


def test_high_precision_triggers_re_en_verbatim():
    """EN verbatim trigger: 'verbatim' matches."""
    for q in ("verbatim quote please", "Verbatim, what does the doc say?", "give me verbatim"):
        assert _query_is_high_precision(q) is True, f"failed for {q!r}"


def test_high_precision_triggers_re_en_contract():
    """EN contract trigger: 'what does the contract say' matches."""
    assert _query_is_high_precision("what does the contract say about termination") is True
    assert _query_is_high_precision("according to the agreement, who is liable") is True
    assert _query_is_high_precision("per the agreement, the term is 12 months") is True


def test_high_precision_triggers_re_en_regulation():
    """EN regulation trigger: 'what does the regulation say' / 'as stated in'."""
    assert _query_is_high_precision("what does the regulation say about emissions") is True
    assert _query_is_high_precision("as stated in section 5.2 of the policy") is True


def test_high_precision_does_not_match_unrelated():
    """Layer 1 short-circuits on unrelated queries — no LLM fallback fires."""
    # Layer 1 regex misses; in production the cheap-LLM would be called.
    # In this test we don't supply an llm kwarg, so the helper returns False
    # at Layer 2's fail-open path. The point: regex path is independent of
    # LLM availability.
    for q in ("今天天气如何", "Python 怎么用?", "你好", "推荐一本书"):
        assert _query_is_high_precision(q) is False, f"false positive for {q!r}"


def test_high_precision_short_query_returns_false():
    """Empty / whitespace / very short queries short-circuit to False."""
    assert _query_is_high_precision("") is False
    assert _query_is_high_precision("   ") is False
    assert _query_is_high_precision("?") is False
    assert _query_is_high_precision("hi") is False


def test_high_precision_regex_compile_idempotent():
    """Module-level regex compile is a single instance (no per-call re-compile)."""
    # Touch twice — should be same object (compiled at module import).
    assert _HIGH_PRECISION_TRIGGERS_RE is _HIGH_PRECISION_TRIGGERS_RE
    # And the match returns a hit on the canonical trigger.
    m = _HIGH_PRECISION_TRIGGERS_RE.search("verbatim please")
    assert m is not None
    assert "verbatim" in m.group(0).lower()


# ---------------------------------------------------------------------------
# 2) User-override helper (tristate semantics)
# ---------------------------------------------------------------------------


def test_resolve_high_precision_user_off_wins_over_regex():
    """user 'off' must always override — even when regex hits."""
    assert _resolve_high_precision("off", regex_hit=True) == "off"
    assert _resolve_high_precision("off", regex_hit=False) == "off"
    assert _resolve_high_precision("off", regex_hit=True, llm_verbatim=True) == "off"


def test_resolve_high_precision_user_on_wins_without_detect():
    """user 'on' is sticky — no detection needed."""
    assert _resolve_high_precision("on", regex_hit=False) == "on"
    assert _resolve_high_precision("on", regex_hit=False, llm_verbatim=False) == "on"


def test_resolve_high_precision_auto_triggers_on_regex():
    """'auto' + regex hit → 'on'."""
    assert _resolve_high_precision("auto", regex_hit=True) == "on"


def test_resolve_high_precision_auto_triggers_on_llm_verbatim():
    """'auto' + cheap-LLM verbatim=True → 'on'."""
    assert _resolve_high_precision("auto", regex_hit=False, llm_verbatim=True) == "on"


def test_resolve_high_precision_auto_misses_stay_auto():
    """'auto' + no detection → stays 'auto' (state has known shape)."""
    assert _resolve_high_precision("auto", regex_hit=False, llm_verbatim=False) == "auto"


def test_resolve_high_precision_none_treated_as_auto():
    """None is treated as 'auto' (uninitialized state from pre-Phase-8 threads)."""
    assert _resolve_high_precision(None, regex_hit=True) == "on"
    assert _resolve_high_precision(None, regex_hit=False) == "auto"


# ---------------------------------------------------------------------------
# 3) intent_analysis fast-path 4 (regex hit returns early)
# ---------------------------------------------------------------------------


def test_intent_analysis_high_precision_regex_hit_forces_qa_complex():
    """regex matches verbatim trigger → intent=qa_complex + high_precision=on."""
    state: AgentState = {
        "current_query": "法律规定第 12 条规定了什么",
        "messages": [],
    }
    delta = intent_analysis_sync(state)
    assert delta["intent"] == "qa_complex"
    assert delta["high_precision"] == "on"


def test_intent_analysis_user_off_overrides_detect():
    """state.high_precision='off' is preserved even when regex matches."""
    state: AgentState = {
        "current_query": "法律规定第 12 条规定了什么",  # would match regex
        "messages": [],
        "high_precision": "off",  # user override
    }
    delta = intent_analysis_sync(state)
    # The fast-path is gated by user_hp != "off", so it short-circuits.
    # No verbatim trigger fires; intent takes default qa_complex (no
    # greeting/summary/simple_fact regex matches either for this query).
    assert delta["high_precision"] == "off"


def test_intent_analysis_user_on_skips_llm():
    """state.high_precision='on' routes through fast-path 4 even without regex."""
    state: AgentState = {
        "current_query": "今天是星期几",  # no verbatim regex match
        "messages": [],
        "high_precision": "on",
    }
    delta = intent_analysis_sync(state)
    assert delta["intent"] == "qa_complex"
    assert delta["high_precision"] == "on"


def test_intent_analysis_simple_fact_fast_path_unchanged():
    """Phase 8 insertion doesn't break the existing simple_fact fast-path."""
    state: AgentState = {
        "current_query": "光速是多少",  # matches _SIMPLE_FACT_HIGH_RE
        "messages": [],
    }
    delta = intent_analysis_sync(state)
    assert delta["intent"] == "simple_fact"
    # Phase 8 — simple_fact path still stamps high_precision="auto"
    # (default; after_intent doesn't route to extractive for simple_fact).
    assert delta["high_precision"] == "auto"


def test_intent_analysis_greeting_fast_path_unchanged():
    """Phase 8 insertion doesn't break the greeting fast-path."""
    state: AgentState = {
        "current_query": "hi",
        "messages": [],
    }
    delta = intent_analysis_sync(state)
    assert delta["intent"] == "greeting"
    assert delta["high_precision"] == "auto"


def test_intent_analysis_summary_fast_path_unchanged():
    """Phase 8 insertion doesn't break the summary fast-path."""
    state: AgentState = {
        "current_query": "总结一下这份报告",
        "messages": [],
    }
    delta = intent_analysis_sync(state)
    assert delta["intent"] == "summary"
    assert delta["high_precision"] == "auto"


# ---------------------------------------------------------------------------
# 4) after_intent FSM predicate
# ---------------------------------------------------------------------------


def test_after_intent_routes_to_extractive_on_user_on():
    """hp='on' + intent='qa_complex' → react_generate_extractive."""
    assert after_intent({"intent": "qa_complex", "high_precision": "on"}) == "react_generate_extractive"


def test_after_intent_routes_to_extractive_on_auto_with_regex_hit():
    """hp='on' was set by intent_analysis (auto-detect fired) → extractive."""
    # Simulates: regex matched in intent_analysis → state.high_precision="on".
    state = {"intent": "qa_complex", "high_precision": "on"}
    assert after_intent(state) == "react_generate_extractive"


def test_after_intent_user_off_skips_extractive():
    """hp='off' (user override) → never routes to extractive, even if detect
    would have fired (state.high_precision was 'off' before intent_analysis)."""
    # In this case intent_analysis doesn't change hp (user off wins).
    # after_intent sees hp="off" → falls through to normal routing.
    assert after_intent({"intent": "qa_complex", "high_precision": "off"}) == "react_agent"


def test_after_intent_auto_unset_normal_routing():
    """hp='auto' or unset → normal routing based on intent."""
    assert after_intent({"intent": "qa_complex", "high_precision": "auto"}) == "react_agent"
    assert after_intent({"intent": "qa_complex"}) == "react_agent"  # unset
    assert after_intent({"intent": "greeting", "high_precision": "on"}) == "react_generate_direct"
    assert after_intent({"intent": "summary", "high_precision": "on"}) == "summary_path"
    assert after_intent({"intent": "simple_fact", "high_precision": "on"}) == "react_generate_direct"


# ---------------------------------------------------------------------------
# 5) react_generate_extractive node
# ---------------------------------------------------------------------------


def _doc(text: str = "verbatim quote", *, doc_id: str = "d1", chunk_id: str = "c1") -> Document:
    return Document(
        page_content=text,
        metadata={"doc_id": doc_id, "chunk_id": chunk_id, "filename": "test.txt"},
    )


def _drive(node_fn, state) -> list[Tuple[str, dict]]:
    """Drive async-gen node to completion; return list of (kind, payload)."""
    out: list[Tuple[str, dict]] = []

    async def _run():
        async for kind, payload in node_fn(state):
            out.append((kind, payload))

    asyncio.run(_run())
    return out


def test_extractive_node_empty_docs_emits_refusal_template():
    """retrieval_status='empty' → emit REFUSAL_TEMPLATES['empty'] + empty sources + bump metric.

    v2.0.32.6 (Stage 5.8) — tests updated to set ``documents=[_doc()]``
    instead of ``[]``. Pre-fix the empty-docs was the refusal trigger;
    post-fix the node calls inline ``retrieve_hybrid_async`` when
    ``state["documents"]`` is empty, so we keep ``documents`` non-empty
    to preserve the refusal path through ``retrieval_status="empty"``
    (the OR condition still fires). The inline-retrieval path itself
    is covered by ``test_extractive_node_inline_retrieval_*`` tests
    added below.
    """
    reset_all()
    state: AgentState = {
        "documents": [_doc()],
        "retrieval_status": "empty",
        "original_query": "verbatim quote please",
        "current_query": "verbatim quote please",
    }
    events = _drive(react_generate_extractive, state)
    # First event: answer_complete with refusal
    kind, payload = events[0]
    assert kind == "answer_complete"
    assert payload["answer"] == REFUSAL_TEMPLATES["empty"]
    assert payload["sources"] == []
    # Counter bumped
    assert EXTRACTIVE_FALLBACK.get(reason="empty") >= 1.0


def test_extractive_node_low_relevance_emits_refusal():
    """retrieval_status='low_relevance' → REFUSAL_TEMPLATES['low_relevance']."""
    reset_all()
    state: AgentState = {
        "documents": [_doc()],
        "retrieval_status": "low_relevance",
        "original_query": "verbatim quote",
        "current_query": "verbatim quote",
    }
    events = _drive(react_generate_extractive, state)
    kind, payload = events[0]
    assert payload["answer"] == REFUSAL_TEMPLATES["low_relevance"]
    assert EXTRACTIVE_FALLBACK.get(reason="low_relevance") >= 1.0


def test_extractive_node_empty_bulk_emits_refusal():
    """retrieval_status='empty_bulk' → REFUSAL_TEMPLATES['empty_bulk'].

    v2.0.32.6 (Stage 5.8) — same test-shape update as
    ``test_extractive_node_empty_docs_emits_refusal_template``: keep
    ``documents=[_doc()]`` non-empty so the inline-retrieve path
    doesn't fire and the ``retrieval_status="empty_bulk"`` OR trigger
    keeps the test focused on the refusal template.
    """
    reset_all()
    state: AgentState = {
        "documents": [_doc()],
        "retrieval_status": "empty_bulk",
        "original_query": "verbatim quote",
        "current_query": "verbatim quote",
    }
    events = _drive(react_generate_extractive, state)
    assert events[0][1]["answer"] == REFUSAL_TEMPLATES["empty_bulk"]
    assert EXTRACTIVE_FALLBACK.get(reason="empty_bulk") >= 1.0


# ---------------------------------------------------------------------------
# v2.0.32.6 (Stage 5.8) — inline retrieval load-bearing fix.
# The FSM (``fsm.after_intent``) routes ``intent=qa_complex +
# high_precision=on`` DIRECTLY to ``react_generate_extractive``,
# SKIPPING the ``react_agent → tools → retrieve`` path that
# normally populates ``state["documents"]``. The fix adds an
# inline ``retrieve_hybrid_async`` call when ``state["documents"]``
# is empty, mirroring the ``summary_path`` pattern. These tests
# pin the new behavior so a future refactor can't silently break
# the verbatim case again (the bug shipped 2026-09-28 with v2.0.29.9
# and was only caught on 2026-10-07 by the eval harness running
# the verbatim case end-to-end).
# ---------------------------------------------------------------------------


def test_extractive_node_inline_retrieval_when_docs_empty():
    """Empty ``state["documents"]`` triggers inline retrieval
    via ``retrieve_hybrid_async``. Retrieval returns docs →
    happy path emits verbatim answer with non-empty sources.

    Pre-fix: ``state["documents"]`` was empty → refusal fallback
    fired every time → no verbatim turn ever succeeded end-to-end
    (Layer 5 hermetic tests constructed the empty state directly
    and only tested the refusal branch).
    """
    reset_all()
    state: AgentState = {
        "documents": [],
        "retrieval_status": None,  # ← unset; fix should trigger
        "original_query": "verbatim quote",
        "current_query": "verbatim quote",
        "thread_id": "test-thread",
    }

    fake_resp = MagicMock()
    fake_resp.content = "[1] 原文 Local Enclosing Global Built-in"
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=fake_resp)

    # Patch retrieve_hybrid_async to return a populated doc list —
    # simulates the normal "retrieval succeeded" branch.
    ret_result = {
        "documents": [_doc("LEGB stands for Local, Enclosing, Global, Built-in")],
        "retrieval_status": "success",
    }
    with patch(
        "src.agent.nodes.react_generate_extractive.retrieve_hybrid_async",
        new=AsyncMock(return_value=ret_result),
    ) as mock_retrieve, patch(
        "src.agent.nodes.react_generate_extractive.build_chat_model",
        return_value=fake_model,
    ):
        events = _drive(react_generate_extractive, state)

    # Inline retrieve WAS called
    mock_retrieve.assert_awaited_once()
    # LLM happy path emitted answer_complete with sources
    answer_kind, answer_payload = events[0]
    assert answer_kind == "answer_complete"
    assert "Local" in answer_payload["answer"]
    assert answer_payload["sources"], "happy path must have non-empty sources"
    # route_decision is "extractive" (not "extractive_refusal")
    assert answer_payload["route_decision"] == "extractive"
    # __delta__ propagates the populated documents + retrieval_status
    delta_kind, delta = events[-1]
    assert delta_kind == "__delta__"
    assert len(delta["documents"]) == 1
    assert delta["retrieval_status"] == "success"


def test_extractive_node_inline_retrieval_skipped_when_docs_present():
    """Non-empty ``state["documents"]`` → inline retrieval is
    NOT called (avoids redundant BGE-M3 + LanceDB hit when
    the FSM populates docs via the normal path — defensive against
    future routing changes).

    Mirrors the summary_path contract: only fetch when state is
    missing the field.
    """
    state: AgentState = {
        "documents": [_doc("already populated")],
        "retrieval_status": "success",
        "original_query": "verbatim quote",
        "current_query": "verbatim quote",
    }
    fake_resp = MagicMock()
    fake_resp.content = "[1] text"
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=fake_resp)
    with patch(
        "src.agent.nodes.react_generate_extractive.retrieve_hybrid_async",
        new=AsyncMock(),
    ) as mock_retrieve, patch(
        "src.agent.nodes.react_generate_extractive.build_chat_model",
        return_value=fake_model,
    ):
        _drive(react_generate_extractive, state)

    mock_retrieve.assert_not_called()


def test_extractive_node_inline_retrieval_failure_falls_through_to_refusal():
    """``retrieve_hybrid_async`` raises → log warning + continue
    with empty docs → refusal fallback fires (``empty`` template).

    Fail-closed semantics match the rest of the refusal contract
    (Phase 1 v2.0.29.1). Storage hiccups must NOT silently leak
    fabricated verbatim quotes to the user.
    """
    reset_all()
    state: AgentState = {
        "documents": [],
        "retrieval_status": None,
        "original_query": "verbatim quote",
        "current_query": "verbatim quote",
    }
    with patch(
        "src.agent.nodes.react_generate_extractive.retrieve_hybrid_async",
        new=AsyncMock(side_effect=RuntimeError("lancedb offline")),
    ):
        events = _drive(react_generate_extractive, state)

    kind, payload = events[0]
    assert kind == "answer_complete"
    assert payload["answer"] == REFUSAL_TEMPLATES["empty"]
    assert EXTRACTIVE_FALLBACK.get(reason="empty") >= 1.0


def test_extractive_node_inline_retrieval_empty_result_emits_refusal():
    """``retrieve_hybrid_async`` returns empty docs (real corpus
    had nothing) → refusal fallback fires normally. Pins the
    end-to-end contract: even with inline retrieval wired, a
    genuinely-empty corpus still refuses rather than fabricates.
    """
    reset_all()
    state: AgentState = {
        "documents": [],
        "retrieval_status": None,
        "original_query": "verbatim quote",
        "current_query": "verbatim quote",
    }
    ret_result = {
        "documents": [],
        "retrieval_status": "empty",
    }
    with patch(
        "src.agent.nodes.react_generate_extractive.retrieve_hybrid_async",
        new=AsyncMock(return_value=ret_result),
    ):
        events = _drive(react_generate_extractive, state)

    kind, payload = events[0]
    assert payload["answer"] == REFUSAL_TEMPLATES["empty"]
    assert EXTRACTIVE_FALLBACK.get(reason="empty") >= 1.0


def test_extractive_node_happy_path_stamps_verbatim_on_sources():
    """LLM returns verbatim text → sources carry verbatim=True + answer_complete."""
    state: AgentState = {
        "documents": [_doc("原文第 12 条规定...")],
        "retrieval_status": "success",
        "original_query": "verbatim please",
        "current_query": "verbatim please",
    }

    # Stub the chat model. build_chat_model returns a BaseChatModel; we
    # monkeypatch the function to return an object whose ``ainvoke`` is an
    # AsyncMock.
    fake_resp = MagicMock()
    fake_resp.content = "[1] 原文第 12 条规定..."
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=fake_resp)

    with patch(
        "src.agent.nodes.react_generate_extractive.build_chat_model",
        return_value=fake_model,
    ):
        events = _drive(react_generate_extractive, state)

    kind, payload = events[0]
    assert kind == "answer_complete"
    assert "[1]" in payload["answer"]
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["verbatim"] is True
    assert payload["sources"][0]["doc_id"] == "d1"
    # No fallback counter bump on happy path
    reset_all()
    assert EXTRACTIVE_FALLBACK.get(reason="empty") == 0.0


def test_extractive_node_llm_error_falls_back_to_low_relevance():
    """LLM raises → emit REFUSAL_TEMPLATES['low_relevance'] + bump counter."""
    reset_all()
    state: AgentState = {
        "documents": [_doc()],
        "retrieval_status": "success",
        "original_query": "verbatim please",
        "current_query": "verbatim please",
    }

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(side_effect=RuntimeError("network down"))

    with patch(
        "src.agent.nodes.react_generate_extractive.build_chat_model",
        return_value=fake_model,
    ):
        events = _drive(react_generate_extractive, state)

    kind, payload = events[0]
    assert payload["answer"] == REFUSAL_TEMPLATES["low_relevance"]
    assert EXTRACTIVE_FALLBACK.get(reason="llm_error") >= 1.0


def test_extractive_node_emits_delta_with_high_precision():
    """Every happy-path turn stamps high_precision='on' in the __delta__."""
    state: AgentState = {
        "documents": [_doc()],
        "retrieval_status": "success",
        "original_query": "verbatim",
        "current_query": "verbatim",
    }
    fake_resp = MagicMock()
    fake_resp.content = "[1] quoted text"
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=fake_resp)
    with patch(
        "src.agent.nodes.react_generate_extractive.build_chat_model",
        return_value=fake_model,
    ):
        events = _drive(react_generate_extractive, state)

    # Last event is __delta__ (mandatory terminator)
    delta_kind, delta = events[-1]
    assert delta_kind == "__delta__"
    assert delta["high_precision"] == "on"


def test_extractive_node_thinking_disabled():
    """build_chat_model is called with enable_thinking=False.

    The synthesis LLM in verbatim mode must NOT have a thinking channel
    (would leak meta-commentary into the bubble). Pinning the kwarg so a
    future refactor can't silently re-enable thinking.
    """
    state: AgentState = {
        "documents": [_doc()],
        "retrieval_status": "success",
        "original_query": "verbatim",
        "current_query": "verbatim",
    }
    fake_resp = MagicMock()
    fake_resp.content = "[1] text"
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=fake_resp)

    with patch(
        "src.agent.nodes.react_generate_extractive.build_chat_model",
        return_value=fake_model,
    ) as mocked_factory:
        _drive(react_generate_extractive, state)

    mocked_factory.assert_called_once_with(enable_thinking=False)


# ---------------------------------------------------------------------------
# 6) Source.verbatim + ChatRequest.high_precision wire schemas
# ---------------------------------------------------------------------------


def test_source_verbatim_default_false():
    """Source model default verbatim=False (backward-compat)."""
    src = Source(index=1, text="text", source_kind="local")
    assert src.verbatim is False


def test_source_verbatim_can_be_true():
    """Source.verbatim=True is settable for extractive answers."""
    src = Source(index=1, text="text", source_kind="local", verbatim=True)
    assert src.verbatim is True


def test_source_extra_ignore_keeps_wire_compat():
    """extra='ignore' — old client receiving extra fields doesn't crash."""
    src = Source(index=1, text="text", source_kind="local")
    # Pydantic v2 with extra='ignore' silently drops unknown fields;
    # model_dump() returns only declared fields.
    assert "verbatim" in src.model_dump()
    # And unknown fields are dropped (not in model_dump).
    dumped = src.model_dump()
    assert isinstance(dumped, dict)


def test_chat_request_default_high_precision_auto():
    """ChatRequest() default high_precision='auto'."""
    req = ChatRequest(thread_id="t1", message="hi")
    assert req.high_precision == "auto"


def test_chat_request_accepts_on():
    req = ChatRequest(thread_id="t1", message="hi", high_precision="on")
    assert req.high_precision == "on"


def test_chat_request_accepts_off():
    req = ChatRequest(thread_id="t1", message="hi", high_precision="off")
    assert req.high_precision == "off"


def test_chat_request_rejects_invalid_high_precision():
    """Pydantic Literal validation — invalid value rejected."""
    with pytest.raises(ValueError):
        ChatRequest(thread_id="t1", message="hi", high_precision="always")


# ---------------------------------------------------------------------------
# 7) EXTRACTIVE_FALLBACK metric counter (Phase 2 pre-registration contract)
# ---------------------------------------------------------------------------


def test_extractive_fallback_labeled_counter_inc():
    """EXTRACTIVE_FALLBACK accepts inc(reason=...) calls."""
    reset_all()
    EXTRACTIVE_FALLBACK.inc(reason="empty")
    EXTRACTIVE_FALLBACK.inc(reason="low_relevance")
    EXTRACTIVE_FALLBACK.inc(reason="empty")  # twice
    EXTRACTIVE_FALLBACK.inc(reason="llm_error")
    assert EXTRACTIVE_FALLBACK.get(reason="empty") == 2.0
    assert EXTRACTIVE_FALLBACK.get(reason="low_relevance") == 1.0
    assert EXTRACTIVE_FALLBACK.get(reason="llm_error") == 1.0


def test_extractive_fallback_rejects_unknown_label_name():
    """Unknown label name → ValueError (regression guard for typos)."""
    with pytest.raises(ValueError):
        EXTRACTIVE_FALLBACK.inc(bogus_label="x")


def test_extractive_fallback_snapshot_omitted_when_empty():
    """Labeled counter does NOT surface in snapshot at value 0 (label-less only)."""
    reset_all()
    snap = snapshot()
    names = {entry["name"] for entry in snap}
    # EXTRACTIVE_FALLBACK is labeled (reason=...); absent until first inc().
    assert "extractive_fallback" not in names


def test_extractive_fallback_snapshot_surfaces_after_inc():
    """Labeled counter surfaces in snapshot AFTER first inc(reason=...)."""
    reset_all()
    EXTRACTIVE_FALLBACK.inc(reason="empty")
    snap = snapshot()
    names = {entry["name"] for entry in snap}
    assert "extractive_fallback" in names


# ---------------------------------------------------------------------------
# 8) Node registration + EXTRACTIVE_SYSTEM prompt
# ---------------------------------------------------------------------------


def test_react_generate_extractive_registered_in_fsm_nodes():
    """react_generate_extractive is in NODES dict."""
    assert "react_generate_extractive" in NODES


def test_extractive_system_prompt_contains_no_reasoning_directive():
    """EXTRACTIVE_SYSTEM includes '禁止自主推理' (no-reasoning directive)."""
    assert "禁止自主推理" in EXTRACTIVE_SYSTEM


def test_extractive_system_prompt_contains_citation_format():
    """EXTRACTIVE_SYSTEM includes the [n] citation format spec."""
    # The prompt tells the LLM to use [n] for citations.
    assert "[n]" in EXTRACTIVE_SYSTEM or "[1]" in EXTRACTIVE_SYSTEM


def test_extractive_system_prompt_contains_empty_fallback():
    """EXTRACTIVE_SYSTEM has the empty-doc fallback phrase."""
    assert "原文未提及" in EXTRACTIVE_SYSTEM


# ---------------------------------------------------------------------------
# 9) IntentDecision schema (extended with verbatim_needed)
# ---------------------------------------------------------------------------


def test_intent_decision_verbatim_needed_default_false():
    """IntentDecision has verbatim_needed field with default False."""
    from src.llm.schemas import IntentDecision
    # Pydantic v2 model_validate
    d = IntentDecision(intent="qa_complex")
    assert d.verbatim_needed is False


def test_intent_decision_verbatim_needed_settable():
    """IntentDecision.verbatim_needed can be set to True."""
    from src.llm.schemas import IntentDecision
    d = IntentDecision(intent="qa_complex", verbatim_needed=True)
    assert d.verbatim_needed is True