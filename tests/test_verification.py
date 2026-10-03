"""v2.0.29.7 (Phase 6) — second-pass verification (rule engine + verifier LLM).

The two-layer verification subsystem (regex-based rule engine +
cheap-model verifier LLM) is the post-generation safety net for
hallucination optimization Phase 6. These tests pin:

  1. **Rule engine** — regex extraction of dates / currency /
     percentages / article numbers from arbitrary text, plus the
     "does this value appear in the doc?" consistency check.
  2. **should_trigger_verification** — zero-cost gate that decides
     whether to run the verifier at all on a given turn.
  3. **verifier LLM** — fail-closed behavior on LLM errors,
     consistent / inconsistent result passthrough (with a stub LLM).
  4. **FSM node** — the ``skipped / consistent / regen /
     fallback_refusal`` branches + wire event payload shape +
     counter bumps.

Per [[yuanrag-hallucination-optimization]] Phase 6: this is
load-bearing for "I made it up" hallucination (LLM cites a doc
that doesn't actually contain the claimed fact). The Phase 1
refusal contract catches "I don't know" cases; this catches
"trust me bro" cases.

What we lock here
-----------------
1. Rule engine extracts structured fields with the expected
   normalization (ISO dates / symbols / CN numerals → ASCII).
2. check_consistency flags every answer field NOT present in the
   doc text (any equivalent surface form counts as present).
3. should_trigger_verification returns True for queries with
   concrete-fact keywords / answers with extractable fields; False
   for chitchat.
4. verify_answer (with stubbed LLM) returns the expected
   VerificationResult + fail-closes on LLM exception.
5. verify_answer_node yields the correct wire event + state delta
   per branch.
6. FSM PREDICATES registers verify_answer with the correct entry
   point (after_react_generate routes through it).
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

# Make the project root importable so we can hit internal modules
# without packaging the verifier into the test tree (matches the
# pattern used by Layer 5 verifiers).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 1. Rule engine — extraction
# ---------------------------------------------------------------------------


def test_rule_engine_extracts_iso_date():
    """ISO date ``2026-09-28`` is extracted with canonical normalized form."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("事件发生在 2026-09-28 这一天。")
    assert len(fields) == 1
    assert fields[0].field_type == "date"
    assert fields[0].raw_text == "2026-09-28"
    assert fields[0].normalized == "2026-09-28"


def test_rule_engine_extracts_cn_date():
    """CN form ``2026年9月28日`` is normalized to ``2026-09-28``."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("2026年9月28日发布")
    assert len(fields) == 1
    assert fields[0].field_type == "date"
    assert fields[0].normalized == "2026-09-28"


def test_rule_engine_extracts_currency_symbol():
    """``¥1,000.50`` is normalized to ``1000.50`` (symbol + commas stripped)."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("价格 ¥1,000.50")
    assert len(fields) == 1
    assert fields[0].field_type == "currency"
    assert fields[0].normalized == "1000.50"


def test_rule_engine_extracts_currency_cnyuan():
    """``100元`` form is normalized to ``100``."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("售价 100元")
    assert any(f.field_type == "currency" and f.normalized == "100" for f in fields)


def test_rule_engine_extracts_percentage():
    """``12.5%`` is normalized to ``12.5%`` (whitespace stripped)."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("12.5% 的用户")
    assert len(fields) == 1
    assert fields[0].field_type == "percentage"
    assert fields[0].normalized == "12.5%"


def test_rule_engine_extracts_article_number():
    """``第 12 条`` is normalized to ASCII digit form ``12``."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("依据第 12 条规定")
    assert len(fields) == 1
    assert fields[0].field_type == "article_number"
    assert fields[0].normalized == "12"


def test_rule_engine_extracts_article_number_cn_numerals():
    """``第十二条`` (CN numerals) is normalized to ASCII ``12``."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("依据第十二条规定")
    assert len(fields) == 1
    assert fields[0].field_type == "article_number"
    assert fields[0].normalized == "12"


def test_rule_engine_extracts_article_number_english():
    """``Article 12`` and ``Section 3.2`` English forms are normalized."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("Per Article 12 and Section 3.2 of the contract")
    article_fields = [f for f in fields if f.field_type == "article_number"]
    assert len(article_fields) == 2
    assert {f.normalized for f in article_fields} == {"12", "3.2"}


def test_rule_engine_empty_text_returns_empty_list():
    """Empty input → empty list (no fields)."""
    from src.agent.verification.rule_engine import extract_fields

    assert extract_fields("") == []
    assert extract_fields(None) == []  # type: ignore[arg-type]


def test_rule_engine_no_fact_returns_empty_list():
    """Free-form text with no extractable facts → empty list."""
    from src.agent.verification.rule_engine import extract_fields

    assert extract_fields("你好,很高兴认识你。") == []


def test_rule_engine_dedupe_same_field():
    """Same field mentioned twice in source text → single entry (location-preserved)."""
    from src.agent.verification.rule_engine import extract_fields

    fields = extract_fields("在 2026-09-28 发布,2026-09-28 这一天值得纪念")
    # Same date appears twice — only one entry due to (type, normalized) dedupe.
    date_fields = [f for f in fields if f.field_type == "date"]
    assert len(date_fields) == 1


# ---------------------------------------------------------------------------
# 2. Rule engine — consistency
# ---------------------------------------------------------------------------


def test_rule_engine_consistent_when_value_in_docs():
    """Value present in doc text → no mismatch."""
    from src.agent.verification.rule_engine import (
        extract_fields, check_consistency,
    )

    answer = "事件发生在 2026-09-28"
    doc = "根据记录,事件确切的日期是 2026-09-28。"
    fields = extract_fields(answer)
    assert check_consistency(fields, doc) == []


def test_rule_engine_consistent_when_canonical_form_in_docs():
    """Answer's normalized form matches even when doc uses a different surface form.

    This is the cross-form match: answer says "¥1,000.50", doc says
    "1000.50" — both normalize to "1000.50" so the rule engine
    treats them as consistent. (Verifier LLM is the second-pass
    safety net for any false positives from this normalization.)
    """
    from src.agent.verification.rule_engine import (
        extract_fields, check_consistency,
    )

    answer = "价格 ¥1,000.50"
    doc = "本次交易总价为 1000.50 元。"
    fields = extract_fields(answer)
    mismatches = check_consistency(fields, doc)
    assert mismatches == [], f"expected consistent, got {mismatches}"


def test_rule_engine_mismatch_when_value_missing():
    """Value NOT in doc text → mismatch flagged."""
    from src.agent.verification.rule_engine import (
        extract_fields, check_consistency,
    )

    answer = "事件发生在 2026-09-28"
    doc = "事件发生在 2025-09-28。"
    fields = extract_fields(answer)
    mismatches = check_consistency(fields, doc)
    assert len(mismatches) == 1
    assert mismatches[0].field.normalized == "2026-09-28"


def test_rule_engine_empty_inputs_no_mismatch():
    """Empty answer or doc → no mismatch (vacuously consistent)."""
    from src.agent.verification.rule_engine import check_consistency, ExtractedField

    assert check_consistency([], "doc text") == []
    assert check_consistency(
        [ExtractedField("date", "2026-09-28", "2026-09-28", 0)], ""
    ) == []


# ---------------------------------------------------------------------------
# 3. should_trigger_verification
# ---------------------------------------------------------------------------


def test_should_trigger_verification_query_with_amount_keyword():
    """Query with concrete-fact keyword → True."""
    from src.agent.verification.verifier import should_trigger_verification

    assert should_trigger_verification("事件是 2026 年的", "这个项目多少钱?") is True


def test_should_trigger_verification_query_with_article_keyword():
    """Query asking for specific clause → True."""
    from src.agent.verification.verifier import should_trigger_verification

    assert should_trigger_verification(
        "看文档", "依据第几条规定?"
    ) is True


def test_should_trigger_verification_answer_with_extractable_fields():
    """Answer contains a date / currency → True (even without keyword match)."""
    from src.agent.verification.verifier import should_trigger_verification

    assert should_trigger_verification(
        "事件发生在 2026-09-28", "介绍一下背景"
    ) is True


def test_should_trigger_verification_neither_returns_false():
    """Chitchat without extractable fields → False (zero-cost path)."""
    from src.agent.verification.verifier import should_trigger_verification

    assert should_trigger_verification("你好", "你好") is False
    assert should_trigger_verification("这是一个普通回答。", "讲个笑话") is False


# ---------------------------------------------------------------------------
# 4. verifier LLM (with stub)
# ---------------------------------------------------------------------------


class _StubLLMConsistent:
    """Stub LLM that returns ``{"consistent": true, ...}`` JSON."""

    async def ainvoke(self, messages):
        return _StubMessage(
            content=json.dumps({
                "consistent": True,
                "mismatches": [],
                "reason": "all claims supported by docs",
            })
        )


class _StubLLMMismatch:
    """Stub LLM that returns ``{"consistent": false, ...}`` with a mismatch."""

    async def ainvoke(self, messages):
        return _StubMessage(
            content=json.dumps({
                "consistent": False,
                "mismatches": [
                    {
                        "claim": "the date is 2026-09-28",
                        "expected_from_docs": "2025-09-28",
                        "actual_in_answer": "2026-09-28",
                    }
                ],
                "reason": "answer's date is one year off",
            })
        )


class _StubLLMInvalidJSON:
    """Stub LLM that returns garbage (forces JSONDecodeError)."""

    async def ainvoke(self, messages):
        return _StubMessage(content="not json")


class _StubLLMRaises:
    """Stub LLM that raises an exception (network / timeout)."""

    async def ainvoke(self, messages):
        raise RuntimeError("simulated upstream 503")


@dataclass
class _StubMessage:
    content: str


@pytest.mark.asyncio
async def test_verifier_llm_consistent_returns_no_mismatch():
    """Stub LLM agrees with rule engine → consistent=True."""
    from src.agent.verification.verifier import verify_answer

    docs = [Document(page_content="事件发生在 2026-09-28。", metadata={})]
    result = await verify_answer(
        question="何时?",
        retrieved_docs=docs,
        first_pass_answer="事件发生在 2026-09-28。",
        llm=_StubLLMConsistent(),
    )
    assert result.consistent is True
    assert result.mismatches == []


@pytest.mark.asyncio
async def test_verifier_llm_mismatch_returns_mismatch_list():
    """Stub LLM flags a mismatch → consistent=False, mismatch carries claim."""
    from src.agent.verification.verifier import verify_answer

    docs = [Document(page_content="事件发生在 2025-09-28。", metadata={})]
    result = await verify_answer(
        question="何时?",
        retrieved_docs=docs,
        first_pass_answer="事件发生在 2026-09-28。",
        llm=_StubLLMMismatch(),
    )
    assert result.consistent is False
    # Rule engine alone would flag this (date 2026-09-28 not in doc).
    # Combined with LLM mismatch, we have 2 mismatches total.
    assert len(result.mismatches) >= 1


@pytest.mark.asyncio
async def test_verifier_llm_invalid_json_treated_as_inconsistent():
    """Stub LLM returns garbage → fail-closed (consistent=False)."""
    from src.agent.verification.verifier import verify_answer

    docs = [Document(page_content="some content", metadata={})]
    result = await verify_answer(
        question="?",
        retrieved_docs=docs,
        first_pass_answer="some answer",
        llm=_StubLLMInvalidJSON(),
    )
    assert result.consistent is False
    # Reason should mention the JSON parse failure (or "verifier LLM").
    assert result.reason is not None
    assert "JSON" in result.reason or "verifier" in result.reason


@pytest.mark.asyncio
async def test_verifier_llm_exception_treated_as_inconsistent():
    """Stub LLM raises → fail-closed (consistent=False, attempts=1)."""
    from src.agent.verification.verifier import verify_answer

    docs = [Document(page_content="some content", metadata={})]
    result = await verify_answer(
        question="?",
        retrieved_docs=docs,
        first_pass_answer="some answer",
        llm=_StubLLMRaises(),
    )
    assert result.consistent is False
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_verifier_strips_markdown_json_fences():
    """Stub LLM returns ```json ... ``` wrapped response → parsed cleanly."""
    from src.agent.verification.verifier import _strip_json_fence

    assert _strip_json_fence('```json\n{"k": 1}\n```') == '{"k": 1}'
    assert _strip_json_fence('```\n{"k": 1}\n```') == '{"k": 1}'
    assert _strip_json_fence('{"k": 1}') == '{"k": 1}'


# ---------------------------------------------------------------------------
# 5. FSM node — branches
# ---------------------------------------------------------------------------


def _drive(node_fn, state):
    """Run an async-gen node and collect (kind, payload) tuples + delta."""
    events = []

    async def _run():
        async for evt in node_fn(state, step=1):
            events.append(evt)
    asyncio.run(_run())
    delta = next((p for k, p in events if k == "__delta__"), None)
    events_only = [(k, p) for k, p in events if k != "__delta__"]
    return events_only, delta


def test_verification_node_skips_when_no_answer():
    """Direct route with no answer → skipped=True."""
    from src.agent.nodes.verify_answer import verify_answer_node

    events, delta = _drive(
        verify_answer_node,
        {"answer": "", "original_query": "你好", "documents": [], "intent": "greeting"},
    )
    assert any(
        k == "verification_result" and p.get("skipped") for k, p in events
    ), f"expected skipped event, got {events}"
    assert delta.get("verification_check") == "skipped"


def test_verification_node_skips_when_should_not_trigger():
    """Chitchat answer with no extractable fields → skipped=True (zero-cost)."""
    from src.agent.nodes.verify_answer import verify_answer_node

    state = {
        "answer": "你好,很高兴认识你。",
        "original_query": "你好",
        "documents": [Document(page_content="ignored", metadata={})],
        "intent": "qa_complex",
    }
    events, delta = _drive(verify_answer_node, state)
    assert any(
        k == "verification_result" and p.get("skipped") for k, p in events
    ), f"expected skipped event, got {events}"
    assert delta.get("verification_check") == "skipped"


def test_verification_node_consistent_emits_grounded():
    """Rule engine + verifier LLM agree → verification_check=grounded."""
    from src.agent.nodes.verify_answer import verify_answer_node

    # The verifier module imports ``build_cheap_model`` lazily inside
    # ``verify_answer`` — patch the factory module instead.
    import src.llm.factory as factory_mod
    original = factory_mod.build_cheap_model
    factory_mod.build_cheap_model = lambda temperature=0.0: _StubLLMConsistent()
    try:
        state = {
            "answer": "事件发生在 2026-09-28。",
            "original_query": "何时?",
            "documents": [Document(page_content="事件发生在 2026-09-28。", metadata={})],
            "intent": "qa_complex",
            "verification_attempts": 0,
        }
        events, delta = _drive(verify_answer_node, state)
    finally:
        factory_mod.build_cheap_model = original

    assert any(
        k == "verification_result" and p.get("consistent") and not p.get("skipped")
        for k, p in events
    ), f"expected consistent event, got {events}"
    assert delta.get("verification_check") == "grounded"


def test_verification_node_inconsistent_emits_regen_directive():
    """Verifier flags mismatches → regen directive stamped on state."""
    from src.agent.nodes.verify_answer import verify_answer_node

    import src.llm.factory as factory_mod
    original = factory_mod.build_cheap_model
    factory_mod.build_cheap_model = lambda temperature=0.0: _StubLLMMismatch()
    try:
        state = {
            "answer": "事件发生在 2026-09-28,价格 ¥1,000.50",
            "original_query": "何时?多少钱?",
            "documents": [Document(page_content="事件发生在 2025-09-28。", metadata={})],
            "intent": "qa_complex",
            "verification_attempts": 0,
        }
        events, delta = _drive(verify_answer_node, state)
    finally:
        factory_mod.build_cheap_model = original

    assert any(
        k == "verification_result" and p.get("regenerated") for k, p in events
    ), f"expected regen event, got {events}"
    # Delta carries the directive + regen marker for the FSM to route back.
    assert delta.get("verification_check") == "regen"
    assert "verification_mismatch_directive" in delta
    assert "2026-09-28" in delta["verification_mismatch_directive"]


def test_verification_node_max_retries_falls_back_to_refusal():
    """After MAX_ATTEMPTS retries → fallback to REFUSAL_TEMPLATES['low_relevance']."""
    from src.agent.nodes.verify_answer import verify_answer_node
    from src.agent.verification.verifier import VERIFICATION_MAX_ATTEMPTS

    import src.llm.factory as factory_mod
    original = factory_mod.build_cheap_model
    factory_mod.build_cheap_model = lambda temperature=0.0: _StubLLMMismatch()
    try:
        state = {
            "answer": "事件发生在 2026-09-28",
            "original_query": "何时?",
            "documents": [Document(page_content="事件发生在 2025-09-28。", metadata={})],
            "intent": "qa_complex",
            # Already at the budget — next attempt would exceed it.
            "verification_attempts": VERIFICATION_MAX_ATTEMPTS,
        }
        events, delta = _drive(verify_answer_node, state)
    finally:
        factory_mod.build_cheap_model = original

    assert any(
        k == "verification_result" and p.get("fallback_refusal") for k, p in events
    ), f"expected fallback_refusal event, got {events}"
    # Delta replaces the answer with the refusal template.
    assert "answer" in delta
    assert "低相关性" in delta["answer"] or "相关性不足" in delta["answer"]


def test_verification_node_bumps_regenerated_counter():
    """Regen path → ``VERIFICATION_REGENERATED{outcome=regenerated}`` bumps."""
    from src.agent.nodes.verify_answer import verify_answer_node
    from src.agent.metrics import VERIFICATION_REGENERATED

    before = VERIFICATION_REGENERATED.get(outcome="regenerated")

    import src.llm.factory as factory_mod
    original = factory_mod.build_cheap_model
    factory_mod.build_cheap_model = lambda temperature=0.0: _StubLLMMismatch()
    try:
        state = {
            "answer": "事件发生在 2026-09-28",
            "original_query": "何时?",
            "documents": [Document(page_content="事件发生在 2025-09-28。", metadata={})],
            "intent": "qa_complex",
            "verification_attempts": 0,
        }
        events, delta = _drive(verify_answer_node, state)
        # Surface the FSM-node branch choice so a regression is debuggable.
        verification_check = (delta or {}).get("verification_check")
        print(
            f"\n  debug: events={[(k, p.get('regenerated'), p.get('fallback_refusal'), p.get('consistent'), p.get('skipped')) for k, p in events if k == 'verification_result']}"
            f" verification_check={verification_check}"
        )
    finally:
        factory_mod.build_cheap_model = original

    after = VERIFICATION_REGENERATED.get(outcome="regenerated")
    assert after == before + 1, f"counter should bump from {before} to {before + 1}, got {after}"


# ---------------------------------------------------------------------------
# 6. FSM predicates — verify_answer is wired
# ---------------------------------------------------------------------------


def test_fsm_predicate_routes_to_verify_answer_when_triggered():
    """after_react_generate returns ``"verify_answer"`` when should_trigger."""
    from src.agent.fsm import after_react_generate

    state = {
        "answer": "事件发生在 2026-09-28",
        "original_query": "何时?",
        "documents": [Document(page_content="ignored", metadata={})],
        "intent": "qa_complex",
    }
    assert after_react_generate(state) == "verify_answer"


def test_fsm_predicate_skips_to_grounding_when_no_trigger():
    """Chitchat → predicate returns should_check_hallucination (None for greeting)."""
    from src.agent.fsm import after_react_generate

    state = {
        "answer": "你好,很高兴认识你。",
        "original_query": "你好",
        "documents": [Document(page_content="ignored", metadata={})],
        "intent": "qa_complex",
    }
    # should_trigger returns False → predicate routes to grounding.
    assert after_react_generate(state) == "check_hallucination"


def test_fsm_predicate_skips_when_no_documents():
    """No documents to verify against → predicate returns should_check_hallucination path."""
    from src.agent.fsm import after_react_generate

    state = {
        "answer": "事件发生在 2026-09-28",
        "original_query": "何时?",
        "documents": [],
        "intent": "qa_complex",
    }
    # No docs → verify_answer doesn't help; route to grounding path.
    next_node = after_react_generate(state)
    assert next_node in (None, "check_hallucination")


def test_fsm_predicate_skips_when_no_answer():
    """No answer (direct / refusal) → predicate returns should_check_hallucination path."""
    from src.agent.fsm import after_react_generate

    state = {
        "answer": "",
        "original_query": "hi",
        "documents": [Document(page_content="ignored", metadata={})],
        "intent": "greeting",
    }
    next_node = after_react_generate(state)
    assert next_node in (None, "check_hallucination")


def test_fsm_predicate_verify_answer_to_react_generate_on_regen():
    """When verify_answer stamps regen directive → FSM routes back to react_generate."""
    from src.agent.fsm import PREDICATES

    predicate = PREDICATES["verify_answer"]
    state = {"verification_mismatch_directive": "fix the date mismatch"}
    assert predicate(state) == "react_generate"

    state_no_directive = {}
    assert predicate(state_no_directive) == "check_hallucination"


def test_fsm_nodes_registry_includes_verify_answer():
    """``NODES["verify_answer"]`` is registered with the verification node fn."""
    from src.agent.fsm import NODES
    from src.agent.nodes.verify_answer import verify_answer_node

    assert "verify_answer" in NODES
    assert NODES["verify_answer"]["fn"] is verify_answer_node


# ---------------------------------------------------------------------------
# 7. Wire event + runner emission
# ---------------------------------------------------------------------------


def test_verification_result_wire_event_round_trips():
    """Pydantic VerificationResultEvent validates + dumps cleanly."""
    from src.agent.wire_protocol import (
        VerificationResultEvent, dump_payload, validate_payload,
    )

    event = VerificationResultEvent(
        consistent=False,
        mismatches_count=2,
        regenerated=True,
        fallback_refusal=False,
        skipped=False,
        reason="date mismatch",
        mismatches=[{"raw": "2026-09-28", "actual": "2026-09-28", "expected": "2025-09-28"}],
    )
    payload = dump_payload(event)
    assert payload["type"] == "verification_result"
    assert payload["consistent"] is False
    assert payload["mismatches_count"] == 2
    assert payload["regenerated"] is True
    # Round-trip via validate_payload
    parsed = validate_payload(payload)
    assert isinstance(parsed, VerificationResultEvent)


def test_events_verification_result_factory_builds_payload():
    """``events.verification_result(...)`` factory matches the wire contract."""
    from src.agent import events

    payload = events.verification_result(
        consistent=True,
        mismatches_count=0,
        skipped=True,
        reason=None,
    )
    assert payload["type"] == "verification_result"
    assert payload["consistent"] is True
    assert payload["skipped"] is True
    assert payload["mismatches_count"] == 0
    # reason=None should be dropped (exclude_none=True)
    assert "reason" not in payload


def test_runner_emits_verification_result_for_fsm_event():
    """runner._emit_fsm_event maps kind=verification_result to the wire factory."""
    from src.agent.runner import _emit_fsm_event

    # Tiny helpers — the runner takes single-cell lists.
    fsm_evt = {
        "kind": "verification_result",
        "consistent": False,
        "mismatches_count": 1,
        "regenerated": True,
        "fallback_refusal": False,
        "skipped": False,
        "reason": "test reason",
        "mismatches": [{"raw": "x", "actual": "y", "expected": "z"}],
    }
    out = _emit_fsm_event(
        fsm_evt,
        accumulated_text=[""],
        last_emitted_text=[""],
        stream_buffer=[""],
        answer_complete_sent=[False],
    )
    assert out is not None
    assert out["type"] == "verification_result"
    assert out["consistent"] is False
    assert out["regenerated"] is True
    assert out["reason"] == "test reason"