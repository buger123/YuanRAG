"""v2.0.29.7 (Phase 6) — FSM node: second-pass verification.

Runs AFTER ``react_generate`` and BEFORE ``check_hallucination``.
If the rule engine + verifier LLM flag mismatches in the first-pass
answer, the FSM routes to a regen pass (up to
``VERIFICATION_MAX_ATTEMPTS``), then falls back to Phase 1's
``REFUSAL_TEMPLATES["low_relevance"]`` if the budget is exhausted.

Yields ``("verification_result", {...})`` wire events so the
frontend can render a ✓/✗ icon (PR-1 ships wire-only; UI lives in
PR-2 / later — backward-compat: omitting the field is safe).

Layered design (per [[yuanrag-hallucination-optimization]] Phase 6)
-------------------------------------------------------------------
1. **Skip path** — when ``should_trigger_verification`` returns False
   (query is chitchat, answer has no extractable fields), yield a
   ``skipped=True`` event and route straight to ``check_hallucination``.
   Zero-cost on this path (no LLM, no regex).
2. **Consistent path** — verifier returns no mismatches → emit a
   ``consistent=True`` event + state delta carrying
   ``verification_check="grounded"``. The FSM continues to the
   grounding node normally.
3. **Regen path** — verifier returns mismatches and the retry budget
   is not exhausted → bump ``VERIFICATION_REGENERATED`` (Phase 2
   pre-registered counter), emit a ``regenerated=True`` event
   carrying the mismatch list, and stamp the FSM state with a
   ``verification_mismatch_directive`` so the FSM re-enters
   ``react_generate`` with a post-script instructing the synthesis
   LLM to fix the mismatches.
4. **Fallback-refusal path** — retry budget exhausted → replace the
   answer with ``REFUSAL_TEMPLATES["low_relevance"]`` and emit a
   ``fallback_refusal=True`` event. ``verification_check="skipped"``
   is stamped so the FSM continues to grounding (the fallback
   answer is a deliberate refusal, not a fabrication).
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Tuple

from langchain_core.documents import Document

# IMPORTANT: import the metrics MODULE (not the Counter symbol). Tests
# elsewhere in this project call ``importlib.reload(src.agent.metrics)``
# to test reload semantics (see test_metrics_aggregation.py::metrics_app
# fixture). A bare ``from src.agent.metrics import VERIFICATION_REGENERATED``
# captures a stale Counter instance when the module is reloaded, so the
# bump here would land in the OLD instance while other tests / the
# /debug/metrics endpoint read from the NEW instance.
#
# We resolve the symbol lazily inside the function body — every bump
# reads the live binding from the current module state. This is the
# same pattern used in v2.0.28.x retell for hot-reload-safe globals
# (per the project's "module identity churn" debugging pitfall).
from src.agent import metrics as _metrics_mod
from src.agent.nodes._docs_for_hallucination import docs_for_hallucination
from src.agent.state import AgentState
from src.agent.verification.verifier import (
    VERIFICATION_MAX_ATTEMPTS,
    VerificationResult,
    should_trigger_verification,
    verify_answer,
)
from src.core.logging import logger
from src.llm.prompts import REFUSAL_TEMPLATES


# SystemMessage injected at index 2 of the synthesis prompt when
# verification flags mismatches. Read by react_generate on its next
# pass via ``state["verification_mismatch_directive"]``. Uses the
# same ``insert(2, ...)`` consecutiveness rule as the Phase 1
# refusal directive (Anthropic tool_use ↔ system ordering per
# v2.0.28.17).
_REGEN_DIRECTIVE_TEMPLATE = (
    "[CRITICAL — 二次校验未通过] 你上一轮的答案与原文资料不一致,以下字段"
    "需要重新核对:\n{mismatch_list}\n\n请重新阅读原文资料并修正以上字段。"
    "如果资料无法支持,请按 refusal_template 拒绝。"
)


def _format_mismatch_list(mismatches: list) -> str:
    """Render a list of Mismatch into a human-readable directive
    payload. Defensive: ``mismatches`` may be empty (rule-engine
    found nothing but verifier LLM failed, etc.); in that case we
    return a generic re-verify instruction."""
    if not mismatches:
        return "(verifier did not enumerate specific claims — re-read all retrieved docs)"
    parts = []
    for m in mismatches[:8]:
        raw = m.field.raw_text
        actual = m.actual
        expected = m.expected if m.expected is not None else "?"
        parts.append(f"- field={raw!r} actual={actual!r} expected={expected!r}")
    return "\n".join(parts)


def _build_wire_payload(
    result: VerificationResult,
    *,
    skipped: bool = False,
    fallback_refusal: bool = False,
    mismatches_payload: list[dict] | None = None,
) -> dict:
    """Build the ``verification_result`` wire payload.

    Centralized so the four yield-sites in this file stay aligned
    with :class:`src.agent.wire_protocol.VerificationResultEvent`.
    """
    if mismatches_payload is None:
        mismatches_payload = [
            {
                "raw": m.field.raw_text,
                "actual": m.actual,
                "expected": m.expected,
                "field_type": m.field.field_type,
            }
            for m in result.mismatches
        ]
    return {
        "consistent": result.consistent,
        "mismatches_count": len(result.mismatches),
        "regenerated": result.regenerated,
        "fallback_refusal": fallback_refusal,
        "skipped": skipped,
        "reason": result.reason,
        "mismatches": mismatches_payload,
    }


async def verify_answer_node(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """FSM node: second-pass verification.

    Async-generator protocol: yields ``("verification_result", ...)``
    wire events (forwarded by the FSM to the runner), then ends
    with the mandatory ``("__delta__", delta_dict)`` terminator.
    The delta carries ``verification_check`` (one of "grounded" /
    "regen" / "skipped") and, on the regen path, a
    ``verification_mismatch_directive`` for react_generate to read
    on its next pass.

    The FSM predicate (see :func:`src.agent.fsm.after_react_generate`)
    decides whether this node runs at all; if
    ``should_trigger_verification`` returns False, we emit a
    skipped event and exit. Otherwise we run rule_engine + LLM
    and choose among the three downstream branches.
    """
    answer_text = state.get("answer") or ""
    query = state.get("original_query") or state.get("current_query") or ""
    docs = docs_for_hallucination(state)
    prev_attempts = int(state.get("verification_attempts", 0) or 0)

    # ---- Skip path: not worth verifying ----
    if not answer_text or not should_trigger_verification(answer_text, query):
        skipped_result = VerificationResult(
            consistent=True,
            mismatches=[],
            reason=None,
            attempts=prev_attempts or 1,
        )
        yield ("verification_result", _build_wire_payload(skipped_result, skipped=True))
        yield ("__delta__", {"verification_check": "skipped"})
        return

    # ---- Run rule_engine + verifier LLM ----
    try:
        result = await verify_answer(
            question=query,
            retrieved_docs=docs,
            first_pass_answer=answer_text,
        )
    except Exception as exc:
        # Last-ditch defense: ``verify_answer`` already catches its
        # own LLM exceptions, so reaching here means an unexpected
        # bug (network, import cycle, etc.). Treat as INCONSISTENT
        # (fail-closed) and route to fallback-refusal path so the
        # user sees a refusal template rather than a possibly-
        # fabricated answer.
        logger.exception("verify_answer_node: verify_answer raised unexpectedly")
        result = VerificationResult(
            consistent=False,
            mismatches=[],
            reason=f"verifier crashed: {type(exc).__name__}: {exc}",
            attempts=prev_attempts or 1,
        )

    # ---- Consistent path ----
    if result.consistent:
        yield ("verification_result", _build_wire_payload(result))
        yield ("__delta__", {
            "verification_check": "grounded",
            "verification_attempts": max(prev_attempts, 1),
        })
        return

    # ---- Fallback-refusal path: retries exhausted ----
    if prev_attempts >= VERIFICATION_MAX_ATTEMPTS:
        fallback = REFUSAL_TEMPLATES["low_relevance"]
        _metrics_mod.VERIFICATION_REGENERATED.inc(outcome="fallback_refusal")
        fallback_result = VerificationResult(
            consistent=False,
            mismatches=result.mismatches,
            reason=result.reason,
            regenerated=False,
            fallback_refusal=True,
            attempts=prev_attempts + 1,
        )
        logger.warning(
            "verify_answer: max retries exhausted (%d >= %d) — fallback to refusal template",
            prev_attempts, VERIFICATION_MAX_ATTEMPTS,
        )
        yield ("verification_result", _build_wire_payload(
            fallback_result, fallback_refusal=True,
        ))
        yield ("__delta__", {
            "answer": fallback,
            "verification_check": "skipped",  # signal refusal fallback to FSM
            "verification_attempts": prev_attempts + 1,
        })
        return

    # ---- Regen path: route back to react_generate ----
    _metrics_mod.VERIFICATION_REGENERATED.inc(outcome="regenerated")
    mismatch_list = _format_mismatch_list(result.mismatches)
    directive = _REGEN_DIRECTIVE_TEMPLATE.format(mismatch_list=mismatch_list)
    regen_result = VerificationResult(
        consistent=False,
        mismatches=result.mismatches,
        reason=result.reason,
        regenerated=True,
        attempts=prev_attempts + 1,
    )
    logger.info(
        "verify_answer: %d mismatches flagged; emitting regen directive (attempt %d)",
        len(result.mismatches), prev_attempts + 1,
    )
    yield ("verification_result", _build_wire_payload(regen_result))
    yield ("__delta__", {
        "verification_check": "regen",
        "verification_attempts": prev_attempts + 1,
        "verification_mismatch_directive": directive,
    })


__all__ = ["verify_answer_node"]