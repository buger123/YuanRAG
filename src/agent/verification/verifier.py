"""v2.0.29.7 (Phase 6) — Second-pass verifier (rule engine + cheap LLM).

Combines the deterministic :mod:`src.agent.verification.rule_engine`
(structured-fact extraction + cross-check) with a cheap-model LLM
call that catches semantic-level hallucination the rule engine
cannot see (concept drift, paraphrase fabrication, entity
confusion). Returns a :class:`VerificationResult` that the FSM
node (``nodes/verify_answer.py``) consumes.

Fail-closed semantics
----------------------
If the verifier LLM raises (timeout, 5xx, JSON parse failure), we
treat the answer as INCONSISTENT (not consistent). The reason:
"verifier hung → assume fabrication" is safer than "verifier hung
→ let the synthesis LLM's word stand". The downstream node reads
this and either regenerates the answer or falls back to the Phase
1 ``REFUSAL_TEMPLATES["low_relevance"]``.

Trigger conditions
------------------
Verification is zero-cost for most turns (the cheap LLM only runs
when ``should_trigger_verification`` returns True). Trigger on:

  * Query contains concrete-fact keywords ("多少 / 几号 / 第 X 条
    / 金额 / 百分比" / "how much" / "Article X" / "which clause").
  * The first-pass answer contains any extractable fact (date /
    currency / percentage / article number).

If neither holds, the answer is a free-form conversation /
explanation that the rule engine doesn't know how to check, so
we skip and route straight to ``check_hallucination``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.metrics import VERIFICATION_REGENERATED
from src.agent.verification.rule_engine import (
    ExtractedField,
    Mismatch,
    check_consistency,
    extract_fields,
)


# Trigger-regex for concrete-fact queries. Catches the most common
# Chinese + English phrasings; we deliberately keep it conservative
# (additive OR) so we never fire on a pure greeting or chitchat.
_TRIGGER_QUERY_RE = re.compile(
    r"(具体|多少|几号|什么时候|哪天|哪些|哪个|条款|规定|金额|百分比|"
    r"第.{1,5}条|多少[次个张元块]|"
    r"when\s+(?:is|was|were|did|does|do)|"
    r"how\s+(?:much|many|old|long|far|often)|"
    r"which\s+(?:article|section|clause|paragraph|version)|"
    r"what\s+(?:date|year|amount|number|percentage|clause))",
    re.IGNORECASE,
)


# System prompt for the verifier LLM. Output is strict JSON so the
# caller can parse without an extra LLM step. We allow ```json
# fences because MiniMax-style proxies sometimes wrap the response
# in markdown code blocks despite the "strict JSON" instruction.
VERIFICATION_SYSTEM = """You are a strict fact-checker. Given:
  (1) the user's question,
  (2) the retrieved documents the system found,
  (3) a first-pass assistant answer,
identify any **factual claims** in the answer that are NOT directly supported
by the retrieved documents. Be conservative: if the answer hedges with
"大概 / 应该是 / 可能" and the docs don't have the exact number, that IS a
mismatch.

Output strict JSON:
{
  "consistent": bool,        // true if all claims supported
  "mismatches": [
    {"claim": "...", "expected_from_docs": "...", "actual_in_answer": "..."}
  ],
  "reason": "one-sentence explanation"
}

Do NOT edit. Do NOT add commentary outside JSON."""


# Max attempts (regen calls) before fallback to REFUSAL_TEMPLATES.
# The first pass is always counted as ``attempts=1``; ``attempts=N``
# means N attempts (first + N-1 regens) before fallback. Set to 2
# per the user's "fail-closed but bounded retries" preference.
VERIFICATION_MAX_ATTEMPTS = 2


@dataclass
class VerificationResult:
    """Aggregate outcome of one verification pass.

    ``consistent``        — True iff rule_engine found no mismatches
                            AND verifier LLM agreed.
    ``mismatches``        — union of rule_engine + verifier LLM
                            findings. Each Mismatch carries the
                            raw claim text and severity.
    ``reason``            — one-sentence summary (typically from the
                            verifier LLM). None when consistent.
    ``regenerated``       — True if downstream FSM passed a regen
                            directive back into react_generate. Set
                            by the FSM node, NOT by this function.
    ``regenerated_answer``— Optional override answer from a regen
                            pass. Set by the FSM node, NOT by this
                            function. Phase 6 PR-1 leaves this None
                            (we don't ship the regen loop in PR-1;
                            PR-2 wires the actual retry).
    ``fallback_refusal``  — True when retries exhausted and the FSM
                            replaced the answer with the Phase 1
                            ``REFUSAL_TEMPLATES["low_relevance"]``
                            fallback. Set by the FSM node.
    ``attempts``          — How many verification passes have run
                            on this turn. 1 for first-pass-only;
                            >1 if the FSM is in a regen loop.
    """

    consistent: bool
    mismatches: list[Mismatch] = field(default_factory=list)
    reason: Optional[str] = None
    regenerated: bool = False
    regenerated_answer: Optional[str] = None
    fallback_refusal: bool = False
    attempts: int = 1


def should_trigger_verification(answer_text: str, query: str) -> bool:
    """Decide whether verification is worth the cheap-LLM cost.

    Force-trigger when:
      - query asks for concrete / specific facts (regex match)
      - answer contains extractable fields (date, currency, %,
        article_number)

    Returns False otherwise. Zero-cost on the False path
    (extract_fields is regex-only, microseconds).
    """
    if query and _TRIGGER_QUERY_RE.search(query):
        return True
    if answer_text and extract_fields(answer_text):
        return True
    return False


def _strip_json_fence(content: str) -> str:
    """Tolerate ```json ... ``` fences the LLM sometimes wraps in.

    The verifier LLM is told to emit strict JSON, but MiniMax-style
    proxies (and Claude under certain system prompts) sometimes add
    markdown fences. We strip the outer fence but otherwise pass
    through; the downstream ``json.loads`` will surface a real
    JSONDecodeError if the body is non-JSON.
    """
    s = content.strip()
    if not s.startswith("```"):
        return s
    # Strip leading ```json or ``` line.
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    # Strip trailing ``` line.
    s = re.sub(r"\s*```\s*$", "", s, flags=re.MULTILINE)
    return s.strip()


async def verify_answer(
    question: str,
    retrieved_docs: list[Document],
    first_pass_answer: str,
    *,
    llm=None,
) -> VerificationResult:
    """Run rule_engine + verifier LLM. Returns merged result.

    Parameters
    ----------
    question : str
        The user's original question (or the corrected_query if
        intent_analysis rewrote it).
    retrieved_docs : list[Document]
        The docs that the synthesis LLM saw at synthesis — same docs
        that backend.retrieve returned (the source of truth for the
        answer). May be a mixture of local + web docs.
    first_pass_answer : str
        The synthesis LLM's raw answer text. We compare against
        ``retrieved_docs`` to detect unsupported claims.
    llm : optional BaseChatModel
        Override the verifier LLM. Defaults to
        ``build_cheap_model(temperature=0.0)`` so the test can
        inject a stub (see ``tests/test_verification.py``).

    Fail-closed: if the verifier LLM raises or returns non-JSON,
    we treat the result as INCONSISTENT (downstream node will
    either regen or fallback-refuse). We do NOT swallow the
    exception silently — the reason is set to the exception class
    + repr so the loguru stream captures it for debugging.
    """
    # Layer 1: rule_engine (no LLM, ~ms).
    answer_fields = extract_fields(first_pass_answer)
    doc_text = "\n\n".join((d.page_content or "") for d in retrieved_docs)
    rule_mismatches = check_consistency(answer_fields, doc_text)

    # Layer 2: verifier LLM (cheap model, ~seconds).
    llm_mismatches: list[Mismatch] = []
    reason: Optional[str] = None
    consistent_from_llm = True

    if llm is None:
        # Lazy import to avoid a hard dependency from tests that
        # only exercise the rule-engine path.
        from src.llm.factory import build_cheap_model
        llm = build_cheap_model(temperature=0.0)

    # Truncate each doc to ~2000 chars; the verifier only needs
    # enough context to cross-check the answer's specific claims.
    docs_blob = "\n\n---\n\n".join(
        f"[{i + 1}] " + (d.page_content or "")[:2000]
        for i, d in enumerate((retrieved_docs or [])[:8])
    )
    user_msg = (
        f"Question:\n{question or ''}\n\n"
        f"Retrieved documents:\n{docs_blob or '(no documents)'}\n\n"
        f"First-pass answer:\n{first_pass_answer or ''}\n\n"
        "Output strict JSON only."
    )

    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=VERIFICATION_SYSTEM),
                HumanMessage(content=user_msg),
            ]
        )
        content = _strip_json_fence((getattr(resp, "content", "") or ""))
        parsed = json.loads(content)
        consistent_from_llm = bool(parsed.get("consistent", False))
        reason_val = parsed.get("reason")
        if isinstance(reason_val, str):
            reason = reason_val
        elif reason_val is not None:
            reason = str(reason_val)
        for m in (parsed.get("mismatches") or []):
            if not isinstance(m, dict):
                continue
            claim = str(m.get("claim") or "?")
            expected = m.get("expected_from_docs") or ""
            actual = m.get("actual_in_answer") or ""
            llm_mismatches.append(
                Mismatch(
                    field=ExtractedField(
                        # Generic placeholder — the LLM doesn't
                        # classify by FieldType. The mismatch is
                        # reported as-is so downstream consumers
                        # can display the raw claim text.
                        field_type="date",
                        raw_text=claim,
                        normalized=actual,
                        location=-1,
                    ),
                    expected=(expected if expected else None),
                    actual=actual,
                    severity="high",
                )
            )
    except (json.JSONDecodeError, Exception) as exc:
        # Fail-closed. Verifier LLM failure → conservative INCONSISTENT.
        # Counter bump via the metric (Phase 2 pre-registered).
        from src.agent.metrics import VERIFICATION_REGENERATED as _VREGEN
        _VREGEN.inc(outcome="verifier_failed")
        consistent_from_llm = False
        reason = f"verifier LLM failed: {type(exc).__name__}: {exc}"

    all_mismatches = rule_mismatches + llm_mismatches
    consistent = (len(all_mismatches) == 0) and consistent_from_llm
    return VerificationResult(
        consistent=consistent,
        mismatches=all_mismatches,
        reason=reason,
        attempts=1,
    )


__all__ = [
    "VERIFICATION_MAX_ATTEMPTS",
    "VERIFICATION_SYSTEM",
    "VerificationResult",
    "should_trigger_verification",
    "verify_answer",
]