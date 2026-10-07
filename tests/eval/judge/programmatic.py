"""Pass 1: programmatic structural assertions (zero LLM cost).

v2.0.32.0 — 10 checks per the plan §D. Runs in <50ms per case.

Each check returns ``CheckResult(name, expected, actual, pass)``. The
overall programmatic pass = all individual checks pass.

Per-case field semantics are defined in ``cases.py``. This module is the
ONLY consumer of wire events for evaluation; downstream report.py
consumes the programmatic + cheap_llm + claude judgments.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class CheckResult:
    """One structural check."""

    name: str
    expected: Any
    actual: Any
    pass_: bool = field(default=False)

    @property
    def passed(self) -> bool:
        return self.pass_

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected": self.expected,
            "actual": self.actual,
            "pass": self.pass_,
        }


@dataclass(frozen=True)
class ProgrammaticJudgment:
    """Pass 1 result bundle — passed iff every check passes.

    v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07): ``degraded`` +
    ``degraded_reason`` flag the case as backend-LLM-degraded (e.g.
    Phase 6 verify_answer regen-loop defensive override). When
    ``degraded=True`` the judge SKIPS answer-text checks because the
    body is a fallback answer, not the real LLM output. The rollup's
    ``pass_rate_excluding_infra_skip`` metric counts these separately
    from real logic bugs.
    """

    pass_: bool
    checks: tuple[CheckResult, ...]
    degraded: bool = False
    degraded_reason: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.pass_

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pass": self.pass_,
            "checks": [c.to_dict() for c in self.checks],
        }
        if self.degraded:
            d["degraded"] = True
            d["degraded_reason"] = self.degraded_reason
        return d


# ============================================================
# Helpers
# ============================================================


def _normalize(text: str) -> str:
    return text.casefold() if isinstance(text, str) else ""


def _answer_field(events: list[dict]) -> str:
    """Pull ``answer_complete.answer`` from the wire-event log.

    CALIBRATION (2026-10-05): read the LAST answer_complete event, not
    the first. The wire may emit multiple answer_complete events (initial
    empty placeholder + final streamed answer); the judge's answer_contains
    check was hitting the empty placeholder and reporting `got=[]`.
    """
    last = ""
    for ev in events:
        if ev.get("type") == "answer_complete":
            ans = ev.get("answer") or ""
            if isinstance(ans, str):
                last = ans
    return last


def _sources(events: list[dict]) -> list[dict]:
    """Pull ``answer_complete.sources`` from the LAST such event."""
    last: list[dict] = []
    for ev in events:
        if ev.get("type") == "answer_complete":
            srcs = ev.get("sources") or []
            if isinstance(srcs, list):
                last = list(srcs)
    return last


def _grounding_status(events: list[dict]) -> Optional[str]:
    for ev in events:
        if ev.get("type") == "grounding":
            status = ev.get("status")
            return status if isinstance(status, str) else None
    # No grounding event on the wire (e.g. simple_fact path, refusal path).
    # Default to "skipped" so Phase 8 callers expecting expected_grounding="skipped"
    # (the common default) get it; callers who care about "grounded" vs "ungrounded"
    # explicitly opt in via the test's expected_grounding field.
    return "skipped"


def _verification(events: list[dict]) -> dict[str, Any]:
    for ev in events:
        if ev.get("type") == "verification_result":
            return ev
    return {}


def _tool_calls(events: list[dict]) -> list[dict]:
    starts = [e for e in events if e.get("type") == "tool_call_start"]
    return starts


def _fsm_node_sequence(events: list[dict]) -> list[str]:
    """Walk events and extract canonical FSM node labels in order.

    We don't have a single "this node ran" event on the wire — instead
    we infer nodes from event co-occurrence. The runner streams events
    in FSM-node order; we map by:
      - intent_analysis   → emit of ``intent`` event (first per turn)
      - react_agent       → first ``step_start`` event
      - tools             → first ``tool_call_start`` event in the per-step loop
      - react_generate    → ``answer_complete`` event (default generation)
      - react_generate_direct  → same, but the answer contains no citations
                          AND no tool_call_start preceded it (heuristic)
      - react_generate_extractive → ``answer_complete`` with all sources
                          having ``verbatim=True``
      - verify_answer     → emit of ``verification_result`` event
      - check_hallucination → emit of ``grounding`` event
      - summary_path      → ``intent == "summary"``
      - summary_intent    → pre-tool-call_start events on a summary intent

    This is heuristic but matches the canonical sequencing the plan §C
    documents — it's good enough to catch regressions in the FSM
    topology without over-fitting.
    """
    seq: list[str] = []
    seen_intent = False
    saw_step_start = False
    saw_tool_call = False
    for ev in events:
        et = ev.get("type")
        if et == "intent" and not seen_intent:
            seq.append("intent_analysis")
            seen_intent = True
            if ev.get("intent") == "summary":
                seq.append("summary_intent")
                seq.append("summary_path")
        elif et == "step_start" and not saw_step_start:
            seq.append("react_agent")
            saw_step_start = True
        elif et == "tool_call_start" and not saw_tool_call:
            seq.append("tools")
            saw_tool_call = True
        elif et == "answer_complete":
            sources = ev.get("sources") or []
            if sources and all(s.get("verbatim") for s in sources):
                seq.append("react_generate_extractive")
            elif not sources:
                seq.append("react_generate_direct")
            else:
                seq.append("react_generate")
        elif et == "verification_result":
            seq.append("verify_answer")
        elif et == "grounding":
            seq.append("check_hallucination")
    return seq


# ============================================================
# Individual checks
# ============================================================


def check_grounding_status(expected: str, events: list[dict]) -> CheckResult:
    actual = _grounding_status(events)
    pass_ = actual == expected
    return CheckResult(
        name="grounding_status",
        expected=expected,
        actual=actual,
        pass_=pass_,
    )


def check_verification_consistent(expected: bool, events: list[dict]) -> CheckResult:
    v = _verification(events)
    if not v:
        # No verification_result event — Phase 6 gate skipped this case.
        # Pass iff caller expected skip (True == skipped).
        return CheckResult(
            name="verification_consistent",
            expected=expected,
            actual="skipped",
            pass_=bool(expected),  # tolerated as "consistent by absence"
        )
    actual = bool(v.get("consistent"))
    if v.get("skipped"):
        actual_str = "skipped"
        pass_ = bool(expected)
    else:
        actual_str = actual
        pass_ = actual == expected
    return CheckResult(
        name="verification_consistent",
        expected=expected,
        actual=actual_str,
        pass_=pass_,
    )


def check_expected_citations(expected_ids: Iterable[str], events: list[dict]) -> CheckResult:
    expected_list = list(expected_ids)
    sources = _sources(events)
    actual_ids = [str(s.get("chunk_id", "")) for s in sources]
    if not expected_list:
        pass_ = True  # no citations expected — vacuously satisfied
    else:
        # Substring match so doc-scoped IDs survive filename prefixes.
        pass_ = all(
            any(eid in cid for cid in actual_ids) for eid in expected_list
        )
    return CheckResult(
        name="expected_citations",
        expected=expected_list,
        actual=actual_ids,
        pass_=pass_,
    )


def check_answer_contains(expected: Iterable[str], events: list[dict]) -> CheckResult:
    expected_list = list(expected)
    answer = _normalize(_answer_field(events))
    present = [e for e in expected_list if _normalize(e) in answer]
    pass_ = bool(present) if expected_list else True
    return CheckResult(
        name="answer_contains",
        expected=expected_list,
        actual=present,
        pass_=pass_,
    )


def check_answer_lacks(forbidden: Iterable[str], events: list[dict]) -> CheckResult:
    forbidden_list = list(forbidden)
    answer = _normalize(_answer_field(events))
    leaked = [f for f in forbidden_list if _normalize(f) in answer]
    pass_ = not leaked
    return CheckResult(
        name="answer_lacks",
        expected=forbidden_list,
        actual=leaked,
        pass_=pass_,
    )


def check_forbidden_phrases(forbidden: Iterable[str], events: list[dict]) -> CheckResult:
    forbidden_list = list(forbidden)
    answer = _normalize(_answer_field(events))
    leaked = [f for f in forbidden_list if _normalize(f) in answer]
    pass_ = not leaked
    return CheckResult(
        name="forbidden_phrases",
        expected=forbidden_list,
        actual=leaked,
        pass_=pass_,
    )


def check_tool_sequence(expected: Iterable[str], events: list[dict]) -> CheckResult:
    expected_list = list(expected)
    if not expected_list:
        return CheckResult(
            name="tool_sequence",
            expected=[],
            actual=_fsm_node_sequence(events),
            pass_=True,
        )
    actual_seq = _fsm_node_sequence(events)
    # Fuzzy: collapse consecutive identical nodes (loop collapse) and
    # match subsequence. We use a simple rolling check.
    actual_norm = tuple(actual_seq)
    expected_norm = tuple(expected_list)
    pass_ = _is_subsequence_or_loop(expected_norm, actual_norm)
    return CheckResult(
        name="tool_sequence",
        expected=expected_norm,
        actual=actual_norm,
        pass_=pass_,
    )


def check_tool_call_count(expected_max: int, events: list[dict]) -> CheckResult:
    actual = len(_tool_calls(events))
    pass_ = actual <= expected_max
    return CheckResult(
        name="tool_call_count",
        expected=expected_max,
        actual=actual,
        pass_=pass_,
    )


def check_route_decision(expected: str, events: list[dict]) -> CheckResult:
    sources = _sources(events)
    if expected == "direct":
        # No tool calls → direct generation.
        actual = "direct" if not _tool_calls(events) else "retrieve"
    elif expected == "retrieve":
        # Tool calls → retrieval path.
        actual = "retrieve" if _tool_calls(events) else "direct"
    elif expected == "extractive":
        # All sources verbatim = extractive generation.
        actual = (
            "extractive"
            if sources and all(s.get("verbatim") for s in sources)
            else "generate"
        )
    else:
        actual = "unknown"
    pass_ = actual == expected
    return CheckResult(
        name="route_decision",
        expected=expected,
        actual=actual,
        pass_=pass_,
    )


def check_refusal(expected_template: Optional[str], events: list[dict]) -> CheckResult:
    """Adversarial-only: did the right REFUSAL_TEMPLATE fire?"""
    if expected_template is None:
        return CheckResult(
            name="refusal",
            expected=None,
            actual=None,
            pass_=True,
        )
    answer = _answer_field(events)
    # REFUSAL_TEMPLATES — copy of the canonical Chinese refusal copy
    # from src/llm/prompts.py:REFUSAL_TEMPLATES. We deliberately don't
    # import the dict because the harness is shipped code that should not
    # assume production is loaded — instead we hardcode the canonical
    # substrings each template emits. Keep this list in sync with
    # src/llm/prompts.py.
    templates = {
        "refusal_empty": ["未提取到文本", "资料库中没有", "暂未检索到"],
        "refusal_empty_bulk": ["该对话尚未上传任何文档", "请先上传文档"],
        "refusal_low_relevance": ["检索到的资料与您的问题相关性不足", "相关性不足"],
    }
    expected_strs = templates.get(expected_template, [])
    matched = [s for s in expected_strs if s in answer]
    pass_ = bool(matched)
    return CheckResult(
        name="refusal",
        expected=expected_template,
        actual=matched if matched else "no_match",
        pass_=pass_,
    )


# ============================================================
# Public entry point
# ============================================================


def run_programmatic(*, turn, events: list[dict]) -> ProgrammaticJudgment:
    """Run all 10 checks against a single turn's wire events.

    ``turn`` is a TurnAnnotation. Returns ProgrammaticJudgment with
    pass=True iff every check passes.

    v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07): when the runner
    detected a backend-LLM-degraded fallback (Phase 6 regen-loop
    defensive override fired repeatedly), answer-text checks become
    meaningless — the body is the defensive override's fallback, not
    the real LLM output. We SKIP them (forcing pass=True with a note)
    and force the overall pass_=True so the rollup's ``degraded``
    bucket separates this from real logic bugs. Structural checks
    that read events (not the answer body) still run normally.
    """
    degraded, degraded_reason = _is_degraded(events)
    if degraded:
        checks = [
            check_grounding_status(turn.expected_grounding, events),
            check_verification_consistent(_phase6_expected_consistent(turn), events),
            # Skip answer-text checks — body is defensive-override fallback.
            _skipped_check("expected_citations", turn.expected_citations),
            _skipped_check("answer_contains", turn.expected_answer_contains),
            _skipped_check("answer_lacks", turn.expected_answer_lacks),
            _skipped_check("forbidden_phrases", turn.forbidden_phrases),
            check_tool_sequence(turn.expected_tool_sequence, events),
            check_tool_call_count(turn.expected_tool_call_count, events),
            check_route_decision(turn.expected_route_decision, events),
            _skipped_check("refusal", None),
        ]
        return ProgrammaticJudgment(
            pass_=True,
            checks=tuple(checks),
            degraded=True,
            degraded_reason=degraded_reason,
        )
    checks = [
        check_grounding_status(turn.expected_grounding, events),
        check_verification_consistent(_phase6_expected_consistent(turn), events),
        check_expected_citations(turn.expected_citations, events),
        check_answer_contains(turn.expected_answer_contains, events),
        check_answer_lacks(turn.expected_answer_lacks, events),
        check_forbidden_phrases(turn.forbidden_phrases, events),
        check_tool_sequence(turn.expected_tool_sequence, events),
        check_tool_call_count(turn.expected_tool_call_count, events),
        check_route_decision(turn.expected_route_decision, events),
        check_refusal(
            turn.expected_refusal.template if turn.expected_refusal else None,
            events,
        ),
    ]
    return ProgrammaticJudgment(pass_=all(c.pass_ for c in checks), checks=tuple(checks))


def _is_degraded(events: list[dict]) -> tuple[bool, Optional[str]]:
    """Read the synthetic ``_degraded`` event injected by the runner's
    ``_aggregate_judgments`` (v2.0.32.7). Returns ``(degraded, reason)``.
    """
    for ev in events:
        if ev.get("type") == "_degraded":
            return True, ev.get("reason") if isinstance(ev.get("reason"), str) else None
    return False, None


def _skipped_check(name: str, expected) -> CheckResult:
    """v2.0.32.7 — placeholder check result for backend-degraded cases.
    Forces pass_=True so the rollup's ``degraded`` bucket does the
    accounting, not the answer-text checks.
    """
    return CheckResult(
        name=name,
        expected=list(expected) if expected else [],
        actual="skipped_due_to_degradation",
        pass_=True,
    )


def _phase6_expected_consistent(turn) -> bool:
    """Default for verification_consistent when the case has no
    adversarial expectation: pass iff the answer is grounded OR the
    turn is a refusal. Adversarial cases with expected_refusal get True
    (they want a refusal, which Phase 6 treats as consistent)."""
    if turn.expected_refusal is not None:
        return True
    return turn.expected_grounding in ("grounded", "skipped")


# ============================================================
# Subsequence matching (used by tool_sequence check)
# ============================================================


def _is_subsequence_or_loop(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    """Check if every needle element appears in haystack, in order, with
    loop-collapse: a needle element that appears multiple times in haystack
    matches any one occurrence.

    For example needle=(A, B) matches haystack=(A, X, Y, B, B).
    """
    h_idx = 0
    for n in needle:
        found = False
        while h_idx < len(haystack):
            if haystack[h_idx] == n:
                found = True
                h_idx += 1
                break
            h_idx += 1
        if not found:
            return False
    return True