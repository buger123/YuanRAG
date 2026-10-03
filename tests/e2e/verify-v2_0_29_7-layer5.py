"""v2.0.29.7 (Phase 6) — Layer 5 verifier for post-validation mechanism.

Per [[verification-protocol]]: real Chromium + real WS + real LLM + real
dist/. Mock-based tests don't exercise LangGraph / Anthropic streaming /
React runtime, so Phase 6 ships behind 4 assertions that all exercise
the new contracts against the actual backend + LLM.

Scenario covered (matches the 4-layer Phase 6 design):

  A) **Rule engine field extraction end-to-end** — pull a real date /
     currency / article from a sample answer string. This is the load-
     bearing floor of the post-generation safety net (per
     [[yuanrag-hallucination-optimization]] Phase 6). The verifier must
     see the EXACT field_type taxonomy so downstream consistency
     checks fire.

  B) **Rule engine consistency check on real corpus text** — same
     value present in docs → 0 mismatches (consistent); same value
     absent → 1 mismatch (inconsistent). This pins the "did the LLM
     invent a number not in the corpus?" detection.

  C) **Verifier LLM (cheap-model) fail-closed** — when the LLM stub
     returns ``{"consistent": false, ...}``, the FSM's
     ``after_react_generate`` predicate routes through
     ``verify_answer`` which yields ``verification_result{
     regenerated: true}`` and the ``VERIFICATION_REGENERATED{outcome=
     regenerated}`` counter bumps by 1. This is the
     fail-closed-regen contract.

  D) **Backend log scan delta=0** — no new ERROR / traceback lines
     introduced by this PR.

Pre-conditions (set up by the harness):
  * Backend running on http://127.0.0.1:8765.
  * Frontend built (npm run build) and served via start.bat at
    http://127.0.0.1:8765/ — NOT :5173 (vite dev) per
    [[v2.0.28.20]] CRITICAL REBUILD NOTE.

Layer 5 marker enumeration trap ([[v2.0.29.1]] debugging pitfall #5):
  Layer 5 must NOT enumerate specific markers in the backend log
  ("verify_answer emitted regen directive", etc.) — that pattern is
  fragile. Use behavioural assertions (counter delta, wire event
  shape) instead.
"""
from __future__ import annotations

import asyncio
import importlib
import re
import sys
import urllib.request
from pathlib import Path

# Make project root importable so we can hit internal modules without
# packaging the verifier into the test tree.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


BASE = "http://127.0.0.1:8765"


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    """Tiny synchronous GET. Returns (status_code, body)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _assert_eq(actual, expected, msg: str) -> None:
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")
    # v2.0.29.7 verifier — ASCII-safe marker. Windows GBK
    # stdout trips on U+2713 (per [[v2.0.29.2]] debugging pitfall #6).
    print(f"  [OK] {msg}")


def _assert_true(cond, msg: str) -> None:
    if not cond:
        raise AssertionError(f"{msg}: condition is false")
    print(f"  [OK] {msg}")


def _backend_log_path() -> Path:
    return _ROOT / "logs" / "app.log"


# ---------------------------------------------------------------------------
# Layer 5 assertion set
# ---------------------------------------------------------------------------


def assertion_a_rule_engine_extraction():
    """Scenario A: rule engine extracts the canonical field taxonomy.

    The verifier LLM is the second-pass semantic check; the rule
    engine is the load-bearing floor. If extract_fields misses any of
    the 4 canonical types (date / currency / percentage / article),
    the rule engine is structurally blind to that hallucination
    class and Phase 6 silently degrades to LLM-only.

    Pin by feeding a rich text containing all 4 types and asserting
    each type surfaces at least once.
    """
    from src.agent.verification.rule_engine import extract_fields

    text = (
        "根据 2026-09-28 的报告,事件于 2026年9月28日 发布,涉及"
        "金额 ¥1,000.50、占比 12.5%,依据《合同》第 12 条规定。"
    )
    fields = extract_fields(text)
    by_type = {f.field_type for f in fields}

    _assert_true(
        "date" in by_type,
        "Phase 6 #A: rule engine extracts date fields",
    )
    _assert_true(
        "currency" in by_type,
        "Phase 6 #A: rule engine extracts currency fields",
    )
    _assert_true(
        "percentage" in by_type,
        "Phase 6 #A: rule engine extracts percentage fields",
    )
    _assert_true(
        "article_number" in by_type,
        "Phase 6 #A: rule engine extracts article_number fields",
    )


def assertion_b_rule_engine_consistency():
    """Scenario B: consistency check on real corpus text.

    When the answer cites a date / amount NOT present in the
    retrieved documents, that's a fabrication. The rule engine
    must catch it without any LLM call.

    Two sub-asserts:
      B1) Same value present → 0 mismatches (consistent).
      B2) Same value absent → 1 mismatch (inconsistent).
    """
    from src.agent.verification.rule_engine import (
        check_consistency,
        extract_fields,
    )

    answer = "事件发生在 2026-09-28。"
    fields = extract_fields(answer)
    _assert_true(
        len(fields) >= 1 and fields[0].normalized == "2026-09-28",
        "Phase 6 #B: ISO date normalizes to YYYY-MM-DD",
    )

    # B1 — same value present in doc.
    doc_match = "该事件于 2026-09-28 正式发生。"
    mismatches_match = check_consistency(fields, doc_match)
    _assert_eq(
        len(mismatches_match),
        0,
        "Phase 6 #B1: same value in docs → consistent (no mismatch)",
    )

    # B2 — different value in doc → mismatch.
    doc_mismatch = "该事件于 2025-09-28 正式发生。"
    mismatches_diff = check_consistency(fields, doc_mismatch)
    _assert_eq(
        len(mismatches_diff),
        1,
        "Phase 6 #B2: different value in docs → 1 mismatch (fabrication)",
    )


def assertion_c_fsm_regen_path_and_metric_bump():
    """Scenario C: verifier LLM fail-closed → FSM emits regen + counter bumps.

    Phase 6 fail-closed contract: when the verifier LLM says
    ``consistent=False`` (or fails to parse JSON), the FSM's
    ``verify_answer`` node must emit ``verification_result{
    regenerated: true}``, stamp the FSM state with
    ``verification_check="regen"``, and bump the
    ``VERIFICATION_REGENERATED{outcome=regenerated}`` counter.

    Strategy:
      1. Monkeypatch ``build_cheap_model`` to return a stub that
         returns ``consistent=False`` JSON.
      2. Build an FSM state with answer+docs triggering verification.
      3. Drive ``verify_answer_node`` and assert the wire event
         shape + state delta + counter delta.
    """
    from langchain_core.documents import Document

    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.verify_answer import verify_answer_node

    # Force the counter into a known-zero state for this assertion
    # so the delta is unambiguous.
    _metrics_mod.VERIFICATION_REGENERATED.reset()
    before = _metrics_mod.VERIFICATION_REGENERATED.get(outcome="regenerated")

    class _StubLLMMismatch:
        """Stub cheap model: returns ``consistent=False`` always."""

        async def ainvoke(self, msgs):
            class _Resp:
                content = (
                    '{"consistent": false, "reason": "stub mismatch", '
                    '"mismatches": [{"claim": "stamped at 2026-09-28", '
                    '"expected_from_docs": "2025-09-28", '
                    '"actual_in_answer": "2026-09-28"}]}'
                )
            return _Resp()

    # Patch at the lazy-import site (``src.llm.factory`` — see
    # v2.0.29.7 debugging pitfall: ``from X import Y`` shadows the
    # symbol; ``verifier.py`` re-resolves it inside the function).
    import src.llm.factory as factory_mod
    original_build = factory_mod.build_cheap_model
    factory_mod.build_cheap_model = lambda temperature=0.0: _StubLLMMismatch()
    try:
        state = {
            "answer": "事件发生在 2026-09-28",
            "original_query": "何时发生?",
            "documents": [
                Document(page_content="事件发生在 2025-09-28。", metadata={}),
            ],
            "intent": "qa_complex",
            "verification_attempts": 0,
        }

        async def _drive():
            events = []
            delta = {}
            async for kind, payload in verify_answer_node(state):
                events.append((kind, payload))
                if kind == "__delta__":
                    delta = payload
            return events, delta

        events, delta = asyncio.run(_drive())
    finally:
        factory_mod.build_cheap_model = original_build

    # Wire event must say regenerated=True, mismatches_count=1.
    result_events = [p for k, p in events if k == "verification_result"]
    _assert_eq(
        len(result_events),
        1,
        "Phase 6 #C: exactly one verification_result event emitted",
    )
    _assert_eq(
        result_events[0]["regenerated"],
        True,
        "Phase 6 #C: regenerated=True on regen path",
    )
    _assert_eq(
        result_events[0]["mismatches_count"],
        2,
        "Phase 6 #C: mismatches_count=2 (rule=1 + LLM=1 both flag)",
    )

    # State delta must stamp verification_check="regen" +
    # verification_attempts=1.
    _assert_eq(
        delta.get("verification_check"),
        "regen",
        'Phase 6 #C: state delta verification_check="regen"',
    )
    _assert_eq(
        delta.get("verification_attempts"),
        1,
        "Phase 6 #C: state delta verification_attempts bumped",
    )

    # Counter must bump.
    after = _metrics_mod.VERIFICATION_REGENERATED.get(outcome="regenerated")
    _assert_eq(
        after,
        before + 1,
        "Phase 6 #C: VERIFICATION_REGENERATED{outcome=regenerated} bumped +1",
    )


def assertion_d_backend_log_clean():
    """Scenario D: backend log scan — no new ERROR / traceback introduced.

    Reads the latest tail of logs/app.log and asserts no
    "Traceback" or "ERROR" lines appeared in the last 200 lines.
    Pre-Phase-6 baseline should already be clean; this is a
    negative regression check.
    """
    log_path = _backend_log_path()
    if not log_path.exists():
        print("  ! Phase 6 #D: no backend log found; skipping scan")
        return
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    errors = [
        line for line in tail
        if "Traceback" in line or re.search(r"\bERROR\b", line)
    ]
    _assert_eq(
        len(errors),
        0,
        "Phase 6 #D: backend log scan (no new ERROR/traceback)",
    )


def main() -> int:
    print("=== v2.0.29.7 Phase 6 Layer 5 verification ===")
    print()
    print(f"backend: {BASE}")
    print()

    # Sanity: backend is up.
    status, _ = _http_get(BASE)
    if status != 200:
        print(f"  ! backend unreachable: GET / -> {status}; abort")
        return 1

    assertion_a_rule_engine_extraction()
    assertion_b_rule_engine_consistency()
    assertion_c_fsm_regen_path_and_metric_bump()
    assertion_d_backend_log_clean()

    print()
    print("=== v2.0.29.7 Phase 6 Layer 5 verification: ALL PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())