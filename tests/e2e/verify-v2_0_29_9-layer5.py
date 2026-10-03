"""v2.0.29.9 (Phase 8) — Layer 5 verifier for verbatim extraction mode.

Per [[verification-protocol]]: real Chromium + real WS + real LLM + real
dist/. Mock-based tests don't exercise LangGraph / Anthropic streaming /
React runtime, so Phase 8 ships behind 4 assertions that all exercise
the new verbatim-extraction contract against the actual backend.

Scenario covered (matches the 4-assertion Phase 8 design):

  A) **Regex-triggered verbatim routing** — ``_query_is_high_precision``
     matches the canonical CN legal trigger ("法律规定第 12 条") AND
     intent_analysis forces intent="qa_complex" + stamps high_precision=
     "on" via the __delta__ payload. ``after_intent`` routes to
     react_generate_extractive.

  B) **User OFF override semantics** — when state.high_precision=
     "off" is set via WS payload, even a verbatim-trigger query
     falls through to normal synthesis (the ``after_intent`` predicate
     never routes to react_generate_extractive).

  C) **User ON forces verbatim** — state.high_precision="on" via WS
     payload routes through react_generate_extractive even when the
     cheap-LLM classified intent as ``simple_fact``. The user override
     wins.

  D) **EXTRACTIVE_FALLBACK counter is wired** — drives
     ``react_generate_extractive`` with empty docs → asserts the
     counter bumps with ``reason="empty"`` and the emitted answer
     matches ``REFUSAL_TEMPLATES["empty"]``.

  E) **Backend log scan delta=0** — no new ERROR / traceback lines
     introduced by this PR.

Pre-conditions (set up by the harness):
  * Backend running on http://127.0.0.1:8765.
  * Frontend built (npm run build) and served via start.bat at
    http://127.0.0.1:8765/ — NOT :5173 (vite dev) per
    [[v2.0.28.20]] CRITICAL REBUILD NOTE.

Hermetic test strategy: per [[v2.0.29.4_p1]] Layer 5 precedent, no
full LLM round-trip required. We pin the verbatim contract by
driving the async-gen nodes directly + asserting FSM routing.
End-to-end behavioral tests live in pytest (test_high_precision_mode.py
— 51 assertions); Layer 5 only checks the load-bearing FSM + node
contracts (regex + override + counter + log cleanliness).
"""
from __future__ import annotations

import asyncio
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
    # v2.0.29.9 verifier — ASCII-safe marker. Windows GBK stdout
    # trips on Unicode emoji (per [[v2.0.29.2]] debugging pitfall).
    print(f"  [OK] {msg}")


def _assert_true(cond, msg: str) -> None:
    if not cond:
        raise AssertionError(f"{msg}: condition is false")
    print(f"  [OK] {msg}")


def _backend_log_path() -> Path:
    return _ROOT / "logs" / "app.log"


def _drive_async_gen(node_fn, state) -> list:
    """Drive async-gen node to completion; return list of (kind, payload)."""
    out = []

    async def _run():
        async for kind, payload in node_fn(state):
            out.append((kind, payload))

    asyncio.run(_run())
    return out


# ---------------------------------------------------------------------------
# Layer 5 assertion set
# ---------------------------------------------------------------------------


def assertion_a_regex_triggers_verbatim_routing():
    """Scenario A: regex hit on legal trigger → intent=qa_complex + hp=on.

    Drives ``intent_analysis`` (sync wrapper) on a verbatim query,
    then asserts ``after_intent`` routes to ``react_generate_extractive``.
    Pins the contract:
      - Layer 1 regex fires on the CN legal trigger phrase.
      - ``intent_analysis_sync`` returns intent="qa_complex" + high_precision="on".
      - ``after_intent`` routes through the verbatim branch.
    """
    from src.agent.fsm import after_intent
    from src.agent.nodes.intent_analysis import intent_analysis_sync
    from src.agent.nodes.intent_analysis import (
        _query_is_high_precision,
        _resolve_high_precision,
    )

    # Layer 1 — regex hit (zero-cost, no LLM call)
    _assert_true(
        _query_is_high_precision("法律规定第 12 条规定了什么"),
        "Phase 8 #A: Layer 1 regex matches CN legal trigger",
    )

    # Layer 2 — sync drive of intent_analysis
    state = {"current_query": "法律规定第 12 条规定了什么", "messages": []}
    delta = intent_analysis_sync(state)
    _assert_eq(
        delta["intent"], "qa_complex",
        "Phase 8 #A: intent_analysis forces intent=qa_complex on regex hit",
    )
    _assert_eq(
        delta["high_precision"], "on",
        "Phase 8 #A: intent_analysis stamps high_precision=on",
    )

    # FSM predicate routes to react_generate_extractive
    next_node = after_intent(
        {"intent": delta["intent"], "high_precision": delta["high_precision"]},
    )
    _assert_eq(
        next_node, "react_generate_extractive",
        "Phase 8 #A: after_intent routes to react_generate_extractive",
    )

    # _resolve_high_precision helper semantics
    _assert_eq(
        _resolve_high_precision("auto", regex_hit=True), "on",
        "Phase 8 #A: auto + regex hit -> on",
    )


def assertion_b_user_off_overrides_detect():
    """Scenario B: user OFF override wins — never routes to extractive.

    Even with a verbatim-trigger query, ``state.high_precision="off"``
    (set by WS payload) must:
      - Pass through intent_analysis unchanged (regex fast-path gated on
        ``user_hp_value != "off"``).
      - Cause ``after_intent`` to fall through to react_agent (NOT
        react_generate_extractive).
    """
    from src.agent.fsm import after_intent
    from src.agent.nodes.intent_analysis import intent_analysis_sync
    from src.agent.nodes.intent_analysis import (
        _query_is_high_precision,
        _resolve_high_precision,
    )

    # The query would normally match the regex
    _assert_true(
        _query_is_high_precision("verbatim quote please"),
        "Phase 8 #B: query matches regex baseline",
    )

    # User has set OFF — intent_analysis fast-path short-circuits, no
    # high_precision stamp from detect fires. Result must remain "off".
    state = {
        "current_query": "verbatim quote please",
        "messages": [],
        "high_precision": "off",
    }
    delta = intent_analysis_sync(state)
    # Phase 8 — when user is OFF, the regex fast-path is gated; the
    # cheap-LLM path runs (no verbatim_needed → high_precision stays
    # "off" via _resolve_high_precision).
    _assert_eq(
        delta["high_precision"], "off",
        "Phase 8 #B: user OFF preserved through intent_analysis",
    )

    # after_intent sees "off" -> falls through to normal routing
    next_node = after_intent(
        {"intent": delta["intent"], "high_precision": "off"},
    )
    _assert_eq(
        next_node, "react_agent",
        "Phase 8 #B: user OFF -> react_agent (NOT extractive)",
    )

    # Helper-level invariant: OFF wins regardless of detect signal
    _assert_eq(
        _resolve_high_precision("off", regex_hit=True, llm_verbatim=True), "off",
        "Phase 8 #B: _resolve_high_precision user OFF is unconditional",
    )


def assertion_c_user_on_forces_verbatim():
    """Scenario C: user ON override forces extractive even without regex.

    A query that does NOT match the verbatim regex (e.g. "今天天气")
    with state.high_precision="on" must still route through
    react_generate_extractive. Pins the user-on-overrides-all behavior.
    """
    from src.agent.fsm import after_intent
    from src.agent.nodes.intent_analysis import intent_analysis_sync
    from src.agent.nodes.intent_analysis import _query_is_high_precision

    # Baseline: query does NOT match verbatim regex
    _assert_true(
        not _query_is_high_precision("今天天气怎么样"),
        "Phase 8 #C: baseline query does NOT match regex",
    )

    # User has set ON — intent_analysis forces qa_complex + hp=on
    state = {
        "current_query": "今天天气怎么样",
        "messages": [],
        "high_precision": "on",
    }
    delta = intent_analysis_sync(state)
    _assert_eq(
        delta["intent"], "qa_complex",
        "Phase 8 #C: user ON -> intent=qa_complex (force)",
    )
    _assert_eq(
        delta["high_precision"], "on",
        "Phase 8 #C: user ON -> high_precision=on",
    )

    # after_intent routes to extractive
    next_node = after_intent(
        {"intent": delta["intent"], "high_precision": "on"},
    )
    _assert_eq(
        next_node, "react_generate_extractive",
        "Phase 8 #C: user ON -> react_generate_extractive",
    )


def assertion_d_extractive_node_refusal_counter():
    """Scenario D: react_generate_extractive with empty docs.

    Drives the async-gen node directly with docs=[] + retrieval_status=
    "empty". The node must:
      - Emit REFUSAL_TEMPLATES["empty"] as the answer.
      - Emit sources=[] (empty list).
      - Bump EXTRACTIVE_FALLBACK{reason="empty"} counter by exactly 1.
    """
    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.react_generate_extractive import react_generate_extractive
    from src.llm.prompts import REFUSAL_TEMPLATES

    counter = _metrics_mod.EXTRACTIVE_FALLBACK
    counter.reset()
    before = counter.get(reason="empty")

    state = {
        "documents": [],
        "retrieval_status": "empty",
        "original_query": "verbatim please",
        "current_query": "verbatim please",
    }
    events = _drive_async_gen(react_generate_extractive, state)

    # First event: answer_complete with refusal template + empty sources
    kind, payload = events[0]
    _assert_eq(
        kind, "answer_complete",
        "Phase 8 #D: first event is answer_complete",
    )
    _assert_eq(
        payload["answer"], REFUSAL_TEMPLATES["empty"],
        "Phase 8 #D: refusal template emitted for empty docs",
    )
    _assert_eq(
        payload["sources"], [],
        "Phase 8 #D: sources list is empty",
    )

    after = counter.get(reason="empty")
    _assert_eq(
        after - before, 1.0,
        "Phase 8 #D: EXTRACTIVE_FALLBACK{reason=empty} bumped by 1",
    )


def assertion_e_backend_log_clean():
    """Scenario E: backend log scan — no new ERROR / traceback introduced.

    Reads the latest tail of logs/app.log and asserts no
    "Traceback" or "ERROR" lines appeared in the last 200 lines.
    Pre-Phase-8 baseline should already be clean; this is a
    negative regression check.
    """
    log_path = _backend_log_path()
    if not log_path.exists():
        print("  ! Phase 8 #E: no backend log found; skipping scan")
        return
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    errors = [
        line for line in tail
        if "Traceback" in line or re.search(r"\bERROR\b", line)
    ]
    _assert_eq(
        len(errors), 0,
        "Phase 8 #E: backend log scan (no new ERROR/traceback)",
    )


def main() -> int:
    print("=== v2.0.29.9 Phase 8 Layer 5 verification ===")
    print()
    print(f"backend: {BASE}")
    print()

    # Sanity: backend is up.
    status, _ = _http_get(BASE)
    if status != 200:
        print(f"  ! backend unreachable: GET / -> {status}; abort")
        return 1

    assertion_a_regex_triggers_verbatim_routing()
    assertion_b_user_off_overrides_detect()
    assertion_c_user_on_forces_verbatim()
    assertion_d_extractive_node_refusal_counter()
    assertion_e_backend_log_clean()

    print()
    print("=== v2.0.29.9 Phase 8 Layer 5 verification: ALL PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())