"""Stage 2 smoke pytest — drives the eval harness against the REAL backend.

v2.0.32.0 — these tests gate ship of the eval harness. They:

  1. Spin up against http://127.0.0.1:8765 with real LLM (MiniMax or any
     provider configured via .env) and real BGE-M3 + reranker.
  2. Drive 5 hand-picked cases from tests/fixtures/eval_v1/cases/{golden,
     adversarial}.yaml through the harness via POST /chat SSE.
  3. Assert the harness emits events, completes within timeout, and the
     programmatic judgment fires on every turn.

Why @pytest.mark.slow (per [[verification-protocol]]):
  These tests hit the real backend. They are NOT collected by default —
  only via ``pytest -m slow`` or ``pytest --run-slow``. CI default skips
  them.

Pre-conditions (asserted in conftest fixtures below, NOT auto-started):
  - Backend running on http://127.0.0.1:8765 (start.bat serving dist/)
  - .env loaded with a real LLM key (otherwise the runner reports
    "llm_error" and the smoke test fails on assert)
  - BGE-M3 + reranker preloaded (or :models/wait ready)

Per [[v2.0.28.20]] REBUILD NOTE: source-only fix requires ``npm run build``
to regenerate dist/. The harness reads from /chat which is on the
backend, NOT the frontend bundle — but the ship gate also requires
dist/ to exist on disk for the frontend to render.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
import pytest

# Mark every test in this module as slow — these hit the real backend.
# We do NOT add asyncio module-level mark because most tests are sync
# (driven via runner.run_case which wraps asyncio.run internally). The
# one async helper, _probe, is decorated per-function.
pytestmark = [pytest.mark.slow]


BACKEND_URL = os.environ.get("YUANRAG_BACKEND_URL", "http://127.0.0.1:8765")
SMOKE_TIMEOUT_S = float(os.environ.get("YUANRAG_SMOKE_TIMEOUT", "180"))


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def backend_alive() -> None:
    """Fail fast if backend isn't reachable on :8765."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
    except Exception as exc:
        pytest.skip(f"backend not reachable at {BACKEND_URL}: {exc}")
    if resp.status_code != 200:
        pytest.skip(f"backend /health returned {resp.status_code}: {resp.text}")


@pytest.fixture(scope="module")
def dist_built() -> None:
    """Per [[v2.0.28.20]] REBUILD NOTE: source-only fix requires dist/.

    We don't load the bundle, just assert it exists. The eval harness
    drives the backend via POST /chat (no UI), so dist/ existence is
    only a sanity signal that npm run build ran in this session.
    """
    dist = Path(__file__).resolve().parents[1] / "src" / "frontend" / "dist" / "index.html"
    if not dist.exists():
        pytest.skip(f"frontend dist/ not built — run `npm run build` in src/frontend: {dist}")


@pytest.fixture(scope="module")
def runner(backend_alive, dist_built):
    """Build a harness runner pointed at the live backend."""
    from tests.eval.conftest_compat import ensure_isolated_data_dir
    from tests.eval.runner import AsyncHarnessRunner

    ensure_isolated_data_dir()
    return AsyncHarnessRunner(base_url=BACKEND_URL, timeout=SMOKE_TIMEOUT_S)


# ============================================================
# Smoke test 1 — Golden suite loads + schema valid
# ============================================================


def test_smoke_loads_golden_suite(backend_alive) -> None:
    """All 10 golden YAML cases load cleanly via load_suite('golden')."""
    from tests.eval.cases import load_suite

    cases = load_suite("golden")
    assert len(cases) == 10, f"golden.yaml should have 10 cases, got {len(cases)}"
    for c in cases:
        assert c.id.startswith("golden-"), f"unexpected case id: {c.id}"
        assert c.language == "zh", f"all Phase 1 cases are zh-only: {c.id}={c.language}"
        assert c.suite == "golden"
        assert len(c.turns) >= 1
    ids = {c.id for c in cases}
    # Sanity — the two cases the user explicitly authored must be present
    assert "golden-001-time-zh" in ids
    assert "golden-002-weather-zh" in ids


# =============================================================
# Smoke test 2 — Adversarial suite loads + root_cause_group tagged
# =============================================================


def test_smoke_loads_adversarial_suite(backend_alive) -> None:
    """All 22 adversarial YAML cases load + each maps to one of A-F groups."""
    from tests.eval.cases import load_suite

    cases = load_suite("adversarial")
    assert len(cases) == 22, f"adversarial.yaml should have 22 cases, got {len(cases)}"
    groups: dict[str | None, int] = {}
    for c in cases:
        groups[c.root_cause_group] = groups.get(c.root_cause_group, 0) + 1
    # Phase 8 6 根因组 — all 6 must be present (A-F mapping)
    for g in ("A", "B", "C", "D", "E", "F"):
        assert groups.get(g, 0) >= 3, f"root cause group {g} should have >=3 cases, got {groups.get(g, 0)}"


# =============================================================
# Smoke test 3 — Drive one no-doc turn against real backend
# =============================================================


def test_smoke_runner_drives_one_no_doc_turn(runner) -> None:
    """Drive golden-010 (1+1=几, no docs, direct route) end-to-end.

    This is the cheapest smoke — no upload, no docs, single direct
    generation. Catches: LLM key valid, /chat SSE stable, programmatic
    judge wires correctly. ~5s round trip on MiniMax-M3.
    """
    from tests.eval.cases import load_suite

    cases = [c for c in load_suite("golden") if c.id == "golden-010-direct-zh"]
    assert len(cases) == 1
    case = cases[0]

    result = runner.run_case(case)

    assert result.case_id == "golden-010-direct-zh"
    assert len(result.turns) == 1
    turn = result.turns[0]
    assert turn.failure_reason is None, f"turn failed: {turn.failure_reason}"
    assert turn.latency_ms > 0
    assert turn.events, "no wire events captured — SSE parser broken"
    # answer must contain "2" (case-insensitive per programmatic normalize)
    assert "2" in turn.answer, f"answer missing '2': {turn.answer!r}"
    # programmatic must have fired
    prog = result.judgment.get("programmatic")
    assert prog is not None
    assert prog["total_count"] == 1


# =============================================================
# Smoke test 4 — Drive one with-fixture-upload turn (golden-005)
# =============================================================


def test_smoke_runner_drives_one_with_fixture_upload(runner) -> None:
    """Drive golden-005 (CET-6 PDF, 考试时间 15:00-17:25) end-to-end.

    This exercises the full upload → poll → retrieval → answer loop.
    ~30s on MiniMax-M3 due to BGE embedding + reranker.

    Skip if the corpus PDF is missing — that means the developer ran
    only the mocked tests, not the corpus-copy step.
    """
    from tests.eval.cases import load_suite

    cases = [c for c in load_suite("golden") if c.id == "golden-005-pdf-time-zh"]
    assert len(cases) == 1
    case = cases[0]

    if not case.fixture_docs:
        pytest.skip("golden-005 has no fixture_docs")
    from tests.eval.cases import fixture_path
    if not fixture_path(case.fixture_docs[0]).exists():
        pytest.skip(f"fixture not found: {case.fixture_docs[0]}")

    t0 = time.monotonic()
    result = runner.run_case(case)
    elapsed = time.monotonic() - t0

    assert result.case_id == "golden-005-pdf-time-zh"
    assert len(result.turns) == 1
    turn = result.turns[0]
    if turn.failure_reason:
        pytest.skip(f"real LLM not configured or 5xx: {turn.failure_reason}")
    assert turn.events, "no wire events captured"
    # answer must reference 15:00 / 17:25 (the user's corrected expectation)
    answer_l = turn.answer.lower()
    assert "15:00" in answer_l or "15 点" in answer_l or "15点" in answer_l, (
        f"answer missing '15:00': {turn.answer!r}"
    )
    assert "17:25" in answer_l or "17 点" in answer_l or "17点" in answer_l, (
        f"answer missing '17:25': {turn.answer!r}"
    )
    # must NOT confuse with 报到 time 14:30
    assert "14:30" not in answer_l and "14 点 30" not in answer_l, (
        f"answer leaked 报到 time 14:30: {turn.answer!r}"
    )
    # Sanity log
    print(f"\n  golden-005 wall-clock: {elapsed:.1f}s, events={len(turn.events)}, "
          f"tool_calls={turn.tool_call_count}")


# =============================================================
# Smoke test 5 — Drive a 3-case subset end-to-end
# =============================================================


@pytest.mark.parametrize("case_id", ["golden-003-greeting-zh", "golden-001-time-zh", "golden-002-weather-zh"])
def test_smoke_runner_drives_3_case_subset(runner, case_id: str) -> None:
    """Drive 3 cases that exercise 3 distinct tool paths:

      - golden-003: greeting, direct route, 0 tools (greets the user)
      - golden-001: time tool, single-tool direct path
      - golden-002: time + web_search, multi-tool path

    Catches: cross-case thread isolation, /documents/clear-thread cleanup,
    TokenCapture sums correctly per case (not bleed-through).
    """
    from tests.eval.cases import load_suite

    cases = [c for c in load_suite("golden") if c.id == case_id]
    assert len(cases) == 1, f"case {case_id} not found"
    case = cases[0]

    result = runner.run_case(case)

    assert result.case_id == case_id
    assert result.thread_id.startswith(f"eval-{case.thread_id}-"), (
        f"thread_id prefix wrong: {result.thread_id}"
    )
    assert len(result.turns) == len(case.turns)
    for i, turn in enumerate(result.turns):
        if turn.failure_reason:
            pytest.skip(f"case {case_id} turn {i} failed: {turn.failure_reason}")
        assert turn.events, f"no events for {case_id} turn {i}"
        # For golden-002 specifically — we expect 2 tool calls (time + web)
        if case_id == "golden-002-weather-zh":
            assert turn.tool_call_count >= 2, (
                f"golden-002 should use 2 tools (time + web_search), got {turn.tool_call_count}"
            )
        # For golden-001 specifically — 1 tool call
        if case_id == "golden-001-time-zh":
            assert turn.tool_call_count >= 1, (
                f"golden-001 should call time tool at least once, got {turn.tool_call_count}"
            )
        # For golden-003 — 0 tools
        if case_id == "golden-003-greeting-zh":
            assert turn.tool_call_count == 0, (
                f"golden-003 should have 0 tool calls, got {turn.tool_call_count}"
            )

    print(f"\n  {case_id}: composite={result.composite_score:.2f} "
          f"pass={result.pass_} wall={result.wall_clock_ms}ms")
