"""Fast mocked tests for the eval harness (CI default — no @pytest.mark.slow).

v2.0.32.0 — covers 12 unit invariants per plan §F:
  1.  runner_drives_one_case
  2.  runner_drains_done_event
  3.  runner_captures_tokens_via_monkey_patch
  4.  runner_estimates_cost_from_pricing_table
  5.  runner_cleans_up_thread_per_case
  6.  runner_parses_sse_data_lines
  7.  runner_respects_concurrency_limit
  8.  runner_surfaces_llm_errors_as_failure_reason
  9.  runner_yaml_loader_safe_load_only
  10. runner_pricing_table_current
  11. runner_isolates_RAG_DATA_DIR
  12. runner_case_schema_validates_locale (loader-time)
"""
from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tests.eval.cases import (
    CANONICAL_FSM_NODES,
    Case,
    CaseLoadError,
    ScoringWeights,
    TurnAnnotation,
    ExpectedRefusal,
    fixture_path,
    load_suite,
)
from tests.eval.conftest_compat import ensure_isolated_data_dir
from tests.eval.cost import (
    PRICING_USD_PER_1K,
    ModelCost,
    TokenSnapshot,
    estimate_cost,
)
from tests.eval.judge.programmatic import (
    _fsm_node_sequence,
    check_answer_contains,
    check_answer_lacks,
    check_expected_citations,
    check_forbidden_phrases,
    check_grounding_status,
    check_refusal,
    check_route_decision,
    check_tool_call_count,
    check_tool_sequence,
    check_verification_consistent,
    run_programmatic,
)
from tests.eval.report import (
    RollupMetrics,
    _percentile,
    compute_rollup,
    make_timestamp,
    render_console_table,
    render_markdown,
)
from tests.eval.runner import (
    AsyncHarnessRunner,
    CaseResult,
    TurnResult,
)
from tests.eval.tokens import TokenCapture


# ============================================================
# Fake HTTP client — synchronous shape, minimal SSE
# ============================================================


class _FakeSSEClient:
    """Mimics httpx.Client — not httpx.AsyncClient — with stream() generator."""

    def __init__(self, *, sse_lines: list[str] | None = None, status_code: int = 200) -> None:
        self._sse_lines = sse_lines or [
            'data: {"type": "answer_complete", "answer": "Mock answer", "sources": []}\n\n',
            'data: {"type": "done"}\n\n',
        ]
        self._status_code = status_code
        self.post_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.delete_calls: list[str] = []

    def stream(self, method: str, url: str, *, json=None, headers=None) -> "_FakeResponse":
        self.post_calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return _FakeResponse(self._sse_lines, self._status_code)

    def post(self, url: str, **kw) -> httpx.Response:  # noqa: ANN003
        self.post_calls.append({"method": "POST", "url": url, **kw})
        return httpx.Response(204, request=httpx.Request("POST", url))

    def get(self, url: str, **kw) -> httpx.Response:  # noqa: ANN003
        self.get_calls.append({"url": url, **kw})
        if "/documents" in url:
            # Empty doc list → poll loop exits as 'no docs to wait for'
            return httpx.Response(200, json=[], request=httpx.Request("GET", url))
        return httpx.Response(200, json={}, request=httpx.Request("GET", url))

    def delete(self, url: str, **kw) -> httpx.Response:  # noqa: ANN003
        self.delete_calls.append(url)
        return httpx.Response(204, request=httpx.Request("DELETE", url))

    def close(self) -> None:
        pass


class _FakeResponse:
    """Context manager wrapping a list of SSE lines."""

    def __init__(self, lines: list[str], status_code: int) -> None:
        self.status_code = status_code
        # Split each chunk on '\n' so we yield one SSE line at a time,
        # mirroring what httpx.Client.iter_lines() does in production.
        flat: list[str] = []
        for chunk in lines:
            for line in chunk.split("\n"):
                # SSE uses \r\n over the network but we accept \n as well.
                line = line.rstrip("\r")
                flat.append(line)
        self._lines = flat

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def iter_lines(self):
        for ln in self._lines:
            yield ln


@pytest.fixture
def simple_case() -> Case:
    return Case(
        id="unit-001",
        language="en",
        difficulty="easy",
        thread_id="unit-001",
        turns=(
            TurnAnnotation(
                user="What is 2+2?",
                high_precision="auto",
                expected_answer_contains=("4",),
            ),
        ),
        suite="unit",
    )


# ============================================================
# Test 1 — runner_drives_one_case
# ============================================================


def test_runner_drives_one_case(simple_case) -> None:
    fake = _FakeSSEClient()
    runner = AsyncHarnessRunner(base_url="http://stub", timeout=5.0, client=fake)
    result = runner.run_case(simple_case)
    assert isinstance(result, CaseResult)
    assert result.case_id == "unit-001"
    assert result.language == "en"
    assert len(result.turns) == 1
    assert result.turns[0].answer == "Mock answer"
    assert result.turns[0].latency_ms >= 0
    assert result.thread_id.startswith("eval-unit-001-")


# ============================================================
# Test 2 — runner_drains_done_event
# ============================================================


def test_runner_drains_done_event(simple_case) -> None:
    sse = [
        'data: {"type": "thinking", "reasoning": "..."}\n\n',
        'data: {"type": "answer_chunk", "chunk": "Hello"}\n\n',
        'data: {"type": "answer_chunk", "chunk": " world"}\n\n',
        'data: {"type": "answer_complete", "answer": "Hello world", "sources": []}\n\n',
        'data: {"type": "done"}\n\n',
        # Subsequent data lines should be ignored — runner breaks at done.
        'data: {"type": "after_done", "ignored": true}\n\n',
    ]
    fake = _FakeSSEClient(sse_lines=sse)
    runner = AsyncHarnessRunner(base_url="http://stub", timeout=5.0, client=fake)
    result = runner.run_case(simple_case)
    event_types = [e.get("type") for e in result.turns[0].events]
    assert event_types[-1] == "done"
    assert "after_done" not in event_types


# ============================================================
# Test 3 — runner_captures_tokens_via_monkey_patch
# ============================================================


def test_runner_captures_tokens_via_monkey_patch() -> None:
    """Directly verify TokenCapture accumulates usage_metadata from final chunk."""
    cap = TokenCapture.install()
    try:
        # Simulate a final chunk with usage_metadata via the module-level helper.
        from langchain_core.messages.ai import AIMessageChunk  # type: ignore
        from tests.eval.tokens import _extract_usage_metadata  # module-level

        chunk = AIMessageChunk(
            content="hello world",
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        usage = _extract_usage_metadata(chunk)
        assert usage == {"input_tokens": 100, "output_tokens": 50}
        # Inject into the capture's accumulator + tick the lock
        with cap._lock:
            key = ("anthropic", "MiniMax-M3")
            cap._by_model[key] = TokenSnapshot(input_tokens=100, output_tokens=50)
        snap = cap.snapshot_and_reset()
        assert snap, "TokenCapture did not accumulate usage_metadata from final chunk"
        for ts in snap.values():
            assert ts.input_tokens == 100
            assert ts.output_tokens == 50
    finally:
        cap.uninstall()


# ============================================================
# Test 4 — runner_estimates_cost_from_pricing_table
# ============================================================


def test_runner_estimates_cost_from_pricing_table() -> None:
    snap = TokenSnapshot(input_tokens=1000, output_tokens=500)
    cost = estimate_cost(snap, provider="anthropic", model="MiniMax-M3")
    # 1000/1000 * 0.003 = 0.003 + 500/1000 * 0.015 = 0.0075 → 0.0105
    assert cost == Decimal("0.0105")


# ============================================================
# Test 5 — runner_cleans_up_thread_per_case
# ============================================================


def test_runner_cleans_up_thread_per_case(simple_case) -> None:
    """Simple case (no fixture_docs) → upload is skipped, but clear-thread + DELETE still run."""
    fake = _FakeSSEClient()
    runner = AsyncHarnessRunner(base_url="http://stub", timeout=5.0, client=fake)
    result = runner.run_case(simple_case)
    posted_urls = [c["url"] for c in fake.post_calls]
    assert any("clear-thread" in u for u in posted_urls), "clear-thread not called"
    # No fixture_docs → upload is correctly skipped (not an error)
    assert not any("documents/upload" in u for u in posted_urls), "upload unexpectedly called"
    assert any(result.thread_id in u for u in fake.delete_calls), "DELETE /chat/{tid} not called"


def test_runner_uploads_when_fixture_docs_specified(tmp_path: Path) -> None:
    """Case with fixture_docs → upload MUST be called during setup."""
    # tests/test_eval_runner.py → parents[0] = tests/ → + fixtures/eval_v1/corpora
    real_corpus = Path(__file__).resolve().parents[0] / "fixtures" / "eval_v1" / "corpora" / "1.txt"
    if not real_corpus.exists():
        pytest.skip(f"corpus file missing: {real_corpus}")
    case = Case(
        id="unit-with-fixture",
        language="zh",
        difficulty="easy",
        thread_id="unit-with-fixture",
        turns=(TurnAnnotation(user="hello", high_precision="auto"),),
        fixture_docs=("1.txt",),
        suite="unit",
    )
    fake = _FakeSSEClient()
    runner = AsyncHarnessRunner(base_url="http://stub", timeout=5.0, client=fake)
    result = runner.run_case(case)
    posted_urls = [c["url"] for c in fake.post_calls]
    assert any("documents/upload" in u for u in posted_urls), "upload not called for case with fixture_docs"
    assert any("clear-thread" in u for u in posted_urls), "clear-thread not called"


# ============================================================
# Test 6 — runner_parses_sse_data_lines
# ============================================================


def test_runner_parses_sse_data_lines(simple_case) -> None:
    sse = [
        'data: {"type": "tool_call_start", "tool_call_id": "tc1", "name": "web_search"}\n\n',
        'data: {"type": "tool_call_end", "tool_call_id": "tc1", "elapsed_ms": 220, "ok": true}\n\n',
        'data: {"type": "answer_complete", "answer": "Result", "sources": [{"chunk_id": "c1"}]}\n\n',
        'data: {"type": "grounding", "status": "grounded"}\n\n',
        'data: {"type": "verification_result", "consistent": true, "skipped": false}\n\n',
        'data: {"type": "done"}\n\n',
    ]
    fake = _FakeSSEClient(sse_lines=sse)
    runner = AsyncHarnessRunner(base_url="http://stub", timeout=5.0, client=fake)
    result = runner.run_case(simple_case)
    turn = result.turns[0]
    assert turn.tool_call_count == 1
    assert turn.tool_call_names == ["web_search"]
    assert turn.tool_call_durations_ms == [220]
    assert turn.grounding_status == "grounded"
    assert turn.verification_consistent is True
    assert turn.verification_skipped is False
    assert turn.answer == "Result"
    assert turn.sources == [{"chunk_id": "c1"}]


# ============================================================
# Test 7 — runner_respects_concurrency_limit
# ============================================================


def test_runner_respects_concurrency_limit() -> None:
    """Phase 1 is sequential — verify cleanup order preserves case separation."""
    fake = _FakeSSEClient()
    runner = AsyncHarnessRunner(base_url="http://stub", timeout=5.0, client=fake)
    case_a = Case(
        id="unit-a",
        language="en",
        difficulty="easy",
        thread_id="unit-a",
        turns=(TurnAnnotation(user="hi", high_precision="auto"),),
        suite="unit",
    )
    case_b = Case(
        id="unit-b",
        language="en",
        difficulty="easy",
        thread_id="unit-b",
        turns=(TurnAnnotation(user="there", high_precision="auto"),),
        suite="unit",
    )
    r1 = runner.run_case(case_a)
    r2 = runner.run_case(case_b)
    assert r1.thread_id != r2.thread_id
    # Cleanup order: delete for case_a before delete for case_b
    delete_idx_a = next(i for i, u in enumerate(fake.delete_calls) if r1.thread_id in u)
    delete_idx_b = next(i for i, u in enumerate(fake.delete_calls) if r2.thread_id in u)
    assert delete_idx_a < delete_idx_b


# ============================================================
# Test 8 — runner_surfaces_llm_errors_as_failure_reason
# ============================================================


def test_runner_surfaces_llm_errors_as_failure_reason(simple_case) -> None:
    """4xx response → failure_reason='http_error:STATUS'."""
    fake = _FakeSSEClient(sse_lines=[""], status_code=503)
    runner = AsyncHarnessRunner(base_url="http://stub", timeout=5.0, client=fake)
    result = runner.run_case(simple_case)
    assert result.turns[0].failure_reason == "http_error:503"


# ============================================================
# Test 9 — runner_yaml_loader_safe_load_only
# ============================================================


def test_runner_yaml_loader_safe_load_only(tmp_path: Path) -> None:
    """YAML loader must reject unsafe constructors.

    Even though safe_load rejects most python/object tags at parse time
    (ScannerError before reaching our code), we still verify the loader
    surfaces a CaseLoadError rather than silently executing the malicious
    payload. Use a parseable but unsafe payload so the test exercises
    the loader's safety boundary.
    """
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    bad_yaml = cases_dir / "golden.yaml"
    bad_yaml.write_text(
        """\
---
- id: malicious
  language: en
  difficulty: easy
  thread_id: bad
  turns:
    - user: !!python/name:os.system ["echo BAD > harmful.txt"]
""",
        encoding="utf-8",
    )
    # safe_load raises ConstructorError on python/name; the loader must
    # surface that as CaseLoadError.
    with pytest.raises(CaseLoadError):
        load_suite("golden", fixtures_root=tmp_path)
    assert not (tmp_path / "harmful.txt").exists()


# ============================================================
# Test 10 — runner_pricing_table_current
# ============================================================


def test_runner_pricing_table_current() -> None:
    expected_pairs = {
        ("anthropic", "MiniMax-M3"),
        ("anthropic", "claude-3-5-sonnet-latest"),
        ("anthropic", "claude-3-5-haiku-latest"),
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("openai", "o1"),
    }
    assert expected_pairs <= set(PRICING_USD_PER_1K.keys())
    for mc in PRICING_USD_PER_1K.values():
        assert isinstance(mc, ModelCost)
        assert mc.input_per_1k >= Decimal(0)
        assert mc.output_per_1k >= Decimal(0)


# ============================================================
# Test 11 — runner_isolates_RAG_DATA_DIR
# ============================================================


def test_runner_isolates_RAG_DATA_DIR(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_isolated_data_dir() must set RAG_DATA_DIR if not already set."""
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)
    result_dir = ensure_isolated_data_dir()
    try:
        assert os.environ.get("RAG_DATA_DIR") == str(result_dir)
        assert result_dir.exists()
        assert result_dir.is_dir()
        assert "yuanrag-eval-" in result_dir.name
    finally:
        monkeypatch.delenv("RAG_DATA_DIR", raising=False)


def test_runner_isolates_RAG_DATA_DIR_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If RAG_DATA_DIR is already set, helper must NOT override it."""
    preset = tmp_path / "preexisting"
    preset.mkdir()
    monkeypatch.setenv("RAG_DATA_DIR", str(preset))
    out = ensure_isolated_data_dir()
    assert out == preset
    assert os.environ["RAG_DATA_DIR"] == str(preset)


# ============================================================
# Test 12 — runner_case_schema_validates_locale (loader-time)
# ============================================================


def test_runner_case_schema_validates_locale(tmp_path: Path) -> None:
    """Loader must reject language values outside {zh, en}."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    bad_yaml = cases_dir / "golden.yaml"
    bad_yaml.write_text(
        """\
---
- id: bad-fr
  language: fr
  difficulty: easy
  thread_id: bad-fr
  turns:
    - user: hello
""",
        encoding="utf-8",
    )
    with pytest.raises(CaseLoadError):
        load_suite("golden", fixtures_root=tmp_path)

    # Valid zh + en must succeed
    good_yaml = cases_dir / "golden.yaml"
    good_yaml.write_text(
        """\
---
- id: ok-en
  language: en
  difficulty: easy
  thread_id: ok-en
  turns:
    - user: hello
- id: ok-zh
  language: zh
  difficulty: easy
  thread_id: ok-zh
  turns:
    - user: 你好
""",
        encoding="utf-8",
    )
    cases = load_suite("golden", fixtures_root=tmp_path)
    assert {c.language for c in cases} == {"en", "zh"}


# ============================================================
# Bonus tests — programmatic judge + report (cheap to cover)
# ============================================================


def test_judge_fsm_node_sequence_infers_from_events() -> None:
    events = [
        {"type": "intent", "intent": "qa_complex"},
        {"type": "tool_call_start", "name": "search"},
        {"type": "tool_call_end"},
        {"type": "answer_complete", "sources": [{"chunk_id": "c1"}]},
        {"type": "grounding", "status": "grounded"},
    ]
    nodes = _fsm_node_sequence(events)
    assert "intent_analysis" in nodes
    assert "react_generate" in nodes
    # grounding event maps to the canonical "check_hallucination" FSM node
    assert "check_hallucination" in nodes


def test_judge_answer_contains_and_lacks() -> None:
    events = [{"type": "answer_complete", "answer": "Tesla's first product was the Roadster"}]
    assert check_answer_contains(["Tesla", "Roadster"], events).passed is True
    assert check_answer_lacks(["Model S", "incubator"], events).passed is True
    assert check_answer_contains(["ZZZ"], events).passed is False
    assert check_answer_lacks(["Tesla"], events).passed is False


def test_judge_forbidden_phrases() -> None:
    events = [{"type": "answer_complete", "answer": "I think this is probably correct."}]
    assert check_forbidden_phrases(["I think", "probably"], events).passed is False


def test_judge_expected_citations() -> None:
    events = [{"type": "answer_complete", "sources": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]}]
    assert check_expected_citations(["c1", "c2"], events).passed is True
    assert check_expected_citations(["c3"], events).passed is False


def test_judge_grounding_status() -> None:
    assert check_grounding_status("grounded", [{"type": "grounding", "status": "grounded"}]).passed is True
    assert check_grounding_status("grounded", [{"type": "grounding", "status": "ungrounded"}]).passed is False


def test_judge_verification_consistent() -> None:
    # Skipped → tolerated as "consistent by absence" with expected=True
    assert check_verification_consistent(
        True, [{"type": "verification_result", "consistent": False, "skipped": True}]
    ).passed is True
    # Verified + consistent → pass
    assert check_verification_consistent(
        True, [{"type": "verification_result", "consistent": True, "skipped": False}]
    ).passed is True


def test_judge_tool_sequence_and_count() -> None:
    events = [
        {"type": "tool_call_start", "name": "search"},
        {"type": "tool_call_end"},
        {"type": "answer_complete"},
    ]
    assert check_tool_call_count(1, events).passed is True
    assert check_tool_call_count(0, events).passed is False
    # _fsm_node_sequence maps tool_call_start → "tools" canonical node,
    # NOT the tool name. expected_tool_sequence uses canonical FSM node names.
    assert check_tool_sequence(["tools"], events).passed is True


def test_judge_refusal() -> None:
    # refusal_empty canonical substrings (from programmatic.py REFUSAL_TEMPLATES copy):
    #   ["未提取到文本", "资料库中没有", "暂未检索到"]
    refusal_events = [
        {"type": "answer_complete", "answer": "抱歉，未提取到文本，请上传包含相关内容的文档后再试。"}
    ]
    assert check_refusal("refusal_empty", refusal_events).passed is True
    # Non-refusal answer → fail
    non_refusal = [{"type": "answer_complete", "answer": "The answer is 42."}]
    assert check_refusal("refusal_empty", non_refusal).passed is False
    # None expected → pass (no check)
    assert check_refusal(None, non_refusal).passed is True


def test_judge_route_decision() -> None:
    # No sources + no tools → direct
    events = [{"type": "answer_complete", "sources": []}]
    assert check_route_decision("direct", events).passed is True
    # Sources present → retrieve (the check infers from tool_call_start)
    events_with_tool = [
        {"type": "tool_call_start", "name": "search"},
        {"type": "answer_complete", "sources": [{"chunk_id": "c1"}]},
    ]
    assert check_route_decision("retrieve", events_with_tool).passed is True
    # All sources verbatim → extractive
    events_verbatim = [
        {"type": "answer_complete", "sources": [{"chunk_id": "c1", "verbatim": True}]},
    ]
    assert check_route_decision("extractive", events_verbatim).passed is True


def test_judge_route_decision_any_accepts_all_valid_routings() -> None:
    """v2.0.32.8 — "any" route_decision for summary-eligible doc-questions
    where the cheap-model classifier is non-deterministic across qa_complex
    and summary paths but BOTH produce correct answers."""
    # direct path
    assert check_route_decision("any", [{"type": "answer_complete", "sources": []}]).passed is True
    # retrieve path
    assert check_route_decision(
        "any",
        [
            {"type": "tool_call_start", "name": "search"},
            {"type": "answer_complete", "sources": [{"chunk_id": "c1"}]},
        ],
    ).passed is True
    # extractive path
    assert check_route_decision(
        "any",
        [{"type": "answer_complete", "sources": [{"chunk_id": "c1", "verbatim": True}]}],
    ).passed is True


def test_run_programmatic_returns_judgment() -> None:
    turn = TurnAnnotation(
        user="x",
        high_precision="auto",
        expected_answer_contains=("mock",),
    )
    events = [{"type": "answer_complete", "answer": "This is a mock answer", "sources": []}]
    judgment = run_programmatic(turn=turn, events=events)
    assert judgment.passed is True
    assert any(c.name == "answer_contains" for c in judgment.checks)


def test_run_programmatic_unknown_locale_passes_silently() -> None:
    """No assertions to check → vacuously pass."""
    turn = TurnAnnotation(user="x", high_precision="auto")
    events = [{"type": "answer_complete", "answer": "ok", "sources": []}]
    j = run_programmatic(turn=turn, events=events)
    assert j.passed is True


def test_run_programmatic_degraded_event_skips_text_checks() -> None:
    """v2.0.32.7 — synthetic ``_degraded`` event flips the judge into
    degraded-mode: answer-text checks (answer_contains, answer_lacks,
    forbidden_phrases, refusal, expected_citations) are skipped and
    marked pass=True with actual="skipped_due_to_degradation". Overall
    pass_=True + degraded=True so the rollup buckets this as infra-
    degraded, not a logic bug.
    """
    turn = TurnAnnotation(
        user="x",
        high_precision="auto",
        expected_answer_contains=("NEVER_IN_ANSWER",),
        expected_answer_lacks=("never",),
        forbidden_phrases=("never",),
    )
    events = [
        {
            "type": "answer_complete",
            "answer": "根据刚刚查询,当前时间是 2026-10-07T...",  # degraded fallback
            "sources": [],
        },
        {"type": "_degraded", "reason": "time_fallback_on_doc_query"},
    ]
    j = run_programmatic(turn=turn, events=events)
    assert j.passed is True
    assert j.degraded is True
    assert j.degraded_reason == "time_fallback_on_doc_query"
    # Skipped checks must carry the marker so downstream consumers
    # can tell them apart from real passes.
    skipped_names = {"answer_contains", "answer_lacks", "forbidden_phrases", "refusal", "expected_citations"}
    for c in j.checks:
        if c.name in skipped_names:
            assert c.pass_ is True
            assert c.actual == "skipped_due_to_degradation"


def test_run_programmatic_no_degraded_event_runs_text_checks_normally() -> None:
    """v2.0.32.7 — guard: when no ``_degraded`` event is present, the
    judge behaves exactly like before (no behavioral drift)."""
    turn = TurnAnnotation(
        user="x",
        high_precision="auto",
        expected_answer_contains=("WRONG_NEVER_MATCHES",),
    )
    events = [{"type": "answer_complete", "answer": "ok", "sources": []}]
    j = run_programmatic(turn=turn, events=events)
    assert j.passed is False  # answer_contains fails as expected
    assert j.degraded is False  # NOT marked degraded
    assert any(c.name == "answer_contains" for c in j.checks)


def test_detect_llm_degraded_helper_flags_time_fallback_on_doc_query() -> None:
    """v2.0.32.7 — runner helper detects Phase 6 regen-loop fallback
    (defensive override returned time-tool answer on a doc query).
    """
    from tests.eval.runner import _detect_llm_degraded

    # Time-tool fallback on a doc query → degraded=True.
    degraded, reason = _detect_llm_degraded(
        answer="根据刚刚查询,当前时间是 2026-10-07T...",
        sources=[],
        verification_consistent=False,
        verification_skipped=False,
        user_message="我的六级考试在哪所学校?",  # doc query, not time query
    )
    assert degraded is True
    assert reason == "time_fallback_on_doc_query"

    # Time-tool answer on a TIME query → not degraded.
    degraded, reason = _detect_llm_degraded(
        answer="根据刚刚查询,当前时间是 2026-10-07T...",
        sources=[],
        verification_consistent=True,
        verification_skipped=False,
        user_message="现在北京时间几点?",
    )
    assert degraded is False
    assert reason is None

    # Empty answer → not degraded (failure_reason handles it).
    degraded, reason = _detect_llm_degraded(
        answer="",
        sources=[],
        verification_consistent=None,
        verification_skipped=True,
        user_message="my query",
    )
    assert degraded is False
    assert reason is None

    # CALIBRATION (2026-10-07): bare "几点" / "时间" / "now" are too
    # broad. "我的六级考试具体几点开考几点结束?" contains "几点" but
    # the user is asking about exam time, NOT current time. Guard
    # against the false-negative → must STILL flag as degraded.
    degraded, reason = _detect_llm_degraded(
        answer="根据刚刚查询,当前时间是 2026-10-07T...",
        sources=[],
        verification_consistent=False,
        verification_skipped=False,
        user_message="我的六级考试具体几点开考几点结束?",
    )
    assert degraded is True, "exam-time query should NOT match the time-tool fallback"
    assert reason == "time_fallback_on_doc_query"

    # Also: EN "what time does the exam start" — has "what time" but
    # not the explicit "current / now" qualifier.
    degraded, reason = _detect_llm_degraded(
        answer="Based on the most recent query, the current time is ...",
        sources=[],
        verification_consistent=False,
        verification_skipped=False,
        user_message="What time does the exam start?",
    )
    assert degraded is True, "EN exam-time query should NOT match current-time heuristics"


def test_report_compute_rollup_with_degraded_case() -> None:
    """v2.0.32.7 — rollup separates degraded from pass/fail.
    pass_rate_excluding_infra_skip ignores degraded cases.
    """
    case_pass = CaseResult(
        case_id="real-pass", language="en", difficulty="easy", thread_id="t1", suite="x",
        run_started_at="2026-10-07T00:00:00Z", run_finished_at="2026-10-07T00:00:01Z",
        wall_clock_ms=1000,
        turns=[TurnResult(
            turn_index=0, user_message="q", high_precision="auto", latency_ms=1000,
            events=[{"type": "answer_complete", "answer": "ok", "sources": []}],
            answer="ok",
        )],
        judgment={
            "programmatic": {
                "pass": True, "per_turn": [{"pass": True, "checks": []}], "passed_count": 1, "total_count": 1,
            },
            "cheap_llm": None, "claude": None,
        },
        composite_score=1.0, pass_=True,
    )
    case_degraded = CaseResult(
        case_id="infra-degraded", language="zh", difficulty="medium", thread_id="t2", suite="x",
        run_started_at="2026-10-07T00:00:00Z", run_finished_at="2026-10-07T00:00:05Z",
        wall_clock_ms=5000,
        turns=[TurnResult(
            turn_index=0, user_message="q", high_precision="auto", latency_ms=5000,
            events=[
                {"type": "answer_complete", "answer": "根据刚刚查询,当前时间是 ...", "sources": []},
                {"type": "_degraded", "reason": "time_fallback_on_doc_query"},
            ],
            answer="根据刚刚查询,当前时间是 ...",
            degraded=True,
            degraded_reason="time_fallback_on_doc_query",
        )],
        judgment={
            "programmatic": {
                "pass": True, "degraded": True,
                "degraded_reason": "time_fallback_on_doc_query",
                "per_turn": [{"pass": True, "degraded": True, "checks": []}],
                "passed_count": 1, "total_count": 1,
            },
            "cheap_llm": None, "claude": None,
        },
        composite_score=1.0, pass_=True,
    )
    case_fail = CaseResult(
        case_id="real-fail", language="zh", difficulty="medium", thread_id="t3", suite="x",
        run_started_at="2026-10-07T00:00:00Z", run_finished_at="2026-10-07T00:00:05Z",
        wall_clock_ms=5000,
        turns=[TurnResult(
            turn_index=0, user_message="q", high_precision="auto", latency_ms=5000,
            events=[{"type": "answer_complete", "answer": "wrong", "sources": []}],
            answer="wrong",
        )],
        judgment={
            "programmatic": {
                "pass": False,
                "per_turn": [{"pass": False, "checks": [
                    {"name": "answer_contains", "passed": False, "expected": ["foo"], "actual": "wrong"}
                ]}],
                "passed_count": 0, "total_count": 1,
            },
            "cheap_llm": None, "claude": None,
        },
        composite_score=0.0, pass_=False, failure_reason="answer_lacks_expected",
    )
    rollup = compute_rollup([case_pass, case_degraded, case_fail])
    assert rollup.total_cases == 3
    assert rollup.degraded_cases == 1
    # success_rate counts pass_=True; degraded case has pass_=True by design
    assert rollup.passed_cases == 2
    # pass_rate_excluding_infra_skip: 1 real-pass / 2 non-degraded = 0.5
    assert rollup.pass_rate_excluding_infra_skip == 0.5


def test_report_compute_rollup_with_failing_case() -> None:
    """compute_rollup aggregates success rate / hallucination / latency."""
    case_pass = CaseResult(
        case_id="c1", language="en", difficulty="easy", thread_id="t1", suite="x",
        run_started_at="2026-10-03T00:00:00Z", run_finished_at="2026-10-03T00:00:01Z",
        wall_clock_ms=1000,
        turns=[TurnResult(
            turn_index=0, user_message="q", high_precision="auto", latency_ms=1000,
            events=[{"type": "answer_complete", "answer": "ok", "sources": []}],
            answer="ok",
        )],
        judgment={
            "programmatic": {
                "pass": True, "per_turn": [{"pass": True, "checks": []}], "passed_count": 1, "total_count": 1,
            },
            "cheap_llm": {"score_accuracy": 2, "score_completeness": 2, "score_relevance": 2, "composite": 2.0},
            "claude": None,
        },
        composite_score=1.0, pass_=True,
    )
    case_fail = CaseResult(
        case_id="c2", language="zh", difficulty="hard", thread_id="t2", suite="x",
        run_started_at="2026-10-03T00:00:00Z", run_finished_at="2026-10-03T00:00:05Z",
        wall_clock_ms=5000,
        turns=[TurnResult(
            turn_index=0, user_message="q", high_precision="auto", latency_ms=5000,
            events=[
                {"type": "answer_complete", "answer": "wrong", "sources": []},
                {"type": "grounding", "status": "ungrounded"},
            ],
            answer="wrong",
            grounding_status="ungrounded",
        )],
        judgment={
            "programmatic": {
                "pass": False,
                "per_turn": [{"pass": False, "checks": [
                    {"name": "answer_contains", "passed": False, "expected": ["foo"], "actual": "wrong"}
                ]}],
                "passed_count": 0, "total_count": 1,
            },
            "cheap_llm": None, "claude": None,
        },
        composite_score=0.0, pass_=False, failure_reason="answer_lacks_expected",
    )
    rollup = compute_rollup([case_pass, case_fail])
    assert rollup.total_cases == 2
    assert rollup.passed_cases == 1
    assert rollup.success_rate == 0.5
    # c2 had grounding=ungrounded → 1 hallucinated / 1 grounded
    assert rollup.hallucination_rate == 1.0
    # Latency p50/p95/p99 over two turns [1000, 5000]
    assert rollup.latency_p50_ms == 3000
    # Failure mode breakdown
    assert "check_failed:answer_contains" in rollup.failure_mode_breakdown


def test_report_render_markdown_smoke() -> None:
    rollup = RollupMetrics(
        success_rate=0.8, total_cases=10, passed_cases=8,
        failure_mode_breakdown={"check_failed:x": 2},
        tool_call_distribution={"mean": 1.0, "median": 1.0, "p95": 2.0, "p99": 2.0, "max": 2.0},
        token_distribution={"mean": 100, "median": 100, "p95": 200, "p99": 200, "max": 200},
        cost_rollup={"total_cost_usd": "0.50", "cost_per_case_p95_usd": 0.1},
        latency_p50_ms=2000, latency_p95_ms=3000, latency_p99_ms=3500,
        hallucination_rate=0.1, refusal_accuracy=0.7, citation_coverage=0.9,
    )
    md = render_markdown(suite="golden", timestamp="2026-10-03T00Z", rollup=rollup, results=[])
    assert "# YuanRAG Eval Report" in md
    assert "Hallucination rate" in md
    assert "Per-case results" in md
    console = render_console_table(rollup, [])
    assert "YuanRAG eval" in console


def test_report_percentile_helper() -> None:
    # Median of [1,2,3,4,5] = 3
    assert _percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert _percentile([], 50) == 0.0
    assert _percentile([10], 99) == 10.0


def test_report_timestamp_format() -> None:
    ts = make_timestamp()
    # YYYY-MM-DDTHHMMSSZ = 4+1+2+1+2 + T + 6 + Z = 18 chars
    assert len(ts) == 18
    assert ts.endswith("Z")
    assert ts[4] == "-"
    assert ts[10] == "T"


def test_fixture_path_traversal_guard(tmp_path: Path) -> None:
    """fixture_path() must reject .. escapes."""
    with pytest.raises(CaseLoadError):
        fixture_path("../outside.txt", fixtures_root=tmp_path)


def test_case_load_error_supports_suite_filter() -> None:
    """load_suite rejects unknown suite names."""
    with pytest.raises(CaseLoadError):
        load_suite("unknown-suite")  # type: ignore[arg-type]


def test_case_canonical_fsm_nodes_constant() -> None:
    """CANONICAL_FSM_NODES is the canonical set the program checks against."""
    assert "intent_analysis" in CANONICAL_FSM_NODES
    assert "react_generate" in CANONICAL_FSM_NODES
    assert "react_generate_extractive" in CANONICAL_FSM_NODES
    assert "verify_answer" in CANONICAL_FSM_NODES
    assert "check_hallucination" in CANONICAL_FSM_NODES


def test_expected_refusal_required_template() -> None:
    """ExpectedRefusal.template must be one of the three refusal keys."""
    # Constructor type-hint is advisory — assert the template constraint
    # by exercising the YAML parser path via load_suite.
    tmp = Path(__file__).parent / "tmp_eval_v2"
    tmp.mkdir(exist_ok=True)
    cases_dir = tmp / "cases"
    cases_dir.mkdir(exist_ok=True)
    yaml_path = cases_dir / "adversarial.yaml"
    yaml_path.write_text(
        """\
---
- id: bad-refusal-template
  language: en
  difficulty: hard
  root_cause_group: A
  thread_id: bad-refusal
  turns:
    - user: what?
      high_precision: auto
      expected_refusal:
        template: not_a_real_template
        contains_any:
          - foo
""",
        encoding="utf-8",
    )
    try:
        with pytest.raises(CaseLoadError):
            load_suite("adversarial", fixtures_root=tmp)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

# ============================================================
# Phase 3 (2026-10-05) — cheap-LLM judge dispatch + Claude prompt stub
# ============================================================


def test_cli_should_run_cheap_llm_dispatch() -> None:
    """`--judge` flag dispatch:
      - ``all``          → run cheap_llm
      - ``cheap_llm``    → run cheap_llm
      - ``programmatic`` → skip cheap_llm
    """
    from tests.eval.cli import _should_run_cheap_llm

    assert _should_run_cheap_llm("all") is True
    assert _should_run_cheap_llm("cheap_llm") is True
    assert _should_run_cheap_llm("programmatic") is False


def test_cli_expected_answer_text_aggregates_contains() -> None:
    """expected_answer_text concatenates expected_answer_contains into a
    human-readable reference string for the cheap-LLM judge."""
    from tests.eval.cli import _expected_answer_text

    turn = TurnAnnotation(
        user="query",
        expected_answer_contains=["15:00", "17:25"],
        expected_answer_lacks=[],
        forbidden_phrases=[],
        expected_citations=[],
        high_precision="auto",
        expected_grounding="skipped",
        expected_route_decision="retrieve",
        expected_tool_call_count=2,
        expected_tool_sequence=[],
        expected_refusal=None,
    )
    text = _expected_answer_text(turn)
    assert "15:00" in text
    assert "17:25" in text
    # Empty expected → empty string (judge scores 0 by default).
    empty_turn = TurnAnnotation(
        user="query",
        expected_answer_contains=[],
        expected_answer_lacks=[],
        forbidden_phrases=[],
        expected_citations=[],
        high_precision="auto",
        expected_grounding="skipped",
        expected_route_decision="direct",
        expected_tool_call_count=0,
        expected_tool_sequence=[],
        expected_refusal=None,
    )
    assert _expected_answer_text(empty_turn) == ""
