"""Async harness runner — drives YuanRAG via POST /chat SSE.

v2.0.32.0 — drives the system under test through the HTTP layer
(``POST /chat`` SSE endpoint), NOT via direct ``stream_agent`` import.
Reason: the wire-format / citation-renumber / i18n-catalog / upload /
clear-thread code paths are the bug surface that the eval harness wants
to cover, and only the HTTP layer exercises all of them.

Per-case workflow:

    1.  Setup thread_id = "eval-<case.thread_id>-<uuid8>"
    2.  Upload corpus docs via POST /documents/upload (multipart form)
    3.  Poll GET /documents?thread_id=... to status="indexed"
    4.  For each turn:
            POST /chat → drain SSE data: lines → record events
            TokenCapture.sum() → /chat
        Run all 3 judge passes on each turn's events.
    5.  Cleanup: POST /documents/clear-thread/{tid} + DELETE /chat/{tid}
    6.  Return CaseResult with all per-turn results + judgments

The runner is concurrency=1 by default. ``asyncio.Semaphore(N)`` wrapper
is left to future PRs — Phase 1 keeps things sequential and deterministic
so timing / latency rollups are reproducible.

Failure modes:
    - HTTP 4xx from /chat → case.failure_reason = "http_error"
    - LLM mid-stream exception → "llm_error"
    - Timeout (default 120s) → "timeout"
    - Document upload failed / never indexed → "upload_failed"
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import httpx

from tests.eval.cases import (
    Case,
    TurnAnnotation,
    fixture_path,
)
from tests.eval.cost import (
    PRICING_USD_PER_1K,
    TokenSnapshot,
    estimate_cost,
)
from tests.eval.tokens import TokenCapture


SCHEMA_VERSION = "1.0"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_BASE_URL = "http://127.0.0.1:8765"


# ============================================================
# Result dataclasses
# ============================================================


@dataclass
class TurnResult:
    turn_index: int
    user_message: str
    high_precision: str
    latency_ms: int
    events: list[dict] = field(default_factory=list)
    tool_call_count: int = 0
    tool_call_durations_ms: list[int] = field(default_factory=list)
    tool_call_names: list[str] = field(default_factory=list)
    any_tool_failed: bool = False
    tokens_by_model: dict[tuple[str, str], TokenSnapshot] = field(default_factory=dict)
    cost_usd: Decimal = Decimal(0)
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    grounding_status: Optional[str] = None
    verification_consistent: Optional[bool] = None
    verification_skipped: bool = True
    failure_reason: Optional[str] = None
    # v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07) — flag for LLM
    # degradation not from eval harness (upload/HTTP/SSE) but from
    # backend regen-loop fallback. Set when the answer_complete
    # event arrived cleanly but the body looks like a defensive
    # override fallback (e.g. "current time" answer to a doc
    # query, or refusal template for a verbatim case). Lets the
    # rollup distinguish quota/regen-degraded failures from real
    # logic bugs when triaging eval results.
    degraded: bool = False
    degraded_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "high_precision": self.high_precision,
            "latency_ms": self.latency_ms,
            "events": self.events,
            "tool_call_count": self.tool_call_count,
            "tool_call_durations_ms": self.tool_call_durations_ms,
            "tool_call_names": self.tool_call_names,
            "any_tool_failed": self.any_tool_failed,
            "tokens": {
                f"{provider}/{model}": {
                    "input": snap.input_tokens,
                    "output": snap.output_tokens,
                }
                for (provider, model), snap in self.tokens_by_model.items()
            },
            "cost_usd": str(self.cost_usd),
            "answer": self.answer,
            "sources": self.sources,
            "grounding_status": self.grounding_status,
            "verification_consistent": self.verification_consistent,
            "verification_skipped": self.verification_skipped,
            "failure_reason": self.failure_reason,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }


@dataclass
class CaseResult:
    case_id: str
    language: str
    difficulty: str
    thread_id: str
    suite: str
    run_started_at: str
    run_finished_at: str
    wall_clock_ms: int
    turns: list[TurnResult] = field(default_factory=list)
    judgment: dict[str, Any] = field(default_factory=dict)
    composite_score: float = 0.0
    pass_: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "case_id": self.case_id,
            "language": self.language,
            "difficulty": self.difficulty,
            "thread_id": self.thread_id,
            "suite": self.suite,
            "run_started_at": self.run_started_at,
            "run_finished_at": self.run_finished_at,
            "wall_clock_ms": self.wall_clock_ms,
            "turns": [t.to_dict() for t in self.turns],
            "judgment": self.judgment,
            "composite_score": self.composite_score,
            "pass": self.pass_,
            "failure_reason": self.failure_reason,
        }


# ============================================================
# Runner
# ============================================================


class AsyncHarnessRunner:
    """Drive one or more Cases via POST /chat SSE."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_S,
        provider: str = "anthropic",
        model: str = "MiniMax-M3",
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._provider = provider
        self._model = model
        self._owns_client = client is None
        self._client = client

    def __enter__(self) -> "AsyncHarnessRunner":
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------

    def run_case(self, case: Case) -> CaseResult:
        """Synchronous wrapper around the async runner — for CLI/tests."""
        return asyncio.run(self._run_case_async(case))

    async def run_case_async(self, case: Case) -> CaseResult:
        """Async variant for embedding in larger asyncio flows."""
        return await self._run_case_async(case)

    async def _run_case_async(self, case: Case) -> CaseResult:
        if self._client is None:
            # Allow bare `await runner.run_case_async(case)` without context.
            self._client = httpx.Client(timeout=self._timeout)
            self._owns_client = True

        case_thread_id = f"eval-{case.thread_id}-{uuid.uuid4().hex[:8]}"
        run_started = _now_iso()
        wall_start = time.monotonic_ns()

        capture = TokenCapture.install()
        try:
            upload_error = await self._upload_corpus(case, case_thread_id)
            if upload_error:
                return self._fail_case(
                    case,
                    case_thread_id,
                    run_started,
                    wall_start,
                    failure_reason="upload_failed",
                    detail=upload_error,
                )

            turn_results: list[TurnResult] = []
            for i, turn in enumerate(case.turns):
                try:
                    res = await self._drive_one_turn(case, turn, i, case_thread_id)
                except Exception as exc:
                    res = TurnResult(
                        turn_index=i,
                        user_message=turn.user,
                        high_precision=turn.high_precision,
                        latency_ms=0,
                        failure_reason=f"runner_error:{type(exc).__name__}",
                    )
                turn_results.append(res)

            await self._cleanup_thread(case_thread_id)
            wall_clock_ms = int((time.monotonic_ns() - wall_start) / 1_000_000)

            # Roll up judgments
            judgment = self._aggregate_judgments(case, turn_results)
            composite_score = self._compute_composite(case, turn_results, judgment)
            pass_ = composite_score >= case.scoring_weights.threshold_pass
            return CaseResult(
                case_id=case.id,
                language=case.language,
                difficulty=case.difficulty,
                thread_id=case_thread_id,
                suite=case.suite,
                run_started_at=run_started,
                run_finished_at=_now_iso(),
                wall_clock_ms=wall_clock_ms,
                turns=turn_results,
                judgment=judgment,
                composite_score=composite_score,
                pass_=pass_,
            )
        finally:
            capture.uninstall()

    # ------------------------------------------------------------
    # Upload / poll / cleanup
    # ------------------------------------------------------------

    async def _upload_corpus(self, case: Case, thread_id: str) -> Optional[str]:
        if not case.fixture_docs:
            return None
        for name in case.fixture_docs:
            try:
                p = fixture_path(name)
            except Exception as exc:
                return f"fixture_resolve:{type(exc).__name__}:{exc}"
            if not p.exists():
                return f"fixture_missing:{name}"
            try:
                with p.open("rb") as f:
                    resp = self._client.post(
                        f"{self._base_url}/documents/upload",
                        files={"file": (p.name, f, "application/octet-stream")},
                        data={"thread_id": thread_id},
                    )
            except Exception as exc:
                return f"upload_http:{type(exc).__name__}:{exc}"
            if resp.status_code >= 400:
                return f"upload_status:{resp.status_code}"
        # Poll for indexed status — default timeout 120s.
        # Eval Stage 5.5 (2026-10-05) bumped from 30s → 120s after
        # observing docling on Python.md (high-density code, ~50KB)
        # routinely takes 35-60s on BGE-M3 CPU. 30s caused 3/15
        # upload_failed in golden.yaml (golden-008-zh/en, 009-zh).
        # 120s gives ~2x headroom for the heaviest corpus files
        # without blocking the runner on actual failures.
        return await self._poll_documents_indexed(thread_id, timeout_s=120.0)

    async def _poll_documents_indexed(
        self, thread_id: str, *, timeout_s: float
    ) -> Optional[str]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                resp = self._client.get(
                    f"{self._base_url}/documents",
                    params={"thread_id": thread_id},
                )
            except Exception as exc:
                return f"poll_http:{type(exc).__name__}:{exc}"
            if resp.status_code >= 400:
                return f"poll_status:{resp.status_code}"
            try:
                docs = resp.json()
            except Exception:
                return "poll_json_parse"
            if not isinstance(docs, list):
                return "poll_shape"
            statuses = [d.get("status") for d in docs if isinstance(d, dict)]
            if not statuses:
                # Empty doc list — nothing to wait for; this can happen in
                # tests where the fake backend returns [].
                return None
            if statuses and all(s == "indexed" for s in statuses):
                return None
            if any(s == "failed" for s in statuses):
                return f"ingest_failed:{statuses}"
            await asyncio.sleep(0.5)
        return "ingest_timeout"

    async def _cleanup_thread(self, thread_id: str) -> None:
        # Best-effort: errors here must NOT fail the case.
        try:
            self._client.post(
                f"{self._base_url}/documents/clear-thread/{thread_id}"
            )
        except Exception:
            pass
        try:
            self._client.delete(f"{self._base_url}/chat/{thread_id}")
        except Exception:
            pass

    # ------------------------------------------------------------
    # Drive one turn
    # ------------------------------------------------------------

    async def _drive_one_turn(
        self,
        case: Case,
        turn: TurnAnnotation,
        turn_idx: int,
        thread_id: str,
    ) -> TurnResult:
        # Send Accept-Language matching case.language. Per src/api/i18n.py
        # default is "zh"; explicit header enforces the case contract.
        headers = {"Accept-Language": case.language}
        payload = {
            "thread_id": thread_id,
            "message": turn.user,
            "stream": True,
            "high_precision": turn.high_precision,
        }

        # Stream POST /chat and parse SSE data: lines. We use httpx's
        # stream() context manager so we can drain cleanly even if the
        # client times out mid-stream.
        latency_start = time.monotonic_ns()
        events: list[dict] = []
        failure_reason: Optional[str] = None
        try:
            with self._client.stream(
                "POST", f"{self._base_url}/chat", json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    failure_reason = f"http_error:{resp.status_code}"
                else:
                    buffer: list[str] = []
                    for line in resp.iter_lines():
                        if line is None:
                            continue
                        if line.startswith("data: "):
                            buffer.append(line[len("data: "):])
                        elif line == "":
                            # End of one SSE event
                            data = "\n".join(buffer).strip()
                            buffer.clear()
                            if not data:
                                continue
                            try:
                                ev = json.loads(data)
                            except Exception:
                                continue
                            events.append(ev)
                            if ev.get("type") == "done":
                                break
        except httpx.ReadTimeout:
            failure_reason = "timeout"
        except Exception as exc:
            failure_reason = f"http_exception:{type(exc).__name__}"
        latency_ms = int((time.monotonic_ns() - latency_start) / 1_000_000)

        # Snapshot token usage since this turn started.
        tokens_by_model = _snapshot_tokens()

        cost_usd = Decimal(0)
        for (provider, model_name), ts in tokens_by_model.items():
            cost_usd += estimate_cost(ts, provider=provider, model=model_name)

        # Compute turn-level rollups from events.
        tool_starts = [e for e in events if e.get("type") == "tool_call_start"]
        tool_ends = {e.get("tool_call_id"): e for e in events if e.get("type") == "tool_call_end"}
        tool_call_count = len(tool_starts)
        tool_call_durations_ms: list[int] = []
        tool_call_names: list[str] = []
        any_tool_failed = False
        for ts in tool_starts:
            tcid = ts.get("tool_call_id")
            te = tool_ends.get(tcid) if tcid else None
            if te and isinstance(te.get("elapsed_ms"), int):
                tool_call_durations_ms.append(te["elapsed_ms"])
            if ts.get("name"):
                tool_call_names.append(ts["name"])
            if te and te.get("ok") is False:
                any_tool_failed = True

        answer = ""
        sources: list[dict] = []
        grounding_status: Optional[str] = None
        verification_consistent: Optional[bool] = None
        verification_skipped = True
        for ev in events:
            if ev.get("type") == "answer_complete":
                ans = ev.get("answer")
                if isinstance(ans, str):
                    answer = ans
                srcs = ev.get("sources")
                if isinstance(srcs, list):
                    sources = srcs
            elif ev.get("type") == "grounding":
                grounding_status = ev.get("status")
            elif ev.get("type") == "verification_result":
                verification_skipped = bool(ev.get("skipped", False))
                verification_consistent = ev.get("consistent")

        # v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07) — surface the
        # backend-degraded flag on the wire so the programmatic
        # judge can skip answer-text checks (they're guaranteed to
        # fail when the LLM emitted a fallback template). Adds a
        # synthetic ``{"type": "_degraded", ...}`` event that the
        # judge recognizes without changing the public wire schema.
        # (Real wire events are NOT polluted — we only inject this
        # event into the events-list we hand to the judge.)
        #
        # Also do the degraded detection HERE (one place, where
        # all the per-turn facts are visible) so the JSONL carries
        # the flag for the rollup.

        # v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07) — detect
        # backend LLM-degraded fallback. When the eval runner gets a
        # clean ``done`` event but the answer body looks like a
        # defensive-override last-ditch fallback (e.g. "current time"
        # as the answer to a PDF-content question), the YAML failure
        # is from quota/regen-loop degradation, NOT from a real
        # YuanRAG logic bug. Triaging this lets us split "code bug"
        # from "external infra flake" without rerunning.
        #
        # Heuristics (conservative; new patterns go here when seen):
        #   - answer starts with "根据刚刚查询" or contains "当前时间"
        #     for non-time queries → defensive_override LLM-fallback
        degraded, degraded_reason = _detect_llm_degraded(
            answer=answer,
            sources=sources,
            verification_consistent=verification_consistent,
            verification_skipped=verification_skipped,
            user_message=turn.user,
        )

        return TurnResult(
            turn_index=turn_idx,
            user_message=turn.user,
            high_precision=turn.high_precision,
            latency_ms=latency_ms,
            events=events,
            tool_call_count=tool_call_count,
            tool_call_durations_ms=tool_call_durations_ms,
            tool_call_names=tool_call_names,
            any_tool_failed=any_tool_failed,
            tokens_by_model=tokens_by_model,
            cost_usd=cost_usd,
            answer=answer,
            sources=sources,
            grounding_status=grounding_status,
            verification_consistent=verification_consistent,
            verification_skipped=verification_skipped,
            failure_reason=failure_reason,
            degraded=degraded,
            degraded_reason=degraded_reason,
        )

    # ------------------------------------------------------------
    # Judgments + composite
    # ------------------------------------------------------------

    def _aggregate_judgments(
        self,
        case: Case,
        turns: list[TurnResult],
    ) -> dict[str, Any]:
        """Run programmatic + cheap-LLM judges per turn.

        Pass 3 (Claude) is left null; operator fills it via
        tests.eval.judge.claudemd in a separate Claude Code session.
        """
        # Lazy import — programmatic imports nothing heavy.
        from tests.eval.judge.programmatic import run_programmatic

        programmatic_per_turn: list[dict] = []
        programmatic_pass_count = 0
        for turn_res, turn_expected in zip(turns, case.turns):
            # v2.0.32.7 — append a synthetic degraded-flag event so
            # the programmatic judge knows to skip answer-text
            # checks on a backend-degraded fallback. Real wire
            # events are NOT polluted; this only mutates the
            # per-turn events list passed to the judge. The original
            # turn_res.events is left intact for the JSONL.
            judge_events = list(turn_res.events)
            if turn_res.degraded:
                judge_events.append({
                    "type": "_degraded",
                    "reason": turn_res.degraded_reason or "unspecified",
                })
            judgment = run_programmatic(turn=turn_expected, events=judge_events)
            # v2.0.32.7 — annotate the per-turn judgment with the
            # degraded flag so the JSONL carries the bucket info.
            j_dict = judgment.to_dict()
            j_dict["degraded"] = turn_res.degraded
            j_dict["degraded_reason"] = turn_res.degraded_reason
            programmatic_per_turn.append(j_dict)
            if judgment.passed:
                programmatic_pass_count += 1

        programmatic_overall = (
            programmatic_pass_count == len(case.turns) if case.turns else False
        )

        return {
            "programmatic": {
                "pass": programmatic_overall,
                "per_turn": programmatic_per_turn,
                "passed_count": programmatic_pass_count,
                "total_count": len(case.turns),
            },
            # Pass 2 is run asynchronously via the runner if --judge cheap_llm
            # is enabled; the CLI does this and updates the CaseResult in
            # place. We leave it null here for now.
            "cheap_llm": None,
            "claude": None,
        }

    def _compute_composite(
        self,
        case: Case,
        turns: list[TurnResult],
        judgment: dict[str, Any],
    ) -> float:
        """Composite score = programmatic_pass (0/1) * w_p +
              cheap_llm composite (0..2) / 2 * w_c."""
        w = case.scoring_weights
        prog = 1.0 if judgment["programmatic"]["pass"] else 0.0
        cheap = judgment.get("cheap_llm")
        if cheap is None:
            # Cheap-LLM skipped (e.g. CLI ran --judge programmatic only)
            # → composite = programmatic only (normalized to 0..1)
            return prog
        cheap_composite_norm = cheap["composite"] / 2.0
        return prog * w.programmatic_pass + cheap_composite_norm * w.cheap_llm_pass

    # ------------------------------------------------------------
    # Failure helper
    # ------------------------------------------------------------

    def _fail_case(
        self,
        case: Case,
        thread_id: str,
        run_started: str,
        wall_start: int,
        *,
        failure_reason: str,
        detail: str = "",
    ) -> CaseResult:
        wall_clock_ms = int((time.monotonic_ns() - wall_start) / 1_000_000)
        return CaseResult(
            case_id=case.id,
            language=case.language,
            difficulty=case.difficulty,
            thread_id=thread_id,
            suite=case.suite,
            run_started_at=run_started,
            run_finished_at=_now_iso(),
            wall_clock_ms=wall_clock_ms,
            turns=[],
            judgment={
                "programmatic": {"pass": False, "per_turn": [], "passed_count": 0, "total_count": 0},
                "cheap_llm": None,
                "claude": None,
            },
            composite_score=0.0,
            pass_=False,
            failure_reason=failure_reason,
        )


# ============================================================
# Helpers
# ============================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07) — see call site in
# ``_drive_one_turn``. Conservative pattern list; add new patterns
# here when a previously-undetected backend-degradation fingerprint
# is observed. Goal is to BIN UP an "infrastructure degraded" flag,
# not to negate a real logic-bug pass — false negatives (missed
# degradation) only defers triage, false positives (real bug misread
# as degraded) is the worse direction; conservative = only flag
# when the answer text is clearly a fallback template.
_DEGRADED_ANSWER_PREFIXES = (
    # Phase 6 verify_answer last-ditch fallback: defensive override
    # noticed the LLM emitted text but did not incorporate the TM data
    # (e.g. user asked about an uploaded PDF, LLM answered "what's the
    # time" — the time-tool result is what TM captured). The fingerprint
    # is "根据刚刚查询,当前时间是" which only fires in that path.
    "根据刚刚查询",
    # English mirror of the above.
    "Based on the most recent query",
    "Per the recent query",
)


def _detect_llm_degraded(
    *,
    answer: str,
    sources: list[dict],
    verification_consistent: Optional[bool],
    verification_skipped: bool,
    user_message: str,
) -> tuple[bool, Optional[str]]:
    """Best-effort detector for backend-degraded fallback answers.

    Returns ``(degraded, reason)``. ``reason`` is a short tag like
    ``"time_fallback_on_doc_query"`` for the rollup's per-bucket
    counter; ``None`` when the answer looks normal.

    Heuristics (each must satisfy BOTH ``answer`` pattern AND the
    semantic shape — e.g. the query must not actually be about time):

    1. **Defensive-override time fallback** — answer starts with
       ``"根据刚刚查询"`` and the user query is NOT asking for the
       current time / date / now. Strong signal that the LLM
       synthesized a non-answer.
    """
    if not answer:
        return False, None

    answer_stripped = answer.strip()
    user_lower = user_message.lower()

    # Heuristic 1: time-tool fallback on a non-time query.
    # Pattern: "根据刚刚查询, 当前时间是 ..." but user wasn't
    # asking for time. This is the dominant fallback fingerprint
    # observed in Stage 5.8 (golden-002/005/006).
    #
    # CALIBRATION (2026-10-07): "几点" / "时间" / "now" are too
    # broad — "我的六级考试具体几点开考几点结束?" matches "几点"
    # but is asking about exam time, not current time. Conservative:
    # only flag when the query is CLEARLY asking for the current /
    # real time / date, not for some other timestamp.
    for prefix in _DEGRADED_ANSWER_PREFIXES:
        if answer_stripped.startswith(prefix):
            time_marker = (
                "当前时间" in answer_stripped
                or "current time" in answer_stripped.lower()
            )
            if not time_marker:
                continue
            user_asks_time = any(
                kw in user_lower
                for kw in (
                    # CN: explicit "now" question (not "几点开考")
                    "现在几点",
                    "现在几点了",
                    "现在北京时间",
                    "现在时间",
                    "现在是几点",
                    "现在几点钟",
                    "现在日期",
                    "今天日期",
                    "今天是几号",
                    "今天星期几",
                    # EN: explicit "now / current" question
                    "current time",
                    "what time is it",
                    "what's the time",
                    "what is the current time",
                    "what time now",
                    "what is today's date",
                    "today's date",
                    "what day is it",
                )
            )
            if not user_asks_time:
                return True, "time_fallback_on_doc_query"
            return False, None

    return False, None


# Make TokenCapture._ACTIVE accessible from inside this module without
# forcing tests to expose it through the public API.
from tests.eval import tokens as _tokens_mod  # noqa: E402


def _snapshot_tokens() -> dict[tuple[str, str], TokenSnapshot]:
    """Read tokens from the currently-installed capture, if any.

    Imports the tokens module's private _ACTIVE pointer rather than
    monkey-patching the class — cleaner separation. Returns empty dict
    if no capture is active.
    """
    active = _tokens_mod._ACTIVE  # type: ignore[attr-defined]
    if active is None:
        return {}
    try:
        return active.snapshot_and_reset()
    except Exception:
        return {}