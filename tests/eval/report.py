"""Report generation: console table + Markdown + JSONL + rollup JSON.

v2.0.32.0 — three output formats per the plan §E:
    1. Console (rich-free, plain ASCII — GBK-safe on Windows per
       existing tests/e2e/ convention)
    2. Markdown report (tests/eval/reports/<timestamp>.md)
    3. JSONL per-case (tests/eval/reports/<timestamp>.jsonl)
    4. Rollup JSON (tests/eval/reports/<timestamp>.rollup.json)

The Markdown + JSONL writers are pure stdlib — no rich / pandas / etc.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

from tests.eval.runner import CaseResult, TurnResult


# ============================================================
# Rollup metrics
# ============================================================


@dataclass
class RollupMetrics:
    success_rate: float
    total_cases: int
    passed_cases: int
    failure_mode_breakdown: dict[str, int]
    tool_call_distribution: dict[str, float]
    token_distribution: dict[str, float]
    cost_rollup: dict[str, Any]
    latency_p50_ms: int
    latency_p95_ms: int
    latency_p99_ms: int
    hallucination_rate: float
    refusal_accuracy: Optional[float]
    citation_coverage: Optional[float]
    # v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07) — backend-degraded
    # bucket. ``degraded_cases`` = cases where the runner detected
    # Phase 6 regen-loop defensive-override fallback (time-tool
    # answer on a doc query, etc.). These are NOT real logic bugs;
    # they're infrastructure degradation (quota / LLM hiccup / regen
    # loop). ``pass_rate_excluding_infra_skip`` is the success rate
    # computed over the NON-degraded subset, so operators can read the
    # "real" pass rate at a glance.
    degraded_cases: int = 0
    pass_rate_excluding_infra_skip: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failure_mode_breakdown": self.failure_mode_breakdown,
            "tool_call_distribution": self.tool_call_distribution,
            "token_distribution": self.token_distribution,
            "cost_rollup": self.cost_rollup,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "hallucination_rate": self.hallucination_rate,
            "refusal_accuracy": self.refusal_accuracy,
            "citation_coverage": self.citation_coverage,
            "degraded_cases": self.degraded_cases,
            "pass_rate_excluding_infra_skip": self.pass_rate_excluding_infra_skip,
        }


def compute_rollup(results: list[CaseResult]) -> RollupMetrics:
    """Aggregate per-run rollups."""
    total = len(results)
    passed = sum(1 for r in results if r.pass_)
    success_rate = passed / total if total else 0.0

    # v2.0.32.7 (Stage 5.8 follow-up, 2026-10-07) — backend-degraded
    # bucket. Read per-turn `degraded` flags (set by the runner's
    # `_detect_llm_degraded` helper + surfaced by programmatic judge).
    # A case counts as degraded if ANY of its turns is degraded. Compute
    # pass_rate_excluding_infra_skip over the non-degraded subset so
    # operators see the "real" pass rate without infrastructure noise.
    degraded_count = 0
    for r in results:
        if any(t.degraded for t in r.turns):
            degraded_count += 1
    non_degraded = [
        r for r in results if not any(t.degraded for t in r.turns)
    ]
    if non_degraded:
        pass_rate_excl = sum(1 for r in non_degraded if r.pass_) / len(non_degraded)
    else:
        pass_rate_excl = None

    # Failure mode breakdown — by failure_reason + per-turn programmatic failure
    failure_modes: Counter[str] = Counter()
    for r in results:
        if r.failure_reason:
            failure_modes[r.failure_reason] += 1
        # Per-turn programmatic failures
        prog = r.judgment.get("programmatic", {})
        per_turn = prog.get("per_turn", [])
        for i, pt in enumerate(per_turn):
            if not pt.get("pass"):
                # Pick the first failing check as the mode tag
                for chk in pt.get("checks", []):
                    if not chk.get("pass"):
                        failure_modes[f"check_failed:{chk['name']}"] += 1
                        break

    # Tool-call distribution across all turns
    tool_counts = [t.tool_call_count for r in results for t in r.turns]
    tool_dist = _distribution_5(tool_counts)

    # Token distribution (sum input + output per turn)
    token_counts = [
        sum(s.input_tokens + s.output_tokens for s in t.tokens_by_model.values())
        for r in results for t in r.turns
    ]
    token_dist = _distribution_5(token_counts)

    # Cost rollup
    total_cost = sum(
        (t.cost_usd for r in results for t in r.turns), Decimal(0)
    )
    per_case_costs = [
        float(sum((t.cost_usd for t in r.turns), Decimal(0)))
        for r in results
    ]
    cost_rollup = {
        "total_cost_usd": str(total_cost),
        "cost_per_case_p95_usd": _percentile(per_case_costs, 95) if per_case_costs else 0.0,
    }

    # Latency per turn
    latencies = [t.latency_ms for r in results for t in r.turns]
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)

    # Hallucination rate — count ungrounded OR (grounded AND cheap-llm
    # accuracy=0). Cases without grounding event are excluded.
    # Phase 3 (2026-10-05): cheap_llm wire shape is ``{pass, per_turn[], composite_mean}``
    # (per-turn judgments, not a single flat judgment). Read per_turn and
    # treat any turn with accuracy=0 as fabrication.
    grounded_count = 0
    hallucinated_count = 0
    for r in results:
        for t_idx, t in enumerate(r.turns):
            if t.grounding_status is None:
                continue
            grounded_count += 1
            if t.grounding_status == "ungrounded":
                hallucinated_count += 1
            elif r.judgment.get("cheap_llm"):
                cheap = r.judgment["cheap_llm"]
                per_turn = cheap.get("per_turn") or []
                # Match per-turn index; if absent, fall back to first turn.
                pt = per_turn[t_idx] if t_idx < len(per_turn) else (per_turn[0] if per_turn else None)
                if pt and pt.get("score_accuracy", 2) == 0:
                    hallucinated_count += 1
    hallucination_rate = hallucinated_count / grounded_count if grounded_count else 0.0

    # Refusal accuracy — adversarial cases expecting a refusal
    refusal_expected = 0
    refusal_correct = 0
    for r in results:
        # Look at programmatic per-turn refusal check
        prog = r.judgment.get("programmatic", {})
        for pt in prog.get("per_turn", []):
            for chk in pt.get("checks", []):
                if chk["name"] == "refusal":
                    expected = chk.get("expected")
                    if expected is not None:
                        refusal_expected += 1
                        if chk.get("pass"):
                            refusal_correct += 1
    refusal_accuracy = (
        refusal_correct / refusal_expected if refusal_expected else None
    )

    # Citation coverage — cases with expected_citations, % where all expected ⊆ actual
    cite_expected = 0
    cite_correct = 0
    for r in results:
        prog = r.judgment.get("programmatic", {})
        for pt in prog.get("per_turn", []):
            for chk in pt.get("checks", []):
                if chk["name"] == "expected_citations":
                    expected = chk.get("expected", [])
                    if expected:
                        cite_expected += 1
                        if chk.get("pass"):
                            cite_correct += 1
    citation_coverage = (
        cite_correct / cite_expected if cite_expected else None
    )

    return RollupMetrics(
        success_rate=success_rate,
        total_cases=total,
        passed_cases=passed,
        failure_mode_breakdown=dict(failure_modes),
        tool_call_distribution=tool_dist,
        token_distribution=token_dist,
        cost_rollup=cost_rollup,
        latency_p50_ms=int(p50),
        latency_p95_ms=int(p95),
        latency_p99_ms=int(p99),
        hallucination_rate=hallucination_rate,
        refusal_accuracy=refusal_accuracy,
        citation_coverage=citation_coverage,
        degraded_cases=degraded_count,
        pass_rate_excluding_infra_skip=pass_rate_excl,
    )


# ============================================================
# Writers
# ============================================================


def write_jsonl(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def write_rollup_json(path: Path, rollup: RollupMetrics, *, suite: str, timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "timestamp": timestamp,
        "suite": suite,
        "metrics": rollup.to_dict(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, *, suite: str, timestamp: str, rollup: RollupMetrics, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(suite=suite, timestamp=timestamp, rollup=rollup, results=results)
    path.write_text(md, encoding="utf-8")


def render_markdown(*, suite: str, timestamp: str, rollup: RollupMetrics, results: list[CaseResult]) -> str:
    """Markdown body — header / rollup table / per-case table / worst-N."""
    lines: list[str] = []
    lines.append(f"# YuanRAG Eval Report — {suite} suite")
    lines.append("")
    lines.append(f"**Timestamp:** {timestamp}  ")
    lines.append(f"**Total cases:** {rollup.total_cases}  ")
    lines.append(f"**Passed:** {rollup.passed_cases}  ")
    lines.append(f"**Success rate:** {rollup.success_rate:.1%}  ")
    if rollup.degraded_cases:
        lines.append(f"**Backend-degraded cases:** {rollup.degraded_cases}  ")
    if rollup.pass_rate_excluding_infra_skip is not None:
        lines.append(
            f"**Pass rate (excl. infra-degraded):** {rollup.pass_rate_excluding_infra_skip:.1%}  "
        )
    lines.append("")

    # Rollup metrics table
    lines.append("## Rollup metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Hallucination rate | {rollup.hallucination_rate:.1%} |")
    lines.append(f"| Latency p50 | {rollup.latency_p50_ms} ms |")
    lines.append(f"| Latency p95 | {rollup.latency_p95_ms} ms |")
    lines.append(f"| Latency p99 | {rollup.latency_p99_ms} ms |")
    if rollup.refusal_accuracy is not None:
        lines.append(f"| Refusal accuracy (adversarial) | {rollup.refusal_accuracy:.1%} |")
    if rollup.citation_coverage is not None:
        lines.append(f"| Citation coverage | {rollup.citation_coverage:.1%} |")
    lines.append(f"| Total cost | ${rollup.cost_rollup['total_cost_usd']} |")
    lines.append(f"| Cost / case p95 | ${rollup.cost_rollup['cost_per_case_p95_usd']:.4f} |")
    if rollup.tool_call_distribution:
        lines.append(f"| Tool calls / p95 | {rollup.tool_call_distribution.get('p95', 0)} |")
    if rollup.token_distribution:
        lines.append(f"| Tokens / p95 | {rollup.token_distribution.get('p95', 0)} |")
    if rollup.pass_rate_excluding_infra_skip is not None:
        lines.append(
            f"| Pass rate (excl. infra-degraded) | {rollup.pass_rate_excluding_infra_skip:.1%} |"
        )
    if rollup.degraded_cases:
        lines.append(f"| Backend-degraded cases | {rollup.degraded_cases} |")
    lines.append("")

    # Failure mode breakdown
    if rollup.failure_mode_breakdown:
        lines.append("## Failure mode breakdown")
        lines.append("")
        lines.append("| Mode | Count |")
        lines.append("|---|---|")
        for mode, count in sorted(rollup.failure_mode_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"| {mode} | {count} |")
        lines.append("")

    # Per-case table
    lines.append("## Per-case results")
    lines.append("")
    lines.append("| case_id | difficulty | lang | pass | composite | tool_calls | latency_ms | cost_usd | degraded |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        last_turn = r.turns[-1] if r.turns else None
        tool_calls = sum(t.tool_call_count for t in r.turns)
        latency = last_turn.latency_ms if last_turn else r.wall_clock_ms
        cost = sum((t.cost_usd for t in r.turns), Decimal(0))
        # v2.0.32.7 — surface the degraded flag in the per-case table
        # so operators can spot infra-degraded rows at a glance.
        any_degraded = any(t.degraded for t in r.turns)
        degraded_marker = "DEGRADED" if any_degraded else "-"
        if any_degraded:
            degraded_marker = f"DEGRADED ({next((t.degraded_reason for t in r.turns if t.degraded), '')})"
        status = "PASS" if r.pass_ else "FAIL"
        if any_degraded:
            status = "INFRA-DEGRADED"  # not a real logic-bug fail
        lines.append(
            f"| {r.case_id} | {r.difficulty} | {r.language} | "
            f"{status} | {r.composite_score:.2f} | "
            f"{tool_calls} | {latency} | {cost} | {degraded_marker} |"
        )
    lines.append("")

    # Worst-N failing cases (top 5). v2.0.32.7 — exclude backend-
    # degraded cases; they're infra noise, not real logic bugs.
    # Degraded cases are surfaced separately in the per-case table
    # (with `INFRA-DEGRADED` marker) and counted in the
    # `Backend-degraded cases` rollup line.
    failing = [
        r for r in results
        if not r.pass_ and not any(t.degraded for t in r.turns)
    ]
    failing.sort(key=lambda r: r.composite_score)
    if failing:
        lines.append("## Worst failing cases")
        lines.append("")
        for r in failing[:10]:
            lines.append(f"### {r.case_id}")
            lines.append("")
            lines.append(f"- **Language:** {r.language}  ")
            lines.append(f"- **Difficulty:** {r.difficulty}  ")
            lines.append(f"- **Suite:** {r.suite}  ")
            lines.append(f"- **Failure reason:** {r.failure_reason or 'check_failed'}  ")
            lines.append(f"- **Composite score:** {r.composite_score:.2f}  ")
            lines.append("")
            prog = r.judgment.get("programmatic", {})
            for i, pt in enumerate(prog.get("per_turn", [])):
                if pt.get("pass"):
                    continue
                lines.append(f"  Turn {i} failures:")
                for chk in pt.get("checks", []):
                    if not chk.get("pass"):
                        lines.append(
                            f"    - `{chk['name']}` expected={chk['expected']!r} "
                            f"actual={chk['actual']!r}"
                        )
                lines.append("")
            # Show last-turn answer excerpt
            if r.turns and r.turns[-1].answer:
                excerpt = r.turns[-1].answer[:300]
                lines.append(f"  Answer excerpt: _{excerpt}…_")
                lines.append("")

    return "\n".join(lines)


def render_console_table(rollup: RollupMetrics, results: list[CaseResult]) -> str:
    """ASCII-safe console summary. Avoids ✓ (U+2713) — Windows GBK trips."""
    lines: list[str] = []
    lines.append(f"=== YuanRAG eval ===")
    lines.append(f"Total cases: {rollup.total_cases} | Passed: {rollup.passed_cases} | Success rate: {rollup.success_rate:.1%}")
    lines.append(f"Hallucination rate: {rollup.hallucination_rate:.1%}")
    lines.append(f"Latency p50/p95/p99: {rollup.latency_p50_ms}/{rollup.latency_p95_ms}/{rollup.latency_p99_ms} ms")
    lines.append(f"Total cost: ${rollup.cost_rollup['total_cost_usd']}")
    if rollup.refusal_accuracy is not None:
        lines.append(f"Refusal accuracy (adv): {rollup.refusal_accuracy:.1%}")
    if rollup.citation_coverage is not None:
        lines.append(f"Citation coverage: {rollup.citation_coverage:.1%}")
    # v2.0.32.7 — surface degraded bucket in console output so the
    # operator's terminal view distinguishes infra-degraded from real
    # logic bugs at a glance.
    if rollup.degraded_cases:
        lines.append(
            f"Backend-degraded cases: {rollup.degraded_cases} | "
            f"Pass rate (excl. infra-degraded): "
            f"{rollup.pass_rate_excluding_infra_skip:.1%}"
            if rollup.pass_rate_excluding_infra_skip is not None
            else f"Backend-degraded cases: {rollup.degraded_cases}"
        )
    lines.append("")
    # v2.0.32.7 — exclude backend-degraded cases from "Worst failing
    # cases" listing; they belong to the INFRA-DEGRADED bucket, not
    # the real logic-bug bucket.
    failing = sorted(
        [
            r for r in results
            if not r.pass_ and not any(t.degraded for t in r.turns)
        ],
        key=lambda r: r.composite_score,
    )
    if failing:
        lines.append("Worst-3 failing cases:")
        for r in failing[:3]:
            fail_reason = next(
                (
                        chk.get("name")
                        for pt in r.judgment.get("programmatic", {}).get("per_turn", [])
                        for chk in pt.get("checks", [])
                        if not chk.get("pass")
                    ),
                r.failure_reason or "?",
            )
            lines.append(
                f"  - {r.case_id} ({r.language}/{r.difficulty}) composite={r.composite_score:.2f} fail={fail_reason}"
            )
    return "\n".join(lines)


# ============================================================
# Helpers
# ============================================================


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _distribution_5(values: Iterable[int]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": float(max(values)),
    }


def make_timestamp() -> str:
    """Filename-safe ISO-8601 timestamp (UTC)."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")