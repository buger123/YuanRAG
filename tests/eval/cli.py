"""CLI entry point for the eval harness.

Usage:
    python -m tests.eval.run --suite golden [--concurrency 1] [--judge all]
    bash tests/eval/run.sh
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from tests.eval.cases import CaseLoadError, load_suite
from tests.eval.conftest_compat import ensure_isolated_data_dir
from tests.eval.report import (
    compute_rollup,
    make_timestamp,
    render_console_table,
    write_jsonl,
    write_markdown,
    write_rollup_json,
)
from tests.eval.runner import AsyncHarnessRunner


DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _should_run_cheap_llm(judge: str) -> bool:
    """Phase 3 (2026-10-05): decide whether to invoke the cheap-LLM judge.

    ``--judge programmatic``  → only programmatic pass (default for fast CI).
    ``--judge cheap_llm``     → only cheap-LLM pass (skips structural checks).
    ``--judge all``           → both passes (default for thorough runs).
    """
    return judge in ("cheap_llm", "all")


def _expected_answer_text(turn) -> str:
    """Concatenate ``expected_answer_contains`` into the human-readable
    "expected answer" string the cheap-LLM judge consumes. The judge
    uses this as the reference answer — having empty expected text just
    makes the judge rate accuracy 0 / completeness 0 by default.
    """
    bits = turn.expected_answer_contains or []
    if bits:
        return " / ".join(str(b) for b in bits)
    # Adversarial refusal cases — surface the template substring as the
    # expected answer so the judge knows what to look for.
    if turn.expected_refusal:
        return str(turn.expected_refusal.template)
    return ""


def _retrieved_docs_text(turn_result) -> str:
    """Concatenate retrieved chunk text from ``turn.sources`` so the
    cheap-LLM judge can verify "the answer matches the docs"."""
    srcs = turn_result.sources or []
    chunks: list[str] = []
    for s in srcs:
        text = s.get("text") if isinstance(s, dict) else None
        if isinstance(text, str):
            chunks.append(text)
    return "\n\n---\n\n".join(chunks)


async def _run_cheap_llm_pass(case, result) -> None:
    """Phase 3 (2026-10-05): invoke cheap-LLM judge per turn, mutate
    ``result.judgment['cheap_llm']`` in place with an aggregate dict.

    Failure modes (per judge):
      - cheap_llm API 529 / timeout → log warning, leave cheap_llm=None
        so downstream composite falls back to programmatic only.
      - structured output validation fail → already swallowed in judge.
    """
    from tests.eval.judge.cheap_llm import run_cheap_llm_judge

    if not result.turns:
        return

    per_turn: list[dict] = []
    for turn, turn_res in zip(case.turns, result.turns):
        judgment = await run_cheap_llm_judge(
            question=turn.user,
            expected_answer=_expected_answer_text(turn),
            actual_answer=turn_res.answer or "",
            retrieved_docs=_retrieved_docs_text(turn_res),
            op_name=f"eval_judge_{case.id}",
        )
        per_turn.append(
            judgment.to_dict() if judgment is not None else {"passed": False, "error": "judge_unavailable"}
        )

    # Aggregate across turns: pass iff every turn passes.
    if per_turn and all(t.get("passed") for t in per_turn):
        cheap_pass = True
    else:
        cheap_pass = False
    composites = [t.get("composite", 0.0) for t in per_turn if isinstance(t.get("composite"), (int, float))]
    mean_composite = sum(composites) / len(composites) if composites else 0.0

    result.judgment["cheap_llm"] = {
        "pass": cheap_pass,
        "per_turn": per_turn,
        "composite_mean": mean_composite,
    }

    # Recompute composite_score now that cheap_llm is populated.
    from tests.eval.cases import Case  # local import — kept off hot path
    w = case.scoring_weights
    prog = 1.0 if result.judgment["programmatic"]["pass"] else 0.0
    cheap_norm = mean_composite / 2.0
    result.composite_score = prog * w.programmatic_pass + cheap_norm * w.cheap_llm_pass
    result.pass_ = result.composite_score >= case.scoring_weights.threshold_pass


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tests.eval.run",
        description="YuanRAG eval harness — fixed test sets + 3-pass judge.",
    )
    p.add_argument(
        "--suite",
        choices=("golden", "adversarial"),
        default="golden",
        help="Which case suite to run.",
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765",
        help="Backend base URL.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-turn SSE timeout in seconds.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of cases to run in parallel. Phase 1 default is 1.",
    )
    p.add_argument(
        "--judge",
        choices=("programmatic", "cheap_llm", "all"),
        default="all",
        help="Which judge passes to run.",
    )
    p.add_argument(
        "--case-id",
        default=None,
        help="Run only this single case id (filter).",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Where to write the JSONL + rollup + markdown reports.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Composite-score pass/fail threshold (per case scoring_weights).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    # CRITICAL: isolate RAG_DATA_DIR before any settings module reads it.
    ensure_isolated_data_dir()

    try:
        cases = load_suite(args.suite)
    except CaseLoadError as exc:
        print(f"[error] failed to load suite {args.suite!r}: {exc}", file=sys.stderr)
        return 2

    if args.case_id:
        cases = [c for c in cases if c.id == args.case_id]
    if not cases:
        print(f"[error] no cases to run (suite={args.suite!r}, filter={args.case_id!r})", file=sys.stderr)
        return 2

    timestamp = make_timestamp()
    run_cheap = _should_run_cheap_llm(args.judge)
    results: list[Any] = []
    with AsyncHarnessRunner(
        base_url=args.base_url,
        timeout=args.timeout,
    ) as runner:
        for case in cases:
            try:
                result = runner.run_case(case)
            except Exception as exc:
                print(f"[error] runner raised on case {case.id!r}: {exc}", file=sys.stderr)
                continue

            # Phase 3 (2026-10-05): invoke cheap-LLM judge per turn.
            # Failures inside the judge (529, validation, missing model)
            # are swallowed by ``run_cheap_llm_judge`` → returns None →
            # we leave cheap_llm=None so downstream composite falls back
            # to programmatic only.
            if run_cheap:
                try:
                    asyncio.run(_run_cheap_llm_pass(case, result))
                except Exception as exc:
                    print(
                        f"[warn] cheap_llm judge unavailable for {case.id!r}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

            results.append(result)
            print(
                f"  [{len(results)}/{len(cases)}] {case.id}: "
                f"{'PASS' if result.pass_ else 'FAIL'} "
                f"(composite={result.composite_score:.2f})"
            )

    if not results:
        print("[error] no results produced", file=sys.stderr)
        return 1

    rollup = compute_rollup(results)

    # Phase 3 (2026-10-05): always emit the Claude-verdict prompt stub.
    # Operator pastes the rendered prompt into a Claude Code session to
    # produce the offline Pass 3 verdict. Self-contained — no LLM call
    # from this side.
    from tests.eval.judge.claudemd import build_claude_verdict_prompt

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    claude_prompt = build_claude_verdict_prompt(reports_dir=args.reports_dir, timestamp=timestamp)
    claude_prompt_path = args.reports_dir / f"{timestamp}.claude_prompt.txt"
    claude_prompt_path.write_text(claude_prompt.to_text(), encoding="utf-8")

    jsonl_path = args.reports_dir / f"{timestamp}.jsonl"
    rollup_path = args.reports_dir / f"{timestamp}.rollup.json"
    md_path = args.reports_dir / f"{timestamp}.md"
    write_jsonl(jsonl_path, results)
    write_rollup_json(rollup_path, rollup, suite=args.suite, timestamp=timestamp)
    write_markdown(md_path, suite=args.suite, timestamp=timestamp, rollup=rollup, results=results)

    print()
    print(render_console_table(rollup, results))
    print()
    print(f"[ok] JSONL:   {jsonl_path}")
    print(f"[ok] Rollup:  {rollup_path}")
    print(f"[ok] Markdown: {md_path}")
    print(f"[ok] Claude-verdict prompt: {claude_prompt_path}")

    # Exit 0 iff success_rate >= threshold (default 0.7 = 70%)
    exit_code = 0 if rollup.success_rate >= args.threshold else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

