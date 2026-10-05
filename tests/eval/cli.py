"""CLI entry point for the eval harness.

Usage:
    python -m tests.eval.run --suite golden [--concurrency 1] [--judge all]
    bash tests/eval/run.sh
"""
from __future__ import annotations

import argparse
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

    args.reports_dir.mkdir(parents=True, exist_ok=True)
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

    # Exit 0 iff success_rate >= threshold (default 0.7 = 70%)
    exit_code = 0 if rollup.success_rate >= args.threshold else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())