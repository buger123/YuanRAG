"""RAG evaluation harness — fixed test sets (golden + adversarial) + 3-pass judge.

v2.0.32.0 — ships as a test-side package, no production-code changes.
Public API:

    from tests.eval.cases import load_case, load_suite
    from tests.eval.runner import AsyncHarnessRunner
    from tests.eval.tokens import TokenCapture
    from tests.eval.cost import estimate_cost
    from tests.eval.judge.programmatic import run_programmatic
    from tests.eval.judge.cheap_llm import run_cheap_llm_judge
    from tests.eval.report import write_reports

CLI entrypoint:
    python -m tests.eval.run --suite golden
    bash tests/eval/run.sh
"""