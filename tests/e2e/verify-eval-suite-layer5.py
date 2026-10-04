"""Stage 2 Layer 5 ship gate for the eval harness.

v2.0.32.0 — per [[development-methodology]] + [[verification-protocol]],
ship gate = real backend + real LLM + dist/ + the eval CLI runs end-to-end
without crashing AND with success_rate >= threshold.

Why this exists as a Python script (not a JS Playwright spec):
  The eval harness drives the backend via POST /chat SSE — there is no
  UI involved. Per [[v2.0.28.20]] REBUILD NOTE, the Layer 5 protocol
  requires dist/ to be built (the ship gate runs the FULL stack the user
  will see in production, even though eval itself doesn't touch the UI).

  We don't need Chromium. We DO need:
    1. dist/ index.html present (built)
    2. Backend reachable on :8765 with /health 200
    3. Real LLM key (otherwise case fail_reason = llm_error)
    4. python -m tests.eval.run --suite golden exits 0
    5. tests/eval/reports/<ts>.{jsonl,rollup.json,md} exist + valid schema

Exit code 0 = ship-ready. Non-zero = block ship.

Usage:
  python tests/e2e/verify-eval-suite-layer5.py
  python tests/e2e/verify-eval-suite-layer5.py --suite adversarial
  python tests/e2e/verify-eval-suite-layer5.py --threshold 0.6 --timeout 240

Pre-conditions (asserted, NOT auto-started):
  - start.bat serving dist/ on :8765 (or equivalent)
  - .env has a real LLM key
  - npm run build has been run (dist/ present)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_URL_DEFAULT = "http://127.0.0.1:8765"
DIST_PATH = REPO_ROOT / "src" / "frontend" / "dist" / "index.html"
REPORTS_DIR = REPO_ROOT / "tests" / "eval" / "reports"


# ================================================================
# Gate helpers — each returns (ok: bool, detail: str)
# ================================================================


def gate_dist_built() -> tuple[bool, str]:
    """Per [[v2.0.28.20]] REBUILD NOTE: dist/ must exist."""
    if DIST_PATH.exists():
        return True, f"dist/ present at {DIST_PATH}"
    return False, (
        f"dist/ NOT FOUND at {DIST_PATH} — run `npm run build` in src/frontend/"
    )


def gate_backend_reachable(url: str) -> tuple[bool, str]:
    """Backend must answer /health 200 on :8765."""
    try:
        resp = httpx.get(f"{url}/health", timeout=5.0)
    except Exception as exc:
        return False, f"backend not reachable at {url}: {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return False, f"backend /health returned {resp.status_code}: {resp.text[:200]}"
    return True, f"backend /health 200 at {url}"


def gate_real_llm(url: str) -> tuple[bool, str]:
    """Hit /models/status — embedder + reranker must report ready."""
    try:
        resp = httpx.get(f"{url}/models/status", timeout=10.0)
    except Exception as exc:
        return False, f"/models/status unreachable: {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return False, f"/models/status returned {resp.status_code}: {resp.text[:200]}"
    try:
        body = resp.json()
    except Exception:
        return False, "/models/status did not return JSON"
    ready = bool(body.get("ready"))
    if not ready:
        return False, f"models not ready: {json.dumps(body)[:200]}"
    return True, f"models ready: embedder={body.get('embedding', {}).get('status')}, reranker={body.get('reranker', {}).get('status')}"


def gate_eval_runs(suite: str, url: str, timeout: float) -> tuple[bool, str]:
    """Run ``python -m tests.eval --suite <suite>`` end-to-end.

    Per cli.py semantics, exit code is:
      0 = success_rate >= threshold (ship-ready)
      1 = success_rate < threshold OR no results
      2 = case load failure / arg error
      Other = uncaught crash

    The ship gate separates "did the eval run" (this gate) from
    "did the system meet threshold" (success_rate gate below). So we
    PASS this gate iff the CLI produced reports on disk — that proves
    the harness drove every case end-to-end without crashing.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Snapshot pre-existing reports so we only check NEW ones.
    before = {p.name for p in REPORTS_DIR.glob("*") if p.is_file()}

    cmd = [
        sys.executable,
        "-m",
        "tests.eval",
        "--suite",
        suite,
        "--base-url",
        url,
        "--timeout",
        str(timeout),
        "--reports-dir",
        str(REPORTS_DIR),
    ]
    env = os.environ.copy()
    # Hermetic RAG_DATA_DIR — never touch the developer's ~/.rag_assistant
    import tempfile
    env["RAG_DATA_DIR"] = tempfile.mkdtemp(prefix="yuanrag-eval-gate-")
    # Force UTF-8 stdout so subprocess prints don't corrupt under Windows GBK
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"  $ {' '.join(cmd)}")
    print(f"    RAG_DATA_DIR={env['RAG_DATA_DIR']}")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout * 12)
    except subprocess.TimeoutExpired:
        return False, f"eval run timed out after {timeout * 12:.0f}s"
    elapsed = time.monotonic() - t0

    # Did the CLI write any new report files? That's the proof of "ran".
    after = {p.name for p in REPORTS_DIR.glob("*") if p.is_file()}
    new_files = sorted(after - before)
    has_jsonl = any(f.endswith(".jsonl") for f in new_files)
    has_rollup = any(f.endswith("rollup.json") for f in new_files)
    has_md = any(f.endswith(".md") for f in new_files)

    if not (has_jsonl and has_rollup and has_md):
        # CLI did not produce a complete report set → crash / arg error.
        return False, (
            f"eval run exit={proc.returncode} after {elapsed:.1f}s — "
            f"reports incomplete: jsonl={has_jsonl} rollup={has_rollup} md={has_md}\n"
            f"  stdout: {proc.stdout[-1000:]}\n"
            f"  stderr: {proc.stderr[-1000:]}"
        )

    if proc.returncode not in (0, 1):
        return False, (
            f"eval run crash exit={proc.returncode} after {elapsed:.1f}s\n"
            f"  stdout: {proc.stdout[-1000:]}\n"
            f"  stderr: {proc.stderr[-1000:]}"
        )

    return True, (
        f"eval run exit={proc.returncode} after {elapsed:.1f}s — "
        f"reports produced: {len(new_files)} files"
    )


def gate_reports_generated(suite: str) -> tuple[bool, str]:
    """Find the most-recent report set and assert all 3 files exist + valid."""
    if not REPORTS_DIR.exists():
        return False, f"reports dir missing: {REPORTS_DIR}"
    files = sorted(REPORTS_DIR.glob(f"*-*-*T*.{suite}.*"), reverse=True)
    if not files:
        # Fall back: glob by extension only
        files = sorted(
            [p for p in REPORTS_DIR.glob("*") if p.is_file() and p.suffix in (".jsonl", ".json", ".md")],
            reverse=True,
        )
    if not files:
        return False, f"no report files in {REPORTS_DIR}"
    jsonl_files = [p for p in files if p.suffix == ".jsonl"]
    rollup_files = [p for p in files if p.suffix == ".json" and "rollup" in p.name]
    md_files = [p for p in files if p.suffix == ".md"]
    if not (jsonl_files and rollup_files and md_files):
        return False, (
            f"missing one of jsonl/rollup/md — "
            f"jsonl={len(jsonl_files)} rollup={len(rollup_files)} md={len(md_files)}"
        )
    # Validate JSONL schema — each line is a CaseResult.to_dict()
    jsonl_path = jsonl_files[0]
    try:
        records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        return False, f"JSONL parse failed: {jsonl_path}: {exc}"
    expected_count = 10 if suite == "golden" else 22
    if len(records) != expected_count:
        return False, (
            f"JSONL record count {len(records)} != expected {expected_count} for suite={suite!r}"
        )
    required_keys = {"case_id", "language", "difficulty", "turns", "judgment", "composite_score", "pass"}
    for i, rec in enumerate(records):
        missing = required_keys - rec.keys()
        if missing:
            return False, f"JSONL[{i}] missing keys {missing} in {jsonl_path}"
    # Validate rollup.json
    rollup_path = rollup_files[0]
    try:
        rollup = json.loads(rollup_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"rollup.json parse failed: {rollup_path}: {exc}"
    if "metrics" not in rollup:
        return False, f"rollup.json missing 'metrics' key: {list(rollup.keys())}"
    return True, (
        f"reports valid: jsonl={jsonl_path.name}({len(records)} records) "
        f"rollup={rollup_path.name} md={md_files[0].name}"
    )


def gate_success_rate(suite: str, threshold: float) -> tuple[bool, str]:
    """Read the most-recent rollup.json; assert success_rate >= threshold."""
    rollup_files = sorted(REPORTS_DIR.glob("*rollup.json"), reverse=True)
    if not rollup_files:
        return False, "no rollup.json to read"
    rollup = json.loads(rollup_files[0].read_text(encoding="utf-8"))
    metrics = rollup.get("metrics", {})
    rate = float(metrics.get("success_rate", 0.0))
    passed = int(metrics.get("passed_cases", 0))
    total = int(metrics.get("total_cases", 0))
    detail = f"success_rate={rate:.2f} ({passed}/{total}) threshold={threshold:.2f}"
    if rate < threshold:
        return False, detail + f" — BELOW threshold"
    return True, detail + f" — PASS"


# ================================================================
# Main
# ================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="verify-eval-suite-layer5",
        description="Layer 5 ship gate for the eval harness.",
    )
    parser.add_argument("--suite", choices=("golden", "adversarial"), default="golden")
    parser.add_argument("--base-url", default=BACKEND_URL_DEFAULT)
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-case SSE timeout in seconds.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum success_rate to ship.")
    parser.add_argument("--skip-real-llm", action="store_true", help="Skip /models/status check (CI without models)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"  v2.0.32.0 EVAL SUITE LAYER 5 SHIP GATE")
    print(f"  suite={args.suite!r}  base_url={args.base_url}  threshold={args.threshold}")
    print(f"  time={datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print("=" * 72)

    gates: list[tuple[str, bool, str]] = []

    print("\n[1/5] dist/ present (per [[v2.0.28.20]] REBUILD NOTE)...")
    ok, detail = gate_dist_built()
    gates.append(("dist_built", ok, detail))
    print(f"       [{'PASS' if ok else 'FAIL'}] {detail}")

    print(f"\n[2/5] backend reachable on {args.base_url}...")
    ok, detail = gate_backend_reachable(args.base_url)
    gates.append(("backend_reachable", ok, detail))
    print(f"       [{'PASS' if ok else 'FAIL'}] {detail}")

    if not args.skip_real_llm:
        print(f"\n[3/5] /models/status ready (real LLM + BGE-M3 + reranker)...")
        ok, detail = gate_real_llm(args.base_url)
        gates.append(("models_ready", ok, detail))
        print(f"       [{'PASS' if ok else 'FAIL'}] {detail}")
    else:
        print(f"\n[3/5] /models/status SKIPPED (--skip-real-llm)")
        gates.append(("models_ready", True, "skipped (--skip-real-llm)"))

    print(f"\n[4/5] eval CLI runs --suite {args.suite} end-to-end...")
    ok, detail = gate_eval_runs(args.suite, args.base_url, args.timeout)
    gates.append(("eval_runs", ok, detail))
    print(f"       [{'PASS' if ok else 'FAIL'}] {detail}")

    print(f"\n[5/5] reports generated + success_rate >= {args.threshold}...")
    if gates[-1][1]:  # only check reports if eval ran OK
        ok, detail = gate_reports_generated(args.suite)
        gates.append(("reports_generated", ok, detail))
        print(f"       [{'PASS' if ok else 'FAIL'}] {detail}")
        if ok:
            ok, detail = gate_success_rate(args.suite, args.threshold)
            gates.append(("success_rate", ok, detail))
            print(f"       [{'PASS' if ok else 'FAIL'}] {detail}")
    else:
        # If eval didn't run, the downstream gates are moot — but we
        # still mark them FAIL so the totals don't lie.
        gates.append(("reports_generated", False, "skipped — eval did not run"))
        gates.append(("success_rate", False, "skipped — eval did not run"))
        print(f"       [FAIL] eval did not run, skipping downstream gates")

    # Print verdict
    passed = sum(1 for _, ok, _ in gates if ok)
    total = len(gates)
    print()
    print("=" * 72)
    if passed == total:
        print(f"  [OK] SHIP READY — {passed}/{total} gates PASSED")
        print("=" * 72)
        return 0
    print(f"  [BLOCK] SHIP BLOCKED — {passed}/{total} gates PASSED ({total - passed} FAILED)")
    print("=" * 72)
    for name, ok, detail in gates:
        if not ok:
            print(f"    FAILED  {name}: {detail}")
    return 1


if __name__ == "__main__":
    import io
    # Force UTF-8 stdout/stderr so we don't crash on Windows GBK.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, io.UnsupportedOperation):
        # Python <3.7 or already-closed stream — fall back to ascii-safe
        pass
    raise SystemExit(main())
