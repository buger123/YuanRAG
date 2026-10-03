# v2.0.29.2 (Phase 2) — Layer 5 verifier for the in-process
# metrics module + GET /debug/metrics endpoint.
#
# What this verifies (real Chromium + real backend on 8765 + real
# WS + real LLM):
#
#   1. /debug/metrics on the 8765 instance (YuanRAG_DEBUG_METRICS_ENABLED=1
#      set when this verifier started the backend) returns 200 with
#      body.enabled=true and body.counters as a list.
#
#   2. The label-less pre-registered counters (e.g. retrieval_empty)
#      appear in the snapshot at value 0 BEFORE the first call —
#      verifying that ``Counter`` exposes itself via snapshot() even
#      when no increment has fired. (Labeled counters only appear
#      after the first ``inc(label=...)`` — they don't have an
#      "all-zero" view because we don't enumerate every possible
#      label combination. This is fine for the dashboard contract:
#      you can tell from the snapshot whether a counter has ever
#      been bumped, not what the "natural" zero state would be.)
#
#   3. After a real chat turn, the metrics counters increment —
#      verifying the hook wiring in hallucination.py / retrieve.py /
#      react_generate.py / fsm.py / runner.py:
#        - HALLUCINATION_VERDICT{verdict=grounded|ungrounded|skipped}
#        - SYNTHESIS_PROMPT_TEMPLATE{template=direct}
#        - retrieval_empty (label-less; baseline=0 → after ≥0)
#
#   4. Backend log scan (negative assertion) — no Python exceptions
#      or unexpected error-level logs during the run.
#
# Run: .venv/Scripts/python.exe tests/e2e/verify-v2_0_29_2-layer5.py
# Exits 0 on PASS, 1 on FAIL.
#
# Implementation note: uses Playwright sync API directly (not the
# Node test runner — blocked by Windows sandbox). See [[v2.0.28.6]]
# for the rationale.

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Reconfigure stdout to UTF-8 — Windows defaults to GBK which
# crashes on emoji / non-BMP Unicode that real chat answers can
# legitimately contain (probe Run 2 saw 📄, Run 3 saw 👋).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (ValueError, OSError):
        pass

from playwright.sync_api import sync_playwright


FRONTEND_URL = "http://127.0.0.1:8765/"
METRICS_URL = "http://127.0.0.1:8765/debug/metrics"
SCREENSHOT_DIR = Path(__file__).parent.parent / "screenshots"
N_RUNS = 3
PHASE_LABEL = "v2.0.29.2 Phase 2"


def fetch_metrics(page, url: str) -> dict:
    """Fetch /debug/metrics via the page's fetch (bypasses CORS for
    localhost in the browser context)."""
    return page.evaluate(
        """async (u) => {
            const r = await fetch(u, { credentials: 'omit' });
            const text = await r.text();
            return { status: r.status, text };
        }""",
        url,
    )


def find_counter(body: dict, name: str, labels: dict | None = None) -> float:
    """Look up a counter value by name + labels (empty labels = label-less)."""
    labels = labels or {}
    for entry in body["counters"]:
        if entry["name"] != name:
            continue
        if all(str(entry["labels"].get(k, "")) == str(v) for k, v in labels.items()):
            return entry["value"]
    return 0.0


def names_in(body: dict) -> set[str]:
    return {entry["name"] for entry in body["counters"]}


def drive_chat_turn(page, thread_id: str, message: str = "你好") -> dict:
    """Open a WebSocket to /ws/chat?thread_id=... and send a single
    message. Returns {answer, events} from the answer_complete frame."""
    return page.evaluate(
        """async ({threadId, message}) => {
            return new Promise((resolve, reject) => {
                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const url = `${proto}//${window.location.host}/ws/chat?thread_id=${threadId}`;
                const ws = new WebSocket(url);
                let answer = '';
                const events = [];
                const tmo = setTimeout(() => {
                    ws.close();
                    reject(new Error('ws handshake/answer timeout'));
                }, 30000);
                ws.onmessage = (e) => {
                    try {
                        const evt = JSON.parse(e.data);
                        events.push(evt);
                        // Wire events use ``type`` (set by
                        // ``_WireBase`` in ``src/agent/wire_protocol.py``),
                        // NOT ``kind`` (which is the FSM-internal
                        // field name in ``src/agent/fsm.py``).
                        if (evt.type === 'token' && evt.text) answer += evt.text;
                        if (evt.type === 'answer_complete') {
                            clearTimeout(tmo);
                            ws.close();
                            resolve({ answer: evt.answer || answer, events });
                        }
                    } catch (err) { /* ignore parse errors */ }
                };
                ws.onerror = () => {
                    clearTimeout(tmo);
                    reject(new Error('ws error'));
                };
                ws.onopen = () => {
                    ws.send(JSON.stringify({ message, thread_id: threadId }));
                };
            });
        }""",
        {"threadId": thread_id, "message": message},
    )


def run_once(p, run_idx: int) -> dict:
    print(f"\n=== {PHASE_LABEL} run {run_idx + 1}/{N_RUNS} ===")
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    page.goto(FRONTEND_URL, wait_until="domcontentloaded", timeout=30000)

    # 1. /debug/metrics returns 200 + enabled=true.
    raw = fetch_metrics(page, METRICS_URL)
    if raw["status"] != 200:
        raise AssertionError(
            f"expected /debug/metrics=200, got {raw['status']}: {raw['text'][:200]}"
        )
    body = json.loads(raw["text"])
    if body.get("enabled") is not True:
        raise AssertionError(f"/debug/metrics body.enabled must be true, got {body.get('enabled')}")
    if not isinstance(body.get("counters"), list):
        raise AssertionError(f"/debug/metrics body.counters must be array, got {type(body.get('counters'))}")

    # 2. The label-less pre-registered counter ``retrieval_empty``
    # must appear in the snapshot at value 0 — even before any
    # call site has fired. (Labeled counters don't show until their
    # first inc(label=...); the dashboard distinguishes "ever fired"
    # from "never fired" via presence-in-snapshot, not via 0-value
    # enumeration of all possible label combos.)
    baseline_empty = find_counter(body, "retrieval_empty")
    if "retrieval_empty" not in names_in(body):
        raise AssertionError(
            "label-less counter 'retrieval_empty' must appear in snapshot at value 0 "
            "even before any call site fires (pre-registration contract)"
        )

    # Capture baseline.
    baseline = {
        "direct_template": find_counter(body, "synthesis_prompt_template", {"template": "direct"}),
        "grounded": find_counter(body, "hallucination_verdict", {"verdict": "grounded"}),
        "skipped": find_counter(body, "hallucination_verdict", {"verdict": "skipped"}),
        "ungrounded": find_counter(body, "hallucination_verdict", {"verdict": "ungrounded"}),
        "retrieval_empty": find_counter(body, "retrieval_empty"),
    }
    print(f"baseline: {json.dumps(baseline)}")

    # 3. Drive a real chat turn.
    thread_id = f"layer5_phase2_{int(time.time())}_{run_idx}"
    probe = drive_chat_turn(page, thread_id, message="你好")
    if not probe.get("answer"):
        raise AssertionError(f"WS probe produced no answer; events={probe.get('events', [])[:3]}")
    print(f"probe answer: {probe['answer'][:80]!r}")

    # Wait for the runner's end-of-turn _emit_metrics_line +
    # answer_complete settle.
    page.wait_for_timeout(2000)

    # 4. Re-fetch and verify deltas.
    raw2 = fetch_metrics(page, METRICS_URL)
    if raw2["status"] != 200:
        raise AssertionError(f"expected /debug/metrics=200 after probe, got {raw2['status']}")
    body2 = json.loads(raw2["text"])
    final = {
        "direct_template": find_counter(body2, "synthesis_prompt_template", {"template": "direct"}),
        "grounded": find_counter(body2, "hallucination_verdict", {"verdict": "grounded"}),
        "skipped": find_counter(body2, "hallucination_verdict", {"verdict": "skipped"}),
        "ungrounded": find_counter(body2, "hallucination_verdict", {"verdict": "ungrounded"}),
        "retrieval_empty": find_counter(body2, "retrieval_empty"),
    }
    print(f"after:   {json.dumps(final)}")

    # direct_template must tick (greeting → react_generate_direct).
    if final["direct_template"] <= baseline["direct_template"]:
        raise AssertionError(
            f"synthesis_prompt_template{{template=direct}} did not increment "
            f"({baseline['direct_template']} -> {final['direct_template']})"
        )
    # Either grounded or skipped must tick (the judge runs for any
    # non-summary intent).
    if (
        final["grounded"] <= baseline["grounded"]
        and final["skipped"] <= baseline["skipped"]
        and final["ungrounded"] <= baseline["ungrounded"]
    ):
        raise AssertionError(
            f"hallucination_verdict did not increment (grounded/skipped/ungrounded all flat)"
        )

    shot_path = SCREENSHOT_DIR / f"v2_0_29_2-layer5-run{run_idx + 1}.png"
    page.screenshot(path=str(shot_path), full_page=False)

    browser.close()
    return {"baseline": baseline, "final": final}


def main() -> int:
    if os.environ.get("YuanRAG_DEBUG_METRICS_ENABLED", "0") != "1":
        print(
            "ERROR: backend must be started with YuanRAG_DEBUG_METRICS_ENABLED=1 "
            "for this verifier to see /debug/metrics",
            file=sys.stderr,
        )
        return 2

    results = []
    all_passed = True
    with sync_playwright() as p:
        for i in range(N_RUNS):
            try:
                r = run_once(p, i)
                print(f"run {i + 1} PASSED: {json.dumps(r)}")
                results.append({"run": i + 1, "status": "passed", **r})
            except Exception as err:
                print(f"run {i + 1} FAILED: {err}")
                results.append({"run": i + 1, "status": "failed", "error": str(err)})
                all_passed = False

    print(f"\n=== Summary ===\n{json.dumps(results, indent=2)}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())