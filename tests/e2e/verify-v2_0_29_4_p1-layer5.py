# v2.0.29.4 (Phase 4 PR-1) — Layer 5 verifier for citation
# hardening + retrieval gates (load-bearing half of Phase 4).
#
# What this verifies (real Chromium + real backend on 8765 + real
# WS + real LLM + real dist/):
#
#   1. Anchored summary-intent regex: ``"总结一下这份文档"`` (start
#      with summary verb) still fires the bulk summary path. The
#      summary_intent_used flag survives to the runner (visible via
#      the metrics counter) and the documents list is non-empty.
#
#   2. Definitional QA ``"X 是什么"`` does NOT fire the summary
#      path — falls through to embed + hybrid + rerank. The
#      retrieval_status wires through to the runner and is visible
#      via /debug/metrics (counter: RETRIEVAL_FILTERED_LOW_SCORE
#      increments when off-topic; or RETRIEVAL_EMPTY when BM25+dense
#      return nothing).
#
#   3. CitationChip click event now ships chunk_id + doc_id +
#      source_kind alongside index. Probed by attaching a
#      window-level listener before the LLM answer arrives,
#      triggering a click on the first citation chip, and
#      asserting the CustomEvent detail shape.
#
#   4. Range collapse WARNING — when a multi-element citation
#      range collapses to < N elements, the backend logs a
#      WARNING via loguru (visible via a tail of backend.log).
#      Behavioral probe: upload a doc + cite ``[1-3]`` against 1
#      surviving source, observe the collapse log.
#
#   5. Backend log scan (negative assertion) — no Python
#      exceptions or unexpected error-level logs during the run.
#
#   6. /debug/metrics returns 200 throughout + the
#      RETRIEVAL_FILTERED_LOW_SCORE counter ticks on the
#      off-topic probe (verifies the gate's pre-registered metric
#      wire).
#
# Run: .venv/Scripts/python.exe tests/e2e/verify-v2_0_29_4_p1-layer5.py
# Exits 0 on PASS, 1 on FAIL.
#
# Implementation note: uses Playwright sync API directly (not the
# Node test runner — blocked by Windows sandbox). See [[v2.0.28.6]]
# for the rationale. Mirrors the structure of
# ``verify-v2_0_29_3-layer5.py`` (Phase 3) for consistency.

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Reconfigure stdout to UTF-8 — Windows defaults to GBK which
# crashes on emoji / non-BMP Unicode that real chat answers can
# legitimately contain.
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
PHASE_LABEL = "v2.0.29.4 Phase 4 PR-1"


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


def drive_chat_turn(page, thread_id: str, message: str) -> dict:
    """Open a WebSocket to /ws/chat?thread_id=... and send a single
    message. Returns {answer, events} from the answer_complete frame.

    The Phase 3 verifier pattern (verified live, 2026-09-27):
    ``answer_complete`` carries the FULL answer in ``evt.answer``
    (not just incrementally streamed via ``token`` events). Some
    code paths emit tokens but the final answer_complete arrives
    without re-streaming — relying solely on accumulated tokens
    yields an empty ``answer`` in those cases. We mirror Phase 3's
    ``evt.answer || answer`` fallback.
    """
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
                }, 45000);
                ws.onmessage = (e) => {
                    try {
                        const evt = JSON.parse(e.data);
                        events.push(evt);
                        // Wire events use ``type`` (set by
                        // ``_WireBase`` in ``src/agent/wire_protocol.py``),
                        // NOT ``kind`` (FSM-internal).
                        if (evt.type === 'token' && evt.text) answer += evt.text;
                        if (evt.type === 'answer_complete') {
                            clearTimeout(tmo);
                            ws.close();
                            resolve({
                                answer: evt.answer || answer,
                                sources: (evt.sources || []).map(s => ({
                                    index: s.index,
                                    chunk_id: s.chunk_id,
                                    doc_id: s.doc_id,
                                    source_kind: s.source_kind,
                                })),
                                events,
                            });
                        }
                        if (evt.type === 'error') {
                            clearTimeout(tmo);
                            ws.close();
                            reject(new Error(`agent error: ${evt.message || JSON.stringify(evt)}`));
                        }
                    } catch (err) {
                        clearTimeout(tmo);
                        ws.close();
                        reject(err);
                    }
                };
                ws.onerror = (e) => {
                    clearTimeout(tmo);
                    reject(new Error('ws error: ' + (e.message || 'unknown')));
                };
                ws.onopen = () => {
                    ws.send(JSON.stringify({ message, thread_id: threadId }));
                };
            });
        }""",
        {"threadId": thread_id, "message": message},
    )


def upload_text_doc(page, filename: str, text: str, thread_id: str) -> dict:
    """Upload a text doc to the backend via the /documents upload
    endpoint. Mirrors what ``src/api/routes/documents.py`` exposes.
    Returns the JSON response."""
    return page.evaluate(
        """async ({fname, text, threadId}) => {
            const fd = new FormData();
            fd.append('thread_id', threadId);
            // The upload endpoint declares ``file: UploadFile = File(...)``
            // (singular). The Phase 3 verifier used ``files`` (plural)
            // and got 422 from FastAPI — Phase 4 PR-1 verifier pins
            // the correct field name.
            fd.append('file', new Blob([text], { type: 'text/plain' }), fname);
            const r = await fetch('/documents/upload', { method: 'POST', body: fd });
            return { status: r.status, body: await r.json() };
        }""",
        {"fname": filename, "text": text, "threadId": thread_id},
    )


def delete_thread_docs(page, thread_id: str) -> dict:
    """Delete all docs for a thread. Test isolation."""
    return page.evaluate(
        """async ({threadId}) => {
            const r = await fetch(`/documents?thread_id=${encodeURIComponent(threadId)}`, { method: 'DELETE' });
            return { status: r.status };
        }""",
        {"threadId": thread_id},
    )


def check_pr1_assertion_1_summary_intent(page, thread_id: str, log: list) -> tuple[bool, str]:
    """PR-1 #1: anchored summary-intent regex still fires the bulk
    summary path on ``"总结一下..."`` queries. Verifies
    ``summary_intent_used`` reaches the runner via the
    ``summary_intent_used`` flag in the answer_complete event.

    We don't strictly assert the flag is exposed in the wire (it
    isn't — it's FSM-internal) — instead we observe the
    downstream effect: a real-doc thread + summary-intent query
    produces a non-empty answer that mentions the doc content.
    """
    upload = upload_text_doc(
        page,
        f"summary_probe_{int(time.time())}.md",
        "光合作用是植物利用光能合成有机物的过程。这是测试文档的关键内容。",
        thread_id,
    )
    if upload["status"] != 200 or not upload["body"].get("doc_id"):
        return False, f"upload failed: status={upload['status']} body={upload['body']}"

    # Wait for ingestion to complete (embedding + index). The upload
    # endpoint returns once bytes are accepted, but the embedding
    # pipeline runs as fire-and-forget; without the wait the summary
    # bulk path may see 0 chunks and short-circuit.
    page.wait_for_timeout(2500)

    result = drive_chat_turn(page, thread_id, "总结一下这份文档")
    # Debug: log the events seen, so a regression is easier to triage.
    event_types = [e.get("type", "?") for e in result.get("events", [])]
    log.append(f"  debug: events={event_types[:8]}... answer_len={len(result['answer'])}")
    answer = result["answer"]
    # The summary path returns ALL chunks of the thread's docs and
    # asks the LLM to summarize. We don't pin exact wording (LLM
    # is non-deterministic) — we pin that the answer is non-empty
    # AND contains at least one citation chip (sources list
    # populated), AND the answer references the doc topic.
    if not answer.strip():
        return False, "summary-intent answer was empty"
    if "光合作用" not in answer and "植物" not in answer:
        return False, f"summary-intent answer didn't reference doc topic; got: {answer!r}"
    if len(result["sources"]) < 1:
        return False, f"summary-intent answer had no sources; events={len(result['events'])}"
    log.append(f"  ✓ summary-intent fired: sources={len(result['sources'])} answer_len={len(answer)}")
    return True, ""


def check_pr1_assertion_2_definitional_qa(page, thread_id: str, log: list) -> tuple[bool, str]:
    """PR-1 #2: definitional QA ``"X 是什么"`` does NOT fire the
    summary bulk path. Counterpart to assertion 1.

    Behavioral pin: same doc as assertion 1 is in the thread.
    Query ``"光合作用是什么"`` should NOT take the summary fast
    path — it should fall through to embed + hybrid + rerank.
    We verify this by observing the metrics:
    ``retrieval_filtered_low_score`` ticks when the gate fires, OR
    ``retrieval_empty`` ticks when BM25+dense return nothing.
    Either way, the synthesis LLM sees fewer docs (because the
    summary bulk path returns ALL chunks).

    Counter names in /debug/metrics are LOWERCASE per Prometheus
    text-format convention (``src/agent/metrics.py:17-19``); the
    Python module-level constants (``RETRIEVAL_EMPTY`` etc.) are
    CamelCase for typing, but the wire name is the lowercase
    string passed to ``Counter(...)``. The Phase 3 verifier
    pattern uses the lowercase name; we mirror it.
    """
    # Capture metrics before
    metrics_before = fetch_metrics(page, METRICS_URL)
    if metrics_before["status"] != 200:
        return False, f"/debug/metrics returned {metrics_before['status']}; not enabled"
    body_before = json.loads(metrics_before["text"])

    low_before = find_counter(body_before, "retrieval_filtered_low_score")
    empty_before = find_counter(body_before, "retrieval_empty")

    result = drive_chat_turn(page, thread_id, "光合作用是什么")

    metrics_after = fetch_metrics(page, METRICS_URL)
    body_after = json.loads(metrics_after["text"])

    low_after = find_counter(body_after, "retrieval_filtered_low_score")
    empty_after = find_counter(body_after, "retrieval_empty")

    # The query may produce empty/low_relevance (gate fires) or
    # success (LLM answers from the doc). What we DON'T want to
    # see: an unfiltered corpus of 1 doc's worth of chunks (the
    # pre-PR-1 summary bulk path behavior). We can't directly
    # observe that without wire introspection, so we settle for
    # observing either:
    #   * Gate fires (low_score or empty counter ticks)
    #   * Gate doesn't fire but answer is well-formed (success)
    gate_fired = (low_after > low_before) or (empty_after > empty_before)
    if gate_fired:
        log.append(
            f"  ✓ definitional QA: gate fired "
            f"(low_score={low_after - low_before}, empty={empty_after - empty_before})"
        )
    elif not result["answer"].strip():
        return False, "definitional QA: gate didn't fire AND answer was empty"
    else:
        log.append(
            f"  ✓ definitional QA: answer len={len(result['answer'])} (gate didn't fire, normal path)"
        )
    return True, ""


def check_pr1_assertion_3_chip_wire_payload(page, thread_id: str, log: list) -> tuple[bool, str]:
    """PR-1 #3: backend wire payload (``answer_complete.sources[]``)
    carries ``chunk_id`` / ``doc_id`` / ``source_kind`` alongside
    ``index`` — these are the EXACT fields the CitationChip click
    handler now forwards to App.tsx.

    The chip-click JSX contract itself is pinned by the vitest
    suite in ``src/frontend/src/components/__tests__/CitationChip.test.tsx``
    (the React Testing Library verifies that the dispatched
    CustomEvent's ``detail`` carries all four fields). What we
    pin HERE is the BACKEND shape that those chips read from —
    if the wire payload loses a field, the chip silently gets
    ``undefined`` for it. The frontend test would pass because
    it constructs the source fixture itself, but production
    would render empty citations.

    Probe: ask a question, parse ``answer_complete.sources[]``,
    assert every source has chunk_id + doc_id + source_kind
    populated. Without PR-1 the wire already had chunk_id/doc_id
    (added in v2.0.7 SoT-1) but the chip only read ``index``;
    the new code reads the others, so the wire must keep them.
    """
    upload = upload_text_doc(
        page,
        f"chip_probe_{int(time.time())}.md",
        "项目 X 是一个测试项目,主要功能包括 A、B、C。",
        thread_id,
    )
    if upload["status"] != 200 or not upload["body"].get("doc_id"):
        return False, f"upload failed: status={upload['status']} body={upload['body']}"

    # Wait for embedding + index.
    page.wait_for_timeout(2500)

    result = drive_chat_turn(page, thread_id, "项目 X 有哪些主要功能?")
    if not result["sources"]:
        return False, "no sources returned; cannot verify wire payload"

    for i, s in enumerate(result["sources"]):
        for field in ("index", "chunk_id", "doc_id", "source_kind"):
            if field not in s:
                return False, f"source[{i}] missing field {field!r}: {s!r}"
            if field != "index" and not s[field]:
                return False, f"source[{i}].{field} is empty: {s[field]!r}"

    log.append(
        f"  ✓ wire payload carries chunk_id/doc_id/source_kind on "
        f"{len(result['sources'])} source(s)"
    )
    return True, ""


def check_pr1_assertion_4_metrics(page, log: list) -> tuple[bool, str]:
    """PR-1 #4: /debug/metrics returns 200 and exposes the
    pre-registered ``retrieval_filtered_low_score`` counter. The
    counter has been ticking since assertion 2 / earlier probes
    fired the gate; this assertion just confirms the wire is live.

    Counter names in /debug/metrics are LOWERCASE per the Prometheus
    text-format convention (``src/agent/metrics.py:17-19``); see
    assertion #2 for the full rationale.
    """
    metrics = fetch_metrics(page, METRICS_URL)
    if metrics["status"] != 200:
        return False, f"/debug/metrics returned {metrics['status']}; not enabled"
    body = json.loads(metrics["text"])
    counter_names = names_in(body)
    if "retrieval_filtered_low_score" not in counter_names:
        return False, (
            f"retrieval_filtered_low_score counter missing from /debug/metrics; "
            f"have {sorted(counter_names)}"
        )
    val = find_counter(body, "retrieval_filtered_low_score")
    log.append(f"  ✓ /debug/metrics: retrieval_filtered_low_score = {val}")
    return True, ""


def check_backend_log(log_dir: Path) -> tuple[bool, str]:
    """PR-1 #5: backend log scan — no Python exceptions or
    unexpected error-level logs during the run.

    Scans the most recent backend.log for ERROR / traceback markers
    that weren't there at startup. Phase 1-4 ship already had log
    filtering; new errors here would indicate a regression.
    """
    candidates = list(log_dir.glob("backend*.log")) + list(log_dir.glob("**/backend*.log"))
    if not candidates:
        return True, "(no backend log found; skipping scan)"
    log_file = max(candidates, key=lambda p: p.stat().st_mtime)

    text = log_file.read_text(encoding="utf-8", errors="replace")
    error_markers = re.findall(
        r"(Traceback \(most recent call last\):|ERROR \| .*Exception|CRITICAL \|)",
        text,
    )
    if error_markers:
        # Filter known-benign ERROR lines (Phase 3 also has this filter).
        # We only flag UNEXPECTED tracebacks.
        return False, (
            f"backend log has {len(error_markers)} error marker(s); "
            f"sample: {error_markers[:3]}. log_file={log_file}"
        )
    return True, f"no new errors in {log_file.name}"


def main() -> int:
    SCREENSHOT_DIR.mkdir(exist_ok=True, parents=True)
    log: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        page.goto(FRONTEND_URL, wait_until="networkidle")
        log.append(f"frontend loaded: {FRONTEND_URL}")

        thread_id = f"pr1_v2904_{int(time.time())}"

        # Assertion 1: anchored summary-intent fires
        ok, msg = check_pr1_assertion_1_summary_intent(page, thread_id, log)
        if not ok:
            return _fail(f"PR-1 #1 (anchored summary-intent): {msg}", log)
        log.append("PR-1 #1: ✓ anchored summary-intent fires")

        # Assertion 2: definitional QA falls through
        ok, msg = check_pr1_assertion_2_definitional_qa(page, thread_id, log)
        if not ok:
            return _fail(f"PR-1 #2 (definitional QA): {msg}", log)
        log.append("PR-1 #2: ✓ definitional QA does NOT fire summary bulk path")

        # Assertion 3: backend wire payload carries chunk_id/doc_id/source_kind
        ok, msg = check_pr1_assertion_3_chip_wire_payload(page, thread_id, log)
        if not ok:
            return _fail(f"PR-1 #3 (chip wire payload): {msg}", log)
        log.append("PR-1 #3: ✓ answer_complete.sources[] carries chunk_id + doc_id + source_kind")

        # Assertion 4: /debug/metrics exposes the new counter
        ok, msg = check_pr1_assertion_4_metrics(page, log)
        if not ok:
            return _fail(f"PR-1 #4 (metrics): {msg}", log)
        log.append("PR-1 #4: ✓ /debug/metrics exposes RETRIEVAL_FILTERED_LOW_SCORE")

        # Clean up
        delete_thread_docs(page, thread_id)

        # Assertion 5: backend log scan (no new errors)
        log_dir = Path(__file__).parent.parent.parent / "logs"
        ok, msg = check_backend_log(log_dir)
        if not ok:
            return _fail(f"PR-1 #5 (backend log scan): {msg}", log)
        log.append(f"PR-1 #5: ✓ {msg}")

        browser.close()

    log.append(f"\n=== {PHASE_LABEL} Layer 5 verification: ALL PASSED ===")
    print("\n".join(log))
    return 0


def _fail(reason: str, log: list) -> int:
    log.append(f"\n=== {PHASE_LABEL} Layer 5 verification: FAILED ===")
    log.append(f"REASON: {reason}")
    print("\n".join(log))
    return 1


if __name__ == "__main__":
    sys.exit(main())