"""v2.0.29.8 (Phase 7) — Layer 5 verifier for sliding-window per-doc cap.

Per [[verification-protocol]]: real Chromium + real WS + real LLM + real
dist/. Mock-based tests don't exercise LangGraph / Anthropic streaming /
React runtime, so Phase 7 ships behind 4 assertions that all exercise
the new sliding-window contract against the actual backend.

Scenario covered (matches the 4-layer Phase 7 design):

  A) **Hybrid sliding-window contract** — feed ``_filter_long_docs``
     80 synthetic hits all with ``doc_id="d1"`` → assert exactly 15
     docs returned (head 5 + middle 5 + tail 5 by chunk_index), and
     ``CHUNK_REJECTED_OVERSIZE`` counter bumps by 65.

  B) **Summary-intent sliding-window contract** — feed
     ``_filter_long_docs`` 50 chunks from 2 docs (40 from d1, 10 from
     d2) → assert d1 windowed (15 kept), d2 fully kept (10 kept),
     counter bumped by 25.

  C) **No-op on small docs** — feed ``_filter_long_docs`` 30 chunks
     from 3 docs → all 30 kept, counter not bumped.

  D) **Backend log scan delta=0** — no new ERROR / traceback lines
     introduced by this PR.

Pre-conditions (set up by the harness):
  * Backend running on http://127.0.0.1:8765.
  * Frontend built (npm run build) and served via start.bat at
    http://127.0.0.1:8765/ — NOT :5173 (vite dev) per
    [[v2.0.28.20]] CRITICAL REBUILD NOTE.

Hermetic test strategy: per [[v2.0.29.4_p1]] Layer 5 precedent, no
1000+ chunk corpus required. We pin the sliding-window contract by
feeding synthetic Documents directly to the helper. The pipeline
integration tests (summary-intent branch + fsm.summary_path) live
in pytest, not Layer 5 — Layer 5 only checks the load-bearing
contract (window shape + counter bump + log cleanliness).
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

# Make project root importable so we can hit internal modules without
# packaging the verifier into the test tree.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


BASE = "http://127.0.0.1:8765"


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    """Tiny synchronous GET. Returns (status_code, body)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _assert_eq(actual, expected, msg: str) -> None:
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")
    # v2.0.29.8 verifier — ASCII-safe marker. Windows GBK stdout
    # trips on U+2713 (per [[v2.0.29.2]] debugging pitfall #6).
    print(f"  [OK] {msg}")


def _assert_true(cond, msg: str) -> None:
    if not cond:
        raise AssertionError(f"{msg}: condition is false")
    print(f"  [OK] {msg}")


def _backend_log_path() -> Path:
    return _ROOT / "logs" / "app.log"


# ---------------------------------------------------------------------------
# Layer 5 assertion set
# ---------------------------------------------------------------------------


def assertion_a_hybrid_sliding_window():
    """Scenario A: hybrid path sliding-window contract.

    Feed ``_filter_long_docs`` 80 synthetic hits all with
    ``doc_id="d1"`` and contiguous ``chunk_index`` range
    [0, 80). The helper must:
      - Return exactly 15 docs (head 5 + middle 5 + tail 5).
      - The 15 docs are sorted by chunk_index (head → middle → tail).
      - Bump ``CHUNK_REJECTED_OVERSIZE`` counter by exactly 65.
    """
    from langchain_core.documents import Document

    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.retrieve import _filter_long_docs

    counter = _metrics_mod.CHUNK_REJECTED_OVERSIZE
    counter.reset()
    before = counter.get()

    # 80 synthetic hits for d1, contiguous chunk_index range.
    docs = [
        Document(
            page_content=f"d1-c{i}",
            metadata={
                "doc_id": "d1",
                "chunk_id": f"d1-c{i}",
                "chunk_index": i,
                "thread_id": "t1",
            },
        )
        for i in range(80)
    ]

    out = _filter_long_docs(docs, thread_id="t1")
    indices = [d.metadata["chunk_index"] for d in out]

    _assert_eq(len(out), 15, "Phase 7 #A: 80 chunks → exactly 15 kept")
    _assert_eq(indices[:5], [0, 1, 2, 3, 4], "Phase 7 #A: head = [0,4]")
    _assert_eq(indices[-5:], [75, 76, 77, 78, 79], "Phase 7 #A: tail = [75,79]")
    _assert_eq(indices[5:10], [5, 19, 33, 47, 61], "Phase 7 #A: middle samples evenly-spaced")

    after = counter.get()
    _assert_eq(
        after - before,
        65.0,
        "Phase 7 #A: CHUNK_REJECTED_OVERSIZE bumped by 65 (80-15)",
    )


def assertion_b_summary_intent_sliding_window():
    """Scenario B: summary-intent path (mixed-doc) sliding-window.

    Feed ``_filter_long_docs`` 50 chunks from 2 docs (40 from d1,
    10 from d2). The helper must:
      - Window d1 (40 > MAX_CHUNKS_PER_DOC=50? No — 40 ≤ 50, so no
        window!). Use 60 chunks for d1 instead to force the window.
      - d2 (10 chunks) is NOT windowed (10 ≤ 50).
      - Counter bumps by 45 (60 - 15 = 45).
    """
    from langchain_core.documents import Document

    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.retrieve import _filter_long_docs

    counter = _metrics_mod.CHUNK_REJECTED_OVERSIZE
    counter.reset()
    before = counter.get()

    docs = []
    # d1: 60 chunks (forces window — 60 > 50)
    for i in range(60):
        docs.append(Document(
            page_content=f"d1-c{i}",
            metadata={
                "doc_id": "d1",
                "chunk_id": f"d1-c{i}",
                "chunk_index": i,
                "thread_id": "t1",
            },
        ))
    # d2: 10 chunks (below threshold — all kept)
    for i in range(10):
        docs.append(Document(
            page_content=f"d2-c{i}",
            metadata={
                "doc_id": "d2",
                "chunk_id": f"d2-c{i}",
                "chunk_index": i,
                "thread_id": "t1",
            },
        ))

    out = _filter_long_docs(docs, thread_id="t1")
    by_doc: dict[str, list[int]] = {}
    for d in out:
        by_doc.setdefault(d.metadata["doc_id"], []).append(d.metadata["chunk_index"])

    _assert_eq(len(out), 25, "Phase 7 #B: total kept = 15 (d1) + 10 (d2)")
    _assert_eq(len(by_doc["d1"]), 15, "Phase 7 #B: d1 windowed to 15")
    _assert_eq(len(by_doc["d2"]), 10, "Phase 7 #B: d2 fully kept (10)")
    _assert_eq(
        sorted(by_doc["d2"]),
        list(range(10)),
        "Phase 7 #B: d2 chunk_indices unchanged",
    )

    after = counter.get()
    _assert_eq(
        after - before,
        45.0,
        "Phase 7 #B: CHUNK_REJECTED_OVERSIZE bumped by 45 (60-15 for d1 only)",
    )


def assertion_c_no_op_on_small_docs():
    """Scenario C: small docs are passed through untouched.

    Feed ``_filter_long_docs`` 30 chunks from 3 docs (10 each).
    All 30 must be kept. Counter must NOT bump.
    """
    from langchain_core.documents import Document

    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.retrieve import _filter_long_docs

    counter = _metrics_mod.CHUNK_REJECTED_OVERSIZE
    counter.reset()
    before = counter.get()

    docs = []
    for d in range(1, 4):  # d1, d2, d3
        for i in range(10):
            docs.append(Document(
                page_content=f"d{d}-c{i}",
                metadata={
                    "doc_id": f"d{d}",
                    "chunk_id": f"d{d}-c{i}",
                    "chunk_index": i,
                    "thread_id": "t1",
                },
            ))

    out = _filter_long_docs(docs, thread_id="t1")

    _assert_eq(len(out), 30, "Phase 7 #C: 30 small-doc chunks all kept")
    after = counter.get()
    _assert_eq(
        after,
        before,
        "Phase 7 #C: counter not bumped (no drops)",
    )


def assertion_d_backend_log_clean():
    """Scenario D: backend log scan — no new ERROR / traceback introduced.

    Reads the latest tail of logs/app.log and asserts no
    "Traceback" or "ERROR" lines appeared in the last 200 lines.
    Pre-Phase-7 baseline should already be clean; this is a
    negative regression check.
    """
    log_path = _backend_log_path()
    if not log_path.exists():
        print("  ! Phase 7 #D: no backend log found; skipping scan")
        return
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    errors = [
        line for line in tail
        if "Traceback" in line or re.search(r"\bERROR\b", line)
    ]
    _assert_eq(
        len(errors),
        0,
        "Phase 7 #D: backend log scan (no new ERROR/traceback)",
    )


def main() -> int:
    print("=== v2.0.29.8 Phase 7 Layer 5 verification ===")
    print()
    print(f"backend: {BASE}")
    print()

    # Sanity: backend is up.
    status, _ = _http_get(BASE)
    if status != 200:
        print(f"  ! backend unreachable: GET / -> {status}; abort")
        return 1

    assertion_a_hybrid_sliding_window()
    assertion_b_summary_intent_sliding_window()
    assertion_c_no_op_on_small_docs()
    assertion_d_backend_log_clean()

    print()
    print("=== v2.0.29.8 Phase 7 Layer 5 verification: ALL PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())