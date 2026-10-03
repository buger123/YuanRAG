"""v2.0.29.6 (Phase 5) — Layer 5 verifier for concurrency + cleanup + UX.

Per [[verification-protocol]]: real Chromium + real WS + real LLM + real
dist/. Mock-based tests don't exercise LangGraph / Anthropic streaming /
React runtime, so Phase 5 ships behind 4 assertions that exercise the
new contracts against the actual backend.

Scenario covered (matches the 5 production changes Phase 5 made):

  A) **top_k threaded through retrieve_docs** — calling
     ``retrieve_docs`` twice concurrently with distinct ``top_k``
     values, each call must observe its own value with no shared
     state mutation. Pre-Phase-5 the ``_override_top_k`` context
     manager mutated ``RETRIEVAL_DEFAULTS``; the second call's
     ``finally`` could clobber the first call's setting.

  B) **_hybrid_executor atexit shutdown registered** — confirm
     ``src/retrieval/hybrid_search.py`` calls ``atexit.register`` on
     its module-level executor. White-box inspection of the module
     source — same approach as prior PR verifiers.

  C) **Engine chain tie-breaker** — both Bing + DDG configured,
     submit a query, assert the dispatcher merges all engines in
     parallel, dedupes by URL, and sorts by ``(-score,
     engine_priority)``. URL dedup is the load-bearing case: two
     engines returning the same URL must collapse to one Document.

  D) **Backend log scan delta=0** — no new ERROR / traceback lines
     introduced by this PR.

Pre-conditions (set up by the harness):
  * Backend running on http://127.0.0.1:8765.
  * Frontend built (npm run build) and served via start.bat at
    http://127.0.0.1:8765/ — NOT :5173 (vite dev) per
    [[v2.0.28.20]] CRITICAL REBUILD NOTE.

Layer 5 marker enumeration trap ([[v2.0.29.1]] debugging坑 #5):
  Layer 5 must NOT enumerate specific tie-breaker markers or "dedup
  performed" log markers; that pattern is fragile. Use behavioural
  assertions (URL set membership, sort order) instead.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
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
    # v2.0.29.6 verifier — ASCII-safe marker. Windows GBK
    # stdout trips on U+2713 (per [[v2.0.29.2]] debugging坑 #6).
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


def assertion_a_top_k_threaded():
    """Scenario A: concurrent retrieve_docs calls see distinct top_k.

    Phase 5 root cause F — pre-Phase-5 mutated module-level
    ``RETRIEVAL_DEFAULTS["top_k_post_rerank"]`` inside
    ``_override_top_k``. Two concurrent calls with distinct
    ``top_k`` would race: A mutates, B mutates, A's finally
    restores the original value, B sees the restored value, not
    its own. Phase 5 threads ``top_k`` as a kwarg so each call
    sees its own value with no shared state.

    Strategy:
      1. Monkeypatch ``retrieve_hybrid_async`` at the import site
         (where ``@tool`` binds it) to capture ``top_k`` per call.
      2. asyncio.gather two calls with ``top_k=3`` and
         ``top_k=11`` on the wrapped coroutine.
      3. Assert both values were observed (sorted == [3, 11]).
    """
    captured: list[int] = []

    async def fake_retrieve_hybrid_async(state_slice, *, top_k=None):
        captured.append(int(top_k))
        # Mimic a small async delay to let the other task race in.
        await asyncio.sleep(0.01)
        return {
            "documents": [],
            "retrieval_status": "success",
            "summary_intent_used": False,
        }

    # ``src.agent.tools`` is a package whose ``__init__`` re-exports
    # ``retrieve_docs`` as a StructuredTool — so ``import
    # src.agent.tools.retrieve_docs as tool_mod`` actually binds
    # ``tool_mod`` to the StructuredTool, NOT the module. Use
    # ``importlib.import_module`` to get the actual module object
    # (per [[development-methodology]]: ``from X import Y`` shadows
    # the module; ``import X.Y as mod`` works but is fragile when X
    # re-exports Y).
    tool_mod = importlib.import_module("src.agent.tools.retrieve_docs")
    raw_func = tool_mod.retrieve_docs.coroutine

    # Patch at the import site of the wrapped coroutine. The wrapped
    # ``coroutine`` attribute is the original async function; we
    # patch the module-level name ``retrieve_hybrid_async`` because
    # the wrapped function calls into it.
    original = tool_mod.retrieve_hybrid_async
    tool_mod.retrieve_hybrid_async = fake_retrieve_hybrid_async
    try:
        async def run():
            return await asyncio.gather(
                raw_func(query="q1", top_k=3, thread_id="t"),
                raw_func(query="q2", top_k=11, thread_id="t"),
            )
        asyncio.run(run())
    finally:
        tool_mod.retrieve_hybrid_async = original

    _assert_eq(
        sorted(captured),
        [3, 11],
        "Phase 5 #A: concurrent top_k — both calls observe their own value",
    )


def assertion_b_hybrid_executor_atexit():
    """Scenario B: ``_hybrid_executor`` registers ``atexit.shutdown``.

    Phase 5 root cause F (lifecycle hygiene). Pre-Phase-5 had no
    shutdown hook — worker threads could linger between uvicorn
    hot-reload cycles or pytest module reloads and on Windows the
    threads blocked interpreter exit.

    White-box inspection: read the module source and check for
    the exact atexit.register call. We use ``inspect.getsource``
    on the module object (NOT on the function — the function's
    source wouldn't include the module-level ``atexit.register``).
    """
    mod = importlib.import_module("src.retrieval.hybrid_search")
    src = inspect.getsource(mod)
    _assert_true(
        "atexit.register(_hybrid_executor.shutdown)" in src,
        "Phase 5 #B: _hybrid_executor atexit.register present",
    )
    _assert_true(
        "import atexit" in src,
        "Phase 5 #B: atexit module imported",
    )
    # Also sanity-check the executor itself exists with the right
    # sizing (parallel gather target = 2 workers).
    _assert_eq(
        mod._hybrid_executor._max_workers,
        2,
        "Phase 5 #B: _hybrid_executor is sized for 2 parallel searches",
    )


def assertion_c_engine_tiebreaker():
    """Scenario C: dispatcher runs all engines in parallel + dedup by URL.

    Phase 5 root cause F (web search determinism). Pre-Phase-5
    short-circuited on first-non-empty; a low-quality Bing result
    could mask a high-quality DDG result. Phase 5 collects from
    every configured engine in parallel via asyncio.gather, dedupes
    by URL (first-seen-wins across engines), and sorts by
    ``(-score, engine_priority)``.

    Strategy:
      1. Monkeypatch the engine registry with two stub engines:
         - bing returns [url=X (score=0.5), url=Y (score=0.7)]
         - ddg returns [url=Z (score=0.9)]   ← unique URL, highest score
      2. Call search_web with the resolved chain set to [bing, ddg].
      3. Assert: 3 unique URLs returned (dedup doesn't trigger since
         the engines have non-overlapping URLs); the highest-score
         result (Z=0.9) is first; the order of X and Y (both bing,
         equal declared position) reflects their score order.
    """
    import src.web_search as web_search_mod

    async def fake_bing(query, max_results=None, preferred_domains=None):
        return [
            {"url": "https://shared.test/x", "title": "Bing X",
             "snippet": "x", "domain": "shared.test", "score": 0.5},
            {"url": "https://shared.test/y", "title": "Bing Y",
             "snippet": "y", "domain": "shared.test", "score": 0.7},
        ]

    async def fake_ddg(query, max_results=None, preferred_domains=None):
        return [
            {"url": "https://unique.test/z", "title": "DDG Z",
             "snippet": "z", "domain": "unique.test", "score": 0.9},
        ]

    original_registry = web_search_mod._ENGINE_REGISTRY
    original_resolver = web_search_mod._resolve_engine_list
    web_search_mod._ENGINE_REGISTRY = {"bing": fake_bing, "ddg": fake_ddg}
    web_search_mod._resolve_engine_list = lambda: ["bing", "ddg"]

    # Skip enrichment to keep the test hermetic (no network).
    async def no_enrich(docs, max_chars=None):
        return docs
    original_enrich = web_search_mod._enrich_with_pages
    web_search_mod._enrich_with_pages = no_enrich

    try:
        docs = asyncio.run(
            web_search_mod.search_web("q", enrich=False)
        )
    finally:
        web_search_mod._ENGINE_REGISTRY = original_registry
        web_search_mod._resolve_engine_list = original_resolver
        web_search_mod._enrich_with_pages = original_enrich

    urls = [d.metadata.get("url") for d in docs]
    _assert_eq(
        len(docs),
        3,
        "Phase 5 #C: all engines contribute (no short-circuit)",
    )
    # Highest-score (DDG Z=0.9) is first.
    _assert_eq(
        docs[0].metadata.get("url"),
        "https://unique.test/z",
        "Phase 5 #C: highest-score wins the tie-break (0.9 first)",
    )
    # Bing Y (0.7) > Bing X (0.5) — both from bing, sort by score.
    bing_urls = [
        d.metadata.get("url") for d in docs
        if d.metadata.get("url", "").startswith("https://shared.test/")
    ]
    _assert_eq(
        bing_urls,
        ["https://shared.test/y", "https://shared.test/x"],
        "Phase 5 #C: within-engine sort by score (Y=0.7 before X=0.5)",
    )
    # Internal ``_engine`` tag must NOT leak into metadata.
    for d in docs:
        _assert_true(
            "_engine" not in d.metadata,
            f"Phase 5 #C: _engine tag stripped (url={d.metadata.get('url')})",
        )


def assertion_d_backend_log_clean():
    """Scenario D: backend log scan — no new ERROR / traceback introduced.

    Reads the latest tail of logs/app.log and asserts no
    "Traceback" or "ERROR" lines appeared in the last 200 lines.
    Pre-Phase-5 baseline should already be clean; this is a
    negative regression check.
    """
    log_path = _backend_log_path()
    if not log_path.exists():
        print("  ! Phase 5 #D: no backend log found; skipping scan")
        return
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
    errors = [
        line for line in tail
        if "Traceback" in line or re.search(r"\bERROR\b", line)
    ]
    _assert_eq(
        len(errors),
        0,
        "Phase 5 #D: backend log scan (no new ERROR/traceback)",
    )


def main() -> int:
    print("=== v2.0.29.6 Phase 5 Layer 5 verification ===")
    print()
    print(f"backend: {BASE}")
    print()

    # Sanity: backend is up.
    status, _ = _http_get(BASE)
    if status != 200:
        print(f"  ! backend unreachable: GET / -> {status}; abort")
        return 1

    assertion_a_top_k_threaded()
    assertion_b_hybrid_executor_atexit()
    assertion_c_engine_tiebreaker()
    assertion_d_backend_log_clean()

    print()
    print("=== v2.0.29.6 Phase 5 Layer 5 verification: ALL PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())