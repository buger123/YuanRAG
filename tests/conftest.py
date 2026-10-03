"""Pytest fixtures.

Provides an isolated, temp-based user data directory for every test so that
no test ever touches the real ``~/.rag_assistant``.

Also resets all path-keyed singletons (``bge_m3``, ``bge_reranker``,
``lancedb_store``, ``history``, ``runner``) so that test N+1 doesn't see
test N's cached resources — this is the "context interference" failure
mode the audit specifically called out.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

# Promote warning categories that almost always indicate real bugs.
#  - RuntimeWarning: a missed ``await`` on an async function returns a
#    coroutine that gets silently consumed as truthy — that's a bug.
#  - DeprecationWarning: a deprecated langchain API call will break
#    in the next minor release; we'd rather fail loudly NOW.
#  - PendingDeprecationWarning: same as above but for an upcoming
#    removal — surfacing it gives us a chance to migrate before the
#    project hits a release where it becomes a hard error.
#  - SyntaxWarning: ambiguous syntax (e.g. missing comma in a tuple)
#    almost always surfaces a real bug.
# Tests that intentionally trip these warnings must filter locally
# with ``pytest.warns(...)`` or ``filterwarnings`` on the test itself.
warnings.filterwarnings("error", category=RuntimeWarning)
warnings.filterwarnings("error", category=DeprecationWarning)
warnings.filterwarnings("error", category=PendingDeprecationWarning)
warnings.filterwarnings("error", category=SyntaxWarning)


@pytest.fixture(autouse=True)
def _isolate_user_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect RAG_DATA_DIR to a temp dir for the duration of the test.

    Also resets every module-level singleton that caches state keyed by the
    resolved user-data path. Without these resets, the second test in a
    suite would silently reuse the first test's LanceDB connection /
    sqlite saver / compiled LangGraph / BGE-M3 model — a classic context-
    interference bug.
    """
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    # Reset cached settings so each test re-resolves against the new dir.
    from config.settings import reset_settings_cache

    reset_settings_cache()

    # Reset path-keyed singletons BEFORE the test runs, so any test that
    # opens them gets a fresh instance pointed at tmp_path.
    _reset_all_singletons()

    yield tmp_path

    # Teardown: reset again so any leftover references can't leak into the
    # next test. Best-effort: some modules may not have a reset_for_tests
    # if they predate the audit (e.g. agent runner's _graph — handled).
    _reset_all_singletons()
    reset_settings_cache()


def bypass_models_loaded_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``stream_agent`` skip the "models still loading" early-return gate.

    ``stream_agent`` (src/agent/runner.py:138) emits ``error`` + ``done``
    and returns immediately when ``models_loaded()`` (BGE-M3) or
    ``model_loaded()`` (BGE reranker) return False — which is always
    the case in the test environment because no real model is loaded.
    The gate short-circuits before the fake graph is even consulted,
    so tests that monkeypatch the graph but DON'T bypass this gate
    "pass" by accident (e.g. test_runner_cancellation's tests assert
    ``done`` is the last event, which the gate's own ``done`` yield
    satisfies — so the tests never actually exercise the code they
    claim to test).

    Tests that need to drive ``stream_agent`` past the gate must call
    this helper with their ``monkeypatch`` fixture before the graph
    fake is set up. Used by:
      - tests/test_runner_cancellation.py
      - tests/test_runner_stream_dedup.py
      - tests/test_v2_0_4_bugfixes.py

    Note: NOT autouse — the gate-firing behavior has its own dedicated
    test (``test_stream_agent_yields_error_when_models_not_loaded`` in
    test_models_loading.py) that must observe the gate firing. Marking
    this autouse would silently flip that test's expectation and break
    the regression it guards.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    monkeypatch.setattr(bge_m3, "models_loaded", lambda *_a, **_kw: True)
    monkeypatch.setattr(bge_reranker, "model_loaded", lambda *_a, **_kw: True)



def _reset_all_singletons() -> None:
    """Reset every known path-keyed singleton. Best-effort."""
    # Local imports so missing modules don't break collection.
    try:
        from src.embeddings import bge_m3

        bge_m3.reset_for_tests()
    except Exception:
        pass
    try:
        from src.reranker import bge_reranker

        bge_reranker.reset_for_tests()
    except Exception:
        pass
    try:
        from src.storage import lancedb_store

        lancedb_store.reset_for_tests()
    except Exception:
        pass
    try:
        from src.storage import checkpointer

        checkpointer.reset_for_tests()
    except Exception:
        pass
    try:
        from src.storage import doc_registry

        # Force-flush any pending state, then drop the in-memory cache so
        # the next test reads from the (reset) tmp_path's registry file.
        doc_registry.flush()
        doc_registry.reset_for_tests()
    except Exception:
        pass
    try:
        from src.agent import runner

        runner.reset_for_tests()
    except Exception:
        pass
    try:
        from src.retrieval import rerank as retrieval_rerank

        if hasattr(retrieval_rerank, "reset_for_tests"):
            retrieval_rerank.reset_for_tests()
    except Exception:
        pass
    try:
        from src.agent.nodes import retrieve as agent_retrieve

        if hasattr(agent_retrieve, "reset_for_tests"):
            agent_retrieve.reset_for_tests()
    except Exception:
        pass
    # The LLM client cache is keyed on (provider, model, temperature,
    # thinking, base_url, api_key). Tests that monkey-patch settings to
    # change any of those must clear the cache or the next test will
    # see the previous test's ChatAnthropic instance.
    try:
        from src.llm import factory

        factory.reset_llm_cache()
    except Exception:
        pass
    # v2.0.28.1 P1-P2 — clear the trace_id ContextVar so a test
    # never inherits a trace_id from a previous test's request (the
    # middleware would normally reset it in ``finally``, but tests
    # that exercise the ContextVar directly without going through
    # the middleware can leak a value across tests).
    try:
        from src.core import trace_id as trace_id_mod

        trace_id_mod.reset_for_tests()
    except Exception:
        pass
    # v2.0.28.6 P1-R4 / test audit — clear the web_search _FETCH_CACHE
    # + DDG per-test state. Without these resets, a test that exercises
    # ``_fetch_url_sync`` leaves entries in ``_FETCH_CACHE`` that the
    # next test will see (False TTL-cached "alive" results, wrong
    # content-type, etc.). ``ddg.reset_for_tests`` mirrors the
    # bge_m3 / bge_reranker pattern — drops the module-level engine
    # cache so a test that monkeypatches ``DDGS`` doesn't leak to the
    # next test.
    try:
        from src.web_search import fetch as web_search_fetch

        if hasattr(web_search_fetch, "_FETCH_CACHE"):
            web_search_fetch._FETCH_CACHE.clear()
    except Exception:
        pass
    try:
        from src.web_search import ddg

        if hasattr(ddg, "reset_for_tests"):
            ddg.reset_for_tests()
    except Exception:
        pass
    # v2.0.28.6 P1-3 — drop the keyring module-level store singletons
    # so a test that swaps ``config.KEYRING_SERVICE`` to a per-test
    # value doesn't keep getting the previous test's store instance.
    try:
        from src.security import keyring_store

        keyring_store.reset_for_tests()
    except Exception:
        pass


# v2.0.29.6 (Phase 5) — opt-in gate for `@pytest.mark.slow` tests.
#
# Tests marked ``slow`` are opt-in: by default they're skipped to keep
# the default `pytest` run cheap. Run with `pytest --run-slow` to
# execute them. Pattern mirrors pytest-asyncio's ``--asyncio-mode``
# style — a CLI flag toggles the gate at collection time.
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.slow (default: skip them)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.slow`` tests unless ``--run-slow`` was passed.

    ``slow`` tests do prompt-statistics + adversarial regex work but
    no real LLM calls. They're skipped by default to keep CI cheap;
    the flag is for pre-release / local dev runs.
    """
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test — pass --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def app_under_test(monkeypatch, tmp_path):
    """Build a FastAPI app bound to a temp ``RAG_DATA_DIR``.

    Used by every test that needs to hit the HTTP layer via ASGITransport.
    Redirects the user-data env var, drops the cached settings, and calls
    ``create_app()`` so each test gets an isolated app / sqlite / LanceDB
    instance pointed at ``tmp_path`` (the autouse ``_isolate_user_data_dir``
    already resets the relevant singletons — we still call
    ``reset_settings_cache`` because settings cache is keyed on env-var
    state, not the path).
    """
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    from config.settings import reset_settings_cache

    reset_settings_cache()
    # Importing here avoids module-level state pollution
    from src.app import create_app

    return create_app()
