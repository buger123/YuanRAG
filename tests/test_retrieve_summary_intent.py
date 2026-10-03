"""Tests for v2.0.28 PR-1 P1-P1 list_documents cache flip.

Pre-PR-1 — the summary-intent hot path in ``src/agent/nodes/retrieve.py:143``
called uncached ``list_documents(thread_id=...)`` on every turn where the
query matched the summary-intent regex. The chunk-list call right after
(line 158) was already cached via ``list_chunks_by_thread_cached``. The
two read sites in the same branch were inconsistent: one cached, one not.

Post-PR-1 — both read sites use cached wrappers. The 5-second TTL on
``list_documents_cached`` is bounded by:

- ``add_chunks`` invalidates the target thread (line 294 of
  ``lancedb_store.py``) — uploading new chunks drops the cache
- ``delete_by_doc_id`` clears the whole cache (line 325) — doc delete
  drops all caches (don't know which thread the doc belonged to)
- ``delete_by_thread_id`` invalidates the target thread
- ``reset_for_tests`` clears the whole cache (test isolation)

So the worst-case scenario is: a user uploads a doc → cache cleared →
summary-intent query 5+ seconds later. The summary-intent regex is
checked by the cheap ``_looks_like_summary_intent`` predicate first; if
no summary-intent match, the cached call is skipped entirely.

The 2 tests below pin:

1. The summary-intent branch in retrieve.py uses the cached wrapper
   (regression guard: a future "optimize" that flips it back to
   uncached would surface here).
2. The ``GET /documents`` wire surface (which powers the sidebar) is
   NOT cached — sidebar users expect delete to be immediately visible
   (no 5s TTL window hiding the deletion).

Note: the GET /documents test is a contract pin, not a behavioral test
of the wire endpoint's HTTP path. It verifies that the ``GET
/documents`` handler in ``src/api/routes/documents.py`` keeps calling
the uncached ``list_documents`` (not the cached ``list_documents_cached``).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. summary-intent branch uses list_documents_cached (the PR-1 fix)
# ---------------------------------------------------------------------------


def test_retrieve_summary_intent_uses_cached_list_documents(monkeypatch):
    """The summary-intent hot path in ``retrieve.py:143`` must call
    ``list_documents_cached`` (not ``list_documents``) per v2.0.28 PR-1.

    Pre-PR-1 — uncached, per-turn full LanceDB scan.
    Post-PR-1 — cached wrapper, 5s TTL + invalidation on add/delete.

    ⚠️ Monkeypatch import-site trap (mirrors the pattern in
    `tests/test_checkpointer_corruption_contract.py` from v2.0.27.2):
    ``retrieve.py`` does ``from src.storage.lancedb_store import
    list_documents_cached`` at module top — that creates a LOCAL
    binding. Patching ``lancedb_store.list_documents_cached`` patches
    the source definition; retrieve.py's reference is unaffected. We
    must patch the USE site: ``src.agent.nodes.retrieve.list_documents_cached``.

    We also stub the chunk path (``list_chunks_by_thread_cached``) so
    retrieve doesn't blow up on the next read after the summary-intent
    branch fires.
    """
    # Stub object returned by both reads.
    fake_doc = {
        "doc_id": "d1",
        "filename": "paper.md",
        "source_type": "file",
        "source_path": "d1/paper.md",
        "chunk_count": 3,
        "ingested_at": "2026-09-21T00:00:00Z",
        "thread_id": "t1",
    }

    # Patch the USE site (where retrieve.py bound the name).
    from src.agent.nodes import retrieve as retrieve_mod

    cached_spy = MagicMock(side_effect=lambda thread_id=None: [fake_doc])
    monkeypatch.setattr(retrieve_mod, "list_documents_cached", cached_spy)

    # Stub the chunk-side cache too (it follows the summary-intent path).
    monkeypatch.setattr(
        retrieve_mod, "list_chunks_by_thread_cached", lambda thread_id=None: []
    )

    # Run the summary-intent branch with a matching query.
    # v2.0.29.4 (Phase 4 PR-1) — the regex is anchored at start
    # (``^(?:...)``), so the query must START with a summary verb
    # for the branch to fire. ``"总结一下这份文档"`` starts with 总结;
    # the OLD ``"请帮我总结..."`` would no longer match because it
    # starts with 请.
    fake_state = {
        "current_query": "总结一下这份文档",
        "original_query": "总结一下这份文档",
        "thread_id": "t1",
        "messages": [],
        "step_count": 1,
    }

    asyncio.run(retrieve_mod.retrieve_hybrid_async(fake_state))

    # PR-1 contract: cached wrapper was invoked.
    assert cached_spy.call_count >= 1, (
        "summary-intent branch must call list_documents_cached (v2.0.28 PR-1). "
        "If this fails, retrieve.py was likely reverted to uncached list_documents."
    )


# ---------------------------------------------------------------------------
# 2. GET /documents wire surface stays uncached (sidebar invariant)
# ---------------------------------------------------------------------------


def test_get_documents_endpoint_does_not_use_cache(monkeypatch):
    """The ``GET /documents`` HTTP handler must keep calling the
    uncached ``list_documents`` (not ``list_documents_cached``).

    Pre-PR-1 — already uncached. PR-1 doesn't change this. The test
    pins the invariant so a future "optimization" that flips it to
    cached doesn't slip through.

    Why the pin matters: the sidebar shows the user-facing document
    list, and a user clicking delete expects the doc to disappear
    immediately. A 5-second TTL would show the deleted doc for up to
    5 seconds — terrible UX. The summary-intent branch can tolerate
    5s staleness because it's a routing hint; the sidebar cannot.

    ⚠️ Same monkeypatch import-site trap as test #1: documents.py
    does ``from src.storage.lancedb_store import list_documents`` at
    module top. We must patch the USE site: ``src.api.routes.documents.list_documents``.
    """
    # Stub object with all required DocumentSummary fields.
    fake_doc = {
        "doc_id": "d1",
        "filename": "paper.md",
        "source_type": "file",
        "source_path": "d1/paper.md",
        "chunk_count": 3,
        "ingested_at": "2026-09-21T00:00:00Z",
        "thread_id": "t1",
    }

    # Patch the USE site (where documents.py bound the name).
    from src.api.routes import documents as docs_mod

    uncached_spy = MagicMock(return_value=[fake_doc])
    cached_spy = MagicMock(return_value=[fake_doc])

    monkeypatch.setattr(docs_mod, "list_documents", uncached_spy)
    # ``list_documents_cached`` may not be imported in documents.py
    # today — that's actually the desired state (handler uses uncached).
    # Attach a spy on the use-site so we can detect any future accidental
    # import that would slip past the next pin.
    #
    # `raising=False` is required because the attribute doesn't exist yet
    # on the module — ``monkeypatch.setattr`` defaults to raising
    # AttributeError for missing attrs (which is the right default for
    # protecting against typos, but not what we want here).
    monkeypatch.setattr(
        docs_mod, "list_documents_cached", cached_spy, raising=False
    )

    # Run the handler. FastAPI handler is ``async def``; calling it as
    # a regular coroutine works because ``list_documents`` is mocked.
    result = asyncio.run(docs_mod.docs(thread_id="t1"))

    # Pin 1: the uncached version was called exactly once.
    assert uncached_spy.call_count == 1, (
        "GET /documents handler must call list_documents (uncached). "
        "If this fails, documents.py was flipped to the cached wrapper — "
        "sidebar delete would appear with 5s delay."
    )

    # Pin 2: the cached wrapper was NEVER called from this path.
    assert cached_spy.call_count == 0, (
        "GET /documents handler must NOT call list_documents_cached. "
        "Cached wrapper is for summary-intent routing hint only; "
        "sidebar wire surface must stay fresh. "
        "If you intentionally want to add cache here, also update Layer 5 PR-1 "
        "assertion that documents delete is immediately visible."
    )

    # Pin 3: result contains the stubbed doc (sanity).
    assert len(result) == 1
    assert result[0].doc_id == "d1"


# ---------------------------------------------------------------------------
# 3. v2.0.29.4 (Phase 4 PR-1) — summary-intent regex tightened.
#
# Pre-PR-1 the regex included broad definitional patterns (``是什么`` /
# ``有哪些`` / ``主要内容``) and was unanchored, so:
#
#   * Definitional QA ("X 是什么", "X 有哪些") fired the summary path
#     and dumped every chunk into the LLM context (waste + known
#     hallucination amplifier — the cheap grader used by the summary
#     branch overflows for 10+ chunks and fail-opens the set anyway).
#   * Mid-sentence mentions ("这个文档的总结是 X") falsely matched.
#
# Post-PR-1 the regex is anchored at start (``^(?:...)``) and those
# broad definitional patterns were removed. Definitional QA must go
# through the normal embed → hybrid → rerank path.
# ---------------------------------------------------------------------------


def test_summary_intent_re_matches_summary_verbs():
    """``总结一下`` should fire the summary path. Anchored at start
    means only queries that BEGIN with a summary verb.

    Pre-PR-1 the regex also matched English patterns unanchored;
    post-PR-1 the regex is Chinese-only and start-anchored. English
    mid-sentence mentions like "what is the summary" no longer fire
    — that's the intended false-positive reduction.
    """
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    assert _looks_like_summary_intent("总结一下这份文档") is True
    assert _looks_like_summary_intent("总结报告") is True
    assert _looks_like_summary_intent("摘要这份文档") is True


def test_summary_intent_re_skips_definitional_qa():
    """Counterpart: definitional QA should NOT fire the summary path.
    Pre-PR-1 these were the worst false-positive — definitional QA
    is exactly when we want a tight top-K, not "dump every chunk."
    """
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    for q in [
        "光合作用是什么",
        "Python 有哪些主要特点",
        "What is photosynthesis",
        "what are the main features of Python",
    ]:
        assert _looks_like_summary_intent(q) is False, (
            f"definitional QA must not fire summary path; got match on {q!r}"
        )


def test_summary_intent_re_skips_mid_sentence_summary():
    """Anchored at start: mid-sentence ``总结`` mentions don't fire."""
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    for q in [
        "你说一下这份报告里的总结怎么写",
        "what is the summary of this chapter",
    ]:
        assert _looks_like_summary_intent(q) is False, (
            f"mid-sentence 'summary' must not fire summary path; got match on {q!r}"
        )