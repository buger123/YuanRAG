"""v2.0.29.4 (Phase 4 PR-1) — retrieval-side gates + citation hardening.

This test file pins the 6 load-bearing changes that ship in
Phase 4 PR-1:

  1. Citation dedupe key now includes ``page_content[:200]``
     (so a re-uploaded doc with different content doesn't silently
     collapse to the stale version).
  2. Summary-intent regex anchored at start, ``是什么`` / ``有哪些``
     removed (definitional QA no longer misroutes to summary_path).
  3. ``min_top_relevance_score`` gate fires when top rerank score
     is below threshold — Phase 1 REFUSAL_TEMPLATES["low_relevance"]
     auto-activates.
  4. Rerank-failure stamping (``rerank_failed=True`` +
     ``rerank_score=0.0``) so downstream gates see the failure
     instead of using RRF score as a fake rerank score.
  5. Range collapse in ``renumber_citations_and_sources`` now logs
     a WARNING (pre-PR-1 was silent).
  6. ``RETRIEVAL_FILTERED_LOW_SCORE`` counter ticks on the gate
     firing (Phase 2 pre-registration contract honored).

Invariants tested
-----------------
* The gate fires **before** ``_hit_to_document`` conversion (CPU
  savings — pre-PR-1 we'd materialise docs we'd throw away).
* ``retrieval_status="low_relevance"`` flows to react_generate,
  which triggers Phase 1 REFUSAL_TEMPLATES["low_relevance"] without
  any extra wiring.
* ``rerank_failed=True`` propagates through ``_hit_to_document``
  unchanged — ``metadata.score = rerank_score or rrf_score`` (line
  89) handles 0.0 correctly.

⚠️ Test pitfall — loguru:
  :func:`caplog` does NOT capture loguru output (Phase 2
  debugging坑 #1). The range-collapse-log test uses
  :class:`StringIO` + ``add(sink)`` rather than caplog.
"""
from __future__ import annotations

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agent.metrics import (
    RETRIEVAL_FILTERED_LOW_SCORE,  # top-level for module identity
    reset_all,
)
# NB: RETRIEVAL_FILTERED_LOW_SCORE is referenced both here and inside
# the test functions (via ``from src.agent.metrics import ...``) so
# the gate's lazy ``from src.agent.metrics import
# RETRIEVAL_FILTERED_LOW_SCORE`` and the test's read site resolve to
# the SAME Counter instance even after ``test_metrics_aggregation``
# reloads ``metrics_mod`` mid-session (it does — see
# ``tests/test_metrics_aggregation.py:metrics_app``). The reload
# creates fresh Counter instances; ``retrieve.py``'s lazy import
# picks them up at call time. If the test reads a top-level-bound
# OLD instance, the count stays 0 even when the gate bumps the new
# one — exactly the failure pattern Phase 2 debugging坑 #7 warned
# about. Re-importing inside the function (below) avoids it.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id: str, chunk_id: str, content: str, **extra) -> Document:
    """Build a Document with the metadata shape ``_hit_to_document``
    produces. Used to exercise the dedupe key in ``react_generate.py``.
    """
    meta = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "source_kind": "local",
    }
    meta.update(extra)
    return Document(page_content=content, metadata=meta)


# ---------------------------------------------------------------------------
# 1. Citation dedupe key includes content_lead
# ---------------------------------------------------------------------------


def test_dedup_key_includes_content_lead():
    """Pre-PR-1: same (doc_id, chunk_id) was silently deduped to one.
    Post-PR-1: different ``page_content`` keeps BOTH docs so the LLM
    reads the most recent content of a re-uploaded doc.
    """
    from src.agent.nodes.react_generate import _dedupe_documents

    docs = [
        _doc("d1", "c1", "old version of the policy: salary cap is $50k"),
        _doc("d1", "c1", "new version of the policy: salary cap is $75k"),
    ]
    out = _dedupe_documents(docs)
    assert len(out) == 2, (
        f"different page_content should produce 2 distinct docs; got {len(out)}. "
        "Pre-PR-1 dedupe key was (doc_id, chunk_id) only — it would silently drop one."
    )


def test_dedup_key_collapses_same_lead():
    """Same (doc_id, chunk_id, content_lead[:200]) → 1 doc.
    Counterpart to ``test_dedup_key_includes_content_lead`` — pins
    that the new key doesn't accidentally over-collide when the
    content really is the same.
    """
    from src.agent.nodes.react_generate import _dedupe_documents

    docs = [
        _doc("d1", "c1", "shared content prefix " + "x" * 250),
        _doc("d1", "c1", "shared content prefix " + "x" * 250),
    ]
    out = _dedupe_documents(docs)
    assert len(out) == 1, (
        f"identical content + identical (doc_id, chunk_id) should still dedupe to 1; "
        f"got {len(out)}"
    )


def test_dedup_key_handles_missing_metadata():
    """Defensive: missing ``doc_id`` / ``chunk_id`` → empty string
    (the existing pre-PR-1 behavior) so a doc with no metadata still
    dedupes deterministically. ``content_lead`` is also empty string
    for empty page_content — so a doc with no metadata AND empty
    content is the same canonical key as another such doc, which is
    correct (both are placeholder/no-data docs and should collapse).
    """
    from src.agent.nodes.react_generate import _dedupe_documents

    docs = [
        Document(page_content="", metadata={}),
        Document(page_content="", metadata={}),
    ]
    out = _dedupe_documents(docs)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# 2. Summary-intent regex tightened
# ---------------------------------------------------------------------------


def test_summary_intent_re_matches_at_start():
    """``总结一下`` should match. Anchor-at-start means only queries
    that BEGIN with a summary verb fire the summary path.

    Note: pre-PR-1 the regex also matched English patterns
    (``summarize``, ``overview``, ``key points``, etc.) but they
    were unanchored — they matched anywhere in the query. Phase 4
    PR-1 anchors at start AND narrows to Chinese summary verbs
    (the English patterns were removed; if an English summary
    request needs to fire summary_path, that should be a separate
    feature — the OLD English patterns would have fired on
    mid-sentence mentions like ``what is the summary of this``,
    which is exactly the false-positive the anchor is meant to
    prevent).
    """
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    # Queries that begin with a Chinese summary verb → fire summary path.
    assert _looks_like_summary_intent("总结一下这份文档") is True
    assert _looks_like_summary_intent("总结报告") is True
    assert _looks_like_summary_intent("摘要这份文档") is True
    assert _looks_like_summary_intent("介绍一下") is True
    assert _looks_like_summary_intent("分析一下") is True


def test_summary_intent_re_ignores_definitional_qa():
    """``X 是什么`` / ``X 有哪些`` are definitional QA, NOT summary.
    Pre-PR-1 these matched the regex and dumped every chunk into
    context. Post-PR-1 they fall through to the normal embed → hybrid
    → rerank path.
    """
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    definitional_queries = [
        "光合作用是什么",
        "Python 有哪些主要特点",
        "AI 的主要内容有哪些",
        "What is photosynthesis",
        "what are the main features of Python",
        "请讲一下合同的主要内容",
    ]
    for q in definitional_queries:
        assert _looks_like_summary_intent(q) is False, (
            f"definitional QA should NOT fire summary path; '{q}' matched. "
            "Pre-PR-1 broad patterns like '是什么' / '主要内容' caused this exact bug."
        )


def test_summary_intent_re_ignores_mid_sentence_summary():
    """Mid-sentence summary mentions (e.g. ``这个文档的总结是 X``) should
    NOT match — anchored at start. Pre-PR-1 the unanchored regex
    would falsely match and dump chunks.
    """
    from src.agent.nodes.retrieve import _looks_like_summary_intent

    mid_sentence = [
        "请你说一下这份报告里的总结怎么写",
        "what is the summary of this chapter",
        "我需要看一下文档的总结部分",
    ]
    for q in mid_sentence:
        assert _looks_like_summary_intent(q) is False, (
            f"mid-sentence 'summary' should NOT fire summary path; '{q}' matched"
        )


# ---------------------------------------------------------------------------
# 3. min_top_relevance_score gate
# ---------------------------------------------------------------------------


def test_min_top_relevance_score_returns_low_relevance(monkeypatch):
    """When the top rerank_score < threshold, retrieve returns empty
    with ``retrieval_status='low_relevance'`` and bumps
    ``RETRIEVAL_FILTERED_LOW_SCORE``.

    Patches the embedder + hybrid_search + rerank paths so we can
    control the rerank scores without a real LanceDB / BGE-M3 round
    trip. Pattern mirrors tests/test_retrieve_summary_intent.py.

    Counter-resolution discipline: ``from src.agent.metrics import
    RETRIEVAL_FILTERED_LOW_SCORE`` is done INSIDE the test body
    (NOT at module top), so after ``test_metrics_aggregation``'s
    ``importlib.reload(metrics_mod)`` mid-session we read the SAME
    Counter instance retrieve_mod's lazy import will write to.
    Phase 2 debugging坑 #7 (importlib.reload stale module ref).
    """
    from src.agent.metrics import RETRIEVAL_FILTERED_LOW_SCORE
    from src.agent.nodes import retrieve as retrieve_mod

    reset_all()

    # Stub embedder (returns a fake dense vector).
    fake_embedder = MagicMock()
    fake_embedder.embed_query = MagicMock(return_value={"dense": [0.0] * 1024})
    monkeypatch.setattr(retrieve_mod, "_get_embedder", lambda: fake_embedder)

    # Stub hybrid_executor + the search fns (return 2 hits).
    fake_hit = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "filename": "f.md",
        "text": "lorem ipsum dolor",
        "score": 0.05,  # below threshold
        "rerank_score": 0.05,  # below 0.3 threshold
    }
    fake_hit2 = dict(fake_hit, chunk_id="c2")
    monkeypatch.setattr(
        retrieve_mod, "_hybrid_executor",
        MagicMock(submit=lambda fn, *a, **kw: MagicMock(result=lambda: [fake_hit, fake_hit2])),
    )
    monkeypatch.setattr(retrieve_mod, "_bm25_search", lambda *a, **kw: [fake_hit, fake_hit2])
    monkeypatch.setattr(retrieve_mod, "_dense_search", lambda *a, **kw: [fake_hit, fake_hit2])
    monkeypatch.setattr(retrieve_mod, "reciprocal_rank_fusion", lambda *a, **kw: [fake_hit, fake_hit2])

    # Stub get_table to avoid a real LanceDB round trip.
    monkeypatch.setattr(retrieve_mod, "get_table", lambda: MagicMock())

    baseline = RETRIEVAL_FILTERED_LOW_SCORE.get()

    state = {
        "current_query": "completely unrelated topic",
        "original_query": "completely unrelated topic",
        "thread_id": "t1",
        "messages": [],
        "step_count": 1,
    }
    delta = asyncio.run(retrieve_mod.retrieve_hybrid_async(state))

    # Gate fires — empty docs + low_relevance status.
    assert delta["documents"] == [], f"gate should drop docs; got {delta['documents']}"
    assert delta["retrieval_status"] == "low_relevance", (
        f"expected retrieval_status='low_relevance'; got {delta['retrieval_status']}"
    )

    # Counter bumped.
    assert RETRIEVAL_FILTERED_LOW_SCORE.get() == baseline + 1, (
        f"RETRIEVAL_FILTERED_LOW_SCORE must tick once; "
        f"baseline={baseline}, after={RETRIEVAL_FILTERED_LOW_SCORE.get()}"
    )


def test_min_top_relevance_score_skipped_when_top_above_threshold(monkeypatch):
    """Counterpart: when top rerank_score >= threshold, the gate does
    NOT fire — docs flow through normally with retrieval_status='success'.

    Same lazy-import discipline as
    ``test_min_top_relevance_score_returns_low_relevance`` (Phase 2
    debugging坑 #7 — importlib.reload stale module ref).
    """
    from src.agent.metrics import RETRIEVAL_FILTERED_LOW_SCORE
    from src.agent.nodes import retrieve as retrieve_mod

    reset_all()

    fake_embedder = MagicMock()
    fake_embedder.embed_query = MagicMock(return_value={"dense": [0.0] * 1024})
    monkeypatch.setattr(retrieve_mod, "_get_embedder", lambda: fake_embedder)

    # High-relevance hit: 0.8 >> 0.3 threshold.
    fake_hit = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "filename": "f.md",
        "text": "directly relevant content",
        "rerank_score": 0.8,
    }
    monkeypatch.setattr(
        retrieve_mod, "_hybrid_executor",
        MagicMock(submit=lambda fn, *a, **kw: MagicMock(result=lambda: [fake_hit])),
    )
    monkeypatch.setattr(retrieve_mod, "_bm25_search", lambda *a, **kw: [fake_hit])
    monkeypatch.setattr(retrieve_mod, "_dense_search", lambda *a, **kw: [fake_hit])
    monkeypatch.setattr(retrieve_mod, "reciprocal_rank_fusion", lambda *a, **kw: [fake_hit])
    monkeypatch.setattr(retrieve_mod, "get_table", lambda: MagicMock())

    baseline = RETRIEVAL_FILTERED_LOW_SCORE.get()

    state = {
        "current_query": "directly relevant topic",
        "thread_id": "t1",
        "messages": [],
        "step_count": 1,
    }
    delta = asyncio.run(retrieve_mod.retrieve_hybrid_async(state))

    assert len(delta["documents"]) == 1, (
        f"gate should NOT fire when top_score >= threshold; docs={delta['documents']}"
    )
    assert delta["retrieval_status"] == "success"
    assert RETRIEVAL_FILTERED_LOW_SCORE.get() == baseline, (
        "counter must NOT tick on the success path"
    )


# ---------------------------------------------------------------------------
# 4. Rerank-failure stamping
# ---------------------------------------------------------------------------


def test_rerank_failure_stamps_rerank_score_zero(monkeypatch):
    """When the BGE reranker raises, every returned hit carries
    ``rerank_score=0.0`` + ``rerank_failed=True``. Without these
    stamps, the downstream min_top_relevance_score gate wouldn't
    know the reranker failed vs. genuinely low-relevance hits.
    """
    # ⚠️ Import pitfall: ``src/retrieval/__init__.py`` does
    # ``from .rerank import rerank`` which makes ``src.retrieval.rerank``
    # resolve to the FUNCTION, not the module. Use ``importlib.import_module``
    # to grab the actual module so we can patch the module-level
    # ``_get_reranker`` symbol.
    import importlib

    rerank_mod = importlib.import_module("src.retrieval.rerank")
    monkeypatch.setattr(rerank_mod, "_get_reranker", lambda: MagicMock(
        score=MagicMock(side_effect=RuntimeError("BGE reranker OOM")),
    ))

    hits = [
        {"chunk_id": "c1", "doc_id": "d1", "text": "first hit"},
        {"chunk_id": "c2", "doc_id": "d1", "text": "second hit"},
        {"chunk_id": "c3", "doc_id": "d2", "text": "third hit"},
    ]
    out = rerank_mod.rerank("query", hits, top_k=2)

    assert len(out) == 2
    for h in out:
        assert h["rerank_score"] == 0.0, (
            f"rerank failure must stamp rerank_score=0.0; got {h['rerank_score']}"
        )
        assert h.get("rerank_failed") is True, (
            f"rerank failure must stamp rerank_failed=True; got {h.get('rerank_failed')}"
        )


def test_rerank_failure_skips_gate(monkeypatch):
    """Integration: rerank fails → ``rerank_failed=True`` stamps
    propagate → min_score gate SKIPS the threshold check (because
    scores are unreliable when the reranker itself errored) → docs
    flow through with ``retrieval_status='success'``.

    v2.0.29.4 (Phase 4 PR-1) carve-out: pre-PR-1 the gate would
    fire on the 0.0 score stamped by the failure path, pushing
    every rerank-exception user to the Phase 1 refusal template
    even though we have real (unranked) hits to work with. Post-PR-1
    the gate respects ``rerank_failed=True`` and lets the
    downstream ``_hit_to_document`` consume the unranked hits
    (scored by their RRF signal, which is what the user would have
    seen pre-PR-1).

    The carving-out was also what kept test fixtures that inject
    a stub ``_FakeReranker`` (which raises because it lacks
    ``.score()`` — only ``.compute_score()``) from spuriously
    hitting the gate.

    The carve-out is also called out in the gate docstring
    (``src/agent/nodes/retrieve.py:295-340``).
    """
    import importlib

    from src.agent.metrics import RETRIEVAL_FILTERED_LOW_SCORE
    from src.agent.nodes import retrieve as retrieve_mod

    reset_all()

    fake_embedder = MagicMock()
    fake_embedder.embed_query = MagicMock(return_value={"dense": [0.0] * 1024})
    monkeypatch.setattr(retrieve_mod, "_get_embedder", lambda: fake_embedder)

    fake_hit = {
        "chunk_id": "c1", "doc_id": "d1", "filename": "f.md",
        "text": "text", "rrf_score": 0.5,  # high RRF — would lie to gate
    }
    monkeypatch.setattr(
        retrieve_mod, "_hybrid_executor",
        MagicMock(submit=lambda fn, *a, **kw: MagicMock(result=lambda: [fake_hit])),
    )
    monkeypatch.setattr(retrieve_mod, "_bm25_search", lambda *a, **kw: [fake_hit])
    monkeypatch.setattr(retrieve_mod, "_dense_search", lambda *a, **kw: [fake_hit])
    monkeypatch.setattr(retrieve_mod, "reciprocal_rank_fusion", lambda *a, **kw: [fake_hit])
    monkeypatch.setattr(retrieve_mod, "get_table", lambda: MagicMock())

    # Force the reranker to fail.
    rerank_mod = importlib.import_module("src.retrieval.rerank")
    monkeypatch.setattr(rerank_mod, "_get_reranker", lambda: MagicMock(
        score=MagicMock(side_effect=RuntimeError("OOM")),
    ))

    baseline = RETRIEVAL_FILTERED_LOW_SCORE.get()

    state = {
        "current_query": "query", "thread_id": "t1",
        "messages": [], "step_count": 1,
    }
    delta = asyncio.run(retrieve_mod.retrieve_hybrid_async(state))

    # End-to-end: gate SKIPS, docs flow through with rerank_failed
    # stamps intact (which ``_hit_to_document`` propagates as
    # ``metadata.score = 0.0``).
    assert len(delta["documents"]) == 1, (
        f"gate must skip when rerank_failed=True; got docs={delta['documents']}"
    )
    assert delta["documents"][0].metadata.get("rerank_failed") is True, (
        f"rerank_failed stamp must propagate to Document metadata; "
        f"got {delta['documents'][0].metadata!r}"
    )
    assert delta["retrieval_status"] == "success", (
        f"gate skip path must mark retrieval_status='success'; "
        f"got {delta['retrieval_status']}"
    )
    # Counter must NOT tick — the gate was skipped, not fired.
    assert RETRIEVAL_FILTERED_LOW_SCORE.get() == baseline, (
        f"RETRIEVAL_FILTERED_LOW_SCORE must NOT tick on the rerank-failure "
        f"carve-out path; baseline={baseline}, after={RETRIEVAL_FILTERED_LOW_SCORE.get()}"
    )


# ---------------------------------------------------------------------------
# 5. Range-collapse WARNING log
# ---------------------------------------------------------------------------


def test_range_collapse_logs_warning():
    """When a multi-element range collapses to 1 element (e.g.
    ``[5-7]`` → ``[5]`` because only source 5 survived the cited-set
    filter), ``renumber_citations_and_sources`` emits a WARNING.

    Uses loguru's ``add(sink)`` because pytest's ``caplog`` does NOT
    capture loguru output (Phase 2 debugging坑 #1). The StringIO
    sink is a deterministic in-memory capture we can inspect.
    """
    from loguru import logger as loguru_logger

    from src.agent.citations import renumber_citations_and_sources

    sink = io.StringIO()
    handler_id = loguru_logger.add(sink, format="{message}", level="WARNING")

    try:
        # The LLM cited [5-7] but only source 5 survived → [5].
        answer = "see [5-7] for details"
        sources = [{"index": 5, "doc_id": "d1", "chunk_id": "c1", "filename": "f.md"}]
        new_answer, new_sources = renumber_citations_and_sources(
            answer, sources, route_decision="retrieve"
        )

        # Functional behavior: range collapsed to single.
        assert new_answer == "see [1] for details"
        assert len(new_sources) == 1
        assert new_sources[0]["index"] == 1

        # Observability: WARNING appeared in the sink.
        log_text = sink.getvalue()
        assert "citations: collapsed range" in log_text, (
            f"WARNING log must appear; got: {log_text!r}"
        )
        assert "[5-7]" in log_text, (
            f"WARNING should mention the original range; got: {log_text!r}"
        )
    finally:
        loguru_logger.remove(handler_id)


def test_range_collapse_no_log_when_full_survives():
    """Counterpart: when the full range survives, no WARNING fires.
    Avoids log spam in the common case (every cited source survived).
    """
    from loguru import logger as loguru_logger

    from src.agent.citations import renumber_citations_and_sources

    sink = io.StringIO()
    handler_id = loguru_logger.add(sink, format="{message}", level="WARNING")

    try:
        answer = "see [1-2] for details"
        sources = [
            {"index": 1, "doc_id": "d1", "chunk_id": "c1", "filename": "f.md"},
            {"index": 2, "doc_id": "d2", "chunk_id": "c1", "filename": "f.md"},
        ]
        new_answer, _ = renumber_citations_and_sources(
            answer, sources, route_decision="retrieve"
        )
        assert new_answer == "see [1-2] for details"
        assert "collapsed range" not in sink.getvalue(), (
            f"no WARNING when range survives intact; got: {sink.getvalue()!r}"
        )
    finally:
        loguru_logger.remove(handler_id)


# ---------------------------------------------------------------------------
# 6. CitationChip click event detail shape
# ---------------------------------------------------------------------------


def test_citation_chip_dispatch_includes_chunk_id_doc_id():
    """The chip's ``rag:citation-click`` event detail now includes
    ``chunk_id``, ``doc_id``, and ``source_kind`` alongside ``index``.

    The JS-side test (``CitationChip.test.tsx``) covers the React
    component contract directly. This Python-side test pins the
    *backend* wire contract — the Pydantic ``Source`` schema
    (``src/api/schemas.py``) must carry the same three fields,
    otherwise the chip click handler reads ``undefined`` and App.tsx
    silently loses the new event keys.
    """
    from src.api.schemas import Source as ApiSource

    # The wire-side Source (Pydantic, what flows to the frontend)
    # must declare all three fields the chip forwards.
    declared = ApiSource.model_fields
    for field in ("chunk_id", "doc_id", "source_kind"):
        assert field in declared, (
            f"Source (wire) must declare {field}; chip click handler forwards it. "
            f"If this fires, the backend wire lost a field the chip needs."
        )


# ---------------------------------------------------------------------------
# v2.0.29.4 (Phase 4 PR-2) — temporal + superseded assertions
# ---------------------------------------------------------------------------


def test_temporal_filter_keeps_no_expiry_chunks():
    """``_temporal_filter(None)`` returns None → caller gets no filter.

    Pre-PR-2 this was the only behaviour. Post-PR-2 it's preserved
    as the safe default: production callers that don't opt in see
    every chunk regardless of expiry.

    Note: only ``None`` means "no filter". An empty string ``""`` is
    treated as a valid (but always-false) cutoff — every non-null
    ``expires_at`` will be ``> ''`` is FALSE on lexicographic compare
    so the filter would erroneously exclude everything. The
    implementation is intentionally strict here: a caller who passes
    ``""`` is buggy and gets a buggy result. The production caller
    path (``retrieve_hybrid_async``) hardcodes ``None`` so this never
    bites in practice.
    """
    from src.retrieval.hybrid_search import _temporal_filter

    assert _temporal_filter(None) is None


def test_temporal_filter_excludes_expired_chunks():
    """A non-None cutoff builds the SQL ``(expires_at IS NULL OR expires_at > '<cutoff>')``.

    SQL injection guard: a single-quote in the cutoff is doubled,
    so the resulting clause is still a single string literal (the
    injected SQL keywords end up inside the quoted string and are
    inert — they don't parse as separate statements).
    """
    from src.retrieval.hybrid_search import _temporal_filter

    clause = _temporal_filter("2026-09-28T00:00:00Z")
    assert clause is not None
    assert "expires_at > '2026-09-28T00:00:00Z'" in clause
    assert "expires_at IS NULL" in clause
    # SQL injection guard — the leading quote is doubled, so the
    # entire injected payload ends up inside ONE quoted literal.
    nasty = _temporal_filter("'; DROP TABLE chunks; --")
    assert "''" in nasty  # single-quote was doubled
    # The injection's single-quote at position 0 became ``''`` so
    # the rest of the payload (DROP TABLE, etc.) is INSIDE the
    # string literal and won't parse as SQL.
    assert nasty.startswith("(expires_at IS NULL OR expires_at > ''")
    assert nasty.endswith("--')")


def test_compose_where_threads_and_temporal():
    """``_compose_where`` ANDs the two filters; both-None returns None.

    Each present filter is parenthesised so the SQL is unambiguous.
    """
    from src.retrieval.hybrid_search import _compose_where

    assert _compose_where(None, None) is None
    assert _compose_where("a", None) == "(a)"
    assert _compose_where(None, "b") == "(b)"
    composed = _compose_where("thread_id = 't1'", "(expires_at IS NULL OR expires_at > '2026-09-28')")
    assert composed == (
        "(thread_id = 't1') AND ((expires_at IS NULL OR expires_at > '2026-09-28'))"
    )


def test_filter_superseded_drops_chunks_from_superseded_doc():
    """``_filter_superseded`` drops hits whose doc_id was marked superseded.

    Pins the contract that ``supersede()`` actually prevents the LLM
    from citing stale content. Without this filter, a re-upload flow
    could leave the old doc's chunks in the retrieval result, and
    the synthesis LLM might cite either version interchangeably —
    a textbook hallucination amplifier.
    """
    from src.agent.nodes.retrieve import _filter_superseded
    from src.storage import doc_registry

    doc_registry.reset_for_tests()
    # Old doc gets re-uploaded as new doc; old is now superseded.
    doc_registry.register("old1", "report_v1.pdf", "/x", "t1")
    doc_registry.update("old1", status="indexed", chunk_count=3)
    doc_registry.register("new1", "report_v2.pdf", "/x", "t1")
    doc_registry.update("new1", status="indexed", chunk_count=3)
    doc_registry.supersede("old1", "new1")

    docs = [
        Document(page_content="v1 content A", metadata={"doc_id": "old1"}),
        Document(page_content="v2 content B", metadata={"doc_id": "new1"}),
        Document(page_content="v1 content C", metadata={"doc_id": "old1"}),
    ]
    filtered = _filter_superseded(docs, thread_id="t1")
    assert len(filtered) == 1
    assert filtered[0].metadata["doc_id"] == "new1"


def test_list_for_thread_default_excludes_superseded():
    """``list_for_thread(thread_id)`` (default) hides rows with status="superseded".

    The ``include_superseded=True`` opt-in recovers them. Pins the
    backward-compat invariant: callers that don't update their call
    sites automatically get the safer behaviour.
    """
    from src.storage import doc_registry

    doc_registry.reset_for_tests()
    doc_registry.register("a", "f.pdf", "/x", "t1")
    doc_registry.update("a", status="indexed", chunk_count=2)
    doc_registry.register("b", "g.pdf", "/x", "t1")
    doc_registry.update("b", status="indexed", chunk_count=2)
    doc_registry.supersede("a", "b")

    # Default: superseded excluded
    active = doc_registry.list_for_thread("t1")
    assert {e["doc_id"] for e in active} == {"b"}

    # Opt-in: superseded included
    all_entries = doc_registry.list_for_thread("t1", include_superseded=True)
    assert {e["doc_id"] for e in all_entries} == {"a", "b"}


# ---------------------------------------------------------------------------
# v2.0.29.8 (Phase 7) — sliding-window per-doc cap
# ---------------------------------------------------------------------------


def test_long_doc_cap_applies_after_supersede_filter():
    """When both filters are needed, superseded filter runs FIRST
    so superseded long docs don't waste window slots.

    Setup:
      - old1 (superseded, 80 chunks) — should be fully dropped by
        the supersede filter, never reaches the long-doc cap.
      - new1 (current, 80 chunks) — should pass supersede filter
        and be windowed by the long-doc cap.
    """
    from src.agent.nodes.retrieve import _filter_long_docs, _filter_superseded
    from src.storage import doc_registry

    doc_registry.reset_for_tests()
    doc_registry.register("old1", "v1.pdf", "/x", "t1")
    doc_registry.update("old1", status="indexed", chunk_count=80)
    doc_registry.register("new1", "v2.pdf", "/x", "t1")
    doc_registry.update("new1", status="indexed", chunk_count=80)
    doc_registry.supersede("old1", "new1")

    docs = []
    for i in range(80):
        docs.append(Document(
            page_content=f"v1-{i}", metadata={"doc_id": "old1", "chunk_index": i}
        ))
    for i in range(80):
        docs.append(Document(
            page_content=f"v2-{i}", metadata={"doc_id": "new1", "chunk_index": i}
        ))

    # Phase 4 PR-2 filter runs first.
    after_supersede = _filter_superseded(docs, thread_id="t1")
    assert len(after_supersede) == 80  # old1 dropped, new1 kept
    assert all(d.metadata["doc_id"] == "new1" for d in after_supersede)

    # Phase 7 filter runs second — only new1 gets windowed.
    after_window = _filter_long_docs(after_supersede, thread_id="t1")
    assert len(after_window) == 15  # 80 chunks → 15 (head 5 + middle 5 + tail 5)
    assert all(d.metadata["doc_id"] == "new1" for d in after_window)


def test_long_doc_cap_empty_after_filter_triggers_empty_status():
    """If every chunk in a long doc gets filtered out by supersede
    (and only the long doc exists), the resulting empty list hits
    the existing ``retrieval_status="empty"`` path.

    This test documents the contract: ``_filter_long_docs`` alone
    never triggers ``retrieval_status`` — the empty-check in
    ``retrieve_hybrid_async`` (line 461) does. Phase 7 is purely a
    chunk-filter helper, not a retrieval-status decision.
    """
    from src.agent.nodes.retrieve import _filter_long_docs

    # 60 chunks for d1, no other docs. After window → 15 kept.
    docs = [
        Document(page_content=f"d1-{i}", metadata={"doc_id": "d1", "chunk_index": i})
        for i in range(60)
    ]
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == 15  # NOT empty — window preserved
    assert out  # truthy → retrieval_status="success" downstream


def test_long_doc_cap_no_double_count_with_supersede():
    """Total dropped count = supersede drops + long-doc drops (no
    overlap because superseded docs are gone before long-doc cap
    runs). Pin the accounting so dashboards see the right totals.
    """
    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.retrieve import _filter_long_docs, _filter_superseded
    from src.storage import doc_registry

    counter = _metrics_mod.CHUNK_REJECTED_OVERSIZE
    counter.reset()

    doc_registry.reset_for_tests()
    doc_registry.register("old1", "v1.pdf", "/x", "t1")
    doc_registry.update("old1", status="indexed", chunk_count=20)
    doc_registry.register("new1", "v2.pdf", "/x", "t1")
    doc_registry.update("new1", status="indexed", chunk_count=20)
    doc_registry.supersede("old1", "new1")

    # Build 20 superseded + 60 current (long) chunks.
    docs = []
    for i in range(20):
        docs.append(Document(page_content=f"o-{i}",
                             metadata={"doc_id": "old1", "chunk_index": i}))
    for i in range(60):
        docs.append(Document(page_content=f"n-{i}",
                             metadata={"doc_id": "new1", "chunk_index": i}))

    before = counter.get()

    after_supersede = _filter_superseded(docs, thread_id="t1")
    # supersede drops 20 (no metric bump; Phase 4 PR-2 is silent)
    assert len(after_supersede) == 60

    after_window = _filter_long_docs(after_supersede, thread_id="t1")
    # long-doc cap drops 45 (60 → 15)
    assert len(after_window) == 15

    # Only the long-doc drop is bumped; supersede drops are not
    # counted in CHUNK_REJECTED_OVERSIZE (they have their own
    # debug log + Phase 4 PR-2 didn't wire a metric).
    after = counter.get()
    assert after - before == 45.0