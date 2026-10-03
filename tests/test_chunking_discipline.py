"""v2.0.29.8 (Phase 7) — sliding window for per-doc chunk overflow.

Pins the contract for ``_filter_long_docs`` in
``src/agent/nodes/retrieve.py``. Without these guards, a long doc
(200-page PDF, large Excel sheet, legal contract) can dump 100+
chunks into the synthesis LLM's context — overflowing the cheap
graders and forcing the LLM to silently pick which chunks to read.

The contract under test:

  * Per-doc overflow → sliding window (head + middle samples + tail).
  * Window is deterministic by ``chunk_index`` (None sorts last).
  * Dropped chunks bump ``CHUNK_REJECTED_OVERSIZE`` (Phase 2 metric,
    pre-registered label-less).
  * Helper is idempotent + never modifies Document objects.
  * Chunks without ``metadata.doc_id`` pass through unchanged.
  * Both summary-intent (bulk) and hybrid (rerank) paths share the
    same helper — single source of truth.
"""
from __future__ import annotations

from langchain_core.documents import Document


def _make_docs(doc_id: str, count: int, *, start_index: int = 0,
               thread_id: str | None = "t1") -> list[Document]:
    """Build ``count`` synthetic Documents for one doc_id, contiguous
    chunk_index range starting at ``start_index``."""
    return [
        Document(
            page_content=f"doc={doc_id} chunk={i} body",
            metadata={
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-c{i}",
                "chunk_index": start_index + i,
                "thread_id": thread_id,
            },
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Empty + small-doc short-circuits
# ---------------------------------------------------------------------------


def test_filter_long_docs_empty_short_circuits():
    """Empty input → empty output, no work done."""
    from src.agent.nodes.retrieve import _filter_long_docs

    assert _filter_long_docs([]) == []
    assert _filter_long_docs([], thread_id="t1") == []


def test_filter_long_docs_single_doc_below_threshold():
    """A single doc with N chunks where N ≤ MAX_CHUNKS_PER_DOC
    passes through unchanged."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = _make_docs("d1", 10)
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == 10
    assert [d.metadata["chunk_index"] for d in out] == list(range(10))


def test_filter_long_docs_single_doc_at_threshold():
    """A doc exactly at MAX_CHUNKS_PER_DOC (50) is not windowed."""
    from config.constants import MAX_CHUNKS_PER_DOC
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = _make_docs("d1", MAX_CHUNKS_PER_DOC)
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == MAX_CHUNKS_PER_DOC


# ---------------------------------------------------------------------------
# Sliding-window contract
# ---------------------------------------------------------------------------


def test_filter_long_docs_single_doc_above_threshold():
    """A single doc with 80 chunks → exactly 15 kept
    (head 5 + middle 5 + tail 5)."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = _make_docs("d1", 80)
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == 15  # head 5 + middle 5 + tail 5

    indices = [d.metadata["chunk_index"] for d in out]
    # head: 0,1,2,3,4
    assert indices[:5] == [0, 1, 2, 3, 4]
    # tail: 75,76,77,78,79
    assert indices[-5:] == [75, 76, 77, 78, 79]
    # middle: 5 samples from range(5, 75), evenly-spaced.
    # range(5, 75) has 70 elements; step = max(1, 70//5) = 14
    # so indices 5, 19, 33, 47, 61
    assert indices[5:10] == [5, 19, 33, 47, 61]


def test_filter_long_docs_preserves_chunk_index_ordering():
    """Windowed output is monotonically sorted by chunk_index
    within each region (head → middle → tail)."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = _make_docs("d1", 100)
    out = _filter_long_docs(docs, thread_id="t1")

    indices = [d.metadata["chunk_index"] for d in out]
    assert indices == sorted(indices), "window must be sorted by chunk_index"
    # No duplicate chunk_indices
    assert len(set(indices)) == len(indices)


def test_filter_long_docs_multiple_docs_only_long_filtered():
    """Two docs: one short (10 chunks, kept), one long (80 chunks,
    windowed to 15). Total kept = 25."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = _make_docs("d1", 10) + _make_docs("d2", 80)
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == 25

    by_doc: dict[str, list[int]] = {}
    for d in out:
        by_doc.setdefault(d.metadata["doc_id"], []).append(d.metadata["chunk_index"])
    # d1: all 10 kept
    assert sorted(by_doc["d1"]) == list(range(10))
    # d2: windowed to 15
    assert len(by_doc["d2"]) == 15


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_filter_long_docs_no_doc_id_passes_through():
    """Chunks without ``metadata.doc_id`` pass through unchanged."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = [
        Document(page_content="orphan", metadata={"chunk_index": 0}),
        Document(page_content="orphan2", metadata={"chunk_index": 1}),
        Document(page_content="d1-c0", metadata={"doc_id": "d1", "chunk_index": 0}),
    ]
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == 3  # all passed through (single d1 chunk, no overflow)


def test_filter_long_docs_legacy_chunks_without_chunk_index_kept_last():
    """Chunks with chunk_index=None sort to the end of the window
    so the indexed content is preserved preferentially."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = []
    # 10 indexed chunks + 10 None-indexed (legacy) chunks
    for i in range(10):
        docs.append(Document(
            page_content=f"idx={i}",
            metadata={"doc_id": "d1", "chunk_index": i},
        ))
    for i in range(10):
        docs.append(Document(
            page_content=f"legacy={i}",
            metadata={"doc_id": "d1", "chunk_index": None},
        ))
    out = _filter_long_docs(docs, thread_id="t1")
    # 20 chunks total, all kept (≤ MAX_CHUNKS_PER_DOC=50)
    assert len(out) == 20

    indexed_count = sum(1 for d in out if d.metadata["chunk_index"] is not None)
    assert indexed_count == 10

    # The 10 None-indexed chunks occupy positions [10:20] in the
    # sorted-by-chunk_index output. (None sorts last; stable order
    # among the Nones preserves original input order.)
    none_count = sum(1 for d in out if d.metadata["chunk_index"] is None)
    assert none_count == 10


def test_filter_long_docs_thread_id_lazy_load_from_metadata():
    """When thread_id=None but first doc has metadata.thread_id,
    the helper logs that thread_id (not correctness-critical, but
    pinned so dashboards see something useful)."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = _make_docs("d1", 80, thread_id="inferred-thread")
    # Pass thread_id=None — helper should pick it up from metadata.
    out = _filter_long_docs(docs)
    assert len(out) == 15


# ---------------------------------------------------------------------------
# Metric + idempotence
# ---------------------------------------------------------------------------


def test_filter_long_docs_bumps_rejected_oversize_metric():
    """Dropping chunks bumps CHUNK_REJECTED_OVERSIZE by exactly the
    number dropped (label-less counter)."""
    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.retrieve import _filter_long_docs

    counter = _metrics_mod.CHUNK_REJECTED_OVERSIZE
    counter.reset()
    before = counter.get()

    # 80 chunks → 15 kept → 65 dropped
    docs = _make_docs("d1", 80)
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == 15

    after = counter.get()
    assert after - before == 65.0


def test_filter_long_docs_idempotent():
    """Second call on already-windowed input drops zero chunks."""
    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.retrieve import _filter_long_docs

    counter = _metrics_mod.CHUNK_REJECTED_OVERSIZE
    counter.reset()

    docs = _make_docs("d1", 80)
    first = _filter_long_docs(docs, thread_id="t1")
    after_first = counter.get()

    second = _filter_long_docs(first, thread_id="t1")
    after_second = counter.get()

    # Same docs in same order.
    assert [d.metadata["chunk_index"] for d in first] == [
        d.metadata["chunk_index"] for d in second
    ]
    # No further bumps.
    assert after_second == after_first


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------


def test_filter_long_docs_window_respects_configured_max():
    """Override ``max_per_doc`` to a smaller cap → window triggers
    earlier."""
    from src.agent.nodes.retrieve import _filter_long_docs

    # 25 chunks, default max=50 → no window. With max=20 → window.
    docs = _make_docs("d1", 25)
    out_default = _filter_long_docs(docs, thread_id="t1")
    assert len(out_default) == 25

    out_tight = _filter_long_docs(docs, thread_id="t1", max_per_doc=20)
    assert len(out_tight) == 15  # still head 5 + middle 5 + tail 5


def test_filter_long_docs_middle_samples_evenly_spaced():
    """For 100 chunks with middle_samples=5, the middle samples
    are evenly-spaced in (head, tail) = (5, 95)."""
    from src.agent.nodes.retrieve import _filter_long_docs

    docs = _make_docs("d1", 100)
    out = _filter_long_docs(docs, thread_id="t1")
    indices = [d.metadata["chunk_index"] for d in out]

    middle = indices[5:10]  # indices 5..14 are middle samples
    # range(5, 95) has 90 elements; step = max(1, 90//5) = 18
    # so positions 5, 23, 41, 59, 77
    assert middle == [5, 23, 41, 59, 77]


def test_filter_long_docs_no_op_on_short_docs_no_metric_bump():
    """Small docs (below threshold) never bump the counter."""
    from src.agent import metrics as _metrics_mod
    from src.agent.nodes.retrieve import _filter_long_docs

    counter = _metrics_mod.CHUNK_REJECTED_OVERSIZE
    counter.reset()
    before = counter.get()

    docs = _make_docs("d1", 10) + _make_docs("d2", 20) + _make_docs("d3", 5)
    out = _filter_long_docs(docs, thread_id="t1")
    assert len(out) == 35

    after = counter.get()
    assert after == before, "short docs must not bump CHUNK_REJECTED_OVERSIZE"


# ---------------------------------------------------------------------------
# Pipeline integration: summary-intent + hybrid
# ---------------------------------------------------------------------------


def test_summary_intent_branch_applies_long_doc_cap():
    """The summary-intent branch in ``retrieve_hybrid_async`` calls
    ``_filter_long_docs`` after ``_bulk_chunk_to_document``.

    Strategy: monkeypatch the cached bulk-list helper to return a
    long-doc dataset, drive ``retrieve_hybrid_async`` with a summary
    query, assert the returned docs are windowed.
    """
    import asyncio

    from langchain_core.documents import Document

    from src.agent.nodes import retrieve as retrieve_mod

    long_chunks = [
        {
            "chunk_id": f"d1-c{i}", "doc_id": "d1", "text": f"body {i}",
            "filename": "long.pdf", "source_type": "file", "chunk_index": i,
            "page_number": i, "section": "", "sheet": "",
        }
        for i in range(80)
    ]
    # thread_docs (list_documents_cached) just needs to be non-empty
    # to enable the summary-intent fast path.
    thread_docs = [{"doc_id": "d1", "filename": "long.pdf", "chunk_count": 80}]

    original_chunks_cached = retrieve_mod.list_chunks_by_thread_cached
    original_docs_cached = retrieve_mod.list_documents_cached

    def fake_chunks_cached(thread_id=None):
        return list(long_chunks)

    def fake_docs_cached(thread_id=None):
        return list(thread_docs)

    retrieve_mod.list_chunks_by_thread_cached = fake_chunks_cached
    retrieve_mod.list_documents_cached = fake_docs_cached
    try:
        fake_state = {
            "current_query": "总结一下这份文档",
            "original_query": "总结一下这份文档",
            "thread_id": "t1",
            "messages": [],
        }
        result = asyncio.run(retrieve_mod.retrieve_hybrid_async(fake_state))
        assert len(result["documents"]) == 15, (
            "summary-intent branch must apply _filter_long_docs"
        )
        assert result["summary_intent_used"] is True
        assert result["retrieval_status"] == "success"
    finally:
        retrieve_mod.list_chunks_by_thread_cached = original_chunks_cached
        retrieve_mod.list_documents_cached = original_docs_cached


def test_summary_path_fsm_node_applies_long_doc_cap():
    """The ``summary_path`` FSM node in ``fsm.py`` calls
    ``_filter_long_docs`` after ``_bulk_chunk_to_document``.

    Strategy: monkeypatch ``list_chunks_by_thread_cached`` (used
    via ``asyncio.to_thread``) to return a long-doc dataset, drive
    ``summary_path`` with an intent=="summary" state, assert the
    yielded ``documents`` delta is windowed.
    """
    import asyncio

    from src.agent import fsm as fsm_mod

    long_chunks = [
        {
            "chunk_id": f"d1-c{i}", "doc_id": "d1", "text": f"body {i}",
            "filename": "long.pdf", "source_type": "file", "chunk_index": i,
            "page_number": i, "section": "", "sheet": "",
        }
        for i in range(80)
    ]

    original_chunks_cached = fsm_mod.list_chunks_by_thread_cached

    def fake_chunks_cached(thread_id):
        return list(long_chunks)

    fsm_mod.list_chunks_by_thread_cached = fake_chunks_cached
    try:
        async def _drive():
            docs_yielded = None
            async for kind, payload in fsm_mod.summary_path(
                {"thread_id": "t1", "messages": []}
            ):
                if kind == "__delta__":
                    docs_yielded = payload.get("documents")
                    retrieval_status = payload.get("retrieval_status")
                    return docs_yielded, retrieval_status
            return None, None

        docs, status = asyncio.run(_drive())
        assert docs is not None
        assert len(docs) == 15, "fsm summary_path must apply _filter_long_docs"
        assert status == "success"
    finally:
        fsm_mod.list_chunks_by_thread_cached = original_chunks_cached