"""retrieve_hybrid node: BGE-M3 + LanceDB hybrid + rerank → top-K Documents."""
from __future__ import annotations

import asyncio
import re
from typing import List, Optional

from langchain_core.documents import Document

from config.constants import MAX_CHUNKS_PER_DOC, RETRIEVAL_DEFAULTS
from src.agent.nodes._recovery import log_node_failure
from src.agent.state import AgentState, Source
from src.core.logging import logger
from src.embeddings.bge_m3 import BGEM3Embedder
from src.retrieval.hybrid_search import (
    _bm25_search,
    _compose_where,
    _dense_search,
    _hybrid_executor,
    _temporal_filter,
    _thread_filter,
    reciprocal_rank_fusion,
)
from src.retrieval.rerank import rerank
from src.storage.lancedb_store import (
    get_table,
    list_chunks_by_thread_cached,
    list_documents,
    list_documents_cached,
)


_embedder: BGEM3Embedder | None = None


def _get_embedder() -> BGEM3Embedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder()
    return _embedder


def reset_for_tests() -> None:
    """Forget the cached embedder. Tests only."""
    global _embedder
    _embedder = None


# Phrasing patterns that mean "operate on the whole uploaded document"
# rather than "find a passage inside it". Matches Chinese and English.
#
# v2.0.29.4 (Phase 4 PR-1) — the regex is now anchored at the start
# of the query (``^(?:...)``) and the broad definitional patterns
# ``是什么`` / ``有哪些`` / ``主要内容`` were REMOVED. Pre-PR-1 those
# broad patterns caused definitional QA like "X 是什么" or "X 有哪些
# 特征" to be misrouted into the summary branch, which dumped every
# chunk of the thread's docs into the context window — a waste of
# tokens AND a known hallucination amplifier (the cheap model used
# to grade documents in the summary branch can overflow and fail-open
# the same set anyway, per the comments at ``summary_intent_used``).
# Definitional QA should always go through the normal embed → hybrid
# → rerank path, which produces a tight top-K instead of dumping
# 50+ chunks into the LLM.
_SUMMARY_INTENT_RE = re.compile(
    r"^(?:总结|摘要|概括|归纳|概述|总览|全文|整篇|介绍|讲讲|讲一下|"
    r"说说|说一下|讲讲内容|列一下|列出来|分析一下|解释一下|解释下)",
    re.IGNORECASE,
)


def _looks_like_summary_intent(query: str) -> bool:
    """True when the user is asking the assistant to operate on the WHOLE
    document (summarize / list / explain "this doc") rather than pull a
    specific fact from it. Used to bypass semantic search (which would
    fail because meta-questions about a doc don't share embedding space
    with the doc's body text) and instead return every chunk of the
    thread's documents.

    v2.0.29.4 (Phase 4 PR-1) — the regex is anchored at the start of
    the query (``^(?:...)``), so mid-sentence mentions like "这个文档
    的第 3 章是什么" no longer match (it should go through the normal
    semantic-search path).
    """
    if not query:
        return False
    return bool(_SUMMARY_INTENT_RE.search(query))


def _hit_to_document(hit: dict) -> Document:
    meta = dict(hit.get("_metadata", {}))
    meta.update(
        {
            "chunk_id": hit.get("chunk_id"),
            "doc_id": hit.get("doc_id"),
            # v2.0.29.8 (Phase 7) — propagate chunk_index so the
            # sliding-window helper (``_filter_long_docs``) can pick
            # head/middle/tail by position. ``_bulk_chunk_to_document``
            # at line 134 already carries this; this brings the hybrid
            # path to parity. Without this, every hit looks like
            # position 0 and the window would degenerate to "all".
            "chunk_index": hit.get("chunk_index"),
            "filename": hit.get("filename"),
            "source_type": hit.get("source_type"),
            "source_path": hit.get("source_path"),
            "chunk_type": hit.get("chunk_type"),
            "page_number": hit.get("page_number"),
            "section": hit.get("section"),
            "sheet": hit.get("sheet"),
            "headers": hit.get("headers"),
            "row_start": hit.get("row_start"),
            "row_end": hit.get("row_end"),
            "score": hit.get("rerank_score", hit.get("rrf_score", 0.0)),
            # v2.0.29.4 (Phase 4 PR-1) — propagate the rerank-failure
            # stamp from ``rerank.py`` so downstream consumers
            # (react_generate, citation helpers, observability
            # dashboards) can distinguish "reranker scored this 0.0"
            # from "reranker failed". Without this, an OOM or BGE
            # network error would look identical to a genuinely
            # irrelevant chunk, and react_generate might cite the
            # failure artifact as if it were relevant evidence.
            "rerank_failed": bool(hit.get("rerank_failed", False)),
        }
    )
    return Document(page_content=hit.get("text", ""), metadata=meta)


def _bulk_chunk_to_document(chunk: dict) -> Document:
    """Convert a raw LanceDB chunk row (from list_chunks_by_thread) into
    a Document. Mirrors ``_hit_to_document`` but skips the score fields
    since the bulk path returns rows in natural order, not ranked."""
    return Document(
        page_content=chunk.get("text", ""),
        metadata={
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "filename": chunk.get("filename"),
            "source_type": chunk.get("source_type"),
            "page_number": chunk.get("page_number"),
            "section": chunk.get("section"),
            "sheet": chunk.get("sheet"),
            "chunk_index": chunk.get("chunk_index"),
            "score": 0.0,
        },
    )


def _filter_superseded(
    docs: List[Document],
    thread_id: Optional[str] = None,
) -> List[Document]:
    """v2.0.29.4 (Phase 4 PR-2) — drop chunks whose doc_id was superseded.

    Reads the doc_registry (which knows the canonical ``superseded_by``
    mapping) and removes any Document whose ``doc_id`` is in the
    superseded set for the thread. A doc that was re-uploaded (new
    doc_id) and explicitly marked old as superseded will not appear
    in retrieved citations — preventing the LLM from citing stale
    content alongside fresh.

    The registry's ``list_for_thread(thread_id)`` already
    default-excludes superseded rows, so the "current" set is the
    complement of the "superseded" set. We build a fast lookup of
    current doc_ids and drop anything not in it.

    Pre-PR-2 the filter didn't exist; callers saw chunks from
    superseded docs leak through to the LLM context. The registry's
    ``include_superseded=False`` default is the data-layer half of
    the fix; this function is the retrieval-layer half.

    Thread safety: registry reads take the registry's own internal
    lock; we copy the result here. No race with concurrent
    ``supersede()`` calls in production (re-upload is a user-driven
    flow, not a hot path).

    Empty docs list short-circuits — the registry lookup is only
    needed if we actually have hits to filter.
    """
    if not docs:
        return docs
    # Lazy import — registry isn't needed for the empty path and the
    # import cost is non-trivial.
    from src.storage.doc_registry import list_for_thread

    # ``include_superseded=False`` (default) returns only the current
    # doc_ids; everything else in the thread is superseded or
    # unrelated. If thread_id is None/empty (global view) we still
    # ask the registry but it returns the full set, which is the
    # correct semantics for the unthreaded case.
    current_entries = list_for_thread(thread_id, include_superseded=False)
    current_doc_ids = {
        str(e.get("doc_id") or "")
        for e in current_entries
        if e.get("doc_id")
    }
    if not current_doc_ids:
        return docs  # Nothing in the registry yet — preserve pre-PR-2 behaviour
    out: List[Document] = []
    dropped = 0
    for d in docs:
        doc_id = str(d.metadata.get("doc_id") or "")
        if doc_id and doc_id not in current_doc_ids:
            dropped += 1
            continue
        out.append(d)
    if dropped:
        # v2.0.29.4 (Phase 4 PR-2) — observability: log the drop
        # count at debug level (NOT warning — superseded drops are
        # expected during re-upload flows, not an error).
        logger.debug(
            f"retrieve: dropped {dropped} chunk(s) from superseded "
            f"doc(s) in thread={thread_id!r}"
        )
    return out


def _filter_long_docs(
    docs: List[Document],
    *,
    thread_id: Optional[str] = None,
    max_per_doc: int = MAX_CHUNKS_PER_DOC,
    head_keep: int = 5,
    tail_keep: int = 5,
    middle_samples: int = 5,
) -> List[Document]:
    """v2.0.29.8 (Phase 7) — sliding window for per-doc chunk overflow.

    For any single ``doc_id`` whose contribution to ``docs`` exceeds
    ``max_per_doc``, replace the full set with a sliding window:
      - head: first ``head_keep`` chunks by chunk_index (lowest index)
      - middle: ``middle_samples`` evenly-spaced chunks (skipping
        the head/tail region)
      - tail: last ``tail_keep`` chunks by chunk_index (highest index)
    Total kept = ``head_keep + middle_samples + tail_keep`` (15 by
    default) regardless of how many chunks the doc actually has.

    Source of truth for "is this doc long?": the candidate set itself
    (the post-rerank top-K on the hybrid path, the bulk chunks list
    on the summary path). The doc_registry ``chunk_count`` is
    approximate due to the 200ms background flusher; the candidate
    set is authoritative at this hook point.

    Chunks are reordered by ``chunk_index`` for stable window
    selection. Chunks with ``chunk_index is None`` (legacy data /
    OCR noise) sort to the end of their doc's chunk list so the
    window preserves the indexed content preferentially.

    Bumps ``CHUNK_REJECTED_OVERSIZE`` per dropped chunk for
    dashboard visibility (Phase 2 metric, pre-registered label-less).
    Drops are logged at debug level (NOT warning — long-doc trimming
    is the expected outcome, not an error).

    Empty docs short-circuits — no work to do.

    Invariants:
      * Document objects are preserved verbatim (no metadata rewrite).
      * Drops never cross doc boundaries (each doc is windowed
        independently).
      * Chunks without ``metadata.doc_id`` are passed through
        unchanged (cannot be grouped, no sliding window applies).
      * Idempotent: calling twice on the same input drops zero
        additional chunks.
    """
    if not docs:
        return docs

    # Group by doc_id. Chunks without doc_id pass through unchanged.
    by_doc: dict[str, List[Document]] = {}
    passthrough: List[Document] = []
    for d in docs:
        doc_id = str(d.metadata.get("doc_id") or "")
        if not doc_id:
            passthrough.append(d)
            continue
        by_doc.setdefault(doc_id, []).append(d)

    if thread_id is None and docs:
        # Best-effort thread_id fallback from first doc's metadata.
        # Used purely for the debug log; not a correctness signal.
        thread_id = str(docs[0].metadata.get("thread_id") or "") or None

    out: List[Document] = []
    total_dropped = 0
    for doc_id, group in by_doc.items():
        # Stable sort by chunk_index. None sorts to the END (so the
        # window preserves the indexed content preferentially).
        group_sorted = sorted(
            group,
            key=lambda d: (
                d.metadata.get("chunk_index") is None,
                d.metadata.get("chunk_index") if d.metadata.get("chunk_index") is not None else 0,
                # ``id`` ensures deterministic ordering when chunk_index
                # ties (Python's sort is stable, but explicit tie-break
                # makes the contract testable).
                id(d),
            ),
        )
        if len(group_sorted) <= max_per_doc:
            out.extend(group_sorted)
            continue

        # Sliding window: head + middle + tail.
        n = len(group_sorted)
        keep_indices: set[int] = set()
        # head
        for i in range(min(head_keep, n)):
            keep_indices.add(i)
        # tail (overlap-safe: starts at n - tail_keep or head_keep,
        # whichever is larger — guarantees no double-count even if
        # head_keep + tail_keep > n)
        tail_start = max(n - tail_keep, head_keep)
        for i in range(tail_start, n):
            keep_indices.add(i)
        # middle: evenly-spaced samples in the (head, tail) interval
        if middle_samples > 0 and n > head_keep + tail_keep:
            window = list(range(head_keep, n - tail_keep))
            if window:
                step = max(1, len(window) // middle_samples)
                sampled = window[::step][:middle_samples]
                keep_indices.update(sampled)

        kept = [group_sorted[i] for i in sorted(keep_indices)]
        out.extend(kept)
        total_dropped += n - len(kept)

    if total_dropped:
        # v2.0.29.8 (Phase 7) — observability. Counter is
        # pre-registered label-less in metrics.py:207-210.
        from src.agent.metrics import CHUNK_REJECTED_OVERSIZE

        CHUNK_REJECTED_OVERSIZE.inc(amount=float(total_dropped))
        logger.debug(
            f"retrieve: dropped {total_dropped} chunk(s) via sliding-window "
            f"(per-doc > {max_per_doc}); thread={thread_id!r}"
        )

    # Passthrough chunks go LAST so that doc-grouped chunks sort to
    # the front (matches the pre-Phase-7 ordering for docs that
    # weren't grouped).
    return out + passthrough


# P2-1: the previous ``retrieve_hybrid`` sync wrapper was dead code
# — LangGraph >=0.2 runs async nodes natively via ``astream`` and the
# graph was rewired to call ``retrieve_hybrid_async`` directly. The
# async name remains the source of truth.


async def retrieve_hybrid_async(state: AgentState, *, top_k: Optional[int] = None) -> dict:
    """Async entry: embed query → hybrid search → rerank → top-K Documents.

    Retrieval is scoped to the active conversation: the ``thread_id`` from
    state is passed to ``hybrid_search`` which applies a ``.where()``
    filter so a file uploaded in conversation A is invisible in conversation B.

    Special case: "summarize / list / explain this document" queries.
    BGE-M3 semantic search returns zero hits for meta-questions like
    "总结一下这个文档的内容" because the query text doesn't share
    embedding space with the document's body. For those queries we
    skip the embed-and-search step entirely and return every chunk of
    the thread's uploaded documents (capped at ``list_chunks_by_thread``'s
    internal ``max_chunks``). The LLM can then read and summarize the
    full content. This branch only fires when (a) the query looks like
    a summary intent AND (b) the thread actually has uploaded documents.
    """
    query = state.get("current_query") or state.get("original_query") or ""
    thread_id = state.get("thread_id") or None

    # ----- Summary-intent fast path -----
    if _looks_like_summary_intent(query) and thread_id:
        try:
            # v2.0.28 P1-P1 — flip from uncached ``list_documents`` to
            # cached ``list_documents_cached``. The summary-intent branch
            # is the per-turn ``route_query`` hot path; it doesn't need
            # fresh reads. Cache invalidation is already wired in
            # ``add_chunks`` (lancedb_store.py:294) and ``delete_by_doc_id``
            # (line 325) so the 5s TTL staleness is bounded. The
            # ``list_chunks_by_thread_cached`` call below (line 158) was
            # already cached — this brings symmetry to the summary-intent
            # branch's two read sites.
            thread_docs = await asyncio.to_thread(
                list_documents_cached, thread_id=thread_id
            )
        except Exception as exc:
            # v2.0.22 (Item 7 Step 11) — uniform logging via the
            # centralized helper (was ``"list_documents failed in
            # summary-intent branch"`` at ``logger.debug`` level,
            # which silently dropped real storage failures out of
            # production logs). Recovery decision (empty docs →
            # fall through to hybrid search) stays inline.
            log_node_failure("retrieve.list_documents", exc)
            thread_docs = []
        if thread_docs:
            try:
                # v2.0.7 perf-3: cached wrapper — saves a 200-500 ms
                # full scan on the summary-intent hot path.
                chunks = await asyncio.to_thread(
                    list_chunks_by_thread_cached, thread_id
                )
            except Exception as exc:
                # v2.0.22 (Item 7 Step 11) — uniform logging.
                log_node_failure("retrieve.list_chunks", exc)
                chunks = []
            if chunks:
                docs: List[Document] = [_bulk_chunk_to_document(c) for c in chunks]
                # v2.0.29.8 (Phase 7) — sliding window for per-doc
                # overflow in the bulk summary path. This is the most
                # common "long doc" scenario: a single 200-page PDF
                # dumps all chunks into the LLM context. Same helper
                # as the hybrid path (lines below); single source of
                # truth for the sliding-window contract.
                docs = _filter_long_docs(docs, thread_id=thread_id)
                logger.info(
                    f"summary-intent: returning {len(docs)} chunks from "
                    f"{len(thread_docs)} thread doc(s) (no embedding)"
                )
                return {
                    "documents": docs,
                    # v2.0.29.2 (Phase 2) — removed dead
                    # ``retrieval_count`` / ``iteration_count``
                    # writes (see empty-path comment for rationale).
                    # Signal to grade_documents that these chunks were
                    # intentionally returned in bulk — don't run the
                    # LLM grader over them (it overflows the cheap
                    # model's output token budget for 10+ chunks and
                    # ends up fail-opening the same set anyway).
                    "summary_intent_used": True,
                    # v2.0.29.1 (Phase 1) — bulk path with chunks
                    # succeeded; mark status as "success" so the
                    # synthesis LLM is NOT forced into the refusal
                    # template. The empty-bulk case below sets
                    # "empty_bulk" instead.
                    "retrieval_status": "success",
                }
            # chunks is empty — fall through to normal hybrid search
            # below. Most likely the upload is still pending; semantic
            # search will probably also return 0, but at least we tried.

    # v2.0.29.1 (Phase 1) — summary-intent path tried but found no
    # chunks. Fall through to hybrid search with an empty-bulk signal
    # so react_generate can inject the REFUSAL_EMPTY_BULK_TEMPLATE if
    # the hybrid path ALSO returns nothing. We deliberately don't set
    # retrieval_status here yet — the hybrid path below will set it.
    # If summary-intent returned 0 chunks AND we're about to do hybrid
    # search anyway, that's a strong hint that the thread has nothing
    # useful, but we want to give hybrid a chance first (it might find
    # something via semantic similarity that bulk-by-chunk missed).

    # ----- Normal embed + hybrid + rerank path -----
    # Parallel BM25 + embed (Phase D2): BM25 only needs the raw query
    # string, so it can start the disk/network read on a worker thread
    # WHILE the embedder is computing the dense vector on another. The
    # previous implementation waited for embed to finish before issuing
    # any search — wasted wall time on a single CPU-bound + IO-bound
    # pair. Net savings: 50-300 ms per retrieval round on a typical
    # thread (BM25 disk-read overlaps with the BGE-M3 inference).
    pre = RETRIEVAL_DEFAULTS["top_k_pre_rerank"]
    # v2.0.29.6 (Phase 5) — accept top_k as an explicit kwarg from
    # ``retrieve_docs`` tool callers. ``None`` (legacy call sites,
    # e.g. summary-intent path that doesn't expose top_k to the LLM)
    # falls back to the module-level default — preserves pre-Phase-5
    # behaviour for any caller that doesn't thread the kwarg.
    post = top_k if top_k is not None else RETRIEVAL_DEFAULTS["top_k_post_rerank"]
    # v2.0.29.4 (Phase 4 PR-2) — temporal_cutoff_iso is None by
    # default, which preserves pre-PR-2 behaviour (no temporal filter).
    # Future operators can supply an ISO timestamp to bound retrieval
    # to non-expired chunks; production upload flows do NOT set this.
    temporal_cutoff_iso: Optional[str] = None
    try:
        table = await asyncio.to_thread(get_table)
        # Compose thread + temporal filters into one where-clause.
        # _compose_where returns None when both are absent so we
        # don't add a spurious ``where(None)`` to LanceDB.
        where = _compose_where(
            _thread_filter(thread_id),
            _temporal_filter(temporal_cutoff_iso),
        )
        # Submit BM25 (text-only) immediately. Dense can't be submitted
        # yet — we need the embedding first.
        bm25_future = _hybrid_executor.submit(
            _bm25_search, table, query, pre, where
        )
        # Run the embed in parallel with BM25.
        emb = await asyncio.to_thread(_get_embedder().embed_query, query)
        dense_vec = emb["dense"]
        # Now submit dense (needs the embed) and wait on both.
        dense_future = _hybrid_executor.submit(
            _dense_search, table, dense_vec, pre, where
        )

        try:
            bm25_hits = bm25_future.result()
        except Exception as exc:
            # v2.0.22 (Item 7 Step 11) — uniform logging.
            log_node_failure("retrieve.bm25_search", exc)
            bm25_hits = []
        try:
            dense_hits = dense_future.result()
        except Exception as exc:
            # v2.0.22 (Item 7 Step 11) — uniform logging.
            log_node_failure("retrieve.dense_search", exc)
            dense_hits = []

        fused = reciprocal_rank_fusion([dense_hits, bm25_hits])[:pre]
        reranked = await asyncio.to_thread(rerank, query, fused, post)
    except Exception as exc:
        # v2.0.22 (Item 7 Step 11) — uniform logging. Recovery
        # decision (empty-result delta) stays inline because it's
        # the only way the outer ``run_fsm`` loop can continue.
        log_node_failure("retrieve", exc)
        # v2.0.29.2 (Phase 2) — observability hook. Two empty
        # paths both bump RETRIEVAL_EMPTY: hybrid search raised
        # before producing a doc list, OR hybrid + rerank returned
        # 0 docs (bumped below). Distinct in the FSM delta but
        # same metric so dashboards show total empty retrievals.
        from src.agent.metrics import RETRIEVAL_EMPTY

        RETRIEVAL_EMPTY.inc()
        return {
            "documents": [],
            # v2.0.29.2 (Phase 2) — removed dead state writes.
            # ``retrieval_count`` and ``iteration_count`` were
            # removed from AgentState in v2.0 (per state.py
            # docstring line 19-21: "ReAct decides / step_count
            # replaces it"). Writing them here meant the runner
            # hydration boundary had to know about two more fields
            # it must drop. Removed at the source — observability
            # moved to RETRIEVAL_EMPTY metric (counter, not state).
            "retrieval_status": "empty",
        }

    # v2.0.29.4 (Phase 4 PR-1) — min_top_relevance_score gate.
    # The gate fires BEFORE ``_hit_to_document`` conversion (cheaper
    # to inspect raw hits than to materialise Document objects we'd
    # throw away). Phase 2 already pre-registered the
    # ``RETRIEVAL_FILTERED_LOW_SCORE`` counter — we just had to wire
    # the ``.inc()`` call site. When the gate fires:
    #
    #   * ``retrieval_status = "low_relevance"`` — Phase 1's
    #     ``REFUSAL_TEMPLATES["low_relevance"]`` template auto-fires
    #     in ``react_generate``. Without this gate, the LLM would
    #     see a corpus of "near-miss" documents and try to answer
    #     the question from them, fabricating a citation to the
    #     closest hit.
    #   * The pre-registered counter ticks so dashboards show the
    #     rate (and operators can tune the threshold).
    #   * ``rerank_failed=True`` hits from ``rerank()`` (Phase 4
    #     PR-1 #6) get ``rerank_score=0.0``. The gate treats those
    #     as "scores unreliable — skip the threshold check" rather
    #     than as low-relevance. Without this carve-out a rerank
    #     exception would always push the user to the refusal
    #     template even though we have real (unranked) hits to work
    #     with. The carve-out is also what keeps test fixtures that
    #     inject a stub ``_FakeReranker`` (which raises because it
    #     lacks ``.score()``) from spuriously hitting the gate.
    from src.agent.metrics import RETRIEVAL_FILTERED_LOW_SCORE

    min_top = RETRIEVAL_DEFAULTS["min_top_relevance_score"]
    # Carve-out: if every hit has ``rerank_failed=True`` we can't
    # trust the score signal at all (all zeros are an artefact of
    # the rerank exception, not a relevance signal). Skip the gate
    # and let the downstream _hit_to_document consume the unranked
    # hits. The reranker exception is already logged in rerank.py
    # so observability isn't lost.
    any_rerank_ok = any(not h.get("rerank_failed", False) for h in reranked)
    if reranked and any_rerank_ok:
        top_score = max(
            (h.get("rerank_score", h.get("rrf_score", 0.0)) for h in reranked),
            default=0.0,
        )
        if top_score < min_top:
            RETRIEVAL_FILTERED_LOW_SCORE.inc()
            # f-string (NOT loguru's lazy ``%s`` / ``%.3f``
            # placeholders) — same discipline as the citations.py
            # collapse log and the Phase 3 #4 debugging坑. A loguru
            # sink with ``format="{message}"`` captures the raw
            # format string for lazy placeholders, which is why the
            # captured stderr in test runs showed ``%.3f`` literal
            # instead of the substituted numbers.
            logger.warning(
                f"retrieve: top rerank_score={top_score:.3f} < "
                f"threshold={min_top:.3f}; returning empty with "
                f"retrieval_status='low_relevance'"
            )
            return {
                "documents": [],
                "retrieval_status": "low_relevance",
                "summary_intent_used": False,
            }

    docs = [_hit_to_document(h) for h in reranked]
    # v2.0.29.4 (Phase 4 PR-2) — drop chunks whose doc_id was
    # superseded (operator re-uploaded with a new doc_id and marked
    # the old one ``status="superseded"``). Run AFTER hit-to-doc
    # conversion (we need doc_id on each Document's metadata) but
    # BEFORE the min_top_relevance_score gate (so a fully-superseded
    # retrieval can still hit the low_relevance path if it drops to
    # zero hits). The rerank_failed carve-out (PR-1 #6) protects
    # against spurious low-relevance classification when every hit
    # was an unranked fallback.
    docs = _filter_superseded(docs, thread_id=thread_id)
    # v2.0.29.8 (Phase 7) — sliding window for per-doc overflow.
    # Hook AFTER supersede filter (so superseded long docs don't
    # waste window slots) and BEFORE the empty check (so a
    # fully-trimmed retrieval still hits ``retrieval_status="empty"``).
    # The hybrid path's candidate set is post-rerank top-K
    # (typically 5-20 docs); per-doc overflow is rare here but the
    # helper handles it the same way as the bulk summary path.
    docs = _filter_long_docs(docs, thread_id=thread_id)
    if not docs:
        # v2.0.29.1 (Phase 1) — hybrid + rerank produced zero hits.
        # This is the "empty" retrieval case: BM25 found nothing AND
        # dense found nothing. Force the refusal template.
        retrieval_status = "empty"
        # v2.0.29.2 (Phase 2) — observability hook.
        from src.agent.metrics import RETRIEVAL_EMPTY

        RETRIEVAL_EMPTY.inc()
    else:
        # v2.0.29.1 (Phase 1) — retrieval returned docs and the gate
        # above (PR-1 #5) didn't fire; mark as success.
        retrieval_status = "success"

    return {
        "documents": docs,
        # v2.0.29.2 (Phase 2) — see the empty-path comment for why
        # we removed the dead ``retrieval_count`` / ``iteration_count``
        # writes. Counter-side observability lives in
        # ``src.agent.metrics.RETRIEVAL_EMPTY``.
        "retrieval_status": retrieval_status,
    }


__all__ = [
    "retrieve_hybrid",
    "retrieve_hybrid_async",
    "_hit_to_document",
    "_bulk_chunk_to_document",
    "_filter_superseded",
    "_filter_long_docs",
    "_looks_like_summary_intent",
    "reset_for_tests",
]


# P2-1: backward-compat alias. Tests that import the sync
# name (e.g. ``from src.agent.nodes.retrieve import retrieve_hybrid``) still
# resolve; the symbol points at the async entry the graph uses.
def retrieve_hybrid(state):
    """Sync wrapper for tests; the graph calls the async entry directly."""
    return asyncio.run(retrieve_hybrid_async(state))
