"""Hybrid retrieval: BM25 (Tantivy via LanceDB FTS) + dense vector → RRF fusion."""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

from config.constants import RRF_K
from src.core.logging import logger
from src.storage.lancedb_store import _safe_identifier, get_table


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Each list is ``[{chunk_id, ...}, ...]`` (top-N retrieval). The fused
    score for a chunk is ``sum(1 / (k + rank))`` across all lists where
    the chunk appears.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            cid = doc.get("chunk_id")
            if cid is None:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id[cid] = doc
    fused_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**by_id[cid], "rrf_score": s} for cid, s in fused_sorted]


def _thread_filter(thread_id: Optional[str]) -> Optional[str]:
    """Return a validated filter expression ``"thread_id = '...'"`` or
    ``None`` if filtering should be skipped. Logs and returns ``None`` on
    unsafe thread_id — the caller falls back to unfiltered retrieval
    rather than crashing (a hostile thread_id is bad, but a hard crash is
    worse than a transient global search).
    """
    if not thread_id:
        return None
    try:
        safe = _safe_identifier(thread_id, kind="thread_id")
    except ValueError as exc:
        logger.warning(f"hybrid_search skipping unsafe thread_id filter: {exc}")
        return None
    return f"thread_id = '{safe}'"


def _temporal_filter(now_iso: Optional[str]) -> Optional[str]:
    """v2.0.29.4 (Phase 4 PR-2) — temporal filter: keep only non-expired chunks.

    Returns the SQL where-clause fragment ``"(expires_at IS NULL OR expires_at > '...')"``
    that selects chunks that either have no expiry set (``expires_at IS NULL``
    — pre-PR-2 legacy rows + production uploads with no expiry configured)
    or have an expiry timestamp strictly in the future relative to ``now_iso``.

    Single-quote injection is guarded via ``replace("'", "''")`` so a
    hostile caller can't smuggle SQL through the ``now_iso`` parameter
    (the value is server-controlled — operators pass an ISO 8601 string
    — but defence-in-depth is cheap here and a future caller could
    accept user input).

    ``None`` (the default) means "no temporal filter" — the caller
    didn't opt into time-bounded retrieval. This is the pre-PR-2
    behaviour and is the safe default until callers explicitly ask
    for time-aware retrieval (e.g. an operator setting an expiry
    horizon on a doc).
    """
    if now_iso is None:
        return None
    safe = now_iso.replace("'", "''")
    return f"(expires_at IS NULL OR expires_at > '{safe}')"


def _compose_where(
    thread_filter: Optional[str],
    temporal_filter: Optional[str],
) -> Optional[str]:
    """v2.0.29.4 (Phase 4 PR-2) — combine thread + temporal filters with AND.

    Each filter can independently be ``None`` (skip that dimension).
    Both None returns ``None`` so the caller can still pass
    ``.where(None)`` to LanceDB without re-introducing a where clause
    (which would conflict with the BM25 query syntax).

    Each present filter is wrapped in parentheses so the SQL
    composition is unambiguous even if a future filter adds its own
    AND/OR clauses internally.
    """
    parts = [f for f in (thread_filter, temporal_filter) if f]
    if not parts:
        return None
    return " AND ".join(f"({p})" for p in parts)


def _dense_search(table, query_dense: list[float], top_k: int, where: Optional[str]) -> list[dict]:
    """LanceDB dense vector search (cosine). Sync; intended to run in a thread."""
    dense_q = table.search(query_dense, vector_column_name="vector").metric("cosine")
    if where is not None:
        dense_q = dense_q.where(where)
    return dense_q.limit(top_k).to_list()


def _bm25_search(table, query_text: str, top_k: int, where: Optional[str]) -> list[dict]:
    """LanceDB Tantivy FTS search. Sync; intended to run in a thread."""
    bm25_q = table.search(query_text, query_type="fts")
    if where is not None:
        bm25_q = bm25_q.where(where)
    return bm25_q.limit(top_k).to_list()


# Public aliases — retrieve_hybrid_async needs to submit BM25 BEFORE the
# query embedding is ready (Phase D2 optimization), so we can't route
# through ``hybrid_search`` (which submits dense + BM25 as a pair and
# waits for the embed to be in hand).
dense_search = _dense_search
bm25_search = _bm25_search


# Process-wide executor for the two hybrid searches. Sized to 2 because
# we issue at most 2 (dense + BM25) at once per query. Re-using across
# calls avoids the cost of thread-pool construction on every turn.
_hybrid_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-search")

# v2.0.29.6 (Phase 5) — register atexit shutdown so the worker pool is
# joined cleanly on interpreter exit. Mirrors the pattern at
# ``src/llm/factory.py:18, 274`` (``atexit.register(close_all_llm_clients)``).
# Without this hook, worker threads can linger between uvicorn hot-reload
# cycles or pytest module reloads and on Windows the threads block the
# interpreter's exit handle.
atexit.register(_hybrid_executor.shutdown)


def hybrid_search(
    query_text: str,
    query_dense: list[float],
    top_k: int = 20,
    thread_id: Optional[str] = None,
    *,
    temporal_cutoff_iso: Optional[str] = None,
) -> list[dict]:
    """Run BM25 + dense in parallel, fuse via RRF, return top-``top_k``.

    When ``thread_id`` is provided and validates, both the dense and BM25
    searches apply ``.where(f"thread_id = '{safe}'")`` so retrieval is
    scoped to the conversation that owns those documents.

    v2.0.29.4 (Phase 4 PR-2) — when ``temporal_cutoff_iso`` is supplied,
    an additional temporal filter ``expires_at IS NULL OR expires_at > '<cutoff>'``
    is composed via ``_compose_where``. ``None`` (default) preserves
    pre-PR-2 behaviour: no temporal filter, all chunks visible regardless
    of expiry. Production callers that don't opt in are unaffected.

    Performance: dense and BM25 are independent disk/network reads on
    different indices, so we issue them concurrently via a shared
    ThreadPoolExecutor. On a typical 20-K-chunk thread this halves
    retrieval wall-time vs. the previous serial implementation
    (~20–100 ms saved per retrieval round, plus IO concurrency headroom
    for parallel chat requests).
    """
    table = get_table()
    thread_where = _thread_filter(thread_id)
    # v2.0.29.4 (Phase 4 PR-2) — compose the AND of thread + temporal
    # filters. Each can independently be None; both None yields None
    # (legacy behaviour). Wrapping each filter in parentheses keeps
    # the SQL unambiguous if either side ever grows an internal
    # AND/OR.
    where = _compose_where(thread_where, _temporal_filter(temporal_cutoff_iso))

    # Submit both queries to the executor concurrently. Each search is
    # wrapped in try/except independently so a failure in one path
    # doesn't kill the other — we still return whatever the surviving
    # path produced.
    dense_future = _hybrid_executor.submit(_dense_search, table, query_dense, top_k, where)
    bm25_future = _hybrid_executor.submit(_bm25_search, table, query_text, top_k, where)

    try:
        dense_hits = dense_future.result()
    except Exception as exc:
        logger.exception(f"Dense search failed: {exc}")
        dense_hits = []

    try:
        bm25_hits = bm25_future.result()
    except Exception as exc:
        logger.exception(f"BM25 search failed: {exc}")
        bm25_hits = []

    fused = reciprocal_rank_fusion([dense_hits, bm25_hits])
    return fused[:top_k]


__all__ = ["reciprocal_rank_fusion", "hybrid_search", "dense_search", "bm25_search", "_compose_where", "_temporal_filter"]
