"""Rerank the fused candidates down to a small top-K via BGE Reranker v2-M3."""
from __future__ import annotations

from src.core.logging import logger
from src.reranker.bge_reranker import BGEReranker


_reranker: BGEReranker | None = None


def _get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker


def reset_for_tests() -> None:
    """Forget the cached reranker instance. Tests only."""
    global _reranker
    _reranker = None


def rerank(query: str, hits: list[dict], top_k: int = 5) -> list[dict]:
    """Score (query, doc) pairs and return the top-``top_k``.

    v2.0.29.4 (Phase 4 PR-1) — on reranker failure, stamp every
    returned hit with ``rerank_failed=True`` + ``rerank_score=0.0``
    so downstream ``_hit_to_document`` (``src/agent/nodes/retrieve.py:73``)
    and the ``min_top_relevance_score`` gate (PR-1 #5) can see the
    failure instead of using the (unranked) RRF score as a fake
    rerank score. Without these stamps, a rerank failure would
    silently let high-RRF but low-relevance hits pass the gate.
    """
    if not hits:
        return []
    texts = [h.get("text", "") for h in hits]
    try:
        scores = _get_reranker().score(query, texts)
    except Exception as exc:
        logger.exception(f"Rerank failed, returning input order: {exc}")
        # Phase 4 PR-1 — stamp failure on every returned hit. The
        # caller (``retrieve_hybrid_async``) consumes ``rerank_score``
        # to decide whether the gate fires, so a silent ``0.0`` would
        # be ambiguous ("did the reranker score this as 0?" vs "did
        # the reranker fail?"). ``rerank_failed=True`` is the explicit
        # signal for downstream code.
        sliced = hits[:top_k]
        for h in sliced:
            h["rerank_score"] = 0.0
            h["rerank_failed"] = True
        return sliced

    for hit, s in zip(hits, scores):
        hit["rerank_score"] = float(s)
    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]


__all__ = ["rerank", "reset_for_tests"]
