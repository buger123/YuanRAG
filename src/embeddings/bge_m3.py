"""BGE-M3 embedding wrapper using FlagEmbedding.

FlagEmbedding's ``BGEM3FlagModel`` produces dense (1024-d), sparse (BM25-like
token weights) and ColBERT vectors in a single forward pass. We use the dense
vector for semantic similarity and the sparse weights for hybrid BM25 fusion.

The model is heavyweight (PyTorch + multi-GB download), so we MUST only
load it once per process. The earlier FastEmbed-based implementation
suffered from two compounding problems:

1. FastEmbed 0.8.0 does NOT support ``BAAI/bge-m3`` — every call to
   ``TextEmbedding(model_name="BAAI/bge-m3")`` raised ``ValueError``, which
   the route handler swallowed silently, and the frontend polled forever.
2. The polling itself re-entered the loader on every request because
   ``functools.lru_cache`` does NOT serialize concurrent calls — it only
   memoizes the result, so N threads racing on a cold cache all run the
   expensive body in parallel.

Fix: a single ``threading.Lock`` around the body (one body run, ever), plus
a ``_loaded`` flag for the cheap status check used by ``/models/status``.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Iterable, List

from src.core.logging import logger

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception:  # pragma: no cover
    BGEM3FlagModel = None  # type: ignore


_DEFAULT_MODEL = "BAAI/bge-m3"

_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize FlagEmbedding.encode_* (not thread-safe)
_model: "BGEM3FlagModel | None" = None
_loaded: bool = False
_loading: bool = False  # True while a load is in flight (else False)
_model_name: str | None = None
_load_error: BaseException | None = None

# QW #7: bounded LRU cache for ``embed_query`` output. Same-query
# replays ("总结一下", "what is RAG?", "你能做什么?") hit the cache
# instead of paying 50-200 ms of BGE-M3 inference per call. Bounded
# to 1024 entries (~256 KB at 1024-d float vectors + sparse dict) so
# a flood of unique queries can't OOM the worker. ``OrderedDict``
# gives us O(1) move-to-end on hit + O(1) popitem(last=False) on
# evict; ``functools.lru_cache`` would also work but doesn't expose
# a clear() hook compatible with ``reset_for_tests`` below.
#
# Why per-process, not per-thread
# -------------------------------
# Embedding is read-mostly and deterministic for a given model
# version, so a shared cache across threads is safe AND beneficial
# (concurrent chat threads on the same query both hit). The lock
# is held across the WHOLE ``embed_query`` call (v2.0.28.5 P1-R1) —
# including inference — so N threads on a cold key collapse to 1
# inference. ``_infer_lock`` inside ``_embed`` already serializes
# FlagEmbedding.encode_* calls, so the cache lock adds no extra
# serialization; it just gates the dedup window.
_QUERY_CACHE_MAXSIZE = 1024
_query_cache: "OrderedDict[str, dict]" = OrderedDict()
_query_cache_lock = threading.Lock()


def _load_blocking(model_name: str) -> BGEM3FlagModel:
    """Actually instantiate the model — called at most once per process."""
    if BGEM3FlagModel is None:
        raise RuntimeError("FlagEmbedding is not installed. pip install FlagEmbedding")
    logger.info(f"Loading BGE-M3 model (FlagEmbedding): {model_name}")
    # use_fp16=True halves VRAM; query_max_length/passage_max_length=512 matches
    # the chunking budget in config/constants.py. return_sparse=True enables
    # hybrid BM25 fusion in the retriever.
    m = BGEM3FlagModel(
        model_name_or_path=model_name,
        use_fp16=True,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        query_max_length=512,
        passage_max_length=512,
    )
    return m


def get_model(model_name: str = _DEFAULT_MODEL) -> BGEM3FlagModel:
    """Return the singleton model, loading it on first call.

    Threading contract: even if N threads race here on a cold cache, exactly
    ONE thread runs the body; the rest block on the lock and then read the
    same instance. After the first successful load, all subsequent callers
    take the fast path without touching the lock.

    IMPORTANT: the load body is synchronous and takes ~10-30 s for BGE-M3.
    Callers that run on the asyncio event loop MUST wrap this in
    ``asyncio.to_thread(...)`` — otherwise the entire server (HTTP, WS,
    polling) freezes for the duration. The ``/models/download`` route uses
    ``asyncio.to_thread``; ``stream_agent`` checks ``models_loaded()``
    first and yields an error event instead of calling this on the hot path.
    """
    global _model, _loaded, _loading, _model_name, _load_error
    if _loaded and _model is not None:
        if model_name != _model_name:
            raise ValueError(
                f"BGE-M3 singleton already loaded as {_model_name!r}; "
                f"cannot switch to {model_name!r} without an explicit reset."
            )
        return _model
    with _lock:
        if _loaded and _model is not None:
            if model_name != _model_name:
                raise ValueError(
                    f"BGE-M3 singleton already loaded as {_model_name!r}; "
                    f"cannot switch to {model_name!r} without an explicit reset."
                )
            return _model
        if not _loaded:
            _loading = True
            _load_error = None
            try:
                _model = _load_blocking(model_name)
                _model_name = model_name
                _loaded = True
            except BaseException as exc:
                # Cache the failure so subsequent callers can surface the
                # reason via load_error() instead of re-running the entire
                # ~30 s load. Do NOT swallow — re-raise so the caller
                # (BackgroundTasks done-callback, /models/download) sees it.
                _load_error = exc
                logger.exception(f"BGE-M3 load failed for {model_name}")
                raise
            finally:
                _loading = False
    return _model


def models_loaded(model_name: str = _DEFAULT_MODEL) -> bool:
    """Cheap status check used by ``/models/status`` — does NOT trigger load.

    Returns True only when a model is fully loaded AND the loaded model
    matches the requested name. (The previous version returned ``_loaded``
    which is True even if ``_model`` is None or the name mismatches — a
    subtle inconsistency that confused tests and could mask real failures.)
    """
    return _loaded and _model is not None and _model_name == model_name


def model_loading() -> bool:
    """True iff a load is currently in progress (else False).

    Cheap status check, no I/O. Used by ``/models/status`` so the frontend
    can show a spinner while the model is loading instead of guessing from
    a stale 'missing' state.
    """
    return _loading


def load_error() -> str | None:
    """Return the last load failure reason (str), or None if the last
    load succeeded or none has been attempted.

    Used by ``/models/status`` so the frontend can distinguish "user
    hasn't clicked Download yet" from "download failed" — the previous
    behavior conflated the two and silently swallowed the reason.
    """
    return str(_load_error) if _load_error is not None else None


def reset_for_tests() -> None:
    """Reset the singleton state. Tests only."""
    global _model, _loaded, _loading, _model_name, _load_error
    _model = None
    _loaded = False
    _loading = False
    _model_name = None
    _load_error = None
    # QW #7: also flush the query cache so the next test doesn't see
    # vectors computed by the previous test's (possibly different)
    # model or settings. Clear is O(N) but N ≤ 1024 and tests run
    # at most once per test — negligible.
    _query_cache.clear()


def release() -> None:
    """Production-grade singleton release. Wired to FastAPI lifespan shutdown.

    Drops the model reference + flushes the query cache + (when CUDA is
    available) releases the cached GPU memory + triggers a Python GC pass.
    Idempotent — safe to call when the model isn't loaded.

    Why a dedicated ``release()`` separate from ``reset_for_tests()``
    ----------------------------------------------------------------
    ``reset_for_tests`` is module-state-only (set globals to None) so it
    stays fast and never imports torch. ``release()`` additionally tries
    to return GPU memory to the OS via ``torch.cuda.empty_cache()`` —
    that matters for long-running GPU boxes where PyTorch's allocator
    holds VRAM until process exit otherwise. The CUDA call is wrapped in
    try/except (torch may not be importable in CPU-only environments).
    """
    global _model, _loaded, _loading, _model_name, _load_error
    _model = None
    _loaded = False
    _loading = False
    _model_name = None
    _load_error = None
    _query_cache.clear()
    # Best-effort VRAM return. Skip silently if torch isn't installed
    # (CPU-only envs) — the test suite doesn't always pull torch.
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("BGE-M3: released cached CUDA memory")
    except Exception:
        # torch not importable (CPU build) or cuda unavailable — nothing to release.
        pass
    import gc

    gc.collect()


class BGEM3Embedder:
    """Embed documents/queries via the singleton BGE-M3 model.

    Output per text::

        {
            "dense": list[float],          # 1024-d normalized
            "sparse": dict[int, float],     # {token_id: weight}, BM25-style
        }
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name

    @staticmethod
    def _to_record_list(output) -> List[dict]:
        """Convert a FlagEmbedding ``encode_*`` output to the
        ``{dense, sparse}`` record list that downstream code consumes.

        Pulled out of ``_embed`` so the sub-batch path (v2.0.7 perf-5)
        can reuse the same coercion without re-inlining the loop.
        """
        dense_vecs = output["dense_vecs"]
        sparse_weights = output["lexical_weights"]
        results = []
        for dv, sw in zip(dense_vecs, sparse_weights):
            # FlagEmbedding returns sparse as {token_str: weight}; we keep
            # the same shape as before (the rest of the pipeline reads the
            # keys as ints, but token strings also work since LanceDB
            # doesn't care about the key type).
            results.append(
                {
                    "dense": dv.tolist() if hasattr(dv, "tolist") else list(dv),
                    "sparse": dict(sw) if sw else {},
                }
            )
        return results

    def _embed(self, texts: List[str], *, is_query: bool) -> List[dict]:
        if not texts:
            return []
        model = get_model(self.model_name)
        # FlagEmbedding's ``encode_queries``/``encode_corpus`` are NOT
        # thread-safe — concurrent calls on the same model instance produce
        # interleaved/wrong embeddings silently. We serialize with a lock
        # that is DISTINCT from the load lock, so an in-flight load does
        # not block inference (or vice versa). See audit findings —
        # concurrent ingest + chat races previously produced semantically
        # wrong vectors.
        #
        # v2.0.7 perf-5: corpus (ingest) batches are processed in 64-chunk
        # sub-batches with the lock released between each, so a long
        # ingest (500-5000 chunks) no longer starves a concurrent chat
        # ``embed_query`` for 30-90 seconds. Queries stay single-shot.
        results: List[dict] = []
        if is_query or len(texts) <= 64:
            with _infer_lock:
                output = (
                    model.encode_queries(
                        texts, return_dense=True, return_sparse=True
                    )
                    if is_query
                    else model.encode_corpus(
                        texts, return_dense=True, return_sparse=True
                    )
                )
            results.extend(self._to_record_list(output))
            return results

        # Corpus path: process in 64-chunk sub-batches. FlagEmbedding
        # internally batches at 64, so the sub-batch boundary matches the
        # kernel batching — no perf regression vs. one mega-call, but the
        # ``_infer_lock`` is released between sub-batches so chat queries
        # can interleave.
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            with _infer_lock:
                output = model.encode_corpus(
                    batch, return_dense=True, return_sparse=True
                )
            results.extend(self._to_record_list(output))
        return results

    def embed_documents(self, texts: Iterable[str]) -> List[dict]:
        """Batch embed a list of documents (used at ingest time)."""
        return self._embed(list(texts), is_query=False)

    def embed_query(self, text: str) -> dict:
        """Embed a single query.

        QW #7: same-query replays hit an in-process LRU cache instead
        of paying BGE-M3 inference again. The cache key is the raw
        query string — BGE-M3 is deterministic so this is safe, and
        the cache is reset by ``reset_for_tests`` along with the
        model singleton. The cache is also busted automatically on
        model swap (different ``_model_name``) via
        ``_query_cache_key``.

        v2.0.28.5 P1-R1 — lock-across-inference dedup. The pre-PR-3
        read path (``_query_cache.get(cache_key)`` outside the lock)
        was GIL-safe for pointer reads but NOT safe for the miss →
        inference → ``__setitem__`` sequence: N threads on the same
        cold key all miss, all run BGE-M3 inference (50-200 ms
        each), all populate.

        First attempted fix (double-checked-locking) didn't work:
        under-lock get returns None for every thread because no one
        has populated yet; releasing the lock and re-checking after
        inference only catches the race AFTER inference ran — N
        inferences have already happened. The correct pattern is to
        hold ``_query_cache_lock`` for the WHOLE embed_query
        including inference. ``_infer_lock`` inside ``_embed``
        already serializes FlagEmbedding.encode_* calls, so holding
        ``_query_cache_lock`` adds zero extra serialization — it
        just gates the cache miss path so N threads on the same
        cold key get serialized into one inference.
        """
        cache_key = self._query_cache_key(text)
        with _query_cache_lock:
            cached = _query_cache.get(cache_key)
            if cached is not None:
                # Move-to-end so the entry is the freshest (true LRU
                # semantics). Holding the lock across get + move_to_end
                # keeps these two ops atomic.
                _query_cache.move_to_end(cache_key)
                return cached
            # Cache miss — run inference UNDER the cache lock. This
            # serializes same-key misses (N threads on one cold key
            # → 1 inference) at the cost of serializing ALL cache
            # misses (different keys block each other). The latter
            # is fine because ``_infer_lock`` inside ``_embed``
            # already serializes FlagEmbedding calls anyway, so
            # ``_query_cache_lock`` adds zero real serialization —
            # it just gates the dedup window.
            result = self._embed([text], is_query=True)[0]
            _query_cache[cache_key] = result
            # Evict the oldest entry once we exceed the cap. We pop
            # BEFORE inserting so the new entry doesn't immediately
            # get evicted by its own insertion.
            while len(_query_cache) > _QUERY_CACHE_MAXSIZE:
                _query_cache.popitem(last=False)
        return result

    @staticmethod
    def _query_cache_key(text: str) -> str:
        """Cache key = ``"<model_name>|<text>"``.

        Mixing the model name in keeps multi-model deployments (or
        a future model swap) from serving stale vectors computed by
        a different model. The cost is one extra string concat per
        call — negligible vs. the embedding compute it saves.
        """
        # ``_model_name`` is set during load; before load the cache
        # is empty anyway (no vectors to look up), so the default
        # ``_DEFAULT_MODEL`` key prefix is correct.
        return f"{_model_name or _DEFAULT_MODEL}|{text}"


__all__ = [
    "BGEM3Embedder",
    "get_model",
    "models_loaded",
    "model_loading",
    "load_error",
    "reset_for_tests",
]