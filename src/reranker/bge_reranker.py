"""BGE Reranker v2-M3 wrapper using FlagEmbedding.

FlagEmbedding's ``FlagReranker`` loads the BGE-reranker-v2-m3 cross-encoder
and returns sigmoid scores per (query, document) pair.

Singleton pattern: same rationale as ``src/embeddings/bge_m3.py`` —
``functools.lru_cache`` doesn't serialize concurrent body execution, so we
use a manual ``threading.Lock`` + ``_loaded`` flag. The earlier FastEmbed
version called a ``TextCrossEncoder`` that didn't exist in FastEmbed 0.8.0
and silently failed.
"""
from __future__ import annotations

import threading
from typing import Iterable, List

from src.core.logging import logger

try:
    from FlagEmbedding import FlagReranker
except Exception:  # pragma: no cover
    FlagReranker = None  # type: ignore


_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize FlagReranker.compute_score (not thread-safe)
_reranker: "FlagReranker | None" = None
_loaded: bool = False
_loading: bool = False  # True while a load is in flight (else False)
_model_name: str | None = None
_load_error: BaseException | None = None


def _load_blocking(model_name: str) -> FlagReranker:
    if FlagReranker is None:
        raise RuntimeError("FlagEmbedding is not installed. pip install FlagEmbedding")
    logger.info(f"Loading BGE reranker model (FlagEmbedding): {model_name}")
    return FlagReranker(
        model_name_or_path=model_name,
        use_fp16=True,
        normalize=True,
        max_length=512,
    )


def get_model(model_name: str = _DEFAULT_MODEL) -> FlagReranker:
    """Return the singleton reranker, loading it on first call.

    IMPORTANT: synchronous body takes ~10-30 s. Callers on the asyncio
    event loop MUST wrap this in ``asyncio.to_thread(...)``. The
    ``/models/download`` route uses ``asyncio.to_thread``; chat callers
    should check ``model_loaded()`` first to avoid blocking the WS.
    """
    global _reranker, _loaded, _loading, _model_name, _load_error
    if _loaded and _reranker is not None:
        if model_name != _model_name:
            raise ValueError(
                f"BGE reranker singleton already loaded as {_model_name!r}; "
                f"cannot switch to {model_name!r} without an explicit reset."
            )
        return _reranker
    with _lock:
        if _loaded and _reranker is not None:
            if model_name != _model_name:
                raise ValueError(
                    f"BGE reranker singleton already loaded as {_model_name!r}; "
                    f"cannot switch to {model_name!r} without an explicit reset."
                )
            return _reranker
        if not _loaded:
            _loading = True
            _load_error = None
            try:
                _reranker = _load_blocking(model_name)
                _model_name = model_name
                _loaded = True
            except BaseException as exc:
                _load_error = exc
                logger.exception(f"BGE reranker load failed for {model_name}")
                raise
            finally:
                _loading = False
    return _reranker


def model_loaded(model_name: str = _DEFAULT_MODEL) -> bool:
    """Cheap status check used by ``/models/status`` — does NOT trigger load.

    Returns True only when the reranker is fully loaded AND the loaded
    model matches the requested name.
    """
    return _loaded and _reranker is not None and _model_name == model_name


def model_loading() -> bool:
    """True iff a load is currently in progress (else False).

    Cheap status check, no I/O. Used by ``/models/status`` so the frontend
    can show a spinner while the reranker is loading.
    """
    return _loading


def load_error() -> str | None:
    """Return the last load failure reason (str), or None if no failure."""
    return str(_load_error) if _load_error is not None else None


def reset_for_tests() -> None:
    """Reset the singleton state. Tests only."""
    global _reranker, _loaded, _loading, _model_name, _load_error
    _reranker = None
    _loaded = False
    _loading = False
    _model_name = None
    _load_error = None


def release() -> None:
    """Production-grade singleton release. Wired to FastAPI lifespan shutdown.

    Mirror of ``src.embeddings.bge_m3.release()`` — drops the model
    reference, returns cached VRAM (when CUDA is available), and
    triggers a Python GC pass. Idempotent.

    See ``src/embeddings/bge_m3.py`` docstring on ``release()`` for the
    rationale on splitting release from reset_for_tests (CUDA empty_cache
    vs no-torch overhead in test paths).
    """
    global _reranker, _loaded, _loading, _model_name, _load_error
    _reranker = None
    _loaded = False
    _loading = False
    _model_name = None
    _load_error = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("BGE-reranker: released cached CUDA memory")
    except Exception:
        pass
    import gc

    gc.collect()


class BGEReranker:
    """Compute reranker scores for (query, doc) pairs."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model_name = model_name

    def score(self, query: str, documents: Iterable[str]) -> List[float]:
        """Score each document against the query, returning one float per doc."""
        docs = list(documents)
        if not docs:
            return []
        reranker = get_model(self.model_name)
        pairs = [(query, d) for d in docs]
        # FlagReranker.compute_score is NOT thread-safe — concurrent calls
        # on the same model instance produce interleaved/wrong scores
        # silently. Serialize via a lock distinct from the load lock.
        with _infer_lock:
            scores = reranker.compute_score(pairs, normalize=True)
        return [float(s) for s in scores]


__all__ = [
    "BGEReranker",
    "get_model",
    "model_loaded",
    "model_loading",
    "load_error",
    "reset_for_tests",
    "release",
]