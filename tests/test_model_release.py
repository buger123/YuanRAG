"""v2.0.28.6 Item 11 PR-4 P1-R4 — BGE-M3 + BGE-reranker ``release()``.

Pre-PR-4, the two singletons in ``src/embeddings/bge_m3.py`` and
``src/reranker/bge_reranker.py`` had ``reset_for_tests()`` (module-
state-only, used by ``conftest._reset_all_singletons``) but no public
``release()``. Production lifespan shutdown never freed GPU memory —
on a long-running GPU box the PyTorch allocator holds VRAM until the
process exits, leading to slow VRAM creep across restarts.

Post-PR-4, both modules expose ``release()`` which:
  1. Drops the model reference + clears caches
  2. Best-effort ``torch.cuda.empty_cache()`` (swallows ImportError
     when torch is absent in CPU-only envs)
  3. Calls ``gc.collect()`` to free any cyclic references

This file pins the release contract for both modules.
"""
from __future__ import annotations

import pytest


def test_bge_m3_release_drops_singleton_ref_and_clears_cache(monkeypatch):
    """After ``release()``, ``models_loaded()`` returns False and the
    query cache is empty.

    The CUDA branch is best-effort: we don't force torch + CUDA to be
    installed in the test env. The non-CUDA branch (state reset +
    cache clear + gc.collect) is the deterministic contract.
    """
    from src.embeddings import bge_m3

    # Mark the singleton as loaded + populate the cache.
    monkeypatch.setattr(bge_m3, "_loaded", True)
    monkeypatch.setattr(bge_m3, "_model", object())  # placeholder
    monkeypatch.setattr(bge_m3, "_model_name", bge_m3._DEFAULT_MODEL)
    bge_m3._query_cache["sentinel"] = {"dense": [0.1] * 16, "sparse": {}}

    assert bge_m3.models_loaded()
    assert len(bge_m3._query_cache) == 1

    bge_m3.release()

    assert not bge_m3.models_loaded()
    assert bge_m3._model is None
    assert bge_m3._model_name is None
    assert len(bge_m3._query_cache) == 0


def test_bge_reranker_release_drops_singleton_ref(monkeypatch):
    """After ``release()``, ``model_loaded()`` returns False.

    Same shape as the BGE-M3 test — verify the reranker's
    module-state reset + CUDA-empty-cache best-effort path.
    """
    from src.reranker import bge_reranker

    monkeypatch.setattr(bge_reranker, "_loaded", True)
    monkeypatch.setattr(bge_reranker, "_reranker", object())  # placeholder
    monkeypatch.setattr(bge_reranker, "_model_name", bge_reranker._DEFAULT_MODEL)

    assert bge_reranker.model_loaded()

    bge_reranker.release()

    assert not bge_reranker.model_loaded()
    assert bge_reranker._reranker is None
    assert bge_reranker._model_name is None


def test_release_is_idempotent_when_not_loaded():
    """Calling ``release()`` on an already-clean singleton is a no-op.

    The lifespan shutdown wires ``release()`` unconditionally; if the
    model was never loaded (e.g. on a startup that failed before
    warmup), the call must not raise.
    """
    from src.embeddings import bge_m3
    from src.reranker import bge_reranker

    # Ensure clean state.
    bge_m3.reset_for_tests()
    bge_reranker.reset_for_tests()

    # No exceptions on double-release.
    bge_m3.release()
    bge_reranker.release()
    bge_m3.release()  # second time, still clean

    assert not bge_m3.models_loaded()
    assert not bge_reranker.model_loaded()