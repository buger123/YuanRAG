"""v2.0.29.6 (Phase 5) — top_k kwarg threading through ``retrieve_docs`` → ``retrieve_hybrid_async``.

Pre-Phase-5 used a context-manager override of the module-level
``RETRIEVAL_DEFAULTS`` dict. Two concurrent ``retrieve_docs`` calls
on different threads would race: A sets ``top_k_post_rerank=10``,
B sets ``top_k_post_rerank=20``, A's ``finally`` restores the
original value → B sees the restored old value, not 20.

Phase 5 threads `top_k` through as an explicit kwarg so each call
sees its own value with no shared state.

These tests pin:
  1. ``retrieve_hybrid_async`` accepts ``top_k`` kwarg.
  2. ``retrieve_hybrid_async`` falls back to default when ``top_k=None``.
  3. ``retrieve_docs`` no longer mutates ``RETRIEVAL_DEFAULTS``.
  4. The ``_override_top_k`` context manager is gone (no
     ``contextlib`` import in the tool module anymore).
  5. Concurrent calls with different ``top_k`` each see their own
     value (no cross-call mutation).
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from src.agent.nodes import retrieve as retrieve_mod
from src.agent.nodes.retrieve import retrieve_hybrid_async


def _read_retrieve_docs_source() -> str:
    """Read the raw ``src/agent/tools/retrieve_docs.py`` source.

    The ``@tool`` decorator wraps the ``retrieve_docs`` async function
    into a ``StructuredTool`` instance whose name shadows the
    original function on the module. We can't reliably introspect
    the wrapped object, so we read the source file directly to check
    for the deleted context manager and the new explicit kwarg.
    """
    # ``retrieve_mod.__file__`` is ``src/agent/nodes/retrieve.py``,
    # so we walk up one level (parent.parent) to ``src/agent`` and
    # descend into ``tools/retrieve_docs.py``.
    src_path = Path(retrieve_mod.__file__).parent.parent / "tools" / "retrieve_docs.py"
    return src_path.read_text(encoding="utf-8")


# ============================================================
# 1. retrieve_hybrid_async signature accepts top_k
# ============================================================


def test_retrieve_hybrid_async_accepts_top_k_kwarg():
    """v2.0.29.6 — ``retrieve_hybrid_async(state, *, top_k=None)`` signature."""
    sig = inspect.signature(retrieve_hybrid_async)
    assert "top_k" in sig.parameters
    # Keyword-only
    assert sig.parameters["top_k"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["top_k"].default is None


# ============================================================
# 2. Default fallback (legacy callers don't pass top_k)
# ============================================================


def test_retrieve_hybrid_async_top_k_none_falls_back_to_default():
    """When ``top_k=None``, the function reads from ``RETRIEVAL_DEFAULTS``."""
    src = inspect.getsource(retrieve_hybrid_async)
    assert "top_k if top_k is not None else RETRIEVAL_DEFAULTS" in src, (
        "retrieve_hybrid_async should fall back to RETRIEVAL_DEFAULTS when top_k is None"
    )


# ============================================================
# 3. retrieve_docs no longer mutates RETRIEVAL_DEFAULTS
# ============================================================


def test_retrieve_docs_does_not_mutate_retrieval_defaults():
    """The tool module must NOT import contextlib for top_k override."""
    src = _read_retrieve_docs_source()
    # The override context manager is the only ``contextlib`` import in
    # the pre-Phase-5 module — its absence confirms the change landed.
    assert "import contextlib" not in src, (
        "retrieve_docs still imports contextlib; _override_top_k must be deleted"
    )
    assert "_override_top_k" not in src, (
        "retrieve_docs still defines _override_top_k; Phase 5 should delete it"
    )
    # And the call site passes top_k as an explicit kwarg.
    assert "retrieve_hybrid_async(state_slice, top_k=top_k)" in src, (
        "retrieve_docs should pass top_k as explicit kwarg, not via context manager"
    )


def test_retrieve_docs_docstring_documents_phase_5():
    """Docstring at the call site mentions Phase 5 + thread-safety."""
    src = _read_retrieve_docs_source()
    assert "v2.0.29.6" in src
    assert "thread-safety" in src or "race" in src


# ============================================================
# 4. Concurrent calls do not see each other's top_k
# ============================================================


def test_concurrent_retrieve_hybrid_async_with_distinct_top_k(monkeypatch):
    """Two concurrent calls with different ``top_k`` values each see their own.

    We patch the ``retrieve_hybrid_async`` symbol at the
    ``src.agent.tools.retrieve_docs`` module level (where it was
    bound at import time) and verify that asyncio.gather'd calls
    each observe their own ``top_k`` without one clobbering the
    other. This is the actual race the Phase 5 fix prevents.
    """
    captured_top_k: list[int] = []

    async def fake_retrieve_hybrid_async(state_slice, *, top_k=None):
        captured_top_k.append(int(top_k))
        # Mimic a small async delay to let the other task race in.
        await asyncio.sleep(0.01)
        return {
            "documents": [],
            "retrieval_status": "success",
            "summary_intent_used": False,
        }

    # Patch the import-site reference. ``@tool`` imports
    # ``retrieve_hybrid_async`` at module top — that binding is the
    # one used at call time. We access the actual module object
    # (not the StructuredTool shadowed by `from ... import`)
    # via ``importlib.import_module``.
    import importlib

    tool_mod_local = importlib.import_module("src.agent.tools.retrieve_docs")

    monkeypatch.setattr(
        tool_mod_local, "retrieve_hybrid_async", fake_retrieve_hybrid_async
    )

    # asyncio.gather two calls with distinct top_k values. Use the
    # ``StructuredTool.coroutine`` attribute to invoke the wrapped
    # async function directly (bypassing Pydantic arg-schema binding,
    # which would reject the ``thread_id`` injection).
    raw_func = tool_mod_local.retrieve_docs.coroutine

    async def run():
        results = await asyncio.gather(
            raw_func(query="q1", top_k=3, thread_id="t"),
            raw_func(query="q2", top_k=11, thread_id="t"),
        )
        return results

    docs_out = asyncio.run(run())

    # Each call should have observed its own top_k, not the other's.
    assert sorted(captured_top_k) == [3, 11], (
        f"Concurrent calls should observe distinct top_k, got {captured_top_k}"
    )
    # Both calls returned the empty-list JSON form.
    assert all(d == "[]" for d in docs_out)