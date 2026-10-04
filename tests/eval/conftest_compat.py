"""Opt-in RAG_DATA_DIR isolation for CLI runs.

The autouse `_isolate_user_data_dir` fixture in tests/conftest.py only
fires inside pytest. CLI runs of `python -m tests.eval.run` would
otherwise inherit the user's `~/.rag_assistant` and pollute their real
documents / history with throwaway test data.

`ensure_isolated_data_dir()` is the entry point: if ``RAG_DATA_DIR`` is
not set in the environment, it creates a temp dir and sets it (with a
warning). Returns the resolved absolute path so callers can use it.

This is a companion module — NOT a pytest fixture. Importable from the
CLI entry point only.
"""
from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path


_TMP_RAG_DATA_DIR: Path | None = None


def ensure_isolated_data_dir(*, prefix: str = "yuanrag-eval-") -> Path:
    """Return a hermetic RAG_DATA_DIR, creating one if not set.

    Order of operations:
      1. If ``RAG_DATA_DIR`` is already in env, return its absolute path.
      2. Otherwise create ``tempfile.mkdtemp(prefix=prefix)``, export it
         in ``os.environ``, and warn the operator. The temp dir is NOT
         cleaned up automatically (so a slow LLM run can't have its
         state yanked mid-flight). Process exit will reclaim it.
    """
    global _TMP_RAG_DATA_DIR
    existing = os.environ.get("RAG_DATA_DIR")
    if existing:
        return Path(existing).resolve()
    if _TMP_RAG_DATA_DIR is None:
        _TMP_RAG_DATA_DIR = Path(tempfile.mkdtemp(prefix=prefix))
        os.environ["RAG_DATA_DIR"] = str(_TMP_RAG_DATA_DIR)
        warnings.warn(
            f"RAG_DATA_DIR not set; eval harness using temp dir "
            f"{_TMP_RAG_DATA_DIR} (real ~/.rag_assistant untouched)",
            stacklevel=2,
        )
    return _TMP_RAG_DATA_DIR


def temp_data_dir() -> Path | None:
    """Public — for tests to verify isolation happened (or not)."""
    return _TMP_RAG_DATA_DIR