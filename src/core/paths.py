"""User data directory management.

All persistent data lives under the directory specified by the
``RAG_DATA_DIR`` environment variable, falling back to
``~/.rag_assistant`` when unset. Relative paths in ``RAG_DATA_DIR`` are
resolved relative to the project root (the directory containing
``src/main.py``) so users can keep all data inside the repo with a
simple ``RAG_DATA_DIR=./data`` in their ``.env``.

Subdirectories are created lazily on first use.
"""
from __future__ import annotations

import os
from pathlib import Path

from config.constants import (
    HISTORY_DB,
    LANCEDB_DIR,
    LOGS_DIR,
    MODELS_DIR,
    UPLOADS_DIR,
)

APP_DIR_NAME = ".rag_assistant"

# Project root = directory containing this package's `src/` folder.
# Use the shared bootstrap module so app.py and main.py agree on what
# "project root" means. ``project_root()`` is the function; call it once
# at import to capture the resolved path.
from src.core.bootstrap import project_root as _project_root_fn  # noqa: E402

_PROJECT_ROOT = _project_root_fn()


def user_data_dir() -> Path:
    """Return the user's app data directory, creating it if missing.

    Resolution order:
    1. ``RAG_DATA_DIR`` env var (absolute → used as-is; relative →
       resolved against the project root).
    2. ``~/.rag_assistant`` (the historical default).
    """
    raw = os.environ.get("RAG_DATA_DIR")
    if raw:
        base = Path(raw)
        if not base.is_absolute():
            base = (_PROJECT_ROOT / base).resolve()
    else:
        base = Path.home() / APP_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def lancedb_path() -> Path:
    """LanceDB root (vector + FTS indices)."""
    p = user_data_dir() / LANCEDB_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def history_db_path() -> Path:
    """Path to SQLite chat history database."""
    p = user_data_dir() / HISTORY_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """Local cache for ONNX models (BGE-M3, BGE Reranker)."""
    p = user_data_dir() / MODELS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def uploads_dir() -> Path:
    """Raw uploads archive (re-ingest safe)."""
    p = user_data_dir() / UPLOADS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    """Application logs."""
    p = user_data_dir() / LOGS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_all_dirs() -> Path:
    """Create all required directories and return the root."""
    base = user_data_dir()
    lancedb_path()
    history_db_path().parent  # ensure parent exists
    models_dir()
    uploads_dir()
    logs_dir()
    return base
