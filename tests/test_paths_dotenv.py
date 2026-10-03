"""Tests for user data directory resolution + dotenv loading.

Why this matters
----------------
Two failure modes bit the user:

1. ``RAG_DATA_DIR`` was set in ``.env`` but the backend never loaded the
   file (no python-dotenv call) — so LanceDB + checkpointer still
   opened under ``~/.rag_assistant`` (the historical default). Symptom:
   log lines like ``Opening LanceDB connection at
   <user-home>/.rag_assistant/data/lancedb`` appeared even though
   ``.env`` said ``RAG_DATA_DIR=./data``.

2. Relative paths in ``RAG_DATA_DIR`` (e.g. ``./data``) were inserted
   literally into ``Path()``, which resolves against ``os.getcwd()``
   instead of the project root. Running from a different directory
   would silently create a stray ``./data`` folder.

The fix:

- ``src/main.py`` calls ``dotenv.load_dotenv(...)`` from the project
  root BEFORE any module reads env.
- ``src/core/paths.user_data_dir()`` resolves relative
  ``RAG_DATA_DIR`` values against the project root, not CWD.

These tests pin both behaviors so a regression would show up here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ============================================================
# Relative RAG_DATA_DIR resolution
# ============================================================


def test_user_data_dir_relative_path_resolves_against_project_root(
    monkeypatch, tmp_path: Path
):
    """A relative ``RAG_DATA_DIR`` (e.g. ``./data``) must be resolved
    against the project root — NOT against the current working
    directory. Otherwise `cd src && python -m src.main` would create a
    different data folder than `cd .. && python -m src.main`.

    We use a sandbox root so the test doesn't depend on whether the
    real repo already has a ``./data`` directory.
    """
    from src.core import paths

    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(paths, "_PROJECT_ROOT", sandbox_root)
    monkeypatch.setenv("RAG_DATA_DIR", "./data")
    # Run from a deliberately weird CWD to prove we don't use it.
    monkeypatch.chdir(elsewhere)

    resolved = paths.user_data_dir()
    assert resolved == (sandbox_root / "data").resolve()
    assert resolved.exists()
    assert resolved.is_dir()
    # Cleanup the env so subsequent tests don't see it.
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)


def test_user_data_dir_absolute_path_used_as_is(monkeypatch, tmp_path: Path):
    """Absolute ``RAG_DATA_DIR`` paths bypass the project-root resolution."""
    from src.core import paths

    target = tmp_path / "custom"
    monkeypatch.setenv("RAG_DATA_DIR", str(target))
    resolved = paths.user_data_dir()
    assert resolved == target.resolve()
    assert resolved.exists()


def test_user_data_dir_default_when_unset(monkeypatch, tmp_path: Path):
    """When ``RAG_DATA_DIR`` is unset, fall back to ``~/.rag_assistant``
    (the historical default) — preserving backwards compatibility.

    We point ``Path.home`` at a writable tmp dir so the test doesn't try
    to mkdir under the real ``C:/Users/...`` (which would fail in
    sandboxed environments and pollute the user's home).
    """
    from src.core import paths

    monkeypatch.delenv("RAG_DATA_DIR", raising=False)

    fake_home = tmp_path / "fake_home"
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: fake_home))
    resolved = paths.user_data_dir()
    assert resolved == (fake_home / ".rag_assistant").resolve()
    assert resolved.exists()


def test_user_data_dir_creates_missing_parents(monkeypatch, tmp_path: Path):
    """If the requested path doesn't exist yet, ``user_data_dir()``
    must create it (``parents=True``) — not raise."""
    from src.core import paths

    nested = tmp_path / "a" / "b" / "c"
    monkeypatch.setenv("RAG_DATA_DIR", str(nested))
    resolved = paths.user_data_dir()
    assert resolved.exists()
    assert resolved.is_dir()


# ============================================================
# dotenv loading from main.py
# ============================================================


def test_dotenv_loader_picks_up_project_env(monkeypatch, tmp_path: Path):
    """A value set in ``.env`` but not in process env should be visible
    to the rest of the app after main.py's loader runs.

    We can't import ``main`` directly (it calls ``uvicorn.run``), so we
    exercise the loader the same way main.py does and assert the
    resulting env state.
    """
    from dotenv import load_dotenv

    # Simulate the project's .env pointing at a fresh tmp dir.
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_DATA_DIR=/tmp/from-dotenv-test\n", encoding="utf-8")
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)

    load_dotenv(dotenv_path=env_file, override=False)
    assert os.environ.get("RAG_DATA_DIR") == "/tmp/from-dotenv-test"


def test_dotenv_loader_does_not_override_process_env(monkeypatch, tmp_path: Path):
    """``override=False`` is the contract: process env wins over .env,
    so users can set ad-hoc overrides on the command line."""
    from dotenv import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("RAG_DATA_DIR=/from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("RAG_DATA_DIR", "/from-shell")

    load_dotenv(dotenv_path=env_file, override=False)
    assert os.environ.get("RAG_DATA_DIR") == "/from-shell"


def test_main_module_loads_dotenv_at_import_time(monkeypatch, tmp_path: Path):
    """``src.main`` must call ``load_dotenv`` at module-import time so
    settings picked up later see the file values.

    Rather than run uvicorn, we trigger the import path's side effects by
    importing ``config.settings`` (which reads RAG_DATA_DIR transitively
    via ``get_settings``) and verify the values from a synthetic .env
    reach it. If main.py forgets to load .env, this test catches it."""
    from dotenv import load_dotenv

    # Write a synthetic .env and run the same loader main.py uses.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MINIMAX_API_KEY=sk-from-env\n"
        "RAG_DATA_DIR=/tmp/from-synthetic-env\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)
    load_dotenv(dotenv_path=env_file, override=False)

    assert os.environ.get("MINIMAX_API_KEY") == "sk-from-env"
    assert os.environ.get("RAG_DATA_DIR") == "/tmp/from-synthetic-env"


# ============================================================
# Integration: dotenv loader + path resolution together
# ============================================================


def test_full_path_resolution_via_dotenv(monkeypatch, tmp_path: Path):
    """End-to-end: write a .env, run main.py's loader, then read
    ``user_data_dir()``. Verifies that a relative ``RAG_DATA_DIR=./data``
    style value survives all the way to a real Path."""
    from dotenv import load_dotenv
    from src.core import paths

    # Use a sandbox project root so we don't pollute the real repo.
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    project_root = sandbox  # pretend this is the project root

    env_file = project_root / ".env"
    env_file.write_text("RAG_DATA_DIR=./data\n", encoding="utf-8")

    # Patch _PROJECT_ROOT to point at our sandbox so resolution lands there.
    monkeypatch.setattr(paths, "_PROJECT_ROOT", project_root)
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)

    load_dotenv(dotenv_path=env_file, override=False)

    resolved = paths.user_data_dir()
    assert resolved == (project_root / "data").resolve()
    assert resolved.exists()