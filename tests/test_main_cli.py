"""v2.0.28.6 Item 11 PR-4 P1-3 — ``src/main.py`` argparse + headless + graceful exit.

Pre-PR-4, ``main.py`` had no CLI surface at all. Operators had to set
``RAG_HOST`` / ``RAG_PORT`` in ``.env`` and re-source the shell. There
was also no headless detect — running on an SSH box (or in CI) would
fire ``webbrowser.open`` and either silently no-op or hang the
shutdown handler. Port-bind failures surfaced as raw stack traces
from uvicorn.

Post-PR-4, ``main.py`` exposes ``--host``, ``--port``, ``--reload``,
``--no-browser``, ``--debug`` argparse flags; ``_should_open_browser()``
suppresses auto-open when ``SSH_CONNECTION`` / ``CI`` is set or on
Linux without ``DISPLAY``; ``OSError`` from uvicorn is caught and
returns exit code 1.

This file pins the CLI contract.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from src.main import _parse_args, _should_open_browser


def test_main_parses_host_override():
    """``--host 0.0.0.0`` must be reflected in the parsed namespace."""
    args = _parse_args(["--host", "0.0.0.0"])
    assert args.host == "0.0.0.0"


def test_main_parses_port_override():
    """``--port 9000`` must be parsed as int (not str)."""
    args = _parse_args(["--port", "9000"])
    assert args.port == 9000
    assert isinstance(args.port, int)


def test_main_default_flags_are_false():
    """No-arg invocation: all flags default to False / None.

    This pins the "no-arg = no override" contract — existing
    ``python -m src.main`` callers (no args) must keep their
    .env-derived behavior.
    """
    args = _parse_args([])
    assert args.host is None
    assert args.port is None
    assert args.reload is False
    assert args.no_browser is False
    assert args.debug is False


def test_main_skips_browser_when_ssh_env_set(monkeypatch):
    """``SSH_CONNECTION=...`` must suppress ``_should_open_browser()``.

    When a user SSHes into a remote box and runs ``python -m src.main``,
    opening the browser on the REMOTE machine is useless — the user
    is on a different machine entirely. The auto-open must be
    suppressed.
    """
    monkeypatch.setenv("SSH_CONNECTION", "192.168.1.1 12345 192.168.1.2 22")
    monkeypatch.delenv("CI", raising=False)
    assert not _should_open_browser()


def test_main_skips_browser_when_ci_env_set(monkeypatch):
    """``CI=1`` must suppress ``_should_open_browser()``.

    GitHub Actions / Jenkins runners have no display server; opening
    a browser there just hangs the uvicorn shutdown handler.
    """
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("CI", "1")
    assert not _should_open_browser()


def test_main_skips_browser_on_linux_without_display(monkeypatch):
    """Linux without ``DISPLAY`` / ``WAYLAND_DISPLAY`` must suppress auto-open.

    A headless Linux server has no display — calling ``webbrowser.open``
    silently fails (or worse, opens an xdg-open dialog the user can't see).
    """
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert not _should_open_browser()


def test_main_allows_browser_on_linux_with_display(monkeypatch):
    """Linux WITH ``DISPLAY`` set must allow auto-open (desktop env)."""
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(sys, "platform", "linux")
    assert _should_open_browser()


def test_main_returns_exit_code_1_on_port_bind_failure(monkeypatch):
    """When ``uvicorn.run`` raises ``OSError`` (port already in use),
    ``main()`` must return 1, not propagate the exception.

    We mock ``uvicorn.run`` to raise, monkeypatch the CLI args so we
    don't trigger the browser thread, and verify the return code.
    """
    monkeypatch.setattr("sys.argv", ["main"])

    def fake_uvicorn_run(*a, **kw):
        raise OSError("address already in use")

    monkeypatch.setattr("src.main.uvicorn.run", fake_uvicorn_run)

    # Suppress the browser-open thread — replace threading.Thread with
    # a no-op stub whose ``.start()`` does nothing. ``main()`` calls
    # ``threading.Thread(...).start()``, so the stub must return an
    # object with a ``start`` attribute (returning None would make
    # ``None.start()`` raise AttributeError).
    class _NoopThread:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

    monkeypatch.setattr("src.main.threading.Thread", _NoopThread)

    from src.main import main

    rc = main()
    assert rc == 1


def test_main_no_browser_flag_skips_browser_thread(monkeypatch):
    """``--no-browser`` must skip the browser-open thread even on a desktop env.

    Without this flag, ``main()`` schedules a daemon thread that
    polls the port and fires ``webbrowser.open``. With the flag,
    the thread must not start (and a log line is emitted instead).
    """
    monkeypatch.setattr("sys.argv", ["main", "--no-browser", "--port", "99999"])
    # Track whether the browser thread was started.
    thread_calls = []

    class _RecordingThread:
        def __init__(self, *a, **kw):
            thread_calls.append((a, kw))

        def start(self):
            pass

    monkeypatch.setattr("src.main.threading.Thread", _RecordingThread)
    monkeypatch.setattr(
        "src.main.uvicorn.run",
        lambda *a, **kw: None,  # return immediately, no server
    )
    monkeypatch.setattr("src.main._should_open_browser", lambda: True)

    from src.main import main

    main()
    # No thread should have been scheduled when --no-browser is set.
    assert thread_calls == []