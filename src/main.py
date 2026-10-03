"""Application entrypoint.

Starts the FastAPI/uvicorn server on the configured host:port and opens
the user's default browser once the server is reachable.

Usage:
    python -m src.main [--host HOST] [--port PORT] [--reload] [--no-browser] [--debug]

Environment variables are loaded from the project-root ``.env`` file at
startup (via python-dotenv) so users can configure ``RAG_DATA_DIR``,
``MINIMAX_*`` and other settings without exporting them in the shell.
Already-exported process env vars take precedence over ``.env``.

v2.0.28.6 P1-3: argparse + headless detect + graceful error trap.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

# Import the bootstrap module FIRST so .env is loaded before any other
# module reads env vars (notably src.core.paths.user_data_dir()).
from src.core import bootstrap  # noqa: F401 — side effect: load_env()
from config.constants import DEFAULT_HOST, DEFAULT_PORT
from config.settings import get_settings
from src.core.logging import logger, setup_logging
from src.core.paths import ensure_all_dirs


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    """Poll until the TCP port is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _should_open_browser() -> bool:
    """Return True iff the current environment is likely a desktop with a browser.

    Heuristics (any one of these suppresses auto-open):
      - ``SSH_CONNECTION`` is set (we're on a remote box — opening a browser
        on the user's machine is impossible)
      - ``CI`` is set (GitHub Actions, Jenkins, etc. — no display)
      - On Linux: ``DISPLAY`` and ``WAYLAND_DISPLAY`` both unset (no X11
        or Wayland server, e.g. a headless server container)

    Returns True on macOS and Windows by default — those platforms don't
    expose a DISPLAY variable and most desktops have a browser.
    """
    if os.environ.get("SSH_CONNECTION"):
        return False
    if os.environ.get("CI"):
        return False
    if sys.platform == "linux" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    return True


def _open_browser_when_ready(url: str, host: str, port: int) -> None:
    if _wait_for_port(host, port, timeout=60.0):
        logger.info(f"Opening browser: {url}")
        try:
            webbrowser.open(url)
        except webbrowser.Error as exc:
            logger.warning(f"Browser open failed: {exc}. Visit {url} manually.")
    else:
        logger.warning(f"Server didn't come up in time. Visit {url} manually.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments. Kept separate from ``main()`` so
    tests can call it directly without spinning up uvicorn."""
    parser = argparse.ArgumentParser(
        prog="yuanrag",
        description="Yuan RAG — local-first RAG chat over your documents.",
    )
    parser.add_argument(
        "--host",
        help="Bind host (overrides RAG_HOST in .env; default 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Bind port (overrides RAG_PORT in .env; default 8765).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (dev only — single worker).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip auto-opening the default browser on startup.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG log level (overrides RAG_LOG_LEVEL).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ensure_all_dirs()
    settings = get_settings()
    setup_logging(level=settings.log_level)
    if args.debug:
        # Reconfigure with DEBUG regardless of .env / settings file.
        setup_logging(level="DEBUG")

    # Echo the resolved data dir so users can see where their state lives.
    data_dir = os.environ.get("RAG_DATA_DIR") or "~/.rag_assistant (default)"
    logger.info(f"User data directory: {data_dir}")

    if settings.allow_remote and settings.host == DEFAULT_HOST:
        logger.warning(
            "ALLOW_REMOTE is set but HOST is 127.0.0.1. Override via user config to expose externally."
        )

    # CLI flags override settings (.env / config file values). argparse
    # defaults are None — leave the value untouched unless the user
    # explicitly passed the flag.
    host = args.host if args.host is not None else settings.host
    port = args.port if args.port is not None else settings.port
    url = f"http://{host}:{port}/"

    logger.info(f"Starting RAG Assistant at {url}")

    # Auto-open browser (non-blocking). Skipped on headless hosts
    # (SSH / CI / Linux without DISPLAY) and when the user passes
    # ``--no-browser``.
    should_open = (
        not args.no_browser
        and _should_open_browser()
    )
    if should_open:
        threading.Thread(
            target=_open_browser_when_ready, args=(url, host, port), daemon=True
        ).start()
    else:
        logger.info("Browser auto-open skipped (headless or --no-browser)")

    # Run server (blocking). ``uvicorn.run`` raises ``OSError`` when
    # the port is already in use or the bind address is invalid — we
    # catch and translate to a friendly exit code so shell scripts /
    # wrappers see a non-zero exit on bind failure rather than a raw
    # stack trace.
    try:
        uvicorn.run(
            "src.app:app",
            host=host,
            port=port,
            log_level=settings.log_level.lower(),
            workers=1,
            reload=args.reload,
            # Avoid uvicorn hijacking loguru; loguru handles its own logging.
            log_config=None,
        )
    except OSError as exc:
        logger.error(f"Server failed to start: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())