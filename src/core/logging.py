"""Structured logging setup using loguru."""
from __future__ import annotations

import sys

from loguru import logger

from src.core.paths import logs_dir


def _inject_trace_id(record) -> None:
    """Loguru patcher: inject ``trace_id`` into every record's ``extra``.

    Called for every log record by the global patcher installed in
    :func:`setup_logging`. Reads from the ContextVar set by
    :class:`src.api.middleware.TraceIdMiddleware` and falls back to
    ``"-"`` if no trace_id is set (startup, shutdown, background
    tasks, tests).

    Without this patcher, loguru's ``{extra[trace_id]}`` format
    placeholder would raise ``KeyError`` when the record's ``extra``
    dict doesn't already contain a ``trace_id`` key (which is the
    default for any logger.* call site that doesn't use
    ``logger.bind(trace_id=...)``).

    Installed via ``logger.configure(patcher=...)`` (NOT ``add(patcher=...)``
    — loguru's ``add()`` doesn't accept that kwarg, the patcher is a
    global logger-level configuration).
    """
    from src.core.trace_id import display_trace_id

    record["extra"]["trace_id"] = display_trace_id()


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru with stderr + rotating file handler.

    Safe to call multiple times — loguru replaces handlers on re-init.

    v2.0.28.1 P1-P2 — both stderr + file formats include
    ``{extra[trace_id]}``. The ``_inject_trace_id`` patcher reads
    from the ContextVar set by :class:`src.api.middleware.TraceIdMiddleware`
    and stamps it into the record's ``extra`` dict, so the format
    placeholder always has a value. Outside any request (startup,
    shutdown, background tasks), the patcher renders ``"-"``.
    """
    # Install the global patcher FIRST so it applies to all sinks
    # added in this call AND any sinks added later (e.g. test sinks).
    # configure() with extra={} resets extras but patcher= is sticky.
    logger.configure(patcher=_inject_trace_id)

    logger.remove()

    # Stderr (colorized in TTY)
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<cyan>trace_id={extra[trace_id]}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )

    # File
    log_file = logs_dir() / "app.log"
    logger.add(
        str(log_file),
        level=level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | trace_id={extra[trace_id]} | {message}"
        ),
    )


__all__ = ["logger", "setup_logging"]
