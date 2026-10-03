"""Per-request ``trace_id`` ContextVar for log correlation.

The middleware in :mod:`src.api.middleware` stamps a unique 12-hex
``trace_id`` into :data:`_trace_id_var` at the start of every HTTP
request and clears it at the end. Any code path that runs inside
the request handler (including ``asyncio.to_thread`` workers — Python
3.7+ ``contextvars`` propagate via ``copy_context`` at task creation)
can read it with :func:`get_trace_id`.

The default loguru format (:mod:`src.core.logging`) renders the value
via ``extra[trace_id]`` so every ``logger.exception`` / ``logger.error``
/ ``logger.warning`` automatically carries the trace_id without
call-site changes. This is the operator-side correlation anchor: given
a trace_id from a client-side error report, an operator can
``grep ``logs/app.log`` and find every log line emitted during that
request, including exceptions in deep call chains.

Why ``ContextVar`` instead of ``threading.local`` / ``request.state``:

* ``threading.local`` doesn't propagate to ``asyncio.to_thread``
  workers (different thread, different local).
* ``request.state`` is only accessible from the handler body, not
  from helper functions called several stack frames deep.
* ``ContextVar`` is task-local by default in asyncio, AND is copied
  to ``asyncio.to_thread`` workers automatically.

Why 12 hex (not full UUID4):

* 48 bits of entropy → 5 simultaneous events have a collision
  probability of ~10^-12 (negligible for in-process log correlation).
* Shorter strings make log lines more readable and copy-paste-friendly
  for bug-report emails.

WS path note:
    The HTTP middleware doesn't cover WebSocket connections
    (``src/api/websocket.py``). WS frames still use ad-hoc
    ``runner.py`` per-call trace_ids for Anthropic request
    correlation. Adding WS-side trace_id is future work (out of scope
    for Item 10 PR-2).
"""
from __future__ import annotations

import contextvars
import uuid


_TRACE_ID_NONE_SENTINEL = "-"

# Module-level ContextVar. ``default=None`` means: outside any request
# (e.g. background tasks, startup, lifespan shutdown), ``get_trace_id()``
# returns ``None`` and the loguru format renders it as the sentinel "-".
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "yuanrag_trace_id", default=None
)


def new_trace_id() -> str:
    """Generate a fresh 12-hex-char trace_id (48 bits entropy)."""
    return uuid.uuid4().hex[:12]


def get_trace_id() -> str | None:
    """Read the current request's trace_id (or ``None`` outside a request)."""
    return _trace_id_var.get()


def set_trace_id(tid: str) -> contextvars.Token:
    """Stamp a trace_id into the current context. Returns a reset token.

    Callers MUST call :func:`reset_trace_id` with the token in a
    ``finally`` block to avoid leaking the trace_id into the next
    request that runs on the same task (e.g. via ``asyncio.gather``
    reuse). The middleware in :mod:`src.api.middleware` follows this
    contract.
    """
    return _trace_id_var.set(tid)


def reset_trace_id(token: contextvars.Token) -> None:
    """Restore the trace_id to whatever it was before :func:`set_trace_id`."""
    _trace_id_var.reset(token)


def reset_for_tests() -> None:
    """Clear the ContextVar back to the default (``None``).

    Called from ``tests/conftest.py`` ``_reset_all_singletons`` so
    tests don't inherit a trace_id from a previous test's request.
    """
    _trace_id_var.set(None)


def display_trace_id() -> str:
    """Return the trace_id for display, or "-" if unset.

    Used by the loguru format string so log lines never carry a
    literal ``"None"`` token (which is confusing in a production log
    grep). The default loguru format renders ``extra[trace_id]`` via
    this helper.
    """
    tid = _trace_id_var.get()
    return tid if tid is not None else _TRACE_ID_NONE_SENTINEL


__all__ = [
    "new_trace_id",
    "get_trace_id",
    "set_trace_id",
    "reset_trace_id",
    "reset_for_tests",
    "display_trace_id",
]
