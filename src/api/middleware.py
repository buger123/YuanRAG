"""ASGI middleware: per-request trace_id + X-Request-ID response header.

Pure ASGI (``__call__(scope, receive, send)``) — **not** Starlette's
``BaseHTTPMiddleware`` — so the same code path covers both regular
HTTP requests AND WebSocket handshakes/frames (which Starlette routes
through the same ASGI app).

Why pure ASGI:
    ``BaseHTTPMiddleware`` is HTTP-only — Starlette's WS upgrade
    bypasses it entirely, leaving WS connections without trace_id.
    A pure ASGI middleware sees every scope (``http`` + ``websocket``)
    and can stamp trace_id into ``contextvars`` before any handler
    code runs, including the WebSocket handshake. WS frame handlers
    (``src/api/websocket.py``) read :func:`src.core.trace_id.get_trace_id`
    to correlate their own log lines.

Why this middleware runs:
    * Stamp ``trace_id = uuid.uuid4().hex[:12]`` into
      :data:`src.core.trace_id._trace_id_var` so every ``logger.exception``
      in this request's call chain (including ``asyncio.to_thread``
      workers — contextvars propagate via ``copy_context``) emits the
      trace_id via the loguru format's ``extra[trace_id]`` placeholder.
    * Read ``X-Request-ID`` from incoming headers if a client (or
      upstream proxy) already provided one — let it round-trip so
      federated systems can correlate. Else generate fresh.
    * Set ``X-Request-ID`` on the response (HTTP scope) and on the
      initial WS handshake accept message so the client can echo it
      back in bug reports.

Why 12 hex:
    Matches the format already used in ``src/storage/checkpointer.py``
    (12 hex, ``uuid.uuid4().hex[:12]``) — operators grepping for a
    trace_id from a client error report will find matches in both
    the HTTP handler logs and the corruption-event logs.

Validation rule for incoming ``X-Request-ID``:
    Length 12 + alphanumeric only — prevents log injection (``\\r\\n``
    smuggling a fake log line) and keeps the round-tripped value
    grep-safe. If the incoming header fails validation, we ignore it
    and generate fresh (this is silent — no 400, no log warning —
    because a poisoned header is more likely a bug than an attack,
    and silently falling back is the safer default).
"""
from __future__ import annotations

import re

from src.core.logging import logger
from src.core.trace_id import (
    new_trace_id,
    reset_trace_id,
    set_trace_id,
)


# Stricter than ``\w`` because we don't want underscores/dashes in
# incoming IDs (would still be safe but not consistent with our own
# 12-hex format). Length-locked to 12 chars to match ``new_trace_id``.
_VALID_INCOMING_RE = re.compile(r"^[0-9a-zA-Z]{1,32}$")
_X_REQUEST_ID_HEADER = b"x-request-id"


def _is_http_scope(scope: dict) -> bool:
    return scope.get("type") == "http"


def _is_websocket_scope(scope: dict) -> bool:
    return scope.get("type") == "websocket"


def _extract_incoming_trace_id(scope: dict) -> str | None:
    """Look for an incoming ``X-Request-ID`` header on the request.

    Returns the validated value (1-32 alphanumeric chars) or ``None``
    if no header present / header fails validation. Headers come in
    as bytes; we lowercase-decode via the comparison loop below.
    """
    for name, value in scope.get("headers", ()):
        if name.lower() == _X_REQUEST_ID_HEADER:
            try:
                decoded = value.decode("latin-1").strip()
            except UnicodeDecodeError:
                return None
            if _VALID_INCOMING_RE.match(decoded):
                return decoded
            return None
    return None


class TraceIdMiddleware:
    """ASGI middleware: per-request trace_id propagation.

    Usage::

        app.add_middleware(TraceIdMiddleware)  # in src/app.py create_app()

    Starlette middleware order is **reverse**: the LAST
    ``add_middleware`` call runs OUTERMOST. To make TraceIdMiddleware
    run first (set trace_id before anything else, get X-Request-ID on
    the final response), add it AFTER any other middleware in source
    order. With only CORSMiddleware today, the source order is::

        app.add_middleware(CORSMiddleware, ...)     # added first
        app.add_middleware(TraceIdMiddleware)        # added second = runs first

    For the typical 4-step HTTP request:

    1. Client sends GET /sessions
    2. ASGI server dispatches scope through middlewares outermost-first
    3. TraceIdMiddleware.__call__ sets trace_id in contextvars, calls next
    4. CORSMiddleware adds CORS headers, calls next
    5. FastAPI handler runs (reads get_trace_id(), logs include trace_id)
    6. Response bubbles back through middlewares
    7. TraceIdMiddleware stamps X-Request-ID on the response, resets trace_id
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Only stamp trace_id for HTTP requests and WS handshakes. Skip
        # ASGI lifespan events (``scope["type"] == "lifespan"``) so
        # startup/shutdown logs don't carry a fake trace_id.
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Try to honor an incoming X-Request-ID for federated tracing.
        # If absent or invalid, generate fresh.
        incoming = _extract_incoming_trace_id(scope)
        tid = incoming if incoming else new_trace_id()

        # Stamp into contextvars. ``asyncio.to_thread`` copies the
        # current context into the worker thread — so even deep
        # synchronous call paths (lancedb / sqlite) see the trace_id
        # when they ``logger.exception``.
        token = set_trace_id(tid)
        try:
            if _is_http_scope(scope):
                await self._serve_http(scope, receive, send, tid)
            else:
                # WebSocket — stamp handshake accept headers + pass
                # through. Frame-level trace_id propagation in WS
                # handlers is handled by ``websocket.py`` reading
                # ``get_trace_id()`` directly (the contextvars are
                # already set by the time the handler runs).
                await self._serve_websocket(scope, receive, send, tid)
        finally:
            reset_trace_id(token)

    async def _serve_http(self, scope, receive, send, tid: str) -> None:
        """Wrap the HTTP send to inject ``X-Request-ID`` on the first
        ``http.response.start`` message.
        """
        first_send_done = False

        async def wrapped_send(message):
            nonlocal first_send_done
            if not first_send_done and message["type"] == "http.response.start":
                # message["headers"] is a list[tuple[bytes, bytes]].
                # Insert our header BEFORE the existing list so it
                # appears first (cosmetic — order doesn't matter
                # semantically). If the application already set an
                # X-Request-ID (e.g. via Response.headers), honor
                # that instead of overwriting.
                existing = message.get("headers", [])
                has_xrid = any(
                    name.lower() == _X_REQUEST_ID_HEADER
                    for name, _ in existing
                )
                if not has_xrid:
                    message["headers"] = [
                        (_X_REQUEST_ID_HEADER, tid.encode("latin-1")),
                        *existing,
                    ]
                first_send_done = True
            await send(message)

        await self.app(scope, receive, wrapped_send)

    async def _serve_websocket(self, scope, receive, send, tid: str) -> None:
        """Stamp ``X-Request-ID`` on the WS handshake accept frame.

        WS doesn't carry headers in normal messages — only in the
        initial handshake accept response. We wrap ``send`` and
        intercept the ``websocket.accept`` message to attach the
        header. After the handshake, WS handlers can read
        ``get_trace_id()`` directly from contextvars for log
        correlation (it's still set because the WS handler runs in
        the same task).
        """

        async def wrapped_send(message):
            if message["type"] == "websocket.accept":
                # ``websocket.accept`` headers are optional; merge ours in.
                existing = list(message.get("headers", ()))
                existing.append((_X_REQUEST_ID_HEADER, tid.encode("latin-1")))
                message["headers"] = existing
            await send(message)

        await self.app(scope, receive, wrapped_send)


__all__ = ["TraceIdMiddleware"]


# v2.0.28.1 P1-P2 — Pure ASGI middleware instead of BaseHTTPMiddleware.
# Why: BaseHTTPMiddleware is HTTP-only — Starlette routes WebSocket
# upgrades around it, leaving WS connections without trace_id in
# their log lines. The pure ASGI ``__call__(scope, receive, send)``
# pattern covers both HTTP and WS scopes uniformly.
#
# v2.0.28.1 P1-P2 — Incoming X-Request-ID validation is silent-drop
# rather than 400. Reasoning: a client sending a malformed header is
# almost always a bug (cookie/domain boundary, header case bug) rather
# than an attack. Returning 400 would break the request entirely for
# what is essentially a cosmetic correlation header. Silent fallback
# to a fresh trace_id preserves the user's primary intent (their
# actual HTTP request) while still letting them see in logs that the
# incoming header was ignored (the fresh trace_id round-trips back
# in the response so they can correlate to the server side).
#
# v2.0.28.1 P1-P2 — Why we don't propagate X-Request-ID into
# upstream HTTP calls (e.g. when YuanRAG makes outbound requests to
# Anthropic API). Outbound propagation is a different concern (use
# httpx middleware / aiohttp tracing) and is out of scope for
# PR-2. Existing ad-hoc correlation IDs in
# ``runner.py:_trace_id_for(exc)`` (which derives from Anthropic's
# request_id) still work for client-side error display.
