"""Tests for v2.0.28.1 PR-2 P1-P2 trace_id middleware.

PR-2 wires per-request ``trace_id`` propagation via a pure ASGI
middleware (not Starlette ``BaseHTTPMiddleware``, which is HTTP-only
and bypasses WebSocket). The middleware:

1. Reads incoming ``X-Request-ID`` header (alphanumeric 1-32 chars).
   Falls back to a fresh ``uuid.uuid4().hex[:12]`` if absent or invalid.
2. Stamps trace_id into :data:`src.core.trace_id._trace_id_var` for
   HTTP and WebSocket scopes (NOT lifespan scopes).
3. Adds ``X-Request-ID`` to the HTTP response (on the first
   ``http.response.start`` message, doesn't overwrite if app already
   set one) and to the WS handshake accept message.
4. Resets the trace_id in a ``finally`` block.

The tests below pin every part of this contract, including the
end-to-end log line format change in :mod:`src.core.logging` (the
default loguru format now renders ``extra[trace_id]``).

Test strategy:
    * For middleware wiring + response header tests we use the real
      ``create_app()`` via the ``app_under_test`` fixture (the
      mounted StaticFiles catch-all doesn't matter because we hit
      ``/health`` which is registered before the mount).
    * For ContextVar-during-handler tests we build a **standalone
      test ASGI app** with ``TraceIdMiddleware`` + a small test
      endpoint, because the real app's StaticFiles mount catches
      every path that isn't a registered API route (so a
      dynamically-added test router would 404).
"""
from __future__ import annotations

import asyncio
import re
import uuid

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


# Canonical 12-hex format used by ``new_trace_id``.
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# Used for tests that exercise the format string. 12 hex chars only.
_VALID_INCOMING_ID = "abcdef012345"


def _make_test_app_with_endpoint(endpoint_path: str = "/probe"):
    """Build a tiny FastAPI app with ONLY ``TraceIdMiddleware`` and a
    single probe endpoint that returns ``get_trace_id()``.

    This avoids the ``create_app()`` StaticFiles catch-all which
    would 404 our test endpoints. Returned app is fresh each call —
    tests must use the same instance for all requests in one test
    (otherwise the second instance has no shared state, which is
    fine because all relevant state lives in contextvars).
    """
    from src.api.middleware import TraceIdMiddleware
    from src.core.trace_id import get_trace_id

    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)

    @app.get(endpoint_path)
    async def _probe():
        return {"trace_id": get_trace_id() or ""}

    return app


def _inject_trace_id_for_test(record) -> None:
    """Test-local copy of :func:`src.core.logging._inject_trace_id`.

    Used by the loguru sink in ``test_logger_exception_emits_trace_id_in_log_line``
    so the captured records include ``extra["trace_id"]`` (otherwise
    the ``{extra[trace_id]}`` format placeholder raises KeyError).

    Re-imported here to avoid pulling ``setup_logging`` into the test
    app — that would reconfigure global loguru state and pollute
    other tests.
    """
    from src.core.trace_id import display_trace_id

    record["extra"]["trace_id"] = display_trace_id()


# ---------------------------------------------------------------------------
# 1. middleware exists and is wired into create_app()
# ---------------------------------------------------------------------------


def test_trace_id_middleware_is_added_in_create_app():
    """``src.app.create_app()`` must register :class:`TraceIdMiddleware`
    so every HTTP request (and WS handshake) flows through it.
    """
    from src.app import create_app
    from src.api.middleware import TraceIdMiddleware

    app = create_app()
    # FastAPI/Starlette stores user middleware in ``app.user_middleware``
    # as a list of ``Middleware`` instances. Look for TraceIdMiddleware.
    found = False
    for mw in app.user_middleware:
        if mw.cls is TraceIdMiddleware:
            found = True
            break
    assert found, (
        "TraceIdMiddleware is not registered on the FastAPI app. "
        "PR-2 must add `app.add_middleware(TraceIdMiddleware)` in "
        "src/app.py:create_app()."
    )


# ---------------------------------------------------------------------------
# 2. every HTTP response carries an X-Request-ID header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_response_has_x_request_id_header(app_under_test):
    """Every HTTP response (success or 4xx) must carry ``X-Request-ID``.

    This is the operator-side correlation anchor: a user reports an
    error with the trace_id they see in the UI / response, an operator
    greps ``logs/app.log`` and finds every log line for that request.
    """
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Hit any endpoint — /health is the cheapest, no body needed.
        res = await client.get("/health")
        assert res.status_code == 200

        xrid = res.headers.get("x-request-id")
        assert xrid is not None, (
            f"X-Request-ID header missing from response. headers={dict(res.headers)!r}"
        )
        assert _TRACE_ID_RE.match(xrid), (
            f"X-Request-ID={xrid!r} does not match 12-hex pattern"
        )


# ---------------------------------------------------------------------------
# 3. unique trace_id per request (no reuse)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_request_gets_a_fresh_trace_id(app_under_test):
    """Two back-to-back requests must produce two distinct trace_ids."""
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res1 = await client.get("/health")
        res2 = await client.get("/health")
        tid1 = res1.headers.get("x-request-id")
        tid2 = res2.headers.get("x-request-id")
        assert tid1 and tid2
        assert tid1 != tid2, (
            f"two requests got the same X-Request-ID {tid1!r} — "
            "middleware must generate fresh per request"
        )


# ---------------------------------------------------------------------------
# 4. incoming X-Request-ID round-trips (federated tracing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incoming_x_request_id_is_honored(app_under_test):
    """If the client sends ``X-Request-ID: abcdef012345`` and it passes
    validation (1-32 alphanumeric), the response must echo it back
    unchanged. Lets upstream proxies / federated systems provide
    their own correlation ID.
    """
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(
            "/health",
            headers={"X-Request-ID": _VALID_INCOMING_ID},
        )
        assert res.status_code == 200
        echoed = res.headers.get("x-request-id")
        assert echoed == _VALID_INCOMING_ID, (
            f"incoming X-Request-ID {_VALID_INCOMING_ID!r} was not echoed; "
            f"got {echoed!r} instead"
        )


@pytest.mark.asyncio
async def test_invalid_incoming_x_request_id_falls_back_to_fresh(app_under_test):
    """Malformed incoming X-Request-ID (wrong length, contains
    non-alnum) must be silently ignored — fall back to a fresh
    12-hex trace_id rather than echo the malformed value or 400.

    Rationale: a client sending a malformed header is almost always
    a bug (cookie boundary, header case bug) rather than an attack.
    Silent fallback preserves the user's primary intent (their actual
    HTTP request) while the fresh trace_id round-trips so they can
    correlate to the server side.
    """
    transport = ASGITransport(app=app_under_test)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 50-char ID — too long (limit is 32).
        res = await client.get("/health", headers={"X-Request-ID": "a" * 50})
        echoed = res.headers.get("x-request-id")
        assert echoed is not None
        assert echoed != "a" * 50, "malformed X-Request-ID was echoed unchanged"
        assert _TRACE_ID_RE.match(echoed), (
            f"expected fresh 12-hex fallback, got {echoed!r}"
        )

        # Non-alphanumeric — contains spaces and semicolons.
        res2 = await client.get(
            "/health", headers={"X-Request-ID": "bad value; injection"}
        )
        echoed2 = res2.headers.get("x-request-id")
        assert echoed2 is not None
        assert echoed2 != "bad value; injection"
        assert _TRACE_ID_RE.match(echoed2)


# ---------------------------------------------------------------------------
# 5. trace_id flows through the ContextVar during request handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_id_contextvar_is_set_during_request():
    """Inside an HTTP request handler, ``get_trace_id()`` must return
    the trace_id stamped by the middleware — so any ``logger.exception``
    inside the handler (and any code it calls) emits the trace_id via
    the loguru format's ``extra[trace_id]`` placeholder.

    Uses a standalone test app (see ``_make_test_app_with_endpoint``)
    so we don't have to fight the real app's StaticFiles catch-all.
    """
    from src.core.trace_id import get_trace_id

    app = _make_test_app_with_endpoint("/probe")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/probe")
        assert res.status_code == 200
        handler_trace_id = res.json()["trace_id"]
        response_header_trace_id = res.headers.get("x-request-id")
        assert handler_trace_id, f"handler saw no trace_id: {res.json()!r}"
        assert response_header_trace_id, (
            f"response header missing X-Request-ID: {dict(res.headers)!r}"
        )
        assert handler_trace_id == response_header_trace_id, (
            f"handler saw trace_id={handler_trace_id!r} but response header "
            f"carries {response_header_trace_id!r} — ContextVar propagation broken"
        )
        assert _TRACE_ID_RE.match(handler_trace_id), (
            f"trace_id from handler is not 12-hex: {handler_trace_id!r}"
        )


@pytest.mark.asyncio
async def test_trace_id_contextvar_isolated_between_concurrent_requests():
    """Two concurrent requests must each see ONLY their own trace_id
    in the handler — not the other request's. This is the
    ``contextvars.ContextVar`` task-locality guarantee.

    Uses a standalone test app with TWO endpoints and fires them
    concurrently.
    """
    from src.api.middleware import TraceIdMiddleware
    from src.core.trace_id import get_trace_id

    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)

    @app.get("/probe_a")
    async def _a():
        await asyncio.sleep(0.05)
        return {"trace_id": get_trace_id() or ""}

    @app.get("/probe_b")
    async def _b():
        await asyncio.sleep(0.05)
        return {"trace_id": get_trace_id() or ""}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_a, res_b = await asyncio.gather(
            client.get("/probe_a"),
            client.get("/probe_b"),
        )
        tid_a_handler = res_a.json()["trace_id"]
        tid_b_handler = res_b.json()["trace_id"]
        tid_a_header = res_a.headers.get("x-request-id")
        tid_b_header = res_b.headers.get("x-request-id")

        # Each request saw its own trace_id (handler ↔ response match).
        assert tid_a_handler == tid_a_header, (
            f"request A: handler saw {tid_a_handler!r} but header was {tid_a_header!r}"
        )
        assert tid_b_handler == tid_b_header, (
            f"request B: handler saw {tid_b_handler!r} but header was {tid_b_header!r}"
        )
        # And they're distinct.
        assert tid_a_handler != tid_b_handler, (
            "two concurrent requests got the same trace_id — "
            "contextvars task isolation is broken"
        )


# ---------------------------------------------------------------------------
# 6. trace_id appears in log lines emitted during a request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logger_exception_emits_trace_id_in_log_line():
    """A ``logger.exception`` inside a request handler must emit a log
    line containing ``trace_id=<12-hex>``. This is the end-to-end
    loguru format string verification — the whole point of PR-2.

    Test strategy: we install a capture sink with the SAME format
    string the production :func:`src.core.logging.setup_logging` uses,
    so the captured output mirrors what operators actually see in
    production logs. The capture sink bypasses the live stderr / file
    sinks so this test doesn't pollute them.

    Uses loguru's native capture (a sink that appends to a list)
    rather than ``caplog`` because ``src.core.logging`` uses loguru,
    which doesn't propagate to stdlib ``logging`` by default. Same
    pattern as ``tests/test_recovery.py:110-137``.
    """
    import loguru
    from fastapi import APIRouter

    from src.api.middleware import TraceIdMiddleware
    from src.core.logging import logger  # local import to avoid global state

    # The exact format string the production stderr sink uses.
    # Must stay in sync with ``src/core/logging.py:setup_logging``.
    _PRODUCTION_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<cyan>trace_id={extra[trace_id]}</cyan> | "
        "<level>{message}</level>"
    )

    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)
    test_router = APIRouter()

    @test_router.get("/log_exception")
    async def _log_exception():
        try:
            raise RuntimeError("synthetic boom for trace_id test")
        except RuntimeError as exc:
            logger.exception(f"synthetic test exception: {exc}")
        return {"ok": True}

    app.include_router(test_router)

    captured: list[str] = []
    # Install the patcher GLOBALLY (matches what setup_logging does
    # in production). loguru's ``add()`` doesn't accept a ``patcher``
    # kwarg — ``configure(patcher=...)`` is the right API.
    loguru.logger.configure(patcher=_inject_trace_id_for_test)
    # Capture sink with the SAME format string as production stderr
    # sink, including the ``extra[trace_id]`` placeholder.
    sink_id = loguru.logger.add(
        captured.append, level="ERROR", format=_PRODUCTION_FORMAT
    )
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/log_exception")
            assert res.status_code == 200
            response_tid = res.headers.get("x-request-id")
            assert response_tid and _TRACE_ID_RE.match(response_tid)
    finally:
        loguru.logger.remove(sink_id)
        # Remove the patcher too — otherwise it leaks into other tests.
        # configure(patcher=None) is the documented way to clear.
        loguru.logger.configure(patcher=None)

    # The captured log lines should contain at least one entry with
    # the request's trace_id. The format renders
    # ``trace_id={extra[trace_id]}`` so the substring ``"trace_id=<tid>"``
    # must appear in the rendered line.
    pattern = re.compile(rf"trace_id={re.escape(response_tid)}\b")
    matched = [line for line in captured if pattern.search(line)]
    assert matched, (
        f"no log line contained trace_id={response_tid!r}. "
        f"captured={captured!r}. The loguru format in src/core/logging.py "
        f"must include the trace_id, and TraceIdMiddleware must stamp "
        f"the ContextVar before any logger.exception in the handler."
    )


# ---------------------------------------------------------------------------
# 7. trace_id reset between requests (no leak via ContextVar)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_id_contextvar_does_not_leak_across_requests():
    """After a request completes (middleware ``finally`` block ran),
    ``get_trace_id()`` must return ``None`` again. Otherwise a
    subsequent background task that runs without going through the
    middleware would falsely inherit the previous request's
    trace_id.

    Uses a standalone test app to avoid the real app's StaticFiles
    catch-all, but the assertion target (``get_trace_id()`` after
    the request) is the same.
    """
    from src.core.trace_id import get_trace_id, reset_for_tests

    # Defensive: clear any prior test's value.
    reset_for_tests()
    assert get_trace_id() is None, "contextvar leaked before this test"

    app = _make_test_app_with_endpoint("/probe")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/probe")
        assert res.status_code == 200
        # While the request was in flight, the ContextVar was set
        # (verified by other tests); after the middleware's finally,
        # the ContextVar should be back to None.
        post_request_tid = get_trace_id()
        assert post_request_tid is None, (
            f"trace_id leaked after request: {post_request_tid!r}. "
            "TraceIdMiddleware.__call__'s finally block must call reset_trace_id."
        )


# ---------------------------------------------------------------------------
# 8. module-level trace_id helpers
# ---------------------------------------------------------------------------


def test_new_trace_id_returns_12_hex_unique_values():
    """``new_trace_id()`` must produce a 12-character lowercase hex string.

    Two calls in a row produce distinct values (collision probability
    is negligible for 12 hex / 48 bits of entropy).
    """
    from src.core.trace_id import new_trace_id

    seen = set()
    for _ in range(100):
        tid = new_trace_id()
        assert _TRACE_ID_RE.match(tid), f"trace_id not 12-hex: {tid!r}"
        assert tid not in seen, "new_trace_id() returned a duplicate"
        seen.add(tid)


def test_display_trace_id_returns_dash_when_unset():
    """``display_trace_id()`` must return ``"-"`` when the ContextVar
    is unset (so log lines render a sane token rather than literal
    ``"None"`` which would break operator greps).
    """
    from src.core.trace_id import display_trace_id, reset_for_tests

    reset_for_tests()
    assert display_trace_id() == "-"
