"""WebSocket chat endpoint — bidirectional streaming of agent events."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.runner import stream_agent
from src.api.i18n import error_message, parse_accept_language
from src.api.ws_security import (
    HEARTBEAT_INTERVAL_SEC,
    check_auth,
    check_message_size,
    check_origin,
    new_rate_limit_bucket,
)
from src.core.logging import logger


router = APIRouter()


# Holds strong references to in-flight ``stream_agent`` generators that
# are still running after their WebSocket consumer disconnected. Without
# this, switching threads (or closing the browser tab) mid-stream would
# let the local ``gen`` variable go out of scope when ``ws_chat``
# returns, the generator would be garbage-collected, and any work it was
# doing would be torn down — including the LangGraph superstep that
# would have written the final checkpoint. The drain task below takes a
# reference too, but we ALSO hold it here as a belt-and-braces guard so
# even an exception path that skips the drain task still keeps the
# generator alive until it finishes naturally.
_background_runs: set[AsyncIterator[dict]] = set()


async def _drain_generator(gen: AsyncIterator[dict]) -> None:
    """Continue pulling events from a ``stream_agent`` generator after
    its WebSocket consumer has disconnected.

    Without this, switching threads mid-stream silently aborts the
    in-flight agent invocation: Starlette calls ``aclose()`` on the
    generator, ``GeneratorExit`` propagates into ``graph.astream``,
    LangGraph cancels its background tasks, and the final checkpoint is
    never written. The user would then reopen that thread and see a
    stale snapshot from before the question was asked — the answer
    appears to have "failed".

    With this drain, the agent run continues to completion regardless
    of whether the client is still listening. The user can switch back
    to that thread (or refresh) and ``GET /sessions/{thread_id}/messages``
    returns the full Q+A as committed by the checkpointer.
    """
    try:
        async for _ in gen:
            pass
    except Exception:
        logger.exception("background agent drain failed")
    finally:
        _background_runs.discard(gen)


async def _safe_send_json(
    websocket: WebSocket,
    payload: dict,
    *,
    validate: bool = True,
) -> tuple[bool, Exception | None]:
    """Send a JSON event, swallowing ``WebSocketDisconnect`` so the
    caller can detach the agent run to the background drain instead of
    aborting it.

    v2.0.17 — when ``validate=True`` (default), the payload is checked
    against the wire schema (see :mod:`src.agent.wire_protocol`) before
    being written. A regression that produces a malformed payload
    (missing required field, wrong type, unknown ``type`` discriminator)
    is caught HERE instead of silently corrupting the frontend's state.
    On schema failure: log + emit a substitute ``error`` event in its
    place, and return ``(False, exc)`` so the caller handles the bad
    frame the same way as a send failure (the agent run keeps going).

    ``validate=False`` skips the check for transport-only frames (the
    heartbeat ``{"type": "ping"}`` — not a content event, not in the
    schema). Use sparingly; default-on is the safe path.

    Returns ``(delivered, exc)`` where ``delivered`` is True iff the
    frame was actually written to the socket. On a disconnect we return
    the exception so the caller can decide whether to keep the
    generator alive or tear it down.
    """
    if validate:
        from pydantic import ValidationError

        from src.agent.wire_protocol import validate_payload

        try:
            validate_payload(payload)
        except ValidationError as exc:
            logger.exception(
                f"wire schema violation (payload dropped): "
                f"type={payload.get('type')!r} errors={exc.errors()}"
            )
            # Replace with a redacted error event so the frontend
            # sees SOMETHING rather than a silent drop. The original
            # event is lost — that's intentional: shipping a malformed
            # frame is worse than not shipping it at all (the frontend
            # would error on parse and lose all subsequent events).
            from src.security.prompt_safety import redact_exception

            redacted = redact_exception(
                RuntimeError(
                    f"internal: outgoing WS event of type "
                    f"{payload.get('type')!r} failed wire-schema "
                    f"validation; see server log for trace_id"
                )
            )
            try:
                await websocket.send_json(redacted)
                return True, exc
            except Exception as send_exc:
                return False, send_exc
    try:
        await websocket.send_json(payload)
        return True, None
    except WebSocketDisconnect as exc:
        return False, exc
    except RuntimeError as exc:
        # ``RuntimeError("Cannot call send once a close has been sent")``
        # is Starlette's way of saying we tried to send on a half-closed
        # socket. Treat it the same as a disconnect.
        return False, exc
    except Exception as exc:
        logger.exception(f"unexpected WebSocket send error: {exc}")
        return False, exc


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket, thread_id: str) -> None:
    # P1-3: Origin / auth gating happens BEFORE ``accept()`` so we can
    # return the correct WS close code. accept() then refuse() would
    # leak a successful handshake to the client.
    origin_err = check_origin(websocket)
    if origin_err:
        await websocket.close(code=1008, reason=origin_err[:123])
        return
    auth_err = check_auth(websocket)
    if auth_err:
        await websocket.close(code=1008, reason=auth_err[:123])
        return

    await websocket.accept()
    # Demoted to DEBUG to match the disconnect log level. At INFO these
    # fire on every page navigation, thread switch, and React StrictMode
    # double-mount — produces lots of noise without useful signal. Bump
    # log_level to DEBUG when investigating WS lifecycle issues.
    logger.debug(f"WebSocket connected: thread_id={thread_id}")

    # PR-4 (v2.0.26.1): thread the client's preferred locale from
    # ``Accept-Language`` into every user-visible error frame so the
    # frontend's SettingsDialog toggle actually flips backend copy.
    # Browsers always send this header, so a missing value is rare —
    # we still fall back to ``parse_accept_language``'s default (zh).
    locale = parse_accept_language(
        websocket.headers.get("accept-language")
    )

    # Per-connection rate limit (token bucket: 5 burst, 1 msg / 2 s).
    # Held in a closure so the inner loop can read+consume without
    # module-level state.
    bucket = new_rate_limit_bucket()

    async def heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
                # ``{"type": "ping"}`` is a transport-only heartbeat,
                # not a content event — bypass wire-schema validation
                # so the heartbeat doesn't pollute the schema with a
                # never-displayed frame type.
                delivered, exc = await _safe_send_json(
                    websocket, {"type": "ping"}, validate=False
                )
                if not delivered:
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("WS heartbeat task crashed")

    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": error_message("ws.invalid_json", locale),
                    }
                )
                continue
            message = data.get("message")
            if not isinstance(message, str) or not message.strip():
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": error_message("ws.empty_message", locale),
                    }
                )
                continue

            # v2.0.29.9 (Phase 8) — verbatim extraction tristate.
            # Per-thread toggle from the frontend's chat input
            # segmented control. ``"auto"`` is the default
            # (hybrid regex + cheap-LLM detect fires inside
            # ``intent_analysis``); ``"on"`` forces extractive mode
            # (bypass ReAct, verbatim quote only); ``"off"`` forces
            # normal synthesis EVEN if auto-detect flags the query.
            # Forwarded to ``stream_agent(..., high_precision=...)``
            # which stamps it into the FSM state before
            # ``intent_analysis`` runs.
            #
            # Backward-compat: pre-Phase-8 clients don't send this
            # field; default "auto" preserves hybrid detect behavior.
            hp_raw = data.get("high_precision")
            if hp_raw not in ("auto", "on", "off"):
                high_precision = "auto"
            else:
                high_precision = hp_raw

            # Size cap (raw frame + parsed message field).
            size_err = check_message_size(raw, message)
            if size_err:
                # 1009 = "message too big" — the standard WS close code
                # for this case so clients can distinguish from a
                # generic policy violation.
                await websocket.close(code=1009, reason=size_err[:123])
                return

            # Rate limit (token bucket). On exhaustion we send a
            # backoff hint but keep the connection open — the user
            # might just be typing fast.
            if not bucket.consume():
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": error_message("ws.rate_limited", locale),
                        "code": "rate_limited",
                    }
                )
                continue

            # We deliberately do NOT wrap ``stream_agent(...)`` in
            # ``aclosing(...)`` here. That was the previous design and
            # it had a subtle bug: when the client disconnected mid-run
            # (most commonly because the user clicked ``+ New chat``
            # while waiting for an answer), Starlette closed the
            # WebSocket and ``aclosing.__aexit__`` threw
            # ``GeneratorExit`` into the generator, which propagated
            # into ``graph.astream`` and aborted the agent run before
            # its final checkpoint was written. The user would then
            # reopen the thread and see no AI reply at all — the
            # "first question fails when I switch threads" bug.
            #
            # Instead we hold a strong reference to the generator and,
            # on disconnect, hand it off to ``_drain_generator`` which
            # lets it run to completion. The checkpointer then commits
            # the final state and the answer is available the next time
            # the user (or anyone) loads that thread's history.
            gen = stream_agent(thread_id, message, high_precision=high_precision)
            _background_runs.add(gen)
            detached = False
            try:
                async for event in gen:
                    delivered, exc = await _safe_send_json(websocket, event)
                    if not delivered:
                        # P1-3: send-side failures no longer abort the
                        # agent run. Detach to the background drain so
                        # the answer still lands in the checkpoint.
                        logger.debug(
                            f"WebSocket send failed mid-stream: thread_id={thread_id} "
                            f"exc={exc!r} — detaching agent to drain in background"
                        )
                        asyncio.create_task(_drain_generator(gen))
                        detached = True
                        return
            except WebSocketDisconnect as exc:
                code = getattr(exc, "code", None)
                reason = getattr(exc, "reason", "") or ""
                logger.debug(
                    f"WebSocket disconnected mid-stream: thread_id={thread_id} "
                    f"code={code} reason={reason!r} — detaching agent to drain in background"
                )
                # Detach: keep the agent running so the answer lands in
                # the checkpoint. The drain task holds the only strong
                # reference once ``_background_runs.discard`` runs in the
                # ``finally`` below.
                asyncio.create_task(_drain_generator(gen))
                detached = True
                return
            except Exception as exc:
                logger.exception(f"WebSocket error: {exc}")
                delivered, _ = await _safe_send_json(
                    websocket, {"type": "error", "message": str(exc)}
                )
            finally:
                # Natural completion (WS closed after ``done`` event) or
                # error path — release our reference. On the detach
                # path, the drain task's argument keeps the generator
                # alive independently.
                if not detached:
                    _background_runs.discard(gen)
    except WebSocketDisconnect as exc:
        # Outer disconnect: client closed before sending any message, or
        # after we returned above. Demoted to DEBUG for the same reason
        # as the inner handler — this fires on every page navigation /
        # thread switch / StrictMode double-mount.
        code = getattr(exc, "code", None)
        reason = getattr(exc, "reason", "") or ""
        logger.debug(
            f"WebSocket disconnected: thread_id={thread_id} code={code} reason={reason!r}"
        )
    except Exception as exc:
        logger.exception(f"WebSocket error: {exc}")
        # P1-4: never echo the raw exception to the client. The outer
        # handler's exception message can contain API keys / proxy URLs
        # (e.g. ``httpx.ConnectError`` includes the URL). Send a Chinese
        # message + trace_id; the full stack is in the server log.
        try:
            from src.security.prompt_safety import redact_exception

            await _safe_send_json(websocket, redact_exception(exc))
        except Exception:
            pass
    finally:
        # Stop the heartbeat; otherwise it leaks a task per connection.
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
