"""WebSocket hardening helpers.

P1-3 audit found the ``/ws/chat`` endpoint had four production-grade
holes:

1. **No Origin allowlist.** Any browser page on any port (including a
   malicious site the user happens to visit) could open a WebSocket to
   the local server and exfiltrate whatever the user asked about. The
   CORS middleware protects ``fetch()`` but does NOT cover WS — that's
   a separate code path that browsers do not enforce the same way.

2. **No rate limit.** A single client could flood ``receive_text`` and
   either starve other users of the WS accept queue or rack up LLM
   tokens. Token bucket (1 msg / 2s, burst 5) caps sustained abuse
   without throttling normal back-and-forth.

3. **No message size cap.** ``receive_text()`` happily accepts
   multi-MB payloads and feeds them into the LLM as a "question" —
   both expensive (token cost) and dangerous (prompt injection
   surface). Cap at 16 KB raw / 4000 chars trimmed.

4. **No heartbeat.** Idle WS connections silently sit forever; on
   reconnect (the browser's standard behavior) the server has no
   signal the client is gone until TCP keepalive kicks in (minutes).
   A 15 s server-side ping keeps dead connections detected early.

5. **Remote-bind requires no auth.** When ``Settings.allow_remote`` is
   True and the bind host isn't localhost, anyone on the network can
   open ``ws://server:8765/ws/chat?thread_id=...`` and use the app as
   that user. ``RAG_AUTH_TOKEN`` (env var) must be supplied via the
   ``Authorization: Bearer ...`` header OR ``?token=...`` query param.
   Localhost-bind (``127.0.0.1`` / ``::1`` / ``localhost``) skips the
   check — the user is already on the box.

Each helper returns ``None`` on accept and a ``reason`` string on
reject; ``ws_chat`` translates that into the appropriate close code
(1008 = policy violation, 1013 = try again later, 1009 = too big).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket

from config.settings import get_settings


# Browsers always send an Origin header for WS handshakes; native clients
# (curl, wscat) typically don't. We accept absent Origin only for
# localhost binds — see ``_origin_allowed`` below.
_LOCAL_ORIGIN_SUFFIXES = (
    "://localhost",
    "://127.0.0.1",
    "://[::1]",
)


def _origin_allowed(origin: Optional[str]) -> bool:
    """Return True iff ``origin`` is acceptable for this connection.

    Policy:

    * ``origin`` empty / None → allowed ONLY on localhost binds. Native
      CLI clients don't send Origin and that's expected.
    * ``origin`` matches ``http://localhost[:port]`` /
      ``http://127.0.0.1[:port]`` / ``http://[::1]:port`` → allowed.
    * Anything else → rejected with a policy-violation close.
    """
    s = get_settings()
    if not origin:
        # No Origin header. OK iff we're bound to localhost; remote binds
        # MUST have an Origin we can check (or a Bearer token — see below).
        return s.host in {"127.0.0.1", "localhost", "::1"}

    if not origin.startswith(("http://", "https://")):
        return False

    for suffix in _LOCAL_ORIGIN_SUFFIXES:
        if origin.startswith("http" + suffix) or origin.startswith("https" + suffix):
            return True
    return False


def check_origin(websocket: WebSocket) -> Optional[str]:
    """Reject the WS handshake if Origin is not allowed.

    Returns ``None`` on accept, a rejection reason string otherwise.
    Must be called BEFORE ``websocket.accept()`` so we can send the
    correct close code.
    """
    origin = websocket.headers.get("origin") or websocket.headers.get(
        "Origin"
    )
    if not _origin_allowed(origin):
        return (
            f"Origin {origin!r} not in allowlist. "
            "This server only accepts connections from localhost."
        )
    return None


# ---- Bearer-token gate for remote binds ---------------------------------


def _extract_bearer(websocket: WebSocket) -> Optional[str]:
    """Pull the bearer token out of either the Authorization header or
    the ``?token=...`` query string. The query form exists for native
    clients (curl, wscat) that can't set arbitrary headers easily."""
    auth = websocket.headers.get("authorization") or websocket.headers.get(
        "Authorization"
    )
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    # Query string fallback — ``websocket.query_params`` is a parsed
    # multidict that includes the raw ``?key=value&...`` URL fragment.
    return websocket.query_params.get("token")


def check_auth(websocket: WebSocket) -> Optional[str]:
    """Enforce bearer token when ``allow_remote`` is on and the bind
    isn't loopback.

    Localhost binds skip the check entirely (the user is on the box;
    requiring a token would break the dev workflow for no gain).
    Native clients without an Origin header on remote binds must
    supply the token via the ``?token=...`` query string.
    """
    s = get_settings()
    if not s.allow_remote:
        return None
    if s.host in {"127.0.0.1", "localhost", "::1"}:
        return None

    expected = os.environ.get("RAG_AUTH_TOKEN")
    if not expected:
        # Misconfiguration: remote bind without a configured token. We
        # could either fail open (let everyone in) or fail closed. Fail
        # closed is the safer default — operator must explicitly set the
        # env var to enable remote access.
        return (
            "Server is configured for remote access but RAG_AUTH_TOKEN "
            "is not set. Set it in .env or the systemd unit and restart."
        )
    provided = _extract_bearer(websocket)
    if not provided or provided != expected:
        return "Missing or invalid bearer token."
    return None


# ---- Token-bucket rate limiter -------------------------------------------


@dataclass
class _Bucket:
    """A leaky token bucket: capacity tokens, refilled at ``rate`` per
    second. ``consume()`` returns True iff a token was available. Tests
    patch the wall clock via ``_now`` so they don't actually sleep."""

    capacity: int = 5
    refill_per_sec: float = 0.5  # 1 token every 2 s → rate = 0.5 / s
    tokens: float = field(default=5.0)
    last_refill: float = field(default_factory=lambda: time.monotonic())

    def _now(self) -> float:
        return time.monotonic()

    def consume(self) -> bool:
        now = self._now()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.refill_per_sec
            )
            self.last_refill = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# One bucket per WS connection; created at handshake time so the burst
# limit is per-client, not global.
def new_rate_limit_bucket() -> _Bucket:
    return _Bucket(capacity=5, refill_per_sec=0.5)


# ---- Message-size cap ----------------------------------------------------


# 16 KB raw is the upper bound; ``message`` field is capped at 4000 chars
# AFTER JSON parsing because most legitimate user questions are < 1 KB
# and anything bigger is almost certainly a paste-bomb or prompt
# injection attempt. Both limits are checked.
MAX_RAW_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4000


def check_message_size(raw: str, message: str) -> Optional[str]:
    """Reject a message that exceeds the size caps.

    Returns a rejection reason string, or ``None`` if the message is OK.
    ``raw`` is the full WS frame as received; ``message`` is the
    already-parsed ``message`` field (we check both because the raw
    frame includes JSON overhead and the parsed field is what actually
    reaches the LLM).
    """
    if len(raw.encode("utf-8")) > MAX_RAW_BYTES:
        return f"Message too large (> {MAX_RAW_BYTES} bytes raw)."
    if len(message) > MAX_MESSAGE_CHARS:
        return f"Message too large (> {MAX_MESSAGE_CHARS} characters)."
    return None


# ---- Heartbeat -----------------------------------------------------------


# 15 s matches the typical reverse-proxy idle timeout. Anything shorter
# wastes bandwidth on benign idle connections; anything longer means a
# dead client sits around for a full timeout cycle before we notice.
HEARTBEAT_INTERVAL_SEC = 15.0


__all__ = [
    "check_origin",
    "check_auth",
    "new_rate_limit_bucket",
    "check_message_size",
    "HEARTBEAT_INTERVAL_SEC",
    "MAX_RAW_BYTES",
    "MAX_MESSAGE_CHARS",
    "_Bucket",
]
