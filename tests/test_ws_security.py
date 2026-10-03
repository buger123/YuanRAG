"""P1-3 — WebSocket hardening (origin, auth, rate limit, size cap)."""
from __future__ import annotations

import asyncio
import os
from typing import Optional

import pytest


# ============================================================
# Helpers — fake ``WebSocket`` that satisfies the duck-typed surface
# our security helpers actually touch (``headers``, ``query_params``,
# ``close``).
# ============================================================


class _FakeHeaders:
    def __init__(self, data: Optional[dict] = None) -> None:
        self._data: dict[str, str] = dict(data or {})

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # Both ``Origin`` and ``Authorization`` are case-insensitive per
        # RFC 7230; we normalize by storing lowercased keys.
        return self._data.get(key.lower(), default)


class _FakeQueryParams:
    def __init__(self, data: Optional[dict] = None) -> None:
        self._data: dict[str, str] = dict(data or {})

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(key, default)


class _FakeWebSocket:
    def __init__(
        self,
        origin: Optional[str] = None,
        authorization: Optional[str] = None,
        query: Optional[dict] = None,
    ) -> None:
        headers: dict[str, str] = {}
        if origin is not None:
            headers["origin"] = origin
        if authorization is not None:
            headers["authorization"] = authorization
        self.headers = _FakeHeaders(headers)
        self.query_params = _FakeQueryParams(query)
        self.closed_with: tuple[int, str] | None = None

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed_with = (code, reason)


# ============================================================
# P1-3a · Origin allowlist
# ============================================================


def test_origin_localhost_http_allowed(monkeypatch):
    """Localhost over http — the dev server case — must accept."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    from config.settings import get_settings, reset_settings_cache

    reset_settings_cache()
    # Default host = 127.0.0.1
    assert get_settings().host == "127.0.0.1"

    ws = _FakeWebSocket(origin="http://localhost:5173")
    assert ws_security.check_origin(ws) is None


def test_origin_localhost_ip_allowed(monkeypatch):
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    from config.settings import reset_settings_cache

    reset_settings_cache()
    ws = _FakeWebSocket(origin="http://127.0.0.1:5173")
    assert ws_security.check_origin(ws) is None


def test_origin_external_rejected(monkeypatch):
    """Anything other than localhost / 127.0.0.1 / [::1] is rejected,
    regardless of the bind host. Browsers DO send Origin even on
    localhost, so a malicious site can't bypass this by sending
    ``Origin: https://evil.com``."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    from config.settings import reset_settings_cache

    reset_settings_cache()
    ws = _FakeWebSocket(origin="https://evil.com")
    err = ws_security.check_origin(ws)
    assert err is not None
    assert "Origin" in err


def test_origin_missing_on_localhost_bind_is_allowed(monkeypatch):
    """Native CLI clients (curl, wscat) don't send Origin. When the
    server is bound to localhost we accept the missing header — the
    user is on the box."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    from config.settings import reset_settings_cache

    reset_settings_cache()
    ws = _FakeWebSocket(origin=None)
    assert ws_security.check_origin(ws) is None


def test_origin_missing_on_remote_bind_rejected(monkeypatch):
    """When bound to a non-loopback host, an absent Origin header is
    rejected. Operators must either bind to localhost or front the
    server with a reverse proxy that adds the Origin header."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_HOST_OVERRIDE_FOR_TEST", "1")
    # Set host to 0.0.0.0 directly via env-injected Settings; we do
    # this by patching the cached Settings instance via reset + reload.
    from config.settings import UserConfig, get_settings, reset_settings_cache, save_user_config

    from src.core.paths import user_data_dir

    save_user_config(
        user_data_dir(),
        UserConfig(host="0.0.0.0", port=9999, allow_remote=True),
    )
    reset_settings_cache()
    assert get_settings().host == "0.0.0.0"

    ws = _FakeWebSocket(origin=None)
    err = ws_security.check_origin(ws)
    assert err is not None
    assert "Origin" in err


# ============================================================
# P1-3b · Bearer-token gate for remote binds
# ============================================================


def test_auth_skipped_on_localhost(monkeypatch):
    """Even if ``RAG_AUTH_TOKEN`` is set, localhost binds skip the
    bearer check — the user is on the box."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_AUTH_TOKEN", "supersecret")
    from config.settings import reset_settings_cache

    reset_settings_cache()
    ws = _FakeWebSocket()
    assert ws_security.check_auth(ws) is None


def test_auth_remote_bind_requires_token(monkeypatch):
    """On a remote bind, missing token = reject."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_AUTH_TOKEN", "supersecret")
    from config.settings import UserConfig, reset_settings_cache, save_user_config
    from src.core.paths import user_data_dir

    save_user_config(
        user_data_dir(),
        UserConfig(host="0.0.0.0", port=9999, allow_remote=True),
    )
    reset_settings_cache()
    ws = _FakeWebSocket()  # no Authorization, no ?token=
    err = ws_security.check_auth(ws)
    assert err is not None
    assert "token" in err.lower()


def test_auth_remote_bind_accepts_correct_bearer(monkeypatch):
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_AUTH_TOKEN", "supersecret")
    from config.settings import UserConfig, reset_settings_cache, save_user_config
    from src.core.paths import user_data_dir

    save_user_config(
        user_data_dir(),
        UserConfig(host="0.0.0.0", port=9999, allow_remote=True),
    )
    reset_settings_cache()
    ws = _FakeWebSocket(authorization="Bearer supersecret")
    assert ws_security.check_auth(ws) is None


def test_auth_remote_bind_accepts_query_token(monkeypatch):
    """Native CLI fallback: ``?token=...`` query string."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_AUTH_TOKEN", "supersecret")
    from config.settings import UserConfig, reset_settings_cache, save_user_config
    from src.core.paths import user_data_dir

    save_user_config(
        user_data_dir(),
        UserConfig(host="0.0.0.0", port=9999, allow_remote=True),
    )
    reset_settings_cache()
    ws = _FakeWebSocket(query={"token": "supersecret"})
    assert ws_security.check_auth(ws) is None


def test_auth_remote_bind_rejects_wrong_token(monkeypatch):
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_AUTH_TOKEN", "supersecret")
    from config.settings import UserConfig, reset_settings_cache, save_user_config
    from src.core.paths import user_data_dir

    save_user_config(
        user_data_dir(),
        UserConfig(host="0.0.0.0", port=9999, allow_remote=True),
    )
    reset_settings_cache()
    ws = _FakeWebSocket(authorization="Bearer wrong")
    assert ws_security.check_auth(ws) is not None


def test_auth_remote_bind_without_configured_token_rejects(monkeypatch):
    """If the operator turned on remote-bind but forgot to set
    ``RAG_AUTH_TOKEN``, fail closed (don't let everyone in)."""
    from src.api import ws_security
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    monkeypatch.delenv("RAG_AUTH_TOKEN", raising=False)
    from config.settings import UserConfig, reset_settings_cache, save_user_config
    from src.core.paths import user_data_dir

    save_user_config(
        user_data_dir(),
        UserConfig(host="0.0.0.0", port=9999, allow_remote=True),
    )
    reset_settings_cache()
    ws = _FakeWebSocket(authorization="Bearer anything")
    err = ws_security.check_auth(ws)
    assert err is not None
    assert "RAG_AUTH_TOKEN" in err


# ============================================================
# P1-3c · Token-bucket rate limiter
# ============================================================


def test_bucket_initial_burst_allowed():
    """Fresh bucket must allow up to ``capacity`` tokens immediately."""
    from src.api.ws_security import new_rate_limit_bucket

    b = new_rate_limit_bucket()
    for _ in range(5):
        assert b.consume() is True
    # 6th must fail — bucket is empty.
    assert b.consume() is False


def test_bucket_refills_over_time():
    """Tokens trickle back at 1 per 2 s. We freeze ``_now`` to assert
    deterministically without sleeping."""
    from src.api.ws_security import _Bucket

    b = _Bucket(capacity=5, refill_per_sec=0.5)
    # Drain.
    for _ in range(5):
        assert b.consume()
    assert b.consume() is False

    # Advance fake clock by 4 s → expect 2 tokens back (0.5 * 4 = 2).
    fake_now = [b.last_refill + 4.0]

    def fake_now_fn() -> float:
        return fake_now[0]

    b._now = fake_now_fn  # type: ignore[assignment]
    assert b.consume() is True
    assert b.consume() is True
    # 3rd must fail (we consumed both refilled tokens).
    assert b.consume() is False


def test_bucket_caps_at_capacity():
    """Refill must not push the bucket above ``capacity`` even after a
    long idle period — otherwise a client that paused could fire the
    full capacity of a multi-second burst in one go."""
    from src.api.ws_security import _Bucket

    b = _Bucket(capacity=5, refill_per_sec=0.5)
    b.last_refill = 0.0
    fake_now = [1000.0]  # Way more refill than capacity could ever hold.

    def fake_now_fn() -> float:
        return fake_now[0]

    b._now = fake_now_fn  # type: ignore[assignment]
    # Drain (5 burst).
    for _ in range(5):
        assert b.consume()
    # 6th must fail; the bucket shouldn't have grown beyond 5.
    assert b.consume() is False


# ============================================================
# P1-3d · Message size cap
# ============================================================


def test_message_size_ok():
    from src.api.ws_security import check_message_size

    assert check_message_size('{"message": "hi"}', "hi") is None
    # Edge of cap: 4000 chars message, ~4096 bytes raw.
    msg = "a" * 4000
    raw = '{"message": "' + msg + '"}'
    assert check_message_size(raw, msg) is None


def test_message_size_rejects_huge_raw():
    from src.api.ws_security import MAX_RAW_BYTES, check_message_size

    raw = "x" * (MAX_RAW_BYTES + 1)
    err = check_message_size(raw, "x" * 100)
    assert err is not None
    assert "raw" in err.lower()


def test_message_size_rejects_huge_message_field():
    from src.api.ws_security import MAX_MESSAGE_CHARS, check_message_size

    # Raw frame can be small (the JSON encoder could compress) but the
    # parsed ``message`` itself is too long for the LLM.
    msg = "a" * (MAX_MESSAGE_CHARS + 1)
    raw = '{"message": "' + msg + '"}'
    err = check_message_size(raw, msg)
    assert err is not None
    assert "character" in err.lower()


# ============================================================
# P1-3e · ``_safe_send_json`` contract
# ============================================================


@pytest.mark.asyncio
async def test_safe_send_json_returns_delivered_true_on_success():
    """A successful ``send_json`` returns ``(True, None)``."""
    from src.api.websocket import _safe_send_json

    sent: list[dict] = []

    class _OkWS:
        async def send_json(self, payload):
            sent.append(payload)

    delivered, exc = await _safe_send_json(
        _OkWS(), {"type": "ping"}, validate=False
    )
    assert delivered is True
    assert exc is None
    assert sent == [{"type": "ping"}]


@pytest.mark.asyncio
async def test_safe_send_json_returns_delivered_false_on_disconnect():
    """A ``WebSocketDisconnect`` from ``send_json`` returns
    ``(False, exc)`` so the caller can decide whether to drain."""
    from fastapi import WebSocketDisconnect

    from src.api.websocket import _safe_send_json

    class _GoneWS:
        async def send_json(self, payload):
            raise WebSocketDisconnect(code=1006)

    delivered, exc = await _safe_send_json(
        _GoneWS(), {"type": "ping"}, validate=False
    )
    assert delivered is False
    assert isinstance(exc, WebSocketDisconnect)


@pytest.mark.asyncio
async def test_safe_send_json_swallows_runtime_error_after_close():
    """Starlette raises ``RuntimeError("Cannot call send once a close
    has been sent")`` on a half-closed socket — treat it as a disconnect."""
    from src.api.websocket import _safe_send_json

    class _HalfClosedWS:
        async def send_json(self, payload):
            raise RuntimeError(
                "Cannot call send once a close message has been sent."
            )

    delivered, exc = await _safe_send_json(
        _HalfClosedWS(), {"type": "ping"}, validate=False
    )
    assert delivered is False
    assert isinstance(exc, RuntimeError)
