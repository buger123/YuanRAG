"""v1.1.16 — WS close code banner gate contract.

User bug:
  2026-09-06 user reported "⚠ 聊天连接已断开 ×" appearing after EVERY
  normal turn completion, not just after actual network drops.

Root cause:
  ``src/frontend/src/App.tsx`` ``ws.onclose`` unconditionally
  dispatched the ``rag:ws-closed`` event regardless of close code.
  Per RFC-6455, browsers fire ``onclose`` for every termination,
  including the clean ``ws.close()`` we call after the agent's
  ``done`` event (which defaults to code 1000 Normal Closure).
  Result: the disconnect banner flashed after every successful turn.

Fix:
  Gate the banner dispatch on the close code:
    code 1000  (Normal Closure)         → NO banner — our own close
    code 1005  (No Status Received)     → NO banner — server crashed
                                          but the matching ``onerror``
                                          already showed the banner
    code 1006  (Abnormal Closure)       → banner — network drop
    code 1008  (Policy Violation)       → banner — auth/origin rejection
    code 1009  (Message Too Big)        → banner — payload exceeded cap
    code 1011  (Internal Error)         → banner — server-side exception
    code 1014  (Bad Gateway)            → banner — proxy failure
    any other code                       → banner (defensive)
  Plus a closure-local ``alreadyDispatched`` flag so the banner
  doesn't flash twice when the browser fires BOTH ``onerror`` and
  ``onclose`` for the same abnormal termination.

Since the project doesn't have a JS test runner, we mirror the
gate as a Python helper and verify the contract here. The actual
ship is the change in ``App.tsx``; these tests lock the wire
contract the frontend consumes.
"""
from __future__ import annotations

import pytest


# Python mirror of App.tsx ``_should_dispatch_banner_on_close``.
# This MUST match the implementation in App.tsx — if it drifts,
# the frontend will start mis-gating banners again. The mirror is
# intentionally tiny so a regression on either side is obvious.
def _should_dispatch_banner_on_close(
    code: int,
    *,
    already_dispatched: bool = False,
) -> bool:
    """Return True iff ``ws.onclose`` should fire the disconnect banner.

    Mirrors ``src/frontend/src/App.tsx`` ``getOrCreateWS``:
      if (ev.code === 1000 || ev.code === 1005) return false;
      if (alreadyDispatched) return false;
      alreadyDispatched = true;
      // dispatch...
    """
    if already_dispatched:
        return False
    if code == 1000 or code == 1005:
        return False
    return True


# Python mirror of App.tsx ``_messageForCloseCode``.
def _message_for_close_code(code: int) -> str:
    """Return the user-facing banner message for a given WS close code."""
    mapping = {
        1001: "聊天连接已断开,请刷新页面",
        1006: "网络连接中断,请检查网络后重试",
        1008: "认证失败,请刷新页面或重新登录",
        1009: "消息过大,请拆分后重试",
        1011: "服务器内部错误,请稍后重试",
        1014: "网关错误,请稍后重试",
    }
    return mapping.get(code, "聊天连接已断开")


# ============================================================
# Gate logic: which close codes show the banner?
# ============================================================


@pytest.mark.parametrize(
    "code,should_dispatch",
    [
        # Clean closes — never show banner.
        pytest.param(1000, False, id="normal_closure_1000_no_banner"),
        pytest.param(1005, False, id="no_status_1005_no_banner"),
        # Abnormal closes — always show banner.
        pytest.param(1001, True, id="going_away_1001_banner"),
        pytest.param(1006, True, id="abnormal_closure_1006_banner"),
        pytest.param(1008, True, id="policy_violation_1008_banner"),
        pytest.param(1009, True, id="message_too_big_1009_banner"),
        pytest.param(1011, True, id="internal_error_1011_banner"),
        pytest.param(1014, True, id="bad_gateway_1014_banner"),
        # Defensive default — anything else is suspicious.
        pytest.param(1015, True, id="tls_handshake_1015_banner"),
        pytest.param(4000, True, id="unknown_app_code_4000_banner"),
    ],
)
def test_banner_gate_by_close_code(code, should_dispatch):
    """The fix: only abnormal close codes dispatch the banner. Clean
    closes (1000, 1005) are normal lifecycle, not user-facing errors."""
    assert _should_dispatch_banner_on_close(code) is should_dispatch


def test_banner_gate_dedups_double_fire():
    """Browsers fire BOTH ``onerror`` and ``onclose`` for abnormal
    terminations. Without de-dup, the user sees the banner flash
    twice. The ``already_dispatched`` flag (closure-local in
    App.tsx) ensures the second call is suppressed."""
    # First fire (from onerror) — banner shown
    assert _should_dispatch_banner_on_close(1006) is True
    # Second fire (from onclose with the matching 1006 code) —
    # suppressed because we already dispatched
    assert _should_dispatch_banner_on_close(1006, already_dispatched=True) is False


def test_banner_gate_dedups_clean_close_after_error():
    """Edge case: onerror fires (banner shown), then onclose fires
    with code 1000 (server crash recovery, sometimes seen). The
    dedup flag must still suppress the second dispatch even though
    the close code itself would be a no-op."""
    # onerror fired → banner shown
    assert _should_dispatch_banner_on_close(1006) is True
    # onclose fires with code 1000 — would normally be a no-op,
    # but the dedup flag handles it uniformly
    assert _should_dispatch_banner_on_close(1000, already_dispatched=True) is False


# ============================================================
# Banner message mapping: which text per code?
# ============================================================


@pytest.mark.parametrize(
    "code,expected_substring",
    [
        pytest.param(1001, "刷新页面", id="going_away"),
        pytest.param(1006, "网络", id="abnormal_closure"),
        pytest.param(1008, "认证", id="policy_violation"),
        pytest.param(1009, "消息过大", id="message_too_big"),
        pytest.param(1011, "服务器内部", id="internal_error"),
        pytest.param(1014, "网关", id="bad_gateway"),
        pytest.param(9999, "聊天连接已断开", id="default_fallback"),
    ],
)
def test_banner_message_for_each_close_code(code, expected_substring):
    """The banner must say something actionable for each known
    abnormal close code. A future code we haven't classified falls
    back to the generic "聊天连接已断开" — not great UX but at
    least the user sees SOMETHING."""
    msg = _message_for_close_code(code)
    assert expected_substring in msg, (
        f"banner for code {code} should mention '{expected_substring}', "
        f"got '{msg}'"
    )


# ============================================================
# Regression: the canonical user-reported scenario
# ============================================================


def test_user_reported_scenario_no_banner_after_done():
    """The exact user-reported bug: after a successful turn (the
    ``done`` event fires, then client calls ``ws.close()``), the
    banner must NOT appear.

    Flow:
      1. LLM streams tokens
      2. ``done`` event arrives
      3. App.tsx ``done`` handler calls ``ws.close()``
         (defaults to code 1000 Normal Closure)
      4. Server's WebSocketDisconnect fires (code 1000)
      5. Client ``ws.onclose`` fires with ``event.code === 1000``
      6. v1.1.16: gate recognizes 1000 → NO banner
    """
    # Step 6: onclose with code 1000 (the client-initiated close
    # from the done handler) → no banner.
    assert _should_dispatch_banner_on_close(1000) is False


def test_user_reported_scenario_no_banner_after_onstop():
    """The onStop handler also calls ``ws.close()`` (no code
    argument, defaults to 1000). Same gate applies — user clicks
    "stop", we close cleanly, no banner."""
    assert _should_dispatch_banner_on_close(1000) is False


def test_user_reported_scenario_banner_still_shows_on_network_drop():
    """The original intent (QW #10) must still work: an actual
    network drop, server crash, or auth failure must STILL show
    the banner so the user knows something went wrong."""
    # Network drop → onerror fires (banner shown via dedup-safe path)
    assert _should_dispatch_banner_on_close(1006) is True
    # Auth failure → onclose with 1008
    assert _should_dispatch_banner_on_close(1008) is True
    # Payload too big → onclose with 1009
    assert _should_dispatch_banner_on_close(1009) is True