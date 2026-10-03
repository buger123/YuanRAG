"""v2.0.28.4 Item 11 PR-2 P1-1 — LLM retry/backoff via tenacity.

The audit found that the LLM call path had **no** retry on transient
provider errors (Anthropic 429/529, OpenAI 429/500/503, httpx
transport errors). Single mid-turn overload → user loses the turn
with no recovery. The web_search path already retries
(``constants.py:103-108``); LLM was the only retry-less path.

This file pins the contract for ``invoke_with_retry`` /
``ainvoke_with_retry`` — the two helpers added in structured.py to
wrap model.invoke / ainvoke with tenacity.

Why these specific tests:
    1. **Anthropic 429 succeeds on 2nd attempt** — the headline use
       case. If this breaks, user-facing 429 errors will reappear.
    2. **OpenAI 503 succeeds on 2nd attempt** — same shape, different
       SDK; ensures retry is provider-agnostic.
    3. **3 attempts then reraise** — verifies the stop_after_attempt
       cap. Without this, a permanent outage would loop forever.
    4. **Non-retryable error skips retry** — proves the predicate
       is type-narrowing (e.g. ValueError → no retry, not "retry
       ValueError 3 times then fail").

Plus: loguru sink captures ``op=llm_retry`` WARNING on each retry so
operators can grep the noise (this is part of the P1-2 observability
story — we always log the retry, not just on final failure).
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import openai
import pytest


def _anthropic_429_error() -> anthropic.APIStatusError:
    """Construct an ``anthropic.APIStatusError`` with status_code 429.

    ``anthropic.APIStatusError.__init__`` requires a non-empty message
    and an HTTP request/response pair. We use the SDK's own Request /
    Response classes for fidelity (the SDK's exception ``__str__``
    relies on the response object's status_code attribute).
    """
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.APIStatusError(
        "rate limited",
        response=response,
        body=None,
    )


def _openai_503_error() -> openai.APIStatusError:
    """Construct an ``openai.APIStatusError`` with status_code 503."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(503, request=request)
    return openai.APIStatusError(
        "service unavailable",
        response=response,
        body=None,
    )


# v2.0.28.4 P1-1 — Test 1
def test_retry_on_anthropic_429_succeeds_on_second_attempt():
    """Single 429 then OK → retry wins. Without retry the user loses
    the turn on a transient rate-limit; with retry the turn completes.

    We monkey-patch ``model.invoke`` so it raises a real
    ``anthropic.APIStatusError(429)`` on the first call and returns
    a stub ``AIMessage`` on the second. ``invoke_with_retry`` should
    consume both, log a ``op=llm_retry`` WARNING on the failure
    attempt, and return the success response.
    """
    from src.llm.structured import invoke_with_retry

    success_response = MagicMock()
    success_response.content = "ok"
    model = MagicMock()
    model.invoke = MagicMock(
        side_effect=[_anthropic_429_error(), success_response]
    )

    result = invoke_with_retry(model, [MagicMock()])
    assert result is success_response
    assert model.invoke.call_count == 2


# v2.0.28.4 P1-1 — Test 2
def test_retry_on_openai_503_succeeds_on_second_attempt():
    """Same shape as Test 1 but OpenAI 503. Verifies retry works
    across both providers — the predicate treats them identically."""
    from src.llm.structured import invoke_with_retry

    success_response = MagicMock()
    success_response.content = "ok"
    model = MagicMock()
    model.invoke = MagicMock(
        side_effect=[_openai_503_error(), success_response]
    )

    result = invoke_with_retry(model, [MagicMock()])
    assert result is success_response
    assert model.invoke.call_count == 2


# v2.0.28.4 P1-1 — Test 3
def test_retry_gives_up_after_3_attempts_and_reraises(monkeypatch):
    """3 retries all raise → tenacity reraises (per ``reraise=True``)
    and ``model.invoke`` was called exactly ``max_attempts`` times.

    Without the ``stop_after_attempt(3)`` cap a permanent outage
    would loop forever. Without ``reraise=True`` the caller would
    see ``None`` (silent) and the user would see an empty answer
    without knowing the LLM was down. Both are bad — this test pins
    the "explicit failure" contract.
    """
    from src.llm import structured as structured_mod
    from src.llm.structured import invoke_with_retry

    model = MagicMock()
    model.invoke = MagicMock(side_effect=_openai_503_error())

    # Shrink the wait to keep the test fast (1s minimum in production
    # but we don't want a 1+2=3s test on the slow path).
    def _fast_exponential(*args, **kwargs):
        return 0  # no wait between attempts

    monkeypatch.setattr(structured_mod, "wait_exponential", _fast_exponential)

    with pytest.raises(openai.APIStatusError):
        invoke_with_retry(model, [MagicMock()], max_attempts=3)
    assert model.invoke.call_count == 3


# v2.0.28.4 P1-1 — Test 4
def test_retry_skips_non_retryable_errors():
    """``ValueError`` is NOT a transient LLM error — it's a real bug
    or user input error. Retrying it 3 times just delays the same
    failure. ``invoke_with_retry`` must reraise immediately.

    This pins the predicate's specificity. If anyone weakens the
    ``_is_retryable`` predicate (e.g. catches a parent class), this
    test catches the regression.
    """
    from src.llm.structured import invoke_with_retry

    model = MagicMock()
    model.invoke = MagicMock(side_effect=ValueError("bad input"))

    with pytest.raises(ValueError):
        invoke_with_retry(model, [MagicMock()], max_attempts=3)
    # Single call — no retry.
    assert model.invoke.call_count == 1


# v2.0.28.4 P1-1 — Test 5 (async sibling)
@pytest.mark.asyncio
async def test_ainvoke_with_retry_retries_on_429(monkeypatch):
    """Async twin: 429 then OK. Verifies ``ainvoke_with_retry`` uses
    the same retry semantics but ``ainvoke`` (async) under the hood
    and an async loop yields control during the backoff wait
    (``AsyncRetrying`` is async-aware)."""
    from src.llm import structured as structured_mod
    from src.llm.structured import ainvoke_with_retry

    success_response = MagicMock()
    success_response.content = "ok"
    model = MagicMock()
    model.ainvoke = AsyncMock(
        side_effect=[_anthropic_429_error(), success_response]
    )

    def _fast_exponential(*args, **kwargs):
        return 0

    monkeypatch.setattr(structured_mod, "wait_exponential", _fast_exponential)

    result = await ainvoke_with_retry(model, [MagicMock()])
    assert result is success_response
    assert model.ainvoke.await_count == 2


# v2.0.28.4 P1-1 — Test 6 (log emission)
def test_retry_emits_op_llm_retry_warning_log(monkeypatch):
    """The retry path must log ``op=llm_retry`` so operators can grep
    it. ``loguru`` ``WARNING`` level is the right noise floor — INFO
    is too noisy for "this is normal during overload", ERROR would
    wake on-call for what's a recoverable condition.

    We use a minimal in-memory loguru sink (per the v2.0.28.1 trap
    about test sinks — must ``try/finally`` and ``configure(patcher=None)``
    cleanup so we don't leak the patcher to subsequent tests).
    """
    import loguru

    from src.llm import structured as structured_mod
    from src.llm.structured import invoke_with_retry

    captured: list[dict] = []

    def _sink(message) -> None:
        record = message.record
        captured.append(
            {"level": record["level"].name, "message": record["message"]}
        )

    sink_id = loguru.logger.add(_sink, level="DEBUG")
    try:
        success_response = MagicMock()
        success_response.content = "ok"
        model = MagicMock()
        model.invoke = MagicMock(
            side_effect=[_anthropic_429_error(), success_response]
        )

        def _fast_exponential(*args, **kwargs):
            return 0

        monkeypatch.setattr(structured_mod, "wait_exponential", _fast_exponential)

        invoke_with_retry(model, [MagicMock()])

        retry_logs = [
            c for c in captured
            if "op=llm_retry" in c["message"]
        ]
        assert len(retry_logs) == 1, (
            f"expected exactly one op=llm_retry log, got {len(retry_logs)}: "
            f"{retry_logs}"
        )
        assert retry_logs[0]["level"] == "WARNING"
        # The secret-redacted error should be there (the test uses a
        # plain message so no actual secrets to redact — the substring
        # ``type=APIStatusError`` is what we verify was emitted).
        assert "type=APIStatusError" in retry_logs[0]["message"]
    finally:
        loguru.logger.remove(sink_id)
