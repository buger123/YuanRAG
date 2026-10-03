"""v2.0.28.5 Item 11 PR-3 F-R6 — Bing 429 short-circuit.

Pre-PR-3, Bing's host loop walked both ``_BING_HOSTS`` regardless of
HTTP status. If the first host returned 429 (rate-limited), the
loop would try the second host anyway — paying another full timeout
even though rate-limit is typically ASN-scoped (both hosts in the
same CDN share rate limits). Post-PR-3, a 429 from the first host
sets a mutable flag, the loop checks the flag at the top of the
next iteration, and short-circuits to ``return []`` so the
dispatcher falls through to the next engine (e.g. DDG).

This file pins the short-circuit + WARNING log behavior.
"""
from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest


class _FakeHTTPError(urllib.error.HTTPError):
    """Stand-in for ``urllib.error.HTTPError`` that bypasses the real
    constructor's strict signature (it requires ``fp``, ``hdrs``, ``url``).
    Only ``.code`` is exercised by ``bing._fetch_one`` — that attribute is
    set explicitly here.

    IMPORTANT: must subclass ``urllib.error.HTTPError`` so that
    ``bing._fetch_one``'s ``except urllib.error.HTTPError as exc`` clause
    actually catches it; a plain ``Exception`` would bubble up and make
    the test fail with "HTTPError raised" instead of the expected
    short-circuit + empty result.
    """

    def __init__(self, code: int):
        # url="" + hdrs=None + fp=None satisfy the parent signature.
        super().__init__(url="", code=code, msg=f"HTTP {code}", hdrs=None, fp=None)
        self.code = code


@pytest.mark.asyncio
async def test_bing_429_from_first_host_short_circuits_chain():
    """When ``_fetch_one`` returns ``[]`` AND signals rate_limited
    for the first host, the second host must NOT be called.

    We patch ``_fetch_one`` to:
      - First call: returns ``[]`` and sets ``rate_limited[0]=True``
      - Second call: would raise ``AssertionError`` if invoked
        (proves the short-circuit).
    """
    from src.web_search import bing

    call_count = {"n": 0}

    def fake_fetch_one(host, query, max_results, timeout, locale="en-US", *, rate_limited=None):
        call_count["n"] += 1
        if rate_limited is not None:
            rate_limited[0] = True  # signal 429
        return []

    with patch.object(bing, "_fetch_one", side_effect=fake_fetch_one):
        results = await bing.search_web_bing("test query", max_results=5)

    assert results == []
    assert call_count["n"] == 1, (
        f"short-circuit failed: expected 1 host call, got {call_count['n']}"
    )


@pytest.mark.asyncio
async def test_bing_no_429_walks_full_chain():
    """Sanity check the OTHER direction: when neither host is
    rate-limited, both hosts are visited until one returns results.

    This guards against over-eager short-circuit (e.g. empty-list
    must NOT trigger the short-circuit — only the explicit
    ``rate_limited`` flag does).
    """
    from src.web_search import bing

    call_count = {"n": 0}

    def fake_fetch_one(host, query, max_results, timeout, locale="en-US", *, rate_limited=None):
        call_count["n"] += 1
        # First host returns empty (no rate-limit signal);
        # second host returns empty too. Both should be called.
        return []

    with patch.object(bing, "_fetch_one", side_effect=fake_fetch_one):
        results = await bing.search_web_bing("test query", max_results=5)

    assert results == []
    assert call_count["n"] == 2, (
        f"expected both hosts called when no rate-limit, got {call_count['n']}"
    )


@pytest.mark.asyncio
async def test_bing_429_emits_op_warning_log():
    """When the first host returns 429, ``op=web_search.bing
    rate_limited`` WARNING must be emitted (operator visibility).

    Without this log, an operator chasing "why did this query
    return nothing?" would not see the rate-limit pressure.

    We exercise the REAL ``_fetch_one`` (not a spy) by patching
    ``urllib.request.urlopen`` at its bing-side module reference to
    return a fake response with status=429. This way the WARNING
    inside ``_fetch_one`` actually fires.
    """
    import loguru

    from src.web_search import bing

    captured: list[dict] = []

    def _sink(message) -> None:
        record = message.record
        captured.append({"level": record["level"].name, "message": record["message"]})

    sink_id = loguru.logger.add(_sink, level="DEBUG")

    try:
        class _FakeResp:
            status = 429

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(req, timeout=None, context=None):
            return _FakeResp()

        with patch.object(bing.urllib.request, "urlopen", side_effect=fake_urlopen):
            await bing.search_web_bing("test query", max_results=5)

        # The 429 WARNING must appear with op= label.
        bing_warnings = [
            c for c in captured
            if c["level"] == "WARNING" and "op=web_search.bing" in c["message"]
            and "rate_limited" in c["message"]
        ]
        assert len(bing_warnings) >= 1, (
            f"expected ≥1 op=web_search.bing WARNING, got: "
            f"{[c for c in captured if c['level'] == 'WARNING']}"
        )
    finally:
        loguru.logger.remove(sink_id)


@pytest.mark.asyncio
async def test_bing_429_via_http_error_also_short_circuits():
    """Some Bing CDN responses raise ``urllib.error.HTTPError``
    instead of returning 429 as a plain status. The HTTPError path
    must also set ``rate_limited[0]=True`` and emit the WARNING.
    """
    import loguru

    from src.web_search import bing

    captured: list[dict] = []

    def _sink(message) -> None:
        record = message.record
        captured.append({"level": record["level"].name, "message": record["message"]})

    sink_id = loguru.logger.add(_sink, level="DEBUG")

    try:
        def fake_urlopen(req, timeout=None, context=None):
            raise _FakeHTTPError(429)

        with patch.object(bing.urllib.request, "urlopen", side_effect=fake_urlopen):
            results = await bing.search_web_bing("test query", max_results=5)

        assert results == []
        # WARNING must still fire via the HTTPError branch.
        warnings = [
            c for c in captured
            if c["level"] == "WARNING" and "op=web_search.bing" in c["message"]
        ]
        assert len(warnings) >= 1, (
            f"expected WARNING via HTTPError path, got: "
            f"{[c for c in captured if c['level'] == 'WARNING']}"
        )
    finally:
        loguru.logger.remove(sink_id)
