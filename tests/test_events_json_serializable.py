"""v1.1.8 regression — every WS event payload MUST round-trip through
``json.dumps`` without raising.

Bug background
--------------
Sept 2026 production stack trace::

    TypeError: Object of type type is not JSON serializable
    File "src/api/websocket.py", line 76, in _safe_send_json
        await websocket.send_json(payload)

The culprit: ``src.agent.events.web_search`` shipped
``{"type": "web_search", "attempted": bool}`` — the Python ``bool``
class object, NOT a bool instance. Starlette's ``send_json`` calls
``json.dumps`` which rejects any value whose ``__class__.__name__``
is ``"type"``. Every time a turn triggered ``search_web`` the WS
send raised and the entire turn's events were lost — the user saw
a stalled UI on web-search fallback turns.

Why v1.1.7 made it visible
-------------------------
The typo was dormant for years because before v1.1.7's relevance
floor, almost every turn stayed on the local-retrieval path and
the graph never reached ``search_web``. Once ``_decide_after_grade``
started routing low-relevance turns to ``search_web`` (Bug 2 fix),
every such turn hit the typo and the WS frame died.

Why a guardrail test, not just one regression test
--------------------------------------------------
The ``events.error(message, **extra)`` builder accepts arbitrary
``**extra`` kwargs; a future caller passing
``events.error("...", code=SomeClass)`` would re-introduce the
same class of bug. A round-trip-every-event test catches ANY
non-JSON-safe value that lands in any event, regardless of which
builder produced it. The cost is one ``json.dumps`` call per event
type — trivial.
"""
from __future__ import annotations

import json

import pytest

from src.agent import events


# ---------------------------------------------------------------------------
# Direct regression: web_search returns bool VALUE, not bool CLASS
# ---------------------------------------------------------------------------


class TestWebSearchEventIsJsonSerializable:
    """v1.1.8 — the ``bool`` typo fix. Pre-3, this test would fail with
    ``Object of type type is not JSON serializable``."""

    def test_web_search_true_is_json_serializable(self):
        """The canonical trigger: ``web_search(True)`` must serialize."""
        ev = events.web_search(True)
        # Sanity: ``attempted`` is now a bool INSTANCE, not the bool class.
        assert ev["attempted"] is True, (
            f"web_search.attempted must be a bool instance, got "
            f"{ev['attempted']!r} (type {type(ev['attempted']).__name__}). "
            "If this is ``<class 'bool'>`` the v1.1.8 fix regressed."
        )
        # The actual user-visible bug: json.dumps used to raise here.
        json.dumps(ev)  # no exception → fix is in place

    def test_web_search_false_is_json_serializable(self):
        ev = events.web_search(False)
        assert ev["attempted"] is False
        json.dumps(ev)

    def test_web_search_coerces_truthy_non_bool_to_true(self):
        """Defensive: ``web_search(1)`` should round-trip as ``true``,
        not leak the integer through as ``1`` (the frontend's
        ``=== true`` check would silently break). The original code
        didn't coerce; the fix preserves the existing semantic of
        ``bool(attempted)`` (truthy → True, falsy → False)."""
        assert events.web_search(1)["attempted"] is True
        assert events.web_search("yes")["attempted"] is True

    def test_web_search_coerces_falsy_non_bool_to_false(self):
        assert events.web_search(0)["attempted"] is False
        assert events.web_search("")["attempted"] is False
        assert events.web_search(None)["attempted"] is False

    def test_web_search_serialized_payload_shape(self):
        """Wire-contract regression: the frontend's React code reads
        ``event.attempted === true`` (strict identity, not just
        truthiness). If ``attempted`` becomes ``1`` or ``"True"``,
        the web-search banner never renders. Lock the JSON shape."""
        ev = events.web_search(True)
        assert json.loads(json.dumps(ev)) == {
            "type": "web_search",
            "attempted": True,  # JSON true, not 1
        }


# ---------------------------------------------------------------------------
# Guardrail: every event builder must round-trip through json.dumps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_factory",
    [
        pytest.param(lambda: events.token("hi"), id="token"),
        pytest.param(lambda: events.reasoning("thinking..."), id="reasoning"),
        pytest.param(lambda: events.web_search(True), id="web_search_true"),
        pytest.param(lambda: events.web_search(False), id="web_search_false"),
        pytest.param(
            lambda: events.answer_complete(
                sources=[
                    {
                        "index": 1,
                        "chunk_id": "",
                        "doc_id": "d1",
                        "filename": "doc.pdf",
                        "page": 1,
                        "section": None,
                        "sheet": None,
                        "text": "excerpt...",
                        "score": 0.85,
                        "url": None,
                        "domain": None,
                        "source_kind": "local",
                    },
                    {
                        "index": 2,
                        "chunk_id": "",
                        "doc_id": "web-abc",
                        "filename": "Example Domain",
                        "page": None,
                        "section": None,
                        "sheet": None,
                        "text": "snippet...",
                        "score": 0.0,
                        "url": "https://example.com/article",
                        "domain": "example.com",
                        "source_kind": "web",
                    },
                ],
                answer="See [1] and [2].",
            ),
            id="answer_complete",
        ),
        pytest.param(lambda: events.grounding("grounded"), id="grounding"),
        pytest.param(lambda: events.error("models loading"), id="error"),
        pytest.param(
            lambda: events.error("rate limit", code="rate_limited"),
            id="error_with_extra",
        ),
        pytest.param(lambda: events.done(), id="done"),
    ],
)
def test_every_event_round_trips_through_json_dumps(event_factory):
    """Guardrail: any event builder that yields a non-JSON-safe value
    breaks the WS send and stalls the UI. ``json.dumps`` is the
    cheapest reproducible check.

    Parametrize over every builder so a new event added later has to
    add its own parameterization — the test stays in sync with the
    contract.
    """
    ev = event_factory()
    # Must not raise. ``TypeError: Object of type type is not JSON
    # serializable`` (the v1.1.8 bug) would raise here.
    serialized = json.dumps(ev)
    # And it must round-trip through json.loads unchanged.
    reloaded = json.loads(serialized)
    assert reloaded == ev


# ---------------------------------------------------------------------------
# Cross-regex: the runner's actual ``web_search`` event (the bug path)
# ---------------------------------------------------------------------------


def test_runner_web_search_event_is_json_serializable_end_to_end():
    """The runner constructs ``events.web_search(bool(delta.get(...)))``
    and forwards the result through the WS. Reconstruct that exact call
    shape and verify the wire payload is JSON-safe.

    Pre-v1.1.8: ``bool`` (the class) was placed in the payload; this
    test would fail with TypeError. Post-v1.1.8: ``bool(attempted)``
    coerces to a bool instance and the payload serializes fine.
    """
    # Simulate what the runner sees on the search_web node path:
    # ``delta.get("web_search_attempted")`` is True / False.
    for raw in (True, False, None, 1, 0):
        ev = events.web_search(bool(raw))
        # The exact path that used to blow up:
        json.dumps(ev)  # no exception

        # And the frontend contract: parsed ``attempted`` must be a
        # JSON boolean, not a class.
        parsed = json.loads(json.dumps(ev))
        assert parsed["attempted"] is bool(raw), (
            f"For raw={raw!r}, parsed.attempted must equal bool(raw); "
            f"got {parsed['attempted']!r}"
        )