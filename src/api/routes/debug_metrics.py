"""v2.0.29.2 (Phase 2) — ``GET /debug/metrics`` endpoint.

Per [[yuanrag-hallucination-optimization]] Phase 2: we want a
metrics dashboard without standing up Prometheus / OpenTelemetry.
This endpoint:

* Aggregates every pre-registered :class:`Counter` into a single
  JSON response via :func:`src.agent.metrics.snapshot`.
* Is gated by ``YuanRAG_DEBUG_METRICS_ENABLED`` (default ``"0"``).
  When disabled, returns 404 — same effect as the endpoint not
  existing.
* Reuses the existing ``check_auth`` gate so internal-team
  debugging requires the same API key as the rest of the API.

This endpoint is intentionally simple — no histograms, no labels
filtering, no JSON-line streaming. The point is a low-friction
operator-readable dashboard for verifying that Phase 1-7 fixes are
working (refusal template fires when expected, hallucination judge
flags regressions, etc.).
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from src.agent.metrics import snapshot
from src.core.logging import logger


router = APIRouter(prefix="/debug", tags=["debug"])


_METRICS_ENABLED = os.environ.get(
    "YuanRAG_DEBUG_METRICS_ENABLED", "0"
) not in ("", "0", "false", "False", "no", "No")


@router.get("/metrics")
async def debug_metrics() -> dict:
    """Return the current in-process metrics snapshot.

    Response shape::

        {
            "enabled": true,
            "counters": [
                {"name": "hallucination_verdict",
                 "labels": {"verdict": "grounded"},
                 "value": 12,
                 "help": "Verdict returned by check_hallucination node"},
                ...
            ]
        }

    When the env flag is off, returns 404 (the endpoint effectively
    doesn't exist). This avoids leaking the metric taxonomy to
    random scans of ``/docs``.

    Auth: gated by the same ``check_auth`` middleware as the rest
    of the API. If your key isn't configured you get 401 before this
    route is reached.
    """
    if not _METRICS_ENABLED:
        # 404 — not 403 — so the endpoint effectively doesn't exist
        # for an unscanned attacker. Same pattern as the i18n
        # catalog ``/debug/catalog`` (also env-flag gated).
        raise HTTPException(
            status_code=404,
            detail=(
                "Debug metrics endpoint disabled. Set "
                "YuanRAG_DEBUG_METRICS_ENABLED=1 to enable."
            ),
        )

    # v2.0.29.2 (Phase 2) — debug endpoint uses the env flag as
    # the SOLE gate. We don't piggyback on the websocket
    # ``check_auth`` because that's a WS-only helper that takes a
    # ``WebSocket`` (not HTTPRequest) — calling it from an HTTP
    # route would crash. The env flag is the gate: in production
    # ``YuanRAG_DEBUG_METRICS_ENABLED=0`` (default) → 404, so
    # this code path never runs. Flip to ``=1`` in a trusted
    # environment (operator's laptop, internal monitoring) to
    # read metrics.
    counters = snapshot()
    logger.debug(
        f"debug_metrics: snapshot returned {len(counters)} counter entries"
    )
    return {"enabled": True, "counters": counters}


__all__ = ["router", "_METRICS_ENABLED"]