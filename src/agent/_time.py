"""Time helpers shared across the agent layer.

Item 7 Step 1 (v2.0.22 FSM cleanup) — centralizes ``now_iso()`` so the
runner, FSM nodes, and answer builders all stamp ``created_at`` with
the same format. UTC by design (canonical across deployments).

Why a single helper rather than ``datetime.now(timezone.utc).isoformat()``
inline at every call site:

* Four call sites used to each carry their own copy; a future change
  to the format (microseconds, local-time zone, etc.) would have to
  remember to update all four or produce inconsistent timestamps
  on the same AIMessage.
* Lets tests patch ``now_iso`` once to freeze time across the agent
  (used by ``test_source_kinds_banner`` and similar).
"""
from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string.

    Used for ``created_at`` injection on assistant AIMessages and
    on ``started_at`` of tool-call lifecycle events. Format is
    ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00`` (Python's default
    ``datetime.isoformat()`` with an explicit ``+00:00`` suffix).
    """
    return datetime.now(timezone.utc).isoformat()


__all__ = ["now_iso"]
