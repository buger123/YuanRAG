"""UUID helpers."""
from __future__ import annotations

import uuid


def new_id() -> str:
    """Return a fresh UUID4 string."""
    return str(uuid.uuid4())


def short_id() -> str:
    """Return a short (8-char) identifier — convenient for thread_ids."""
    return uuid.uuid4().hex[:8]


__all__ = ["new_id", "short_id"]
