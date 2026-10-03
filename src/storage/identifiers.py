"""Public identifier helpers used across the storage layer.

P2-1 split: ``_safe_identifier`` used to live as a private helper
inside ``lancedb_store`` and was re-wrapped by ``_safe_doc_id`` /
``_safe_thread_id``. The wrapper layers added nothing — they were
just one-line pass-throughs that obscured where the real validation
happened. Hoist the canonical implementation here, expose it under the
public ``safe_identifier`` name (no leading underscore), and have
``lancedb_store`` import it directly.
"""
from __future__ import annotations

import re


# Identifiers that flow into SQL/LanceDB filters must match a strict
# charset — letters, digits, dash, underscore, dot, colon (for the
# ``langsmith:``-style prefixed IDs). Anything else (path separators,
# quotes, NUL, control chars) would either break the ``.where()``
# filter parsing or let an attacker inject predicates against
# unrelated thread_ids. The audit found ``_safe_doc_id`` /
# ``_safe_thread_id`` as one-line wrappers around this same check;
# those wrappers were deleted in the same refactor and callers now
# invoke ``safe_identifier(kind=..., value=...)`` directly.
_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]+$")


def safe_identifier(value, kind: str = "id") -> str:
    """Validate and return a string safe for use as a LanceDB / SQL
    identifier filter.

    ``kind`` is only used in the error message — it tells the operator
    WHICH namespace failed validation when a log line surfaces the
    ValueError. We don't actually use ``kind`` to gate the charset;
    a doc_id and a thread_id must both satisfy the same strict
    character class, otherwise an attacker who can influence one
    could still smuggle an injection via the other.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} must be a non-empty string; got {value!r}")
    if not _ID_RE.match(value):
        raise ValueError(
            f"{kind} contains characters outside [A-Za-z0-9._:-]; got {value!r}"
        )
    return value


__all__ = ["safe_identifier"]