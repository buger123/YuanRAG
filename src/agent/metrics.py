"""Phase 2 — In-process metrics for the hallucination optimization project.

Per [[yuanrag-hallucination-optimization]] Phase 2: we explicitly chose
NOT to introduce Prometheus / OpenTelemetry. Instead, this module
provides:

  * An in-process :class:`Counter` registry (monotonic, with optional
    labels) — zero external dependencies.
  * Optional JSON-line emission to a dedicated loguru sink so the
    values are visible in the existing ``logs/app.log`` rotation
    (no new infrastructure to operate).
  * A :func:`snapshot` aggregator that powers the
    ``GET /debug/metrics`` endpoint (env-flag gated).

Counter conventions
-------------------
Names follow Prometheus text-format convention (lowercase, underscores,
no dots) so they map cleanly to any future scraping. Each counter
carries an optional set of label names; ``inc()`` passes label values
that get concatenated into the storage key. Counters are thread-safe
via :class:`threading.Lock` — needed because the FSM runner can
increment from any thread that owns the websocket handler.

Pre-registered counters
-----------------------
Counters are pre-declared at module load even when their call sites
land in later phases (Phase 4 / 6 / 7). This is intentional:

  * ``/debug/metrics`` can show the full taxonomy without crashing
    when a Phase-4 call site hasn't fired yet.
  * Tests can verify the pre-registration contract without
    depending on the call site being wired.

Activation model
----------------
All counters are ALWAYS defined and ALWAYS accept ``inc()`` calls.
JSON-line emission is gated by the ``YuanRAG_METRICS_EMIT``
environment variable (default ``"0"``) — keeping the runtime
overhead near zero in production while allowing opt-in for
debugging or for the post-Phase-2 metrics dashboard.
"""
from __future__ import annotations

import os
import threading
from typing import Iterable

from src.core.logging import logger as _log


# ---------------------------------------------------------------------------
# Counter primitive
# ---------------------------------------------------------------------------


class Counter:
    """Monotonic counter with optional labels.

    The label-tuple design mirrors Prometheus text-format:
    ``my_counter{label_a="x",label_b="y"}`` is stored as a single
    key ``"my_counter|x|y"``. ``inc()`` is the only mutator; there
    is no ``dec()`` because all metrics in this project are
    monotonic event counters (no prototype pollution that could
    decrease them).

    Thread safety: per-Counter :class:`threading.Lock` covers the
    inner dict mutation. Increment is O(1) under the lock; the FSM
    runner calls ``inc()`` from many threads (one per WS handler),
    so the lock is required.
    """

    __slots__ = (
        "name",
        "help_text",
        "label_names",
        "_values",
        "_lock",
    )

    def __init__(self, name: str, help_text: str, label_names: tuple = ()):
        if not name:
            raise ValueError("counter name must be non-empty")
        if any(not ln.isidentifier() for ln in label_names):
            raise ValueError(f"label names must be identifiers: {label_names}")
        self.name = name
        self.help_text = help_text
        self.label_names = label_names
        # key -> float. Keys are "name|lab1|lab2..." for labeled
        # counters; just "name" for label-less ones.
        self._values: dict[str, float] = {}
        self._lock = threading.Lock()

    def _key(self, labels: dict) -> str:
        if not self.label_names:
            return self.name
        # Stable order: use declared label_names.
        return "|".join([self.name] + [str(labels.get(ln, "")) for ln in self.label_names])

    def inc(self, amount: float = 1.0, **labels) -> None:
        """Increment this counter by ``amount`` (default 1.0).

        ``labels`` keyword arguments correspond to ``label_names``;
        unknown labels raise ``ValueError`` so we catch typos in
        call sites at increment time, not at debug time.
        """
        if amount < 0:
            raise ValueError(f"counter {self.name} is monotonic; got {amount}")
        unknown = set(labels) - set(self.label_names)
        if unknown:
            raise ValueError(
                f"counter {self.name} got unknown labels {sorted(unknown)}; "
                f"declared: {list(self.label_names)}"
            )
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, **labels) -> float:
        """Read the current value for a label combination."""
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def snapshot(self) -> list[dict]:
        """Return a serializable view of this counter.

        Format: ``[{"name": ..., "labels": {...}, "value": ...}, ...]``
        suitable for direct JSON emission.

        Label-less counters always surface themselves in the snapshot
        (even at value 0) so the dashboard can show the full metric
        taxonomy from boot — verifying the pre-registration contract.
        Labeled counters only surface AFTER the first
        ``inc(label=...)`` because we don't enumerate every possible
        label combination.
        """
        with self._lock:
            items = list(self._values.items())
        out = []
        if not self.label_names:
            # Label-less: emit a single zero entry if no values yet,
            # so the dashboard sees the counter before its first fire.
            if not items:
                out.append({"name": self.name, "labels": {}, "value": 0.0})
        for key, value in items:
            if "|" in key:
                parts = key.split("|")
                name = parts[0]
                labels = dict(zip(self.label_names, parts[1:]))
            else:
                name = key
                labels = {}
            out.append({"name": name, "labels": labels, "value": value})
        return out

    def reset(self) -> None:
        """Reset all values for this counter. Tests only."""
        with self._lock:
            self._values.clear()


# ---------------------------------------------------------------------------
# Pre-registered counters (Phase 2 scope)
# ---------------------------------------------------------------------------


# Phase 1 / 2: shipped counters (call sites wired in this batch)
HALLUCINATION_VERDICT = Counter(
    "hallucination_verdict",
    "Verdict returned by check_hallucination node",
    label_names=("verdict",),  # grounded | ungrounded | skipped
)
RETRIEVAL_EMPTY = Counter(
    "retrieval_empty",
    "Retrieval returned 0 docs (hybrid + rerank)",
)
SYNTHESIS_TM_IGNORED = Counter(
    "synthesis_tm_ignored",
    "Synthesis LLM produced output without citing an available ToolMessage",
    label_names=("tool",),
)
SYNTHESIS_PROMPT_TEMPLATE = Counter(
    "synthesis_prompt_template",
    "Which prompt template react_generate walked",
    label_names=("template",),  # generate | direct | refusal_empty | ...
)
GROUNDING_EVENT_EMITTED = Counter(
    "grounding_event_emitted",
    "grounding wire event emitted by check_hallucination",
    label_names=("status",),
)

# Phase 4 forward-looking counters (call sites wired in Phase 4 / 6 / 7)
RETRIEVAL_FILTERED_LOW_SCORE = Counter(
    "retrieval_filtered_low_score",
    "Chunks dropped because top rerank_score < threshold (Phase 4)",
)
RETRIEVAL_FILTERED_EXPIRED = Counter(
    "retrieval_filtered_expired",
    "Chunks dropped because metadata.expires_at < now() (Phase 4)",
)
VERIFICATION_REGENERATED = Counter(
    "verification_regenerated",
    "Verification agent regenerated the answer after rule-check fail (Phase 6)",
    label_names=("outcome",),  # regenerated | fallback_refusal | verifier_failed
)
CHUNK_REJECTED_OVERSIZE = Counter(
    "chunk_rejected_oversize",
    "Chunks dropped at retrieval time because doc exceeded MAX_CHUNKS_PER_DOC (Phase 7)",
)

# v2.0.29.9 (Phase 8) — verbatim extraction fallback counter. Fires when
# ``react_generate_extractive`` short-circuits to a refusal template
# (reason=empty / empty_bulk / low_relevance) OR when the extractive
# LLM raises an exception (reason=llm_error). The ``reason`` label lets
# dashboards separate "user asked a verbatim question but no docs
# matched" (reason=low_relevance) from "extractive LLM crashed"
# (reason=llm_error) — different operator responses for each.
EXTRACTIVE_FALLBACK = Counter(
    "extractive_fallback",
    "Phase 8 react_generate_extractive fallback to REFUSAL_TEMPLATES "
    "(reason=empty|empty_bulk|low_relevance|llm_error)",
    label_names=("reason",),
)


_ALL_COUNTERS: tuple[Counter, ...] = (
    HALLUCINATION_VERDICT,
    RETRIEVAL_EMPTY,
    SYNTHESIS_TM_IGNORED,
    SYNTHESIS_PROMPT_TEMPLATE,
    GROUNDING_EVENT_EMITTED,
    RETRIEVAL_FILTERED_LOW_SCORE,
    RETRIEVAL_FILTERED_EXPIRED,
    VERIFICATION_REGENERATED,
    CHUNK_REJECTED_OVERSIZE,
    EXTRACTIVE_FALLBACK,
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def snapshot() -> list[dict]:
    """Aggregate every counter into a single serializable list.

    Returns a flat list (no grouping) because some counters are
    label-less and others are labeled — emitting two different
    structures would force the ``/debug/metrics`` endpoint to do
    post-processing. Flat list is also what loguru JSON lines
    expect.
    """
    out: list[dict] = []
    for c in _ALL_COUNTERS:
        for entry in c.snapshot():
            out.append({**entry, "help": c.help_text})
    return out


def reset_all() -> None:
    """Reset every counter. Tests only."""
    for c in _ALL_COUNTERS:
        c.reset()


# ---------------------------------------------------------------------------
# JSON-line emission to loguru
# ---------------------------------------------------------------------------


_EMIT_ENABLED = os.environ.get("YuanRAG_METRICS_EMIT", "0") not in ("", "0", "false", "False")


def _emit_metrics_line() -> None:
    """Emit one JSON line per counter to the loguru stream.

    Called by callers who want a metrics dump (e.g. the runner at
    end-of-turn). The line goes through loguru so it shares the
    existing file rotation + trace_id injection — no new file
    sink needed.

    Format: ``{"event": "metrics", "counters": [...]}`` so loguru's
    parsers / downstream tools can recognize it as a structured
    metric event rather than free-form log text.
    """
    if not _EMIT_ENABLED:
        return
    payload = {"event": "metrics", "counters": snapshot()}
    _log.bind(payload=payload).info("metrics_snapshot")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _all_counters() -> Iterable[Counter]:
    """Return the internal counter tuple. Tests only."""
    return _ALL_COUNTERS


__all__ = [
    "Counter",
    "HALLUCINATION_VERDICT",
    "RETRIEVAL_EMPTY",
    "SYNTHESIS_TM_IGNORED",
    "SYNTHESIS_PROMPT_TEMPLATE",
    "GROUNDING_EVENT_EMITTED",
    "RETRIEVAL_FILTERED_LOW_SCORE",
    "RETRIEVAL_FILTERED_EXPIRED",
    "VERIFICATION_REGENERATED",
    "CHUNK_REJECTED_OVERSIZE",
    "EXTRACTIVE_FALLBACK",
    "snapshot",
    "reset_all",
    "_emit_metrics_line",
]