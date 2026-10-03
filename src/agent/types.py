"""Shared FSM / wire-protocol types.

Centralizes typed-dict definitions that were scattered across
``fsm.py``, ``events.py`` and ``nodes/*`` modules so a single
``from src.agent.types import X`` import makes the contract visible
to callers (instead of three modules each defining their own
copy of the same shape).

v2.0.22 (Item 7 Step 7) — introduced :class:`NodeSpec` so the
``NODES`` registry can declare per-step FSM-managed wire events
(``step_start`` / ``step_end``) declaratively, while per-call DATA
events (``intent`` / ``grounding`` / ``tool_call_start``) are still
yielded by the node function itself (the plan's "single source of
truth" rule).

v2.0.22 (Item 7 Step 10) — added :data:`RouteDecision` literal so
``_intent_to_route_decision`` (react_generate), the
``FSMEvent.route_decision`` field (fsm), and the citation helpers
(citations) all share the same closed vocabulary, instead of each
module declaring ``str | None`` / ``Optional[str]`` and silently
treating typos like ``"retreive"`` as ``"direct"``.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Literal, TypedDict


# Closed vocabulary for the v1.1.3-vintage routing decision. The
# generate node (``_intent_to_route_decision``) maps v2.0 intent →
# this literal; the FSM forwards it on the ``answer_complete`` event
# (kept in ``FSMEvent.route_decision``); the runner reads it and
# passes it to :func:`filter_sources_to_cited` /
# :func:`renumber_citations_and_sources` which short-circuit to
# ``[]`` when it isn't ``"retrieve"``. Keeping this Literal here
# (instead of three ``str``-typed declarations) lets a single
# ``assert route_decision in (None, *get_args(RouteDecision))`` at
# the helper entry catch typos loudly rather than silently treating
# them as the ``direct`` branch.
RouteDecision = Literal["direct", "retrieve"]


# An async-generator node implementation. Each ``NODES`` entry's
# ``fn`` must match this shape: ``async def (state, *, step) ->
# AsyncIterator[(kind, payload)]`` that yields zero or more
# ``(kind, payload)`` event tuples and ends with exactly one
# ``("__delta__", delta_dict)`` terminator. See
# ``fsm._run_node`` for the protocol enforcement.
NodeFn = Callable[..., AsyncIterator[tuple[str, Any]]]


class NodeSpec(TypedDict, total=False):
    """Per-node FSM-managed wire events.

    Pre-Step-7, ``NODES`` was ``dict[str, Callable]`` — the entry
    was just the node function. The FSM main loop then hardcoded
    four ``if current == "react_agent" / "intent_analysis" /
    "check_hallucination"`` branches to emit ``step_start`` /
    ``step_end`` / ``intent`` / ``grounding`` events. That's the
    P0-B1 audit finding: "看似 data-driven, 实际硬编码".

    Step 7 collapses this: ``NODES["x"]`` is a :class:`NodeSpec`
    declaring:

    * ``fn`` — the async-generator node impl (single source of
      truth for per-call data events; the node yields its own
      ``intent`` / ``grounding`` / ``tool_call_start`` events
      before ``__delta__``).
    * ``emits_pre`` — list of wire event kinds the FSM emits
      BEFORE the node runs (currently just ``"step_start"`` for
      ``react_agent`` so the frontend's "thinking..." spinner
      lights up at the right moment).
    * ``emits_post`` — list of wire event kinds the FSM emits
      AFTER the node + merge (currently just ``"step_end"`` for
      ``react_agent`` so the per-step border renders after the
      tool calls settle).

    All fields are ``total=False`` because Step 12+ may add
    ``emits_recovery`` / etc.; absent key = no event emitted.
    """

    fn: NodeFn
    emits_pre: list[str]
    emits_post: list[str]


__all__ = ["NodeFn", "NodeSpec", "RouteDecision"]