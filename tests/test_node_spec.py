"""v2.0.22 (Item 7 Step 7) — NODES registry shape + dispatch contract.

Pre-Step-7, ``src/agent/fsm.py::NODES`` was ``dict[str, Callable]``
— the entry was just the node function. ``run_fsm`` then hardcoded
four ``if current == "react_agent" / "intent_analysis" /
"check_hallucination"`` branches to emit ``step_start`` /
``step_end`` / ``intent`` / ``grounding`` events (the P0-B1 audit
finding: "看似 data-driven, 实际硬编码").

Post-Step-7, ``NODES`` is ``dict[str, NodeSpec]`` with three fields:

* ``fn`` — the async-generator node impl (Step 6 protocol).
* ``emits_pre`` — list of wire event kinds the FSM emits BEFORE
  the node runs (currently ``["step_start"]`` for ``react_agent``
  only).
* ``emits_post`` — list of wire event kinds the FSM emits AFTER
  the node + merge (currently ``["step_end"]`` for ``react_agent``
  only).

Per-call DATA events (``intent`` / ``grounding`` / ``tool_call_start``
/ ``web_search``) are yielded by the node itself BEFORE
``__delta__`` — the plan's "single source of truth" rule.

These tests pin the new contract so a future refactor can't
silently regress it:
1. Every ``NODES`` entry is a :class:`NodeSpec` dict (has ``fn``).
2. ``emits_pre`` / ``emits_post`` are lists of strings (or absent).
3. ``run_fsm`` yields ``step_start`` BEFORE ``react_agent`` and
   ``step_end`` AFTER (via spec dispatch, not ``if current == ...``).
4. ``run_fsm`` does NOT have any ``if current == ...`` hardcoded
   branches left (source-level grep).
5. The data-driven wire events (``intent`` / ``grounding`` /
   ``tool_call_start`` / ``web_search``) are yielded by the
   node function itself, not synthesized by ``run_fsm``.
6. The order of events on a ``react_agent`` step is
   ``step_start`` → node events (tokens / tool_call_start /
   web_search) → ``__delta__`` → ``step_end``.
"""
from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Tuple

import pytest

from src.agent.fsm import FSMEvent, NODES, _run_node, run_fsm
from src.agent.types import NodeSpec
from src.agent.nodes.intent_analysis import intent_analysis
from src.agent.nodes.hallucination import check_hallucination_async
from src.agent.nodes.react_agent import react_agent


# Canonical NODES registry — keeps the test self-documenting; the
# runtime source-of-truth is ``src.agent.fsm.NODES``. Must match
# test_node_protocol.py's EXPECTED_NODES so a node addition shows
# up in both suites.
EXPECTED_NODE_NAMES = {
    "intent_analysis",
    "react_generate_direct",
    "summary_path",
    "react_agent",
    "tools",
    "react_generate",
    "check_hallucination",
    "verify_answer",  # v2.0.29.7 Phase 6 — second-pass verification node
    "react_generate_extractive",  # v2.0.29.9 Phase 8 — verbatim extraction node
}


# ---------------------------------------------------------------------------
# 1) NODES shape: dict[str, NodeSpec] with required ``fn`` key
# ---------------------------------------------------------------------------


def test_nodes_registry_complete():
    """Sanity — every node the FSM routes through is registered."""
    assert set(NODES) == EXPECTED_NODE_NAMES


@pytest.mark.parametrize("name", sorted(EXPECTED_NODE_NAMES))
def test_node_spec_has_fn_key(name):
    """Every NODES entry must be a NodeSpec dict with a ``fn`` key.

    Catches a refactor that puts the function directly in NODES
    (pre-Step-7 shape ``dict[str, Callable]``) — the FSM would
    crash on ``spec["fn"]`` because the entry isn't subscriptable
    as a dict of fields, but is the function itself.
    """
    spec = NODES[name]
    assert isinstance(spec, dict), (
        f"{name!r} NODES entry is not a NodeSpec dict; Step 7 requires "
        f"``dict[str, NodeSpec]`` shape. Got: {type(spec).__name__}"
    )
    assert "fn" in spec, (
        f"{name!r} NodeSpec missing required ``fn`` key; got keys: "
        f"{sorted(spec.keys())}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_NODE_NAMES))
def test_node_spec_emits_pre_post_are_lists_or_absent(name):
    """``emits_pre`` and ``emits_post``, if present, must be lists
    of strings (the wire event kinds the FSM yields declaratively).

    Absent keys are fine (``total=False`` TypedDict) — the FSM uses
    ``spec.get("emits_pre", []) or ()`` so missing = empty loop.
    But if present, they MUST be ``list[str]`` so the FSM can
    iterate without crashing.
    """
    spec = NODES[name]
    for key in ("emits_pre", "emits_post"):
        if key not in spec:
            continue
        val = spec[key]
        assert isinstance(val, list), (
            f"{name!r}.{key} must be a list; got {type(val).__name__}"
        )
        for kind in val:
            assert isinstance(kind, str), (
                f"{name!r}.{key} entry must be a wire event kind "
                f"(str); got {type(kind).__name__}: {kind!r}"
            )


def test_node_specs_for_react_agent_have_step_boundary_events():
    """``react_agent`` must declare ``step_start`` in emits_pre
    and ``step_end`` in emits_post — those are the FSM-managed
    per-step border events that used to be hardcoded in ``run_fsm``.
    """
    spec = NODES["react_agent"]
    assert spec.get("emits_pre") == ["step_start"], (
        f"react_agent.emits_pre must be ['step_start']; got {spec.get('emits_pre')!r}"
    )
    assert spec.get("emits_post") == ["step_end"], (
        f"react_agent.emits_post must be ['step_end']; got {spec.get('emits_post')!r}"
    )


def test_other_nodes_have_no_fsm_managed_pre_post():
    """Only ``react_agent`` has ``step_start`` / ``step_end``
    boundary events today. Every other node has empty (or absent)
    ``emits_pre`` / ``emits_post`` because their progress is
    already conveyed via token events or no progress UI at all.
    """
    for name in EXPECTED_NODE_NAMES - {"react_agent"}:
        spec = NODES[name]
        pre = spec.get("emits_pre", []) or []
        post = spec.get("emits_post", []) or []
        assert pre == [], (
            f"{name!r} unexpectedly has emits_pre={pre!r}; only "
            f"react_agent should have FSM-managed pre-step events."
        )
        assert post == [], (
            f"{name!r} unexpectedly has emits_post={post!r}; only "
            f"react_agent should have FSM-managed post-step events."
        )


# ---------------------------------------------------------------------------
# 2) run_fsm uses spec dispatch — no `if current == ...` hardcoded branches
# ---------------------------------------------------------------------------


def test_run_fsm_has_no_hardcoded_current_branches_in_source():
    """Source-level guard: ``run_fsm`` body must not contain any
    ``if current == "..."`` branches (those are the P0-B1 hardcoded
    emission sites that Step 7 deletes).

    Uses AST instead of raw grep so a future docstring reference
    to the old behavior doesn't trigger a false positive.
    """
    import ast

    from src.agent import fsm as fsm_mod

    src = open(fsm_mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)

    # Find the ``run_fsm`` function definition.
    run_fsm_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_fsm":
            run_fsm_node = node
            break
    assert run_fsm_node is not None, "could not locate run_fsm in fsm.py"

    # Walk the body and look for ``if current == "..."`` patterns.
    # We accept ``if current is None`` (the loop terminator) and
    # ``if current is not None`` (the loop condition) but reject
    # string equality.
    for node in ast.walk(run_fsm_node):
        if isinstance(node, ast.Compare):
            left = node.left
            if isinstance(left, ast.Name) and left.id == "current":
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(
                        comparator.value, str
                    ):
                        pytest.fail(
                            f"fsm.py::run_fsm still has "
                            f"`if current == {comparator.value!r}` — "
                            f"this is a P0-B1 hardcoded emission site "
                            f"that Step 7 should have deleted."
                        )


# ---------------------------------------------------------------------------
# 3) Node-level: data events are yielded by nodes, not synthesized by FSM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_analysis_node_yields_intent_event():
    """``intent_analysis`` (greeting fast-path) MUST yield the
    ``intent`` wire event BEFORE ``__delta__`` — the node is the
    wire event's single source of truth post-Step-7.
    """
    events: list = []
    async for kind, payload in intent_analysis({"current_query": "hi"}):
        events.append((kind, payload))
    kinds = [k for k, _ in events]
    # intent event BEFORE __delta__.
    assert "intent" in kinds, (
        f"intent_analysis must yield an `intent` wire event; got kinds: {kinds}"
    )
    assert "__delta__" in kinds
    assert kinds.index("intent") < kinds.index("__delta__"), (
        f"`intent` must precede `__delta__`; got order: {kinds}"
    )

    # Inspect the intent payload — must carry intent_value + corrected_query.
    intent_payload = next(p for k, p in events if k == "intent")
    assert intent_payload.get("intent_value") == "greeting"
    assert "corrected_query" in intent_payload


@pytest.mark.asyncio
async def test_check_hallucination_node_yields_grounding_event():
    """``check_hallucination_async`` MUST yield the ``grounding``
    wire event BEFORE ``__delta__`` — post-Step-7 the node owns
    this event, not the FSM main loop.
    """
    events: list = []
    async for kind, payload in check_hallucination_async({
        "messages": [],
        "documents": [],
        "answer": "",
    }):
        events.append((kind, payload))
    kinds = [k for k, _ in events]
    # The skipped branch is the only one that runs with empty docs
    # / empty answer, but every branch yields both grounding and
    # __delta__.
    assert "grounding" in kinds, (
        f"check_hallucination_async must yield a `grounding` wire "
        f"event; got kinds: {kinds}"
    )
    assert "__delta__" in kinds
    assert kinds.index("grounding") < kinds.index("__delta__"), (
        f"`grounding` must precede `__delta__`; got order: {kinds}"
    )

    # The grounding payload must carry the status field the runner
    # reads for the events.grounding(...) builder.
    grounding_payload = next(p for k, p in events if k == "grounding")
    assert "status" in grounding_payload
    # Empty docs + empty answer → "skipped" branch.
    assert grounding_payload["status"] == "skipped"


@pytest.mark.asyncio
async def test_react_agent_node_yields_tool_call_start_per_tool_call():
    """``react_agent`` MUST yield one ``tool_call_start`` event
    per AIMessage.tool_call, each stamped with the FSM-passed
    ``step`` so the runner can pair start/end.

    Pre-Step-7, the FSM main loop walked ``last AIMessage.tool_calls``
    in a hardcoded ``if current == "react_agent"`` branch and
    synthesized these events itself. Step 7 moved that walk into
    the node (the source-of-truth rule).

    This test uses a mocked model that returns an AIMessage with
    two tool_calls so we can verify both events fire AND both
    carry the FSM-supplied step number.
    """
    from unittest.mock import AsyncMock, MagicMock
    from langchain_core.messages import AIMessage
    import src.agent.nodes.react_agent as ra

    response = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc1", "name": "retrieve_docs", "args": {"query": "x"}},
            {"id": "tc2", "name": "web_search", "args": {"query": "y"}},
        ],
    )

    fake_model = MagicMock()
    fake_model.bind_tools = MagicMock(return_value=fake_model)
    fake_model.ainvoke = AsyncMock(return_value=response)

    orig = ra.build_chat_model
    ra.build_chat_model = lambda **kw: fake_model
    try:
        events: list = []
        async for kind, payload in react_agent(
            {"messages": [], "step_count": 0, "needs_current_time": False},
            step=4,
        ):
            events.append((kind, payload))
    finally:
        ra.build_chat_model = orig

    # Find tool_call_start events.
    tc_starts = [p for k, p in events if k == "tool_call_start"]
    assert len(tc_starts) == 2, (
        f"react_agent must yield one tool_call_start per tool_call; "
        f"got {len(tc_starts)} events: {tc_starts}"
    )
    # Each carries the FSM-supplied step number so the frontend
    # can pair start with end on the same step.
    for tc_evt in tc_starts:
        assert tc_evt.get("step") == 4, (
            f"tool_call_start must carry the FSM-supplied step "
            f"(4); got: {tc_evt.get('step')}"
        )
        assert tc_evt.get("tool_call_id"), tc_evt
        assert tc_evt.get("tool_name"), tc_evt
        assert tc_evt.get("started_at"), tc_evt

    # web_search event fires after the matching tool_call_start.
    web_search_events = [p for k, p in events if k == "web_search"]
    assert len(web_search_events) == 1, (
        f"react_agent must yield exactly one web_search event for "
        f"the matching tool_call; got {len(web_search_events)}: {web_search_events}"
    )
    assert web_search_events[0].get("attempted") is True


# ---------------------------------------------------------------------------
# 4) End-to-end: run_fsm uses spec dispatch for step_start / step_end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fsm_emits_step_start_step_end_via_spec_dispatch(monkeypatch):
    """End-to-end: drive ``run_fsm`` with a faked ``react_agent``
    NodeSpec that yields one ``__delta__`` (no tool_calls), and
    verify the FSM yields ``step_start`` BEFORE the node and
    ``step_end`` AFTER — via spec dispatch (the node doesn't yield
    either), NOT from a hardcoded ``if current == "react_agent"``
    branch.

    Pre-Step-7, ``run_fsm`` hardcoded ``step_start``/``step_end``
    in the main loop. Post-Step-7, those come from
    ``spec.get("emits_pre")`` / ``spec.get("emits_post")``.

    We monkeypatch the NODES so we don't need a real LLM.
    """

    async def fake_react_agent_node(state, *, step=0):
        yield ("__delta__", {"messages": []})

    async def fake_intent_node(state, *, step=0):
        # The first node — needs to advance to react_agent.
        yield ("__delta__", {"intent": "qa_complex"})

    # Predicate that loops forever on react_agent (so we can
    # verify the boundary events on the FIRST react_agent call
    # without needing to also drive react_generate).
    monkeypatch.setattr(
        "src.agent.fsm.NODES",
        {
            "intent_analysis": NodeSpec(fn=fake_intent_node, emits_pre=[], emits_post=[]),
            "react_agent": NodeSpec(
                fn=fake_react_agent_node,
                emits_pre=["step_start"],
                emits_post=["step_end"],
            ),
        },
    )
    monkeypatch.setattr(
        "src.agent.fsm.PREDICATES",
        {
            "intent_analysis": lambda s: "react_agent",
            "react_agent": lambda s: None,  # stop after first step
        },
    )

    events: list = []
    async for evt in run_fsm({"step_count": 0}, max_steps=10):
        events.append(evt)

    kinds = [e["kind"] for e in events]

    # ``__delta__`` is the FSM/node protocol terminator (Step 6)
    # — it never reaches the caller as a wire event. The FSM
    # consumes it inside ``_run_node`` and applies the payload to
    # ``merge()``; only non-``__delta__`` events are forwarded
    # to the caller. So a minimal fake node yielding only
    # ``__delta__`` produces no forwarded event.
    #
    # With minimal fakes (intent_analysis yields only __delta__,
    # react_agent yields only __delta__), the only wire events
    # are the FSM-managed ``step_start`` / ``step_end`` from
    # react_agent's spec:
    assert kinds == ["step_start", "step_end"], (
        f"Expected [step_start, __delta__, step_end] for "
        f"react_agent iteration; got: {kinds}"
    )

    # And the step_start / step_end carry the FSM-supplied node +
    # step fields (the runner only reads step, but node is kept
    # for debug / observability).
    step_start_evt = events[0]
    assert step_start_evt["kind"] == "step_start"
    # ``intent_analysis`` runs first (step_count 0 → 1), so
    # ``react_agent`` is the SECOND step → step 2.
    assert step_start_evt["step"] == 2
    step_end_evt = events[-1]
    assert step_end_evt["kind"] == "step_end"
    assert step_end_evt["step"] == 2


# ---------------------------------------------------------------------------
# 5) NodeSpec type itself is importable from src.agent.types
# ---------------------------------------------------------------------------


def test_node_spec_type_is_exposed():
    """The new public type is re-exported from ``src.agent.types``
    so callers can ``from src.agent.types import NodeSpec`` to type
    their monkeypatched entries.
    """
    from src.agent import types as types_mod

    assert hasattr(types_mod, "NodeSpec")
    assert hasattr(types_mod, "NodeFn")
    assert "NodeSpec" in types_mod.__all__
    assert "NodeFn" in types_mod.__all__


__all__ = [
    "test_nodes_registry_complete",
    "test_node_spec_has_fn_key",
    "test_node_spec_emits_pre_post_are_lists_or_absent",
    "test_node_specs_for_react_agent_have_step_boundary_events",
    "test_other_nodes_have_no_fsm_managed_pre_post",
    "test_run_fsm_has_no_hardcoded_current_branches_in_source",
    "test_intent_analysis_node_yields_intent_event",
    "test_check_hallucination_node_yields_grounding_event",
    "test_react_agent_node_yields_tool_call_start_per_tool_call",
    "test_run_fsm_emits_step_start_step_end_via_spec_dispatch",
    "test_node_spec_type_is_exposed",
]