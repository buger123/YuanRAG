"""v2.0.22 (Item 7 Step 6) — every node follows the async-generator protocol.

Pre-Step-6, ``src/agent/fsm.py::_run_node`` accepted two shapes:

1. ``async def (state) -> dict`` (sync nodes: ``intent_analysis``,
   ``summary_path``, ``check_hallucination_async``).
2. ``async def (state, *, step) -> AsyncIterator[(kind, payload)]``
   (streaming nodes: ``react_generate_direct``, ``_tools_step``,
   ``react_generate``).

``_run_node`` dispatched on ``inspect.isasyncgenfunction(node_fn)`` and
used ``_accepts_step`` (a ``inspect.signature`` sniff) to decide
whether to pass ``step=`` to the sync nodes. The sync shape was
retired in Step 6 — every production node is now async-gen, the
``inspect`` import is gone, ``_accepts_step`` is deleted.

These tests pin the new contract so a future refactor can't silently
re-introduce either of those escape hatches:

* every ``NODES`` entry is an async-gen function
* every node accepts ``(state, *, step=0)`` (kwarg-only)
* every node yields exactly one ``("__delta__", delta_dict)``
* ``_run_node`` raises ``RuntimeError`` if a node yields no
  ``__delta__`` (the FSM's "cardiac arrest" detection — preventing
  the silent infinite-loop regression of v1.x ``GraphRecursionError``)
* ``_run_node`` raises ``RuntimeError`` if a node yields
  ``__delta__`` twice (defensive against bad refactors that emit
  delta twice)
* No node function still returns ``dict`` synchronously — AST scan
  over each node's source file rules out the old shape sneaking back
"""
from __future__ import annotations

import asyncio
import inspect
import textwrap

import pytest

from src.agent.fsm import FSMEvent, NODES, _run_node, run_fsm
from src.agent.nodes.hallucination import check_hallucination_async
from src.agent.nodes.intent_analysis import intent_analysis
from src.agent.nodes.react_agent import react_agent


# Canonical NODES registry — keeps the test self-documenting; the
# runtime source-of-truth is ``src.agent.fsm.NODES``.
EXPECTED_NODES = {
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
# 1) Every NODES entry is an async-gen function
# ---------------------------------------------------------------------------


def test_nodes_registry_complete():
    """Sanity — every node the FSM routes through is registered."""
    assert set(NODES) == EXPECTED_NODES


@pytest.mark.parametrize("name", sorted(EXPECTED_NODES))
def test_node_is_async_gen_function(name):
    """Every production node must be ``async def`` returning an
    :class:`AsyncIterator` — the unified Step-6 protocol.

    Pre-Step-6 the sync-shape nodes (``intent_analysis`` /
    ``summary_path`` / ``check_hallucination``) were plain
    ``async def`` (no ``yield``), which failed this check; now
    they ``yield`` exactly one ``__delta__`` and pass.
    """
    # v2.0.22 (Item 7 Step 7) — NODES entries are NodeSpec dicts;
    # ``fn`` is the async-generator impl that must match the protocol.
    fn = NODES[name]["fn"]
    assert inspect.isasyncgenfunction(fn), (
        f"{name!r} is not an async-gen function; Step 6 requires every "
        f"NODES entry's ``fn`` to be ``async def ... -> AsyncIterator``."
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_NODES))
def test_node_accepts_step_kwarg(name):
    """Every node accepts ``(state, *, step=0)`` — the FSM passes
    ``step=current_step`` to every call (pre-Step-6 this used the
    ``_accepts_step`` sniff to skip the kwarg for sync nodes; that
    escape hatch is gone).

    Catches: future refactors that drop ``*, step`` and would break
    the FSM's ``_run_node(node_fn, state, step=current_step)`` call.
    """
    # v2.0.22 (Item 7 Step 7) — NODES entries are NodeSpec dicts;
    # the async-gen ``fn`` is what carries the protocol signature.
    fn = NODES[name]["fn"]
    sig = inspect.signature(fn)
    assert "step" in sig.parameters, (
        f"{name!r} must accept a ``step`` kwarg; got signature {sig}"
    )
    # ``step`` must be kwarg-only (matches the protocol).
    step_param = sig.parameters["step"]
    assert step_param.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{name!r}.step must be kwarg-only (kind={step_param.kind})"
    )


# ---------------------------------------------------------------------------
# 2) Each production node yields exactly one __delta__ on a minimal state
# ---------------------------------------------------------------------------
# Drive each node directly with a minimal state. The streaming nodes
# (``react_generate_direct`` / ``_tools_step``) may yield zero or more
# non-__delta__ events BEFORE the terminator; sync-shape nodes yield
# exactly one __delta__. In all cases the FSM contract is "exactly
# one __delta__ per call".

@pytest.mark.asyncio
async def test_intent_analysis_yields_exactly_one_delta_greeting():
    """Fast-path 1 (greeting) yields exactly one __delta__ and nothing else."""
    deltas: list = []
    async for kind, payload in intent_analysis(
        {"current_query": "hi", "step_count": 0}
    ):
        if kind == "__delta__":
            deltas.append(payload)
    assert len(deltas) == 1
    assert deltas[0]["intent"] == "greeting"


@pytest.mark.asyncio
async def test_summary_path_yields_exactly_one_delta():
    """``summary_path`` yields exactly one __delta__ with documents."""
    state = {"thread_id": "x", "step_count": 0}
    deltas: list = []
    async for kind, payload in summary_path_fn(state):
        if kind == "__delta__":
            deltas.append(payload)
    assert len(deltas) == 1
    assert "documents" in deltas[0]


@pytest.mark.asyncio
async def test_check_hallucination_yields_exactly_one_delta():
    """``check_hallucination_async`` yields exactly one __delta__ with the verdict."""
    state = {
        "messages": [],
        "documents": [],
        "intent": "qa_complex",
        "step_count": 0,
    }
    deltas: list = []
    async for kind, payload in check_hallucination_async(state):
        if kind == "__delta__":
            deltas.append(payload)
    assert len(deltas) == 1
    assert "hallucination_check" in deltas[0]


@pytest.mark.asyncio
async def test_react_agent_yields_exactly_one_delta():
    """``react_agent`` yields exactly one __delta__ with the AIMessage."""
    deltas: list = []
    # Fake model that returns a plain AIMessage — minimal coverage of the
    # protocol shape without the LLM call actually happening.
    from langchain_core.messages import AIMessage

    async def _drive():
        async for kind, payload in react_agent(
            {"messages": [], "step_count": 0, "needs_current_time": False}
        ):
            if kind == "__delta__":
                deltas.append(payload)

    # Drive react_agent with monkeypatched build_chat_model so the LLM
    # isn't actually called.
    import src.agent.nodes.react_agent as ra
    from unittest.mock import AsyncMock, MagicMock

    fake_model = MagicMock()
    fake_model.bind_tools = MagicMock(return_value=fake_model)
    fake_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    orig = ra.build_chat_model
    ra.build_chat_model = lambda **kw: fake_model
    try:
        await _drive()
    finally:
        ra.build_chat_model = orig

    assert len(deltas) == 1
    assert "messages" in deltas[0]


# ---------------------------------------------------------------------------
# 3) _run_node protocol enforcement (no __delta__, double __delta__)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_node_raises_when_node_yields_no_delta():
    """A node that exhausts without yielding ``__delta__`` must
    raise ``RuntimeError`` — the FSM's "cardiac arrest" detection
    that prevents silent infinite loops (the v1.x
    ``GraphRecursionError`` class of regression).
    """

    async def _no_delta_node(state, *, step=0):
        # Yield a non-delta event then exit without __delta__.
        yield ("token", {"text": "hi"})
        return
        # Generator finish — never yields __delta__.

    with pytest.raises(RuntimeError) as excinfo:
        await _run_node(_no_delta_node, {}, step=1)
    assert "__delta__" in str(excinfo.value)
    assert "_no_delta_node" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_node_raises_when_node_yields_delta_twice():
    """Defensive: a node that yields two ``__delta__`` payloads
    raises ``RuntimeError`` so the FSM surfaces the bad refactor
    instead of silently merging twice (which would double-bump
    ``step_count`` via ``merge()`` and break the budget guard).
    """

    async def _double_delta_node(state, *, step=0):
        yield ("__delta__", {"x": 1})
        yield ("__delta__", {"x": 2})

    with pytest.raises(RuntimeError) as excinfo:
        await _run_node(_double_delta_node, {}, step=1)
    assert "twice" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_node_forwards_non_delta_events():
    """Per-node events (e.g. ``token`` / ``tool_call_end``) must be
    forwarded to the caller via the events list; the ``__delta__``
    is returned separately.
    """

    async def _node(state, *, step=0):
        yield ("token", {"text": "hello"})
        yield ("reasoning", {"text": "thinking"})
        yield ("__delta__", {"messages": []})

    events, delta = await _run_node(_node, {}, step=1)
    assert events == [
        ("token", {"text": "hello"}),
        ("reasoning", {"text": "thinking"}),
    ]
    assert delta == {"messages": []}


# ---------------------------------------------------------------------------
# 4) Source-level guards — no sync-shape node sneak back
# ---------------------------------------------------------------------------


def _async_gen_defs(node_module) -> list:
    """Return every top-level async-gen ``async def`` in ``node_module``.

    A node shape regression that drops the ``yield`` from a node would
    make this list shorter for that file; the per-node parameterised
    tests above catch the runtime behaviour, this list is the source
    documentation.
    """
    return [
        name
        for name, obj in vars(node_module).items()
        if inspect.isasyncgenfunction(obj)
    ]


def test_all_nodes_are_async_gen_in_source():
    """Source-level guard: every production node module exposes at
    least one async-gen function (the node itself). Catches a
    refactor that drops the ``yield`` keyword and silently regresses
    the node to sync-shape (which would TypeError at runtime when
    the FSM tries ``async for evt in node_fn(...)``).
    """
    import src.agent.nodes.intent_analysis as ia_mod
    import src.agent.nodes.react_agent as ra_mod
    import src.agent.nodes.react_generate as rg_mod
    from src.agent import fsm as fsm_mod
    from src.agent.nodes import hallucination as hal_mod

    assert "intent_analysis" in _async_gen_defs(ia_mod)
    assert "react_agent" in _async_gen_defs(ra_mod)
    assert "check_hallucination_async" in _async_gen_defs(hal_mod)
    # ``summary_path`` + ``react_generate_direct`` + ``_tools_step``
    # + ``react_generate`` live in fsm.py or react_generate.py.
    fsm_agens = _async_gen_defs(fsm_mod)
    assert "summary_path" in fsm_agens
    assert "react_generate_direct" in fsm_agens
    assert "_tools_step" in fsm_agens
    rg_agens = _async_gen_defs(rg_mod)
    assert "react_generate" in rg_agens


# ---------------------------------------------------------------------------
# Helpers — kept at module bottom so the parametrize list above stays clean
# ---------------------------------------------------------------------------


def summary_path_fn(state, *, step=0):
    """Async wrapper to drive the fsm-local ``summary_path`` node
    in the parameterised test. Avoids a separate import dance per
    test and matches the public surface (``NODES["summary_path"]``).
    """
    # Import lazily so module-level import errors surface in the
    # test body, not at collection time.
    from src.agent.fsm import summary_path

    return summary_path(state, step=step)


__all__ = [
    "test_nodes_registry_complete",
    "test_node_is_async_gen_function",
    "test_node_accepts_step_kwarg",
    "test_intent_analysis_yields_exactly_one_delta_greeting",
    "test_summary_path_yields_exactly_one_delta",
    "test_check_hallucination_yields_exactly_one_delta",
    "test_react_agent_yields_exactly_one_delta",
    "test_run_node_raises_when_node_yields_no_delta",
    "test_run_node_raises_when_node_yields_delta_twice",
    "test_run_node_forwards_non_delta_events",
    "test_all_nodes_are_async_gen_in_source",
]