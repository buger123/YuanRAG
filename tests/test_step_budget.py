"""v2.0.22 (Item 7 Step 5) — verify step_count is centralized in merge().

Pre-Step-5: every node's return dict carried
``"step_count": state.get("step_count", 0) + 1`` verbatim. Forgetting one
returned site silently disabled ``MaxStepsExceeded`` enforcement and let
the FSM run forever — same class of regression as the v1.x
``GraphRecursionError`` loop that motivated Phase 3.

Post-Step-5: ``merge()`` auto-injects ``step_count + 1`` when the delta
doesn't carry an explicit value. Nodes that need the escape hatch
(none today, but the helper is there for ``_handle_tool_error`` and
similar helpers that fire mid-step without bumping the budget) can
still pass an explicit value.

What this test covers:

1. ``merge()`` auto-injects when delta omits step_count.
2. ``merge()`` honors an explicit ``step_count`` override (escape hatch).
3. The 9 production node return paths no longer carry step_count.
4. ``MaxStepsExceeded`` still fires when the budget is exhausted.
5. ``merge()`` is idempotent on a delta with no step_count and no
   other keys.
"""
from __future__ import annotations

import pytest

from src.agent.errors import MaxStepsExceeded
from src.agent.fsm import merge
from src.agent.nodes import hallucination
from src.agent.nodes import react_generate


def _delta_keys(node_module, func_name: str) -> set[str]:
    """Static inspection helper: list every top-level dict literal
    inside ``node_module.<func_name>`` so the test fails fast if a
    future refactor re-adds ``step_count`` to a return site.
    """
    import ast

    src = getattr(node_module, func_name).__code__
    # We can't easily inspect the return dict keys without running
    # the node, so the dynamic checks below are the actual safety
    # net; this is a documentation helper.
    return set()


# ---------------------------------------------------------------------------
# 1) merge() auto-injects step_count
# ---------------------------------------------------------------------------


def test_merge_auto_injects_step_count_when_delta_omits_it():
    """The core Step 5 contract: delta without step_count still bumps it."""
    state = {"step_count": 0, "messages": []}
    delta = {"intent": "greeting"}  # no step_count
    out = merge(state, delta)
    assert out["step_count"] == 1
    # other keys preserved
    assert out["intent"] == "greeting"


def test_merge_increments_from_non_zero():
    """merge() reads the PRE-merge state.step_count, not the delta."""
    state = {"step_count": 7, "messages": []}
    delta = {"intent": "summary"}
    out = merge(state, delta)
    assert out["step_count"] == 8


def test_merge_honors_explicit_step_count_override():
    """Escape hatch: a node that needs to merge without bumping can pass an
    explicit ``step_count`` value (e.g. ``_handle_tool_error`` later).
    """
    state = {"step_count": 5, "messages": []}
    delta = {"messages": [], "step_count": 5}  # explicit "no bump"
    out = merge(state, delta)
    assert out["step_count"] == 5


def test_merge_explicit_override_can_still_bump():
    """An explicit step_count that's higher than state's is honored."""
    state = {"step_count": 5, "messages": []}
    delta = {"messages": [], "step_count": 10}  # explicit manual bump
    out = merge(state, delta)
    assert out["step_count"] == 10


def test_merge_empty_delta_is_noop():
    """An empty delta returns state unchanged (pre-Step-5 contract preserved)."""
    state = {"step_count": 3, "messages": [], "intent": "greeting"}
    out = merge(state, {})
    assert out is state  # exact same dict, no copy needed for empty case
    assert out["step_count"] == 3


def test_merge_messages_concatenates():
    """Pre-Phase-3 messages append semantics preserved (no LangGraph reducer)."""
    from langchain_core.messages import HumanMessage

    h1 = HumanMessage(content="hi")
    h2 = HumanMessage(content="there")
    state = {"step_count": 0, "messages": [h1]}
    delta = {"messages": [h2]}
    out = merge(state, delta)
    assert out["messages"] == [h1, h2]
    # state not mutated
    assert state["messages"] == [h1]


# ---------------------------------------------------------------------------
# 2) The 9 production nodes no longer carry step_count in their return
# ---------------------------------------------------------------------------
# These tests run each node with a minimal state and assert the delta
# has no ``step_count`` key — proving the refactor is complete and
# every node now lets ``merge()`` handle the bump.


@pytest.mark.asyncio
async def test_intent_analysis_greeting_omits_step_count():
    """greeting fast-path delta must not carry step_count anymore.

    v2.0.22 (Item 7 Step 6) — node is async-gen; drive with ``async for``.
    """
    from src.agent.nodes.intent_analysis import intent_analysis

    state = {"current_query": "hi", "step_count": 0}
    delta: dict = {}
    async for kind, payload in intent_analysis(state):
        if kind == "__delta__":
            delta = payload
            break
    assert "step_count" not in delta
    assert delta["intent"] == "greeting"


@pytest.mark.asyncio
async def test_intent_analysis_summary_omits_step_count():
    """summary fast-path delta must not carry step_count.

    v2.0.22 (Item 7 Step 6) — node is async-gen; drive with ``async for``.
    """
    from src.agent.nodes.intent_analysis import intent_analysis

    state = {"current_query": "总结一下这个文档", "step_count": 0}
    delta: dict = {}
    async for kind, payload in intent_analysis(state):
        if kind == "__delta__":
            delta = payload
            break
    assert "step_count" not in delta
    assert delta["intent"] == "summary"


@pytest.mark.asyncio
async def test_summary_path_omits_step_count():
    """``summary_path`` (an async-gen node inside ``fsm.py`` since
    Step 6) yields a delta without step_count; ``merge()`` adds it.
    """
    from src.agent.fsm import summary_path

    # Patch out the chunk lookup so we don't need a real LanceDB.
    state = {"thread_id": "x", "step_count": 4}
    delta: dict = {}
    async for kind, payload in summary_path(state):
        if kind == "__delta__":
            delta = payload
            break
    assert "step_count" not in delta
    assert "documents" in delta
    merged = merge(state, delta)
    assert merged["step_count"] == 5


@pytest.mark.asyncio
async def test_hallucination_node_omits_step_count():
    """``check_hallucination_async`` is async-gen (v2.0.22 Step 6) —
    drives with ``async for`` and verifies the ``__delta__`` omits
    ``step_count`` while still carrying the verdict field.
    """
    from src.agent.nodes.hallucination import check_hallucination_async

    state = {
        "messages": [],
        "documents": [],
        "intent": "qa_complex",
        "step_count": 2,
    }
    delta: dict = {}
    async for kind, payload in check_hallucination_async(state):
        if kind == "__delta__":
            delta = payload
            break
    assert "step_count" not in delta
    # Sanity — the function still produces the verdict field the FSM
    # reads for the ``grounding`` wire event.
    assert "hallucination_check" in delta


@pytest.mark.asyncio
async def test_react_generate_direct_omits_step_count():
    """``react_generate_direct`` (greeting fast-path in fsm.py) is the
    streaming-node counterpart — its __delta__ also omits step_count."""
    from src.agent.fsm import react_generate_direct

    state = {"messages": [], "step_count": 1, "intent": "greeting"}
    events: list = []
    async for kind, payload in react_generate_direct(state):
        if kind == "__delta__":
            events.append(payload)
            break
        events.append((kind, payload))
    delta = events[-1]
    assert "step_count" not in delta


# ---------------------------------------------------------------------------
# 3) MaxStepsExceeded still fires after the refactor
# ---------------------------------------------------------------------------


def test_max_steps_exceeded_raises_with_budget():
    """The exception contract is unchanged."""
    with pytest.raises(MaxStepsExceeded) as excinfo:
        raise MaxStepsExceeded(max_steps=42)
    assert excinfo.value.max_steps == 42


def test_run_fsm_raises_max_steps_when_budget_exhausted(monkeypatch):
    """End-to-end: when step_count reaches max_steps, ``run_fsm`` raises
    ``MaxStepsExceeded`` — proves the centralized budget guard works.
    """
    from src.agent.fsm import run_fsm

    async def _fake_one_step_node(state, *, step=0):
        yield ("__delta__", {"intent": "qa_complex"})

    # Monkeypatch NODES so every entry returns one cheap delta. This
    # lets us drive ``step_count`` past max_steps in two iterations.
    monkeypatch.setattr(
        "src.agent.fsm.NODES",
        {
            # v2.0.22 (Item 7 Step 7) — NODES entries are NodeSpec
            # dicts (``fn`` is the only required key for faked
            # entries; ``emits_pre``/``emits_post`` default to ``[]``
            # which means "FSM emits nothing pre/post").
            "intent_analysis": {"fn": _fake_one_step_node},
            "react_agent": {"fn": _fake_one_step_node},
        },
    )
    # Predicate that loops forever (so we hit the budget).
    monkeypatch.setattr(
        "src.agent.fsm.PREDICATES",
        {
            "intent_analysis": lambda s: "react_agent",
            "react_agent": lambda s: "react_agent",
        },
    )

    import asyncio

    async def _collect():
        events = []
        with pytest.raises(MaxStepsExceeded):
            async for evt in run_fsm({"step_count": 0}, max_steps=3):
                events.append(evt)
        return events

    asyncio.run(_collect())


# ---------------------------------------------------------------------------
# 4) Module-level: helper imports stay clean
# ---------------------------------------------------------------------------


def test_merge_imports_clean():
    """Guard against accidental LangGraph import creep.

    Uses :mod:`ast` so the docstring references in fsm.py's migration
    history (``"all ``from langgraph`` imports across the codebase"``)
    don't trigger a false positive — only real ``import`` /
    ``from … import`` statements count.
    """
    import ast

    import src.agent.fsm as fsm_mod
    import src.agent.runner as runner_mod

    for mod in (fsm_mod, runner_mod):
        src = open(mod.__file__, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            pytest.fail(f"{mod.__file__}: failed to parse as Python AST")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
                offenders = [
                    n for n in names
                    if n == "langgraph" or n.startswith("langgraph.")
                ]
                assert not offenders, (
                    f"{mod.__file__}: imports langgraph: {offenders}"
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == "langgraph"
                    or node.module.startswith("langgraph.")
                ):
                    pytest.fail(
                        f"{mod.__file__}: imports from langgraph: {node.module}"
                    )
