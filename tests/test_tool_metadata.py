"""v2.0.22 (Item 7 Step 8) — tool metadata drives per-tool wire events.

Pre-Step-8, ``src/agent/nodes/react_agent.py`` had a hardcoded
``if tc_name == "web_search": yield ("web_search", {...})``
branch — the P0-B3 audit finding: "FSM 本应 tool-name-agnostic,
这一处字面量是 leak; 重命名 / 加新搜索类工具必须改 fsm.py"
(this was moved to ``react_agent.py`` in Step 7, still a leak).

Post-Step-8, :data:`src.agent.tools.TOOL_METADATA` declares each
tool's extra wire events:

    TOOL_METADATA = {
        "web_search":     {"emits": ["web_search"]},
        "retrieve_docs":  {"emits": []},
        "get_current_time": {"emits": []},
    }

``react_agent`` walks ``response.tool_calls``, yields the default
``tool_call_start``, then looks up ``TOOL_METADATA[tc_name]["emits"]``
to decide which extra events to yield. Adding a new search-class
tool (e.g. ``web_search_tier2``) is now a one-line metadata entry
— no node edits, no FSM edits.

These tests pin the contract:

1. ``TOOL_METADATA`` covers every tool in :data:`ALL_TOOLS`.
2. ``web_search`` metadata emits a ``web_search`` event (v1.1.8
   wire contract — frontend uses ``attempted=True``).
3. ``retrieve_docs`` / ``get_current_time`` emit no extra events.
4. ``react_agent`` yields ``web_search`` event when LLM calls
   ``web_search`` (full stack test with mocked model).
5. ``react_agent`` does NOT yield ``web_search`` event for other
   tools.
6. The hardcoded ``tc_name == "web_search"`` string comparison
   is gone from ``react_agent.py`` (source-level guard).
7. Adding a new tool only requires ``TOOL_METADATA`` entry — no
   node edits (forward-compatibility test using a fake
   ``web_search_tier2`` entry).
"""
from __future__ import annotations

import ast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# 1) TOOL_METADATA registry shape
# ---------------------------------------------------------------------------


def test_tool_metadata_covers_all_tools():
    """Every tool in ALL_TOOLS must have a TOOL_METADATA entry.

    Without a metadata entry the FSM would silently skip the tool's
    extra events (defensive ``.get(tc_name) or {}`` in react_agent)
    — which would be a silent regression if someone added a tool
    but forgot to register its metadata.
    """
    from src.agent.tools import ALL_TOOLS, TOOL_METADATA

    for tool in ALL_TOOLS:
        assert tool.name in TOOL_METADATA, (
            f"Tool {tool.name!r} is in ALL_TOOLS but missing from "
            f"TOOL_METADATA. Add an entry — even ``emits=[]`` is "
            f"fine for tools with no extra events (the entry is "
            f"required so future schema additions don't silently "
            f"miss this tool)."
        )


def test_tool_metadata_values_are_dicts_with_emits_list():
    """Each TOOL_METADATA entry must be a dict with an ``emits`` key
    that is a list of strings (wire event kinds). Defensive against
    a typo like ``emits="web_search"`` (string) which would crash
    the ``for evt_kind in meta["emits"]:`` loop with TypeError.
    """
    from src.agent.tools import TOOL_METADATA

    for tool_name, meta in TOOL_METADATA.items():
        assert isinstance(meta, dict), (
            f"TOOL_METADATA[{tool_name!r}] must be a dict; got "
            f"{type(meta).__name__}"
        )
        assert "emits" in meta, (
            f"TOOL_METADATA[{tool_name!r}] missing required "
            f"``emits`` key"
        )
        emits = meta["emits"]
        assert isinstance(emits, list), (
            f"TOOL_METADATA[{tool_name!r}]['emits'] must be a "
            f"list; got {type(emits).__name__}"
        )
        for kind in emits:
            assert isinstance(kind, str), (
                f"TOOL_METADATA[{tool_name!r}]['emits'] entry "
                f"must be str (wire event kind); got "
                f"{type(kind).__name__}: {kind!r}"
            )


def test_web_search_metadata_emits_web_search_event():
    """``web_search`` tool's metadata MUST include
    ``emits=['web_search']`` — this is the v1.1.8 wire contract
    that makes the frontend render the search pill.
    """
    from src.agent.tools import TOOL_METADATA

    assert TOOL_METADATA["web_search"]["emits"] == ["web_search"]


def test_other_tools_have_empty_emits():
    """``retrieve_docs`` and ``get_current_time`` don't have a
    per-tool wire event — their progress is conveyed solely via
    ``tool_call_start`` / ``tool_call_end``.
    """
    from src.agent.tools import TOOL_METADATA

    assert TOOL_METADATA["retrieve_docs"]["emits"] == []
    assert TOOL_METADATA["get_current_time"]["emits"] == []


# ---------------------------------------------------------------------------
# 2) Source-level guard: hardcoded `tc_name == "web_search"` is gone
# ---------------------------------------------------------------------------


def test_react_agent_no_hardcoded_web_search_string_in_source():
    """Source-level guard: ``react_agent.py`` body must not contain
    the pre-Step-8 hardcoded ``if tc_name == "web_search"``
    pattern. The string comparison must live ONLY in comments
    (referencing history) or in the TOOL_METADATA dict.

    Uses AST walk to find ``Compare(left=Name("tc_name"))`` with
    a string comparator, so a future comment referencing the old
    behavior doesn't trigger a false positive.
    """
    import src.agent.nodes.react_agent as ra_mod

    src = open(ra_mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            left = node.left
            if isinstance(left, ast.Name) and left.id == "tc_name":
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(
                        comparator.value, str
                    ):
                        pytest.fail(
                            f"react_agent.py still has "
                            f"`if tc_name == {comparator.value!r}` — "
                            f"this is the P0-B3 hardcoded tool name "
                            f"comparison that Step 8 deleted in "
                            f"favor of TOOL_METADATA lookup."
                        )


# ---------------------------------------------------------------------------
# 3) End-to-end: react_agent yields web_search event when LLM calls web_search
# ---------------------------------------------------------------------------


async def _drive_react_agent_with_tool_call(
    monkeypatch,
    tool_name: str,
    tool_call_id: str = "tc1",
    tool_args: dict | None = None,
):
    """Drive ``react_agent`` with a fake model that returns one
    AIMessage with the given tool_call. Returns the list of
    ``(kind, payload)`` events yielded by the node so the caller
    can assert on the wire sequence.

    Pre-Step-7 used the same helper; kept in-test (rather than
    conftest) because it's only used here and test_v2_bugfixes
    already has its own copy.
    """
    import src.agent.nodes.react_agent as ra

    response = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": tool_name,
                "args": tool_args or {},
            }
        ],
    )
    fake_model = MagicMock()
    fake_model.bind_tools = MagicMock(return_value=fake_model)
    fake_model.ainvoke = AsyncMock(return_value=response)

    orig = ra.build_chat_model
    ra.build_chat_model = lambda **kw: fake_model
    try:
        events: list = []
        async for kind, payload in ra.react_agent(
            {"messages": [], "step_count": 0, "needs_current_time": False},
            step=1,
        ):
            events.append((kind, payload))
    finally:
        ra.build_chat_model = orig
    return events


@pytest.mark.asyncio
async def test_react_agent_yields_web_search_event_for_web_search_tool():
    """Full-stack: when LLM calls ``web_search``, ``react_agent``
    yields ``("tool_call_start", ...)`` AND
    ``("web_search", {"attempted": True})``.

    Pre-Step-8 the web_search event was emitted by a hardcoded
    ``if tc_name == "web_search"`` branch. Post-Step-8 the event
    comes from ``TOOL_METADATA["web_search"]["emits"]`` — same
    wire output, different dispatch path.
    """
    events = await _drive_react_agent_with_tool_call(
        monkeypatch=None,  # unused; we patch inline in the helper
        tool_name="web_search",
        tool_call_id="tc-ws",
        tool_args={"query": "双鱼座 特点"},
    )

    kinds = [k for k, _ in events]
    # The default tool_call_start + the metadata-driven web_search
    # event both fire, in order.
    assert "tool_call_start" in kinds
    assert "web_search" in kinds
    assert kinds.index("tool_call_start") < kinds.index("web_search"), (
        f"tool_call_start must precede web_search; got order: {kinds}"
    )

    # The web_search payload matches v1.1.8 wire contract.
    web_search_payload = next(p for k, p in events if k == "web_search")
    assert web_search_payload.get("attempted") is True

    # The tool_call_start carries the FSM-supplied step number so
    # the frontend pairs start with end.
    tc_start = next(p for k, p in events if k == "tool_call_start")
    assert tc_start["tool_name"] == "web_search"
    assert tc_start["tool_call_id"] == "tc-ws"
    assert tc_start["step"] == 1


@pytest.mark.asyncio
async def test_react_agent_does_not_yield_web_search_for_retrieve_docs():
    """When LLM calls ``retrieve_docs``, ``react_agent`` MUST NOT
    yield the ``web_search`` event — only ``tool_call_start``.

    Pre-Step-8 the hardcoded check was ``tc_name == "web_search"``
    which correctly skipped other tools, but at the cost of
    stringly-typed dispatch. Post-Step-8 the same correctness
    comes from ``TOOL_METADATA["retrieve_docs"]["emits"] == []``.
    """
    events = await _drive_react_agent_with_tool_call(
        monkeypatch=None,
        tool_name="retrieve_docs",
        tool_call_id="tc-rd",
        tool_args={"query": "腾讯音乐 营收"},
    )

    kinds = [k for k, _ in events]
    assert "tool_call_start" in kinds
    assert "web_search" not in kinds, (
        f"retrieve_docs must not trigger web_search event; "
        f"got kinds: {kinds}"
    )


@pytest.mark.asyncio
async def test_react_agent_does_not_yield_web_search_for_get_current_time():
    """``get_current_time`` is the third tool in ALL_TOOLS;
    verify it doesn't trigger the web_search event either."""
    events = await _drive_react_agent_with_tool_call(
        monkeypatch=None,
        tool_name="get_current_time",
        tool_call_id="tc-time",
        tool_args={},
    )

    kinds = [k for k, _ in events]
    assert "tool_call_start" in kinds
    assert "web_search" not in kinds


# ---------------------------------------------------------------------------
# 4) Forward-compat: adding a new search-class tool is metadata-only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_search_tool_emits_web_search_via_metadata(monkeypatch):
    """Forward-compatibility: adding a hypothetical
    ``web_search_tier2`` tool with the same metadata as
    ``web_search`` triggers the web_search event WITHOUT any
    node code change.

    This is the core Step 8 win: pre-Step-8 would require editing
    ``react_agent.py`` to add a new ``if tc_name == "web_search_tier2":``
    branch. Post-Step-8 only ``TOOL_METADATA`` changes.

    We monkeypatch ``TOOL_METADATA`` with the hypothetical new
    entry to prove the dispatch is metadata-driven (not name-matched).
    """
    from src.agent import tools as tools_pkg

    # Inject the hypothetical tool into TOOL_METADATA.
    monkeypatch.setitem(
        tools_pkg.TOOL_METADATA,
        "web_search_tier2",
        {"emits": ["web_search"]},
    )

    events = await _drive_react_agent_with_tool_call(
        monkeypatch=None,
        tool_name="web_search_tier2",
        tool_call_id="tc-tier2",
        tool_args={"query": "test"},
    )

    kinds = [k for k, _ in events]
    assert "tool_call_start" in kinds
    assert "web_search" in kinds, (
        f"web_search_tier2 with metadata emits=['web_search'] "
        f"must trigger web_search event; got kinds: {kinds}"
    )


# ---------------------------------------------------------------------------
# 5) Unregistered tool name: defensively skip (no crash, no event)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregistered_tool_name_does_not_crash():
    """Defensive: if the LLM somehow calls a tool that isn't in
    ``TOOL_METADATA``, ``react_agent`` must NOT crash — it should
    silently skip the metadata-driven events and emit only the
    default ``tool_call_start``.

    Without this guard, ``TOOL_METADATA[tc_name]`` would raise
    KeyError and break the FSM mid-step. The actual code uses
    ``TOOL_METADATA.get(tc_name) or {}`` which returns empty
    when the tool isn't registered.
    """
    events = await _drive_react_agent_with_tool_call(
        monkeypatch=None,
        tool_name="nonexistent_tool",
        tool_call_id="tc-x",
        tool_args={},
    )

    kinds = [k for k, _ in events]
    assert "__delta__" in kinds, (
        "FSM must still receive __delta__ even for unregistered tools"
    )
    # Only tool_call_start + __delta__ — no extras, no crash.
    assert "tool_call_start" in kinds


# ---------------------------------------------------------------------------
# 6) __all__ exposure
# ---------------------------------------------------------------------------


def test_tool_metadata_is_exported_from_tools_package():
    """``TOOL_METADATA`` and ``ToolMetadata`` are exported from
    ``src.agent.tools`` so callers can monkeypatch them (the
    forward-compat test above does this).
    """
    from src.agent import tools as tools_pkg

    assert "TOOL_METADATA" in tools_pkg.__all__
    assert "ToolMetadata" in tools_pkg.__all__


__all__ = [
    "test_tool_metadata_covers_all_tools",
    "test_tool_metadata_values_are_dicts_with_emits_list",
    "test_web_search_metadata_emits_web_search_event",
    "test_other_tools_have_empty_emits",
    "test_react_agent_no_hardcoded_web_search_string_in_source",
    "test_react_agent_yields_web_search_event_for_web_search_tool",
    "test_react_agent_does_not_yield_web_search_for_retrieve_docs",
    "test_react_agent_does_not_yield_web_search_for_get_current_time",
    "test_new_search_tool_emits_web_search_via_metadata",
    "test_unregistered_tool_name_does_not_crash",
    "test_tool_metadata_is_exported_from_tools_package",
]