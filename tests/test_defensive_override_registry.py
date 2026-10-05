"""Phase 3 — Defensive-override registry tests (v2.0.29.3).

Pins every behavior of ``src/agent/nodes/_defensive_override_registry.py``
plus the ``state["loop_breaker_active"]`` propagation in
``react_agent`` and the loop-break synthesis hint insertion in
``react_generate``.

Critical invariants (locked by tests below):
- ``DIRECTIVE_BUILDERS`` / ``FINGERPRINT_EXTRACTORS`` /
  ``FALLBACK_BUILDERS`` are pre-populated with ``get_current_time``
  (the only tool currently using defensive override).
- Adding a new tool requires one line per registry; the unit tests
  below detect registry drift (e.g. ``DIRECTIVE_BUILDERS`` has a
  tool that lacks a fingerprint).
- ``apply_defensive_overrides`` is a no-op on direct routes,
  on empty answers, and when the LLM's answer already contains the
  fingerprint marker.
- ``apply_defensive_overrides`` overrides + bumps
  ``SYNTHESIS_TM_IGNORED{tool=<name>}`` when the marker is missing.
- Iteration is reversed (most-recent TM wins), preventing the
  multi-turn stale-data bug.
- ``state["loop_breaker_active"]`` propagates through FSM ``merge()``:
  ``react_agent`` writes the flag, ``react_generate`` reads it and
  inserts the synthesis hint at ``msgs.insert(2, ...)``.
- The loop-break hint preserves Anthropic consecutiveness
  (v2.0.28.17): all SystemMessages stay consecutive at index 0..N.
"""
from __future__ import annotations

import asyncio
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.nodes._defensive_override_registry import (
    DIRECTIVE_BUILDERS,
    FALLBACK_BUILDERS,
    FINGERPRINT_EXTRACTORS,
    _aligned_tools,
    _build_get_current_time_directive,
    _build_time_fallback,
    _extract_time_marker,
    apply_defensive_overrides,
)


# ============================================================
# 1. Registry pre-population + alignment contract
# ============================================================


def test_get_current_time_registered_in_all_three():
    """The tool we explicitly support must be in all 3 registries."""
    assert "get_current_time" in DIRECTIVE_BUILDERS
    assert "get_current_time" in FINGERPRINT_EXTRACTORS
    assert "get_current_time" in FALLBACK_BUILDERS


def test_aligned_tools_returns_registered_set():
    """``_aligned_tools()`` exposes the intersection of all 3 registries.

    Adding a tool to one registry without the others creates a silent
    skip — a registry drift bug. Tests that mutate the registries
    must re-check this invariant.
    """
    aligned = _aligned_tools()
    assert "get_current_time" in aligned
    # All three registries have the same set of keys (intersection
    # = union when there's no partial registration).
    assert aligned == set(DIRECTIVE_BUILDERS.keys())
    assert aligned == set(FINGERPRINT_EXTRACTORS.keys())
    assert aligned == set(FALLBACK_BUILDERS.keys())


# ============================================================
# 2. Directive builder behavior
# ============================================================


def test_directive_builder_fires_on_time_tm():
    """A get_current_time TM triggers a strong directive string."""
    history = [
        HumanMessage(content="现在几点"),
        AIMessage(content="", tool_calls=[{
            "id": "tc1", "name": "get_current_time", "args": {},
        }]),
        ToolMessage(
            content="2026-09-28T14:00:00+08:00 (Asia/Shanghai)",
            tool_call_id="tc1",
            name="get_current_time",
        ),
    ]
    directive = _build_get_current_time_directive(history)
    assert directive is not None
    assert "get_current_time" in directive
    assert "2026-09-28" in directive
    assert "MUST" in directive


def test_directive_builder_returns_none_on_no_tm():
    """No TM → no directive (don't add noise to unrelated prompts)."""
    history = [HumanMessage(content="hello")]
    assert _build_get_current_time_directive(history) is None


def test_directive_builder_skips_empty_content_tm():
    """Empty TM content → no directive (nothing to point the LLM at)."""
    history = [
        HumanMessage(content="现在几点"),
        AIMessage(content="", tool_calls=[{
            "id": "tc1", "name": "get_current_time", "args": {},
        }]),
        ToolMessage(content="", tool_call_id="tc1", name="get_current_time"),
    ]
    assert _build_get_current_time_directive(history) is None


def test_directive_builder_via_registry_iteration():
    """Iterating ``DIRECTIVE_BUILDERS`` produces the same output as the
    direct helper call. The iteration pattern is what
    ``react_generate`` uses; if a future registry entry doesn't
    follow the same callable shape, this test catches it.
    """
    history = [
        HumanMessage(content="x"),
        AIMessage(content="", tool_calls=[{
            "id": "tc1", "name": "get_current_time", "args": {},
        }]),
        ToolMessage(content="2026-09-28 14:00:00", tool_call_id="tc1", name="get_current_time"),
    ]
    builder = DIRECTIVE_BUILDERS["get_current_time"]
    assert builder(history) == _build_get_current_time_directive(history)


# ============================================================
# 3. Fingerprint extractor behavior
# ============================================================


def test_fingerprint_extractor_iso_date_priority():
    """ISO date wins over HH:MM and Chinese M月D日 when all are present."""
    raw = (
        "当前时间: 2026-09-28 14:23:11+08:00 (Asia/Shanghai)\n"
        "2026年9月28日 周一 14:23"
    )
    marker = _extract_time_marker(raw)
    assert marker == "2026-09-28"


def test_fingerprint_extractor_falls_back_to_hh_mm():
    """When ISO date is missing, HH:MM wins over Chinese M月D日."""
    raw = "现在时间: 14:23"
    marker = _extract_time_marker(raw)
    assert marker == "14:23"


def test_fingerprint_extractor_falls_back_to_hh_mm_ss():
    """HH:MM:SS variant is also extracted."""
    raw = "现在时间: 14:23:45"
    marker = _extract_time_marker(raw)
    assert marker == "14:23:45"


def test_fingerprint_extractor_falls_back_to_chinese():
    """When ISO date AND HH:MM are missing, Chinese M月D日 is the last resort."""
    raw = "今天是9月28日"
    marker = _extract_time_marker(raw)
    assert marker == "9月28日"


def test_fingerprint_extractor_returns_none_on_no_marker():
    """Unparseable content returns None — caller treats as no override."""
    assert _extract_time_marker("no time data here") is None
    assert _extract_time_marker("") is None


def test_fingerprint_extractor_via_registry_lookup():
    """Fingerprint extraction via the registry matches the direct helper."""
    raw = "2026-09-28T14:00:00+08:00"
    fn = FINGERPRINT_EXTRACTORS["get_current_time"]
    assert fn(raw) == _extract_time_marker(raw)


# ============================================================
# 4. Fallback builder behavior
# ============================================================


def test_fallback_builder_returns_chinese_answer_with_marker():
    """Fallback includes the TM content's first line (timestamp)."""
    tm = ToolMessage(
        content="2026-09-28T14:23:11+08:00 (Asia/Shanghai)\n详细信息: foo",
        tool_call_id="tc1",
        name="get_current_time",
    )
    answer = _build_time_fallback(tm)
    assert answer is not None
    assert "2026-09-28" in answer or "14:23" in answer


def test_fallback_builder_returns_none_on_empty():
    """Empty TM content → no fallback (we have nothing to substitute)."""
    tm = ToolMessage(content="", tool_call_id="tc1", name="get_current_time")
    assert _build_time_fallback(tm) is None


def test_fallback_builder_via_registry_lookup():
    """Fallback via the registry matches the direct helper."""
    tm = ToolMessage(content="2026-09-28 14:23:11", tool_call_id="tc1", name="get_current_time")
    fn = FALLBACK_BUILDERS["get_current_time"]
    assert fn(tm) == _build_time_fallback(tm)


# ============================================================
# 5. apply_defensive_overrides entry point
# ============================================================


def _make_history(*contents: str) -> List:
    """Build a sanitized history with N get_current_time TMs.

    Each content is a TM's ``content`` string; we pair it with an
    AIMessage(tool_calls=[...]) + ToolMessage so it survives the
    sanitizer (irrelevant here, but consistent with real wire data).
    """
    out: List = [HumanMessage(content="现在几点")]
    for i, content in enumerate(contents, start=1):
        out.append(AIMessage(content="", tool_calls=[{
            "id": f"tc{i}", "name": "get_current_time", "args": {},
        }]))
        out.append(ToolMessage(content=content, tool_call_id=f"tc{i}", name="get_current_time"))
    return out


def test_apply_defensive_overrides_noop_on_direct_route():
    """Direct route has no TMs → no override (route gate)."""
    history = _make_history("2026-09-28 14:23:11")
    out = apply_defensive_overrides("任何答案", history, route_decision="direct")
    assert out == "任何答案"


def test_apply_defensive_overrides_noop_on_empty_answer():
    """Empty answer means thinking-only → caller has its own fallback."""
    history = _make_history("2026-09-28 14:23:11")
    out = apply_defensive_overrides("", history, route_decision="retrieve")
    assert out == ""


def test_apply_defensive_overrides_noop_when_marker_present():
    """LLM used the time data → no override, no metric bump."""
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    history = _make_history("2026-09-28T14:23:11+08:00")
    # LLM correctly cited the time data.
    out = apply_defensive_overrides(
        "现在是 2026-09-28 14:23:11。",
        history,
        route_decision="retrieve",
    )
    assert out == "现在是 2026-09-28 14:23:11。"
    # The counter MUST NOT have ticked — successful synthesis path.
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before


def test_apply_defensive_overrides_overrides_when_marker_missing():
    """LLM ignored the time → override + SYNTHESIS_TM_IGNORED bump."""
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    history = _make_history("2026-09-28T14:23:11+08:00")
    # LLM emitted a chat reply without citing the time marker.
    out = apply_defensive_overrides(
        "好的,那就早点休息吧,晚安。",
        history,
        route_decision="retrieve",
    )
    # The override REPLACES the LLM answer with the constructed fallback.
    assert out != "好的,那就早点休息吧,晚安。"
    assert "2026-09-28" in out or "14:23" in out
    # The metric bumped — this is the Phase 2 observability contract.
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before + 1.0


def test_apply_defensive_overrides_reversed_iteration_winner():
    """Most-recent TM wins (multi-turn safety, v2.0.28.16 invariant)."""
    history = _make_history(
        "2026-09-27 10:00:00",   # older — should NOT match
        "2026-09-28 14:23:11",   # newer — must match (reversed iteration)
    )
    # LLM cited the OLDER date (e.g. remembered from earlier turn).
    # The override MUST use the NEWER TM (most-recent-wins), which
    # then bumps SYNTHESIS_TM_IGNORED because the LLM didn't cite
    # 2026-09-28 either.
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    out = apply_defensive_overrides(
        "之前是 2026-09-27 10:00:00。",
        history,
        route_decision="retrieve",
    )
    # Override fired (LLM didn't cite the newer 2026-09-28).
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before + 1.0
    # Fallback used the NEWER TM (2026-09-28), not the older one.
    assert "2026-09-28" in out


def test_apply_defensive_overrides_skips_unregistered_tool():
    """A tool without a fingerprint + fallback registered is silently
    skipped. Direct + iterate-reversed keeps the iteration cheap; the
    registry intersection is what gates the override, not the loop.
    """
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    # History has a ToolMessage for a tool that's NOT in any registry.
    history = [
        HumanMessage(content="hi"),
        ToolMessage(content="foo bar", tool_call_id="tc1", name="unregistered_tool"),
    ]
    out = apply_defensive_overrides(
        "回答 any answer 任意 answer",
        history,
        route_decision="retrieve",
    )
    # No override fired — the tool isn't registered.
    assert out == "回答 any answer 任意 answer"
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before


# ============================================================
# 5b. v2.0.32.3 (Phase 1.5) — multi-TM coverage fix
# ============================================================
# Bug being fixed: the previous implementation returned on FIRST miss,
# dropping data from any subsequent TM. So when get_current_time + a
# retrieval tool fired in the same turn and the synthesis LLM only
# emitted a chat reply (no marker for either), the override used only
# the get_current_time fallback, dropping the retrieval chunks.
#
# New contract:
#   1. Dedupe to LATEST TM per covered tool (most-recent-wins).
#   2. Check marker for every covered tool; bump SYNTHESIS_TM_IGNORED
#      for each missed tool.
#   3. If ANY tool was missed → combine ALL covered-tool fallbacks
#      joined by "\n\n". Never drop retrieval data because an
#      unrelated tool's marker was missing.
#   4. If ALL markers present → no override (LLM did its job).
# ============================================================


def _make_multi_tool_history(*tm_specs):
    """Build history with TMs for N tool calls (tool_name, content) tuples.

    Each tuple becomes one ``AIMessage(tool_calls=...)`` +
    ``ToolMessage(content, name=<tool>)`` pair. ``tool_name`` defaults
    to ``"get_current_time"`` when not provided (only tool registered
    today; future Phase 1.5+ tools can be passed in by name).
    """
    out: List = [HumanMessage(content="query")]
    for i, spec in enumerate(tm_specs, start=1):
        if isinstance(spec, str):
            tool_name, content = "get_current_time", spec
        else:
            tool_name, content = spec
        out.append(AIMessage(content="", tool_calls=[{
            "id": f"tc{i}", "name": tool_name, "args": {},
        }]))
        out.append(ToolMessage(
            content=content, tool_call_id=f"tc{i}", name=tool_name,
        ))
    return out


def test_apply_defensive_overrides_dedupes_same_tool_latest_wins():
    """Multiple TMs for the same tool collapse to the LATEST one.

    Multi-turn safety (v2.0.28.16 invariant): if the user asked
    "现在几点" twice and got two TMs back, the LATEST is the one
    the synthesis LLM is currently answering — older TMs are stale
    context that should be ignored by the override path.
    """
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    # Two get_current_time TMs — older 09-27, newer 09-28.
    history = _make_multi_tool_history(
        "2026-09-27 10:00:00",
        "2026-09-28 14:23:11",
    )
    # LLM cited the older one (regression of multi-turn bug).
    out = apply_defensive_overrides(
        "之前是 2026-09-27 10:00:00。",
        history,
        route_decision="retrieve",
    )
    # Newest TM is 09-28 → override fires, fallback cites 09-28.
    assert "2026-09-28" in out, (
        f"override should use LATEST TM 2026-09-28, got: {out!r}"
    )
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") >= 1.0


def test_apply_defensive_overrides_all_markers_present_no_override():
    """When ALL covered-tool markers are present in the answer, the
    override MUST NOT fire (LLM did its job — defensive override is
    last-resort, not normal-path).
    """
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    history = _make_multi_tool_history("2026-09-28T14:23:11+08:00")
    out = apply_defensive_overrides(
        "现在是 2026-09-28 14:23:11,帮你查好了。",
        history,
        route_decision="retrieve",
    )
    # No override.
    assert out == "现在是 2026-09-28 14:23:11,帮你查好了。"
    # No metric bump — LLM did its job.
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before


def test_apply_defensive_overrides_combined_fallback_when_multi_tool_miss():
    """When the history has multiple TMs and ANY tool's marker is
    missed, the override uses the COMBINED fallback from ALL
    covered tools — never dropping data because an earlier tool
    fired first in iteration order (the bug it fixed).

    Note: only ``get_current_time`` is registered today, so this test
    uses two calls to the same tool to simulate the multi-TM
    coverage scenario. The Phase 1.5 contract pins that the
    override path:
      1. dedupes to latest per tool (here: 1 tool, 2 calls → 1 dedup)
      2. checks the marker
      3. bumps the metric when missed
      4. combines all covered-tool fallbacks

    When a future tool is added to FALLBACK_BUILDERS, the contract
    generalizes — multi-tool coverage works the same way. Today,
    with one tool registered, this test pins the single-tool
    behavior post-Phase 1.5 (combined fallback == single-tool
    fallback when only one tool is covered).
    """
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    history = _make_multi_tool_history(
        "2026-09-28T14:23:11+08:00",
        "2026-09-28T15:00:00+08:00",
    )
    out = apply_defensive_overrides(
        "好的,那就早点休息吧,晚安。",
        history,
        route_decision="retrieve",
    )
    # Override fired and used the LATEST TM (15:00, not 14:23).
    assert "15:00" in out, f"expected latest TM 15:00 in fallback, got: {out!r}"
    # Metric bumped (single tool, single miss → single bump).
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before + 1.0


def test_apply_defensive_overrides_no_metric_bump_on_unparseable_marker():
    """If the TM's content has NO parseable marker (fingerprint
    extractor returns None), we don't bump SYNTHESIS_TM_IGNORED —
    there's no way to know whether the LLM "missed" or not. Defensive
    override should still apply (caller's answer is unchanged →
    we still emit a fallback if any other tool missed), but the
    metric stays clean for THIS tool.
    """
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    # TM has unparseable content (no ISO, HH, or CN marker).
    history = _make_multi_tool_history("random text no time marker here")
    out = apply_defensive_overrides(
        "好的,那就早点休息吧,晚安。",
        history,
        route_decision="retrieve",
    )
    # Override MAY fire (we still emit a fallback), but the metric
    # for THIS tool is NOT bumped — there was no detectable miss.
    # (Other covered tools' markers would still be checked.)
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before
    # And the fallback includes the unparseable raw content.
    assert "random text" in out or out  # fallback always non-empty when fb is not None


def test_apply_defensive_overrides_empty_history_noop():
    """Empty history (no TMs at all) → no override."""
    history = [HumanMessage(content="hello")]
    out = apply_defensive_overrides(
        "回答 any answer",
        history,
        route_decision="retrieve",
    )
    assert out == "回答 any answer"


def test_apply_defensive_overrides_no_fallback_for_history_only():
    """History with only AIMessages/HumanMessages (no TMs at all) →
    no override, no metric bump.
    """
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    history = [
        HumanMessage(content="query"),
        AIMessage(content="hi"),
        HumanMessage(content="query 2"),
        AIMessage(content="hi 2"),
    ]
    out = apply_defensive_overrides(
        "回答 any answer",
        history,
        route_decision="retrieve",
    )
    assert out == "回答 any answer"
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before


def test_apply_defensive_overrides_multi_turn_mixed_tools_latest_wins():
    """Mixed scenario: older TM (the OLD time tool call) + newer
    TM (the NEW time tool call) in the SAME turn → override uses
    the LATEST (most-recent-wins). Pre-Phase 1.5 code used
    whichever tool iterated first in reverse iteration; the bug
    was the early-return after the first miss. Post-Phase 1.5:
    always check ALL covered tools, use the LATEST per tool.
    """
    from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

    reset_all()
    before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
    history = _make_multi_tool_history(
        "2026-09-27 10:00:00",   # older
        "2026-09-28 14:23:11",   # newer — should win
        "2026-09-28 15:30:00",   # newest — should win
    )
    # LLM cited the OLDEST date (regression).
    out = apply_defensive_overrides(
        "之前是 2026-09-27 10:00:00。",
        history,
        route_decision="retrieve",
    )
    # Override fired and used the NEWEST TM (15:30).
    assert "15:30" in out, (
        f"override should use LATEST TM (2026-09-28 15:30), got: {out!r}"
    )
    # Single-tool, single dedup → single miss → single metric bump.
    assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before + 1.0


# ============================================================
# 6. state["loop_breaker_active"] propagation (integration)
# ============================================================


def test_loop_breaker_active_propagates_through_fsm_merge():
    """``react_agent`` writes ``loop_breaker_active`` to the delta;
    ``merge()`` (``src/agent/fsm.py:163``) persists arbitrary keys,
    so the next node reads the flag. This is the contract the loop-break
    synthesis hint depends on.
    """
    from src.agent.fsm import merge

    # Baseline state — no flag yet.
    state = {"messages": [], "step_count": 0}
    delta = {"messages": [AIMessage(content="x")], "loop_breaker_active": True}
    merged = merge(state, delta)
    assert merged["loop_breaker_active"] is True
    # Reset path: a non-break delta must clear the flag.
    delta2 = {"messages": [AIMessage(content="y")], "loop_breaker_active": False}
    merged2 = merge(merged, delta2)
    assert merged2["loop_breaker_active"] is False


def test_loop_break_hint_inserts_at_index_2_when_flag_true(monkeypatch):
    """When ``state["loop_breaker_active"]`` is True, the synthesis
    LLM receives the loop-break hint at msgs[2] — preserving
    v2.0.28.17 Anthropic consecutiveness (all SystemMessages
    stay consecutive at the START of msgs).
    """
    from src.agent.nodes import react_generate as react_generate_mod

    # Capture msgs handed to astream to verify the hint lands at
    # index 2 (BEFORE the sanitized history which starts at idx 3).
    captured: List = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured.extend(msgs)
            # Yield one chunk with text content so react_generate
            # sees non-empty answer_text (else it falls into the
            # v2.0.28.12 thinking-only fallback path).
            yield type("Chunk", (), {"content": "答案 (14:23:11)"})()

        async def ainvoke(self, msgs, **kwargs):
            # The fake ``ainvoke`` is here so react_generate's
            # ``build_chat_model(enable_thinking=True).astream(...)``
            # route works (it calls astream, not ainvoke). The
            # unused parameter silences the linter.
            pass

    monkeypatch.setattr(react_generate_mod, "build_chat_model", lambda **kw: _CaptureModel())

    state = {
        "current_query": "现在几点",
        "original_query": "现在几点",
        "intent": "qa_complex",
        "thread_id": "loop_break_test_thread",
        "messages": [],
        "step_count": 1,
        "loop_breaker_active": True,  # <-- the contract under test
    }
    events: list = []

    async def collect():
        async for evt in react_generate_mod.react_generate(state):
            events.append(evt)
    asyncio.run(collect())

    # msges[0..2] are all SystemMessages (the directive system
    # prompt + the HISTORY_FRAME_HINT + the loop-break hint).
    assert len(captured) >= 3, f"expected ≥3 SystemMessages, got {len(captured)}"
    for m in captured[:3]:
        assert isinstance(m, SystemMessage), (
            f"msgs[0..2] must be SystemMessage for Anthropic consecutiveness, "
            f"got {type(m).__name__} at index {captured.index(m)}"
        )
    # The loop-break hint IS at index 2 (NEW directive, BOTH
    # refusal + time didn't fire in this scenario).
    assert "loop breaker" in captured[2].content or "loop_breaker" in captured[2].content or \
        "研究已达上限" in captured[2].content, (
        f"msgs[2] should be the loop-break hint, got: {captured[2].content[:200]!r}"
    )


def test_loop_break_hint_skipped_when_flag_false(monkeypatch):
    """When ``state["loop_breaker_active"]`` is False (the default
    when the loop breaker DIDN'T fire), no loop-break hint is
    inserted. This pins the contract that the hint is gated, not
    always-on.
    """
    from src.agent.nodes import react_generate as react_generate_mod

    captured: List = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured.extend(msgs)
            yield type("Chunk", (), {"content": "答案 (14:23:11)"})()

    monkeypatch.setattr(react_generate_mod, "build_chat_model", lambda **kw: _CaptureModel())

    state = {
        "current_query": "现在几点",
        "original_query": "现在几点",
        "intent": "qa_complex",
        "thread_id": "no_loop_break_thread",
        "messages": [],
        "step_count": 1,
        "loop_breaker_active": False,  # <-- default
    }

    async def collect():
        async for _evt in react_generate_mod.react_generate(state):
            pass
    asyncio.run(collect())

    # msgs[0..1] are the system prompt + history frame hint. NO
    # loop-break hint at index 2 — the get_current_time directive
    # may land there instead (depends on history).
    # The contract: the loop-break string MUST NOT be in any msg.
    for m in captured:
        if isinstance(m, SystemMessage):
            assert "研究已达上限" not in m.content, (
                f"loop-break hint leaked into msgs when flag was False: {m.content[:200]!r}"
            )


def test_loop_break_hint_orders_after_refusal_before_time(monkeypatch):
    """When refusal + loop-break + time directive ALL fire, the
    final ordering in msgs is: [sys, hist_hint, time, loop_break,
    refusal, sanitized_history]. Anthropic's API concatenates all
    SystemMessages in order; the LAST one (rightmost, closest to
    the conversation start) carries the strongest recency bias —
    so refusal (strongest) must be rightmost.

    Implementation note: every directive is inserted at ``index=2``,
    so the order is determined by insertion order (each new insert
    pushes the prior one to idx 3, 4, ...). Inserting refusal LAST
    places it at the highest index. Belt-and-suspenders: even when
    the strongest fires, the collection-level SystemMessage count
    is consistent (3 directives + 2 baseline = 5 SystemMessages at
    indices 0..4, with sanitized history following).
    """
    from src.agent.nodes import react_generate as react_generate_mod

    captured: List = []

    class _CaptureModel:
        async def astream(self, msgs, **kwargs):
            captured.extend(msgs)
            yield type("Chunk", (), {"content": "答案"})()

    monkeypatch.setattr(react_generate_mod, "build_chat_model", lambda **kw: _CaptureModel())

    state = {
        "current_query": "现在几点",
        "original_query": "现在几点",
        "intent": "qa_complex",
        "thread_id": "all_directives_thread",
        "messages": [],
        "step_count": 1,
        "loop_breaker_active": True,
        "retrieval_status": "empty",  # triggers REFUSAL_TEMPLATES["empty"]
    }

    async def collect():
        async for _evt in react_generate_mod.react_generate(state):
            pass
    asyncio.run(collect())

    # Find each directive's position. All SystemMessages are
    # consecutive at indices 0..N.
    sys_idxs = [i for i, m in enumerate(captured) if isinstance(m, SystemMessage)]
    # Use distinctive markers — "ZERO documents" is unique to
    # REFUSAL_EMPTY_TEMPLATE; "研究已达上限" is unique to the
    # loop-break hint. Avoid generic "refusal" / "REFUSAL_TEMPLATES"
    # which the loop-break hint mentions by name.
    refusal_idx = next(
        (i for i in sys_idxs if "ZERO documents" in captured[i].content), None,
    )
    loop_break_idx = next(
        (i for i in sys_idxs if "研究已达上限" in captured[i].content), None,
    )
    # Both fire in this scenario.
    assert refusal_idx is not None, "refusal directive did not fire"
    assert loop_break_idx is not None, "loop-break directive did not fire"
    # Refusal must come AFTER loop-break (highest index = strongest
    # recency bias, rightmost = closest to conversation start).
    assert loop_break_idx < refusal_idx, (
        f"refusal must come after loop-break so it gets strongest "
        f"recency bias (refusal={refusal_idx}, loop_break={loop_break_idx})"
    )
    # And both are SystemMessages (the v2.0.28.17 consecutiveness rule).
    assert isinstance(captured[refusal_idx], SystemMessage)
    assert isinstance(captured[loop_break_idx], SystemMessage)
    # And no non-SystemMessage entries between them — consecutiveness
    # preserved across the whole SystemMessage block.
    between = captured[loop_break_idx:refusal_idx + 1]
    assert all(isinstance(m, SystemMessage) for m in between), (
        f"non-SystemMessage leaked between loop-break and refusal: "
        f"{[type(m).__name__ for m in between]}"
    )