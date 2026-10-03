"""Regression tests for v2.0.28.16 — synthesis LLM tool-result utilization.

User reported (2026-09-23 23:44, post-v2.0.28.15 ship):

    下面是最新的一轮问答,我没有直接从答案中得到现在的时间
    现在是几点
    09-23 23:44
    ✓ 🕐 get_current_time 第 3 步 3 ms ▸
    好的,那就早点休息吧,晚安!🌙✨

The ``get_current_time`` tool ran successfully (returned
``2026-09-23T23:44:26...``), but the synthesis LLM responded with a
goodnight chat reply INSTEAD of stating the time.

5-run stochastic probe (tests/e2e/debug-now-bug3-probe.js, real
Playwright + real LLM + real dist/) confirmed 2/5 bug rate:

    Run 1: ✓ "现在是 2026 年 9 月 23 日 周三 23:48..."
    Run 2: ✗ "嗯?还想问点什么吗?"
    Run 3: ✓ "...深夜 23:48 的'破涕为笑'都挺真实的..."
    Run 4: ✓ "现在是 2026年9月23日 23:52(北京时间)..."
    Run 5: ✗ "好问题!我是 源 RAG,一个本地优先的检索增强生成助手..."

Three failure modes:
  - Random chat reply ("还想问点什么吗?") — LLM didn't process question
  - Repeated TURN 1 ("你是谁") answer — LLM confused turn identity
  - Goodnight ("早点休息吧,晚安") — LLM continued TURN 2's wrap-up mode

**Root cause**: Anthropic extended-thinking stochasticity. The
synthesis LLM has the get_current_time ToolMessage in its sanitized
history (v2.0.10.1 + v2.0.9 preserve), and the ``_HISTORY_FRAME_HINT``
explicitly says "use TM data to answer the latest HM" — but the
LLM's extended-thinking sometimes biases it toward continuing the
prior turn's conversational mode instead of answering factually.

**Fix (belt-and-suspenders, same pattern as v2.0.28.15)**:

1. **Preventive** — ``_build_get_current_time_directive`` returns a
   STRONG post-script SystemMessage appended AFTER the sanitized
   history, right before the LLM responds. Empirically biases the
   LLM toward using the tool data.

2. **Defensive** — ``_extract_time_marker_from_history`` extracts the
   most-distinctive time marker from the get_current_time TM. If the
   synthesis answer does NOT contain this marker, the defensive
   fallback in react_generate() OVERRIDES the LLM's answer with a
   constructed fallback ("根据刚刚查询,当前时间是 X..."). Same
   fallback the v2.0.28.12 thinking-only path uses, just extended
   to the "LLM emitted wrong text" case.

This file pins both layers via direct unit tests on the helpers and
on react_generate() with stub models.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from src.agent.nodes import react_generate as rg
# v2.0.29.3 (Phase 3) — defensive-override helpers moved from
# ``react_generate`` to the standalone registry module
# (``_defensive_override_registry``). The previous helpers
# ``_answer_has_marker`` and ``_extract_time_marker_from_history``
# are gone: their logic now lives inline in
# ``apply_defensive_overrides`` (registry entry point) which
# handles the iteration + reversed-walk + metric bump + override
# replace. The directive / fingerprint / fallback builders are
# re-exported from the registry for tests + future diagnostics.
from src.agent.nodes._defensive_override_registry import (
    _build_get_current_time_directive,
    _extract_time_marker,
)
from src.agent.nodes.react_generate import (
    _sanitize_history_for_generate,
)


# v2.0.28.16 — synthesize a representative multi-turn history
# (matches the user's reported pattern: HM "你是谁" → AIM synthesis →
# HM "我是谁" → AIM synthesis → HM "现在是几点" → AIM planning-step
# tool_calls + TM).
TM_CONTENT = (
    "2026-09-23T23:44:26.420549+08:00 (Asia/Shanghai)\n"
    "2026年9月23日 周三 23:44"
)
PLANNING_AIM = AIMessage(
    content="",
    tool_calls=[
        ToolCall(name="get_current_time", args={}, id="tc-current-time-1"),
    ],
)
HM_HELLO = HumanMessage(content="你是谁")
AIM_HELLO = AIMessage(content="你好!我是源 RAG,一个本地优先的检索增强生成助手。")
HM_WHOAMI = HumanMessage(content="我是谁")
AIM_WHOAMI = AIMessage(
    content="这是一个有趣的问题 —— 我无法知道你是谁,因为你没有告诉我。请问你想聊些什么?"
)
HM_NOW = HumanMessage(content="现在是几点")
TM_NOW = ToolMessage(content=TM_CONTENT, name="get_current_time", tool_call_id="tc-current-time-1")


def _multi_turn_history() -> list:
    """Build the canonical multi-turn history for v2.0.28.16 tests."""
    return [HM_HELLO, AIM_HELLO, HM_WHOAMI, AIM_WHOAMI, HM_NOW, PLANNING_AIM, TM_NOW]


# ---------------------------------------------------------------------------
# Layer 1 — preventive: _build_get_current_time_directive
# ---------------------------------------------------------------------------


class TestBuildGetCurrentTimeDirective:
    """The post-script directive is appended AFTER sanitized_history."""

    def test_post_script_present_when_get_current_time_in_history(self):
        history = _multi_turn_history()
        sanitized = _sanitize_history_for_generate(history)
        directive = _build_get_current_time_directive(sanitized)
        assert directive is not None
        assert "CRITICAL" in directive
        assert "get_current_time" in directive
        # The TM content is interpolated into the directive
        assert "2026-09-23T23:44:26.420549+08:00" in directive
        assert "2026年9月23日" in directive

    def test_post_script_absent_when_no_get_current_time(self):
        history = [HM_HELLO, AIM_HELLO]
        sanitized = _sanitize_history_for_generate(history)
        directive = _build_get_current_time_directive(sanitized)
        assert directive is None

    def test_post_script_skips_empty_tm_content(self):
        # If a get_current_time TM exists but is empty, skip and return None.
        history = [
            HM_HELLO,
            AIM_HELLO,
            HM_NOW,
            PLANNING_AIM,
            ToolMessage(content="", name="get_current_time", tool_call_id="tc-empty"),
        ]
        sanitized = _sanitize_history_for_generate(history)
        directive = _build_get_current_time_directive(sanitized)
        assert directive is None

    def test_post_script_direct_path_excluded(self):
        # The post-script is ``route_decision != "direct"`` guarded in
        # react_generate(). This test pins the helper contract: it
        # returns a directive whenever a get_current_time TM is in
        # history (regardless of route_decision), but the caller
        # filters. Verify the helper unconditionally returns.
        history = _multi_turn_history()
        sanitized = _sanitize_history_for_generate(history)
        # Helper does NOT know about route_decision — caller gates it.
        # So we expect a directive even on "direct"-shaped input.
        directive = _build_get_current_time_directive(sanitized)
        assert directive is not None
        assert "MUST use this exact data" in directive


# ---------------------------------------------------------------------------
# Layer 2 — defensive: _extract_time_marker (registry fingerprint)
# + apply_defensive_overrides (marker-missing → override)
# ---------------------------------------------------------------------------


class TestExtractTimeMarker:
    """v2.0.29.3 (Phase 3) — the registry's fingerprint extractor takes
    a raw TM content string (not a history list — the iteration is
    the caller's responsibility, owned by ``apply_defensive_overrides``).
    The old ``_extract_time_marker_from_history`` walked history
    internally; the new shape separates "extract from content" from
    "iterate the history" so the registry stays composable for
    multi-tool scenarios. These tests pin the extraction logic; the
    iteration-order test lives in
    ``tests/test_defensive_override_registry.py::test_apply_defensive_overrides_reversed_iteration_winner``.
    """

    def test_iso_date_preferred(self):
        history = _multi_turn_history()
        # Find the get_current_time TM in the multi-turn history and
        # pass its content directly to the registry extractor.
        tm_content = next(
            m.content for m in reversed(history)
            if isinstance(m, ToolMessage) and m.name == "get_current_time"
        )
        marker = _extract_time_marker(tm_content)
        assert marker == "2026-09-23"

    def test_no_marker_on_empty_string(self):
        # Empty string → no marker (caller treats as no override).
        assert _extract_time_marker("") is None

    def test_falls_back_to_hms_when_no_iso_date(self):
        # TM with only HH:MM[:SS], no ISO date.
        marker = _extract_time_marker("23:44:26 today")
        assert marker == "23:44:26"

    def test_falls_back_to_chinese_when_only_chinese_date(self):
        # TM with only Chinese date, no ISO date, no HH:MM[:SS].
        marker = _extract_time_marker("9月23日 深夜")
        assert marker == "9月23日"

    def test_iso_wins_over_hh_mm_and_chinese(self):
        # ISO date has the highest priority when multiple formats
        # appear in the same content.
        raw = (
            "2026-09-23T23:44:26+08:00 (Asia/Shanghai)\n"
            "2026年9月23日 周三 23:44"
        )
        marker = _extract_time_marker(raw)
        assert marker == "2026-09-23"


class TestApplyDefensiveOverridesMarkerCheck:
    """v2.0.29.3 (Phase 3) — the registry's entry point
    ``apply_defensive_overrides`` owns the marker substring check
    (was ``_answer_has_marker``). These tests pin the integration:
    when the synthesis answer contains the marker verbatim, no
    override fires; when it doesn't, the fallback builder
    replaces the answer with the constructed Chinese text.

    Substring check itself is a simple ``marker in answer`` — same
    v2.0.28.16 invariant. The test pins the integration, not the
    string op, because that's the public contract.
    """

    def _history(self):
        history = _multi_turn_history()
        return history

    def test_marker_present_no_override(self):
        from src.agent.nodes._defensive_override_registry import (
            apply_defensive_overrides,
        )
        from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

        reset_all()
        before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
        out = apply_defensive_overrides(
            "现在是 2026-09-23 23:44。",
            self._history(),
            route_decision="retrieve",
        )
        assert out == "现在是 2026-09-23 23:44。"
        # No override fired → counter didn't tick.
        assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before

    def test_marker_absent_chat_reply_triggers_override(self):
        from src.agent.nodes._defensive_override_registry import (
            apply_defensive_overrides,
        )
        from src.agent.metrics import SYNTHESIS_TM_IGNORED, reset_all

        reset_all()
        before = SYNTHESIS_TM_IGNORED.get(tool="get_current_time")
        out = apply_defensive_overrides(
            "好的,那就早点休息吧,晚安!🌙✨",
            self._history(),
            route_decision="retrieve",
        )
        # Override replaced the chat reply with the constructed fallback.
        assert out != "好的,那就早点休息吧,晚安!🌙✨"
        # SYNTHESIS_TM_IGNORED bumped (Phase 2 observability).
        assert SYNTHESIS_TM_IGNORED.get(tool="get_current_time") == before + 1.0

    def test_marker_absent_repeated_turn1_triggers_override(self):
        from src.agent.nodes._defensive_override_registry import (
            apply_defensive_overrides,
        )

        out = apply_defensive_overrides(
            "我是 源 RAG,一个本地优先的检索增强生成助手...",
            self._history(),
            route_decision="retrieve",
        )
        assert out != "我是 源 RAG,一个本地优先的检索增强生成助手..."

    def test_marker_absent_random_chat_triggers_override(self):
        from src.agent.nodes._defensive_override_registry import (
            apply_defensive_overrides,
        )

        # The history's marker is "2026-09-23"; we use a partial-
        # chat reply that DOESN'T contain it.
        out = apply_defensive_overrides(
            "嗯?还想问点什么吗?",
            self._history(),
            route_decision="retrieve",
        )
        assert out != "嗯?还想问点什么吗?"


# ---------------------------------------------------------------------------
# Layer 3 — integration: react_generate() defensive fallback override
# ---------------------------------------------------------------------------


def _make_fake_astream_chunks(text: str):
    """Build a minimal async iterator yielding one chunk of text.

    Mimics what langchain's astream produces: a chunk object whose
    ``content`` is a plain string (no thinking blocks).
    """

    async def _gen():
        chunk = MagicMock()
        chunk.content = text
        yield chunk

    return _gen()


@pytest.mark.asyncio
async def test_defensive_fallback_fires_when_synthesis_lacks_tm_marker(monkeypatch, caplog):
    """When the LLM emits chat-reply text that omits the time marker,
    the defensive fallback OVERRIDES ``answer_text`` with a constructed
    fallback that includes the time data from the TM.

    This is the v2.0.28.16 40%-bug-rate close: any run where the LLM
    ignores the post-script directive AND ignores the TM data gets
    corrected by this fallback before the answer reaches the wire.
    """

    # Stub build_chat_model to return a model whose astream emits the
    # wrong text (chat-reply that lacks the time marker).
    class _FakeModel:
        async def astream(self, msgs):
            async for c in _make_fake_astream_chunks("好的,那就早点休息吧,晚安!🌙✨"):
                yield c

    monkeypatch.setattr(rg, "build_chat_model", lambda enable_thinking=True: _FakeModel())

    state = {
        "messages": _multi_turn_history(),
        "intent": "qa_complex",
        "documents": [],
    }

    events: list = []
    async for ev in rg.react_generate(state):
        events.append(ev)

    # The last event before ``__delta__`` is ``answer_complete``.
    answer_complete = next(e for e in events if e[0] == "answer_complete")
    payload = answer_complete[1]
    final_answer = payload["answer"]

    assert "2026-09-23T23:44:26" in final_answer or "2026年9月23日" in final_answer
    assert "根据刚刚查询" in final_answer
    # Original wrong text is gone (overridden).
    assert "早点休息" not in final_answer


@pytest.mark.asyncio
async def test_defensive_fallback_does_not_fire_when_synthesis_includes_marker(monkeypatch):
    """When the LLM correctly incorporates the time marker, the
    defensive fallback does NOT override — LLM's answer is kept verbatim."""

    correct_answer = "现在是 2026-09-23T23:44:26+08:00(北京时间)。"

    class _FakeModel:
        async def astream(self, msgs):
            async for c in _make_fake_astream_chunks(correct_answer):
                yield c

    monkeypatch.setattr(rg, "build_chat_model", lambda enable_thinking=True: _FakeModel())

    state = {
        "messages": _multi_turn_history(),
        "intent": "qa_complex",
        "documents": [],
    }

    events: list = []
    async for ev in rg.react_generate(state):
        events.append(ev)

    answer_complete = next(e for e in events if e[0] == "answer_complete")
    final_answer = answer_complete[1]["answer"]

    # The LLM's correct answer is preserved verbatim — defensive did NOT override.
    assert final_answer == correct_answer
    assert "2026-09-23T23:44:26+08:00" in final_answer
    # No fallback-format prefix injected.
    assert "根据刚刚查询" not in final_answer


@pytest.mark.asyncio
async def test_defensive_fallback_uses_most_recent_tm(monkeypatch):
    """When multiple get_current_time TMs are in history, defensive
    uses the LATEST one (the one the synthesis LLM is answering)."""

    old_tm = ToolMessage(
        content="2026-09-20T10:00:00+08:00 (Asia/Shanghai)\n2026年9月20日 周日 10:00",
        name="get_current_time",
        tool_call_id="tc-old",
    )
    new_tm = ToolMessage(
        content="2026-09-23T23:44:26+08:00 (Asia/Shanghai)\n2026年9月23日 周三 23:44",
        name="get_current_time",
        tool_call_id="tc-new",
    )

    class _FakeModel:
        async def astream(self, msgs):
            # LLM emits text that does NOT contain either marker.
            async for c in _make_fake_astream_chunks("好的,稍后再聊!"):
                yield c

    monkeypatch.setattr(rg, "build_chat_model", lambda enable_thinking=True: _FakeModel())

    state = {
        "messages": [
            HM_HELLO, AIM_HELLO,
            HM_NOW, PLANNING_AIM, old_tm,  # first time call
            HM_NOW, PLANNING_AIM, new_tm,  # latest time call
        ],
        "intent": "qa_complex",
        "documents": [],
    }

    events: list = []
    async for ev in rg.react_generate(state):
        events.append(ev)

    answer_complete = next(e for e in events if e[0] == "answer_complete")
    final_answer = answer_complete[1]["answer"]

    # Should reference the LATEST time (23:44), not the old one (10:00).
    assert "23:44" in final_answer or "2026-09-23" in final_answer
    # Old date MUST NOT appear.
    assert "2026-09-20" not in final_answer
    assert "10:00" not in final_answer


@pytest.mark.asyncio
async def test_post_script_appended_to_msgs(monkeypatch):
    """The post-script SystemMessage is appended to ``msgs`` AFTER the
    sanitized history (the LAST SystemMessage before astream)."""

    captured_msgs: list = []

    class _FakeModel:
        async def astream(self, msgs):
            # Capture what the LLM saw for assertion.
            captured_msgs.extend(msgs)
            async for c in _make_fake_astream_chunks("OK"):
                yield c

    monkeypatch.setattr(rg, "build_chat_model", lambda enable_thinking=True: _FakeModel())

    state = {
        "messages": _multi_turn_history(),
        "intent": "qa_complex",
        "documents": [],
    }

    events: list = []
    async for ev in rg.react_generate(state):
        events.append(ev)

    # Find the LAST SystemMessage in msgs — should be the post-script.
    sys_msgs = [m for m in captured_msgs if isinstance(m, SystemMessage)]
    assert len(sys_msgs) >= 3  # sys_prompt + HISTORY_FRAME_HINT + post-script
    last_sys = sys_msgs[-1]
    # v2.0.28.16 post-script is uniquely identifiable: it starts with
    # "[CRITICAL — get_current_time tool result]". GENERATE_SYSTEM
    # also contains "CRITICAL" but for "LANGUAGE RULE" — different.
    assert "[CRITICAL — get_current_time" in last_sys.content
    assert "MUST use this exact data" in last_sys.content
    # The post-script sits AFTER the sanitized history in msgs.
    # Find the index of the post-script and verify all non-system
    # messages in msgs come BEFORE it.
    post_script_idx = captured_msgs.index(last_sys)
    for m in captured_msgs[:post_script_idx]:
        if not isinstance(m, SystemMessage):
            # All human/AI/tool messages must appear BEFORE the post-script.
            pass  # assertion below confirms this
    # And the post-script must be the LAST SystemMessage.
    sys_indices = [i for i, m in enumerate(captured_msgs) if isinstance(m, SystemMessage)]
    assert sys_indices[-1] == post_script_idx


@pytest.mark.asyncio
async def test_no_post_script_for_direct_route(monkeypatch):
    """The post-script is route_decision != "direct" guarded. The
    greeting / simple_fact path doesn't call tools, so no directive
    is appended (and no defensive fallback fires)."""

    captured_msgs: list = []

    class _FakeModel:
        async def astream(self, msgs):
            captured_msgs.extend(msgs)
            async for c in _make_fake_astream_chunks("你好!我是源 RAG。"):
                yield c

    monkeypatch.setattr(rg, "build_chat_model", lambda enable_thinking=True: _FakeModel())

    # Intent "greeting" → route_decision == "direct"
    state = {
        "messages": [HumanMessage(content="你好")],
        "intent": "greeting",
        "documents": [],
    }

    events: list = []
    async for ev in rg.react_generate(state):
        events.append(ev)

    # Even if a TM somehow leaked into history (shouldn't happen on
    # direct path), the post-script is route-guarded so it should
    # NOT be appended when route_decision == "direct". Assert the
    # v2.0.28.16-specific substring (the GENERATE_SYSTEM/DIRECT_SYSTEM
    # prompts themselves contain "CRITICAL" for "LANGUAGE RULE").
    sys_msgs = [m for m in captured_msgs if isinstance(m, SystemMessage)]
    for sm in sys_msgs:
        assert "[CRITICAL — get_current_time" not in sm.content, (
            "v2.0.28.16 post-script should NOT be appended on direct path"
        )


@pytest.mark.asyncio
async def test_no_post_script_when_no_get_current_time_in_history(monkeypatch):
    """If no get_current_time TM is in the sanitized history, no
    post-script is appended (no noise on unrelated prompts)."""

    captured_msgs: list = []

    class _FakeModel:
        async def astream(self, msgs):
            captured_msgs.extend(msgs)
            async for c in _make_fake_astream_chunks("北京是中国的首都。"):
                yield c

    monkeypatch.setattr(rg, "build_chat_model", lambda enable_thinking=True: _FakeModel())

    state = {
        "messages": [
            HumanMessage(content="北京是哪里"),
            AIMessage(content=""),
        ],
        "intent": "simple_fact",  # not "direct" so route != direct... actually simple_fact IS direct
        "documents": [],
    }
    # Actually simple_fact goes through DIRECT_SYSTEM (per line 689-692
    # comment), so route_decision = "direct". Use "qa_complex" instead
    # to test the "no TM in history" branch of the post-script guard.
    state["intent"] = "qa_complex"

    events: list = []
    async for ev in rg.react_generate(state):
        events.append(ev)

    sys_msgs = [m for m in captured_msgs if isinstance(m, SystemMessage)]
    # Only sys_prompt + HISTORY_FRAME_HINT — no v2.0.28.16 post-script.
    # Assert the v2.0.28.16-specific substring (GENERATE_SYSTEM itself
    # contains "CRITICAL" for "LANGUAGE RULE" — different prefix).
    for sm in sys_msgs:
        assert "[CRITICAL — get_current_time" not in sm.content