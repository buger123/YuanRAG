"""Regression tests for v2.0.28.17 — fix v2.0.28.16 SystemMessage ordering bug.

User report (2026-09-25, post-v2.0.28.16 ship):

    点击历史对话出现的是一个新对话的样式,后台提示如下
    2026-09-25 17:51:07.924 | ERROR | src.agent.nodes._recovery:llm_call_failure:124 |
      react_generate failed: Received multiple non-consecutive system messages.
    ValueError: Received multiple non-consecutive system messages.

**Root cause**: v2.0.28.16 appended the post-script ``SystemMessage`` to
the END of ``msgs`` (after sanitized history). Anthropic's API requires
ALL system messages to be CONSECUTIVE at the START of the messages
list (see ``langchain_anthropic/chat_models.py:527``). With any
HumanMessage/AIMessage/ToolMessage between the leading SystemMessages
and the appended one, the API raises ``ValueError``.

**Worse, the bug was MASKED**: the defensive fingerprint override
(see v2.0.28.16) fires AFTER the streaming loop, overriding the LLM's
output with a constructed fallback when the answer lacks the time
marker. When the LLM call raises ValueError, ``llm_call_failure``
returns an apology string. The defensive override sees the apology
doesn't contain the time marker and replaces it with "根据刚刚查询,
当前时间是 X...". **User saw correct answer but the LLM itself never
ran.** Silent failure masked by defensive override.

**Fix**: ``msgs.insert(2, SystemMessage(content=time_directive))``
instead of ``msgs.append(...)``. The directive now sits at index 2
(BEFORE sanitized history), keeping msgs[0..2] all SystemMessage.
System messages stay consecutive; sanitized history still follows.

This file pins the new contract via direct unit tests on the
resulting ``msgs`` list structure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from src.agent.nodes import react_generate as rg
from src.agent.nodes.react_generate import (
    _sanitize_history_for_generate,
)


TM_CONTENT = (
    "2026-09-23T23:44:26.420549+08:00 (Asia/Shanghai)\n"
    "2026年9月23日 周三 23:44"
)
HM_HELLO = HumanMessage(content="你是谁")
AIM_HELLO = AIMessage(content="你好!我是源 RAG。")
HM_WHOAMI = HumanMessage(content="我是谁")
AIM_WHOAMI = AIMessage(content="不知道你是谁。请问你想聊些什么?")
HM_NOW = HumanMessage(content="现在是几点")
PLANNING_AIM = AIMessage(
    content="",
    tool_calls=[ToolCall(name="get_current_time", args={}, id="tc-current-time-1")],
)
TM_NOW = ToolMessage(
    content=TM_CONTENT, name="get_current_time", tool_call_id="tc-current-time-1"
)


def _multi_turn_history() -> list:
    """Build the canonical multi-turn history with a get_current_time TM."""
    return [HM_HELLO, AIM_HELLO, HM_WHOAMI, AIM_WHOAMI, HM_NOW, PLANNING_AIM, TM_NOW]


def _make_fake_astream_chunks(text: str):
    async def _gen():
        chunk = MagicMock()
        chunk.content = text
        yield chunk

    return _gen()


# ---------------------------------------------------------------------------
# Regression tests — pin the consecutive-system-messages invariant
# ---------------------------------------------------------------------------


class TestSystemMessagesConsecutiveInvariant:
    """Anthropic API raises ``ValueError`` if any SystemMessage appears
    AFTER a non-SystemMessage in the messages list. The v2.0.28.16
    ``msgs.append(SystemMessage(...))`` violated this; v2.0.28.17 fix
    uses ``msgs.insert(2, SystemMessage(...))``. These tests pin the
    new invariant by driving ``react_generate`` and inspecting the
    ``msgs`` passed to the LLM."""

    @pytest.mark.asyncio
    async def test_post_script_systemmessage_stays_consecutive(self, monkeypatch):
        """When a get_current_time TM is in history, the post-script
        SystemMessage is INSERTED at index 2 — BEFORE sanitized history.
        Result: msgs[0..2] are all SystemMessage (consecutive), then
        HumanMessage/AIMessage/ToolMessage follow."""

        captured_msgs: list = []

        class _FakeModel:
            async def astream(self, msgs):
                captured_msgs.extend(msgs)
                async for c in _make_fake_astream_chunks("OK"):
                    yield c

        monkeypatch.setattr(
            rg, "build_chat_model", lambda enable_thinking=True: _FakeModel()
        )

        state = {
            "messages": _multi_turn_history(),
            "intent": "qa_complex",
            "documents": [],
        }

        async for _ in rg.react_generate(state):
            pass

        # Find the SystemMessage with the v2.0.28.16 post-script directive.
        sys_msgs = [m for m in captured_msgs if isinstance(m, SystemMessage)]
        assert len(sys_msgs) >= 3, (
            f"expected ≥3 SystemMessages (sys_prompt + _HISTORY_FRAME_HINT + "
            f"post-script), got {len(sys_msgs)}"
        )
        last_sys = sys_msgs[-1]
        assert "[CRITICAL — get_current_time" in last_sys.content

        # The CORE invariant: every SystemMessage must come BEFORE any
        # non-SystemMessage. Verify by finding the LAST SystemMessage's
        # index and asserting no non-SystemMessage appears at an earlier
        # index.
        last_sys_idx = captured_msgs.index(last_sys)
        for m in captured_msgs[: last_sys_idx + 1]:
            if not isinstance(m, SystemMessage):
                # Found a non-system message BEFORE the last SystemMessage.
                # That would mean a SystemMessage follows it → non-consecutive.
                pytest.fail(
                    f"non-SystemMessage at index {captured_msgs.index(m)} comes "
                    f"BEFORE the last SystemMessage at index {last_sys_idx} → "
                    f"Anthropic API will raise 'multiple non-consecutive system messages'"
                )

        # Direct check: msgs[0..2] all SystemMessage.
        assert isinstance(captured_msgs[0], SystemMessage)
        assert isinstance(captured_msgs[1], SystemMessage)
        assert isinstance(captured_msgs[2], SystemMessage)
        # msgs[3] must NOT be SystemMessage (would be redundant but
        # consecutive; the real risk is a SystemMessage at idx ≥ 3 with
        # non-system messages before).
        if len(captured_msgs) > 3:
            assert not isinstance(captured_msgs[3], SystemMessage) or all(
                isinstance(m, SystemMessage) for m in captured_msgs[3:]
            ), (
                "messages[3:] should start with sanitized history (HumanMessage "
                "or AIMessage, NOT a SystemMessage — that would still be "
                "consecutive but is not where v2.0.28.16's directive lives)"
            )

    @pytest.mark.asyncio
    async def test_post_script_inserted_at_index_2(self, monkeypatch):
        """The directive SystemMessage must be at index 2 (right after
        sys_prompt + _HISTORY_FRAME_HINT), NOT appended at the end."""

        captured_msgs: list = []

        class _FakeModel:
            async def astream(self, msgs):
                captured_msgs.extend(msgs)
                async for c in _make_fake_astream_chunks("OK"):
                    yield c

        monkeypatch.setattr(
            rg, "build_chat_model", lambda enable_thinking=True: _FakeModel()
        )

        state = {
            "messages": _multi_turn_history(),
            "intent": "qa_complex",
            "documents": [],
        }

        async for _ in rg.react_generate(state):
            pass

        # Find the post-script SystemMessage.
        post_script_idx = None
        for i, m in enumerate(captured_msgs):
            if isinstance(m, SystemMessage) and "[CRITICAL — get_current_time" in m.content:
                post_script_idx = i
                break
        assert post_script_idx is not None, "post-script SystemMessage not found"
        # Must be at index 2 (right after the two consecutive SystemMessages).
        assert post_script_idx == 2, (
            f"post-script must be inserted at index 2 for consecutiveness, "
            f"got index {post_script_idx} (pre-v2.0.28.17 bug: appended at end "
            f"→ index {len(captured_msgs) - 1})"
        )

    @pytest.mark.asyncio
    async def test_no_post_script_when_no_get_current_time(self, monkeypatch):
        """Without a get_current_time TM, no post-script SystemMessage
        is added — msgs stays at exactly 2 SystemMessages (sys_prompt +
        _HISTORY_FRAME_HINT)."""

        captured_msgs: list = []

        class _FakeModel:
            async def astream(self, msgs):
                captured_msgs.extend(msgs)
                async for c in _make_fake_astream_chunks("北京是中国的首都。"):
                    yield c

        monkeypatch.setattr(
            rg, "build_chat_model", lambda enable_thinking=True: _FakeModel()
        )

        state = {
            "messages": [HM_HELLO, AIM_HELLO],
            "intent": "qa_complex",
            "documents": [],
        }

        async for _ in rg.react_generate(state):
            pass

        sys_msgs = [m for m in captured_msgs if isinstance(m, SystemMessage)]
        # Exactly 2 SystemMessages — no post-script added.
        assert len(sys_msgs) == 2
        # No post-script directive anywhere.
        for m in captured_msgs:
            if isinstance(m, SystemMessage):
                assert "[CRITICAL — get_current_time" not in m.content

    @pytest.mark.asyncio
    async def test_no_post_script_on_direct_path(self, monkeypatch):
        """The post-script is route_decision != "direct" guarded. On the
        greeting / simple_fact path, no directive is added regardless
        of whether a TM is somehow in history."""

        captured_msgs: list = []

        class _FakeModel:
            async def astream(self, msgs):
                captured_msgs.extend(msgs)
                async for c in _make_fake_astream_chunks("你好!我是源 RAG。"):
                    yield c

        monkeypatch.setattr(
            rg, "build_chat_model", lambda enable_thinking=True: _FakeModel()
        )

        state = {
            "messages": [HumanMessage(content="你好")],
            "intent": "greeting",  # route_decision == "direct"
            "documents": [],
        }

        async for _ in rg.react_generate(state):
            pass

        for m in captured_msgs:
            if isinstance(m, SystemMessage):
                assert "[CRITICAL — get_current_time" not in m.content

    @pytest.mark.asyncio
    async def test_sanitized_history_not_dropped(self, monkeypatch):
        """Sanity check: v2.0.28.17's `insert(2, ...)` does NOT drop any
        sanitized history messages — total count must match."""

        captured_msgs: list = []

        class _FakeModel:
            async def astream(self, msgs):
                captured_msgs.extend(msgs)
                async for c in _make_fake_astream_chunks("OK"):
                    yield c

        monkeypatch.setattr(
            rg, "build_chat_model", lambda enable_thinking=True: _FakeModel()
        )

        history = _multi_turn_history()
        state = {
            "messages": history,
            "intent": "qa_complex",
            "documents": [],
        }

        async for _ in rg.react_generate(state):
            pass

        # Sanitized history may differ in count from raw history (v2.0.28.11
        # strips trailing AIMessages), but the count must be ≥ 1 (at minimum
        # the LAST HumanMessage survives).
        sanitized_history = _sanitize_history_for_generate(history)
        # msgs = [sys_prompt, HISTORY_HINT, post-script?, *sanitized_history]
        # Post-script IS added (get_current_time TM in history).
        expected_count = 3 + len(sanitized_history)
        assert len(captured_msgs) == expected_count

        # All sanitized_history messages must be present in msgs.
        for m in sanitized_history:
            assert m in captured_msgs