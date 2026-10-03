"""Regression tests for LLM reasoning persistence across sessions.

User-facing bug
---------------
The model's extended-thinking blocks (Anthropic ``thinking`` /
``redacted_thinking``) were only ever visible during the live stream.
When the user reloaded the page or switched to the thread from the
sidebar, ``GET /sessions/{thread_id}/messages`` would return the
answer text but no reasoning — so the 💭 drawer was always empty on
historical messages, which read like the model never thought.

Why
---
The LangGraph ``messages`` channel only carries LangChain
``BaseMessage`` objects; there's no first-class slot for "the model's
private reasoning". The runner side already emitted reasoning as a
``reasoning`` WS event during streaming, but it was never written
back into the persisted AIMessage, so the SqliteSaver snapshot
couldn't replay it.

Fix
---
The generate node splits the LLM's structured content blocks and
attaches the reasoning text to ``AIMessage.additional_kwargs["reasoning"]``
— same trick already used for ``sources``. ``_serialize_messages``
then extracts it on history reload. ``MessageRecord`` carries the
new ``reasoning`` field, and the frontend's ``loadHistory`` mapper
picks it up.

These tests verify that round-trip.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# _split_text_and_thinking — pure helper, mirrors runner._extract_text_and_thinking
# ---------------------------------------------------------------------------


class TestSplitTextAndThinking:
    """Direct unit tests for the shared ``src.agent._thinking_split``
    helper that walks Anthropic's structured content blocks. (Phase C5
    moved this from generate.py / runner.py into a leaf module to
    avoid the prior circular-import risk.)"""

    def _split(self, content):
        from src.agent._thinking_split import split_text_and_thinking

        return split_text_and_thinking(content)

    def test_string_content_passes_through(self):
        """OpenAI / non-thinking providers: AIMessage.content is a plain
        string. We should return it verbatim and an empty reasoning."""
        text, think = self._split("Hello world.")
        assert text == "Hello world."
        assert think == ""

    def test_list_with_only_text_block(self):
        """Anthropic without extended-thinking: list of ``{type:text}``
        blocks. Same result as plain string."""
        content = [{"type": "text", "text": "Hello world."}]
        text, think = self._split(content)
        assert text == "Hello world."
        assert think == ""

    def test_list_with_thinking_and_text_blocks(self):
        """The Anthropic thinking case: ``content`` is a list with both
        ``thinking`` and ``text`` blocks. We must split — putting the
        full list into ``answer_text`` would render raw dicts in the UI
        and put the thinking in the wrong place."""
        content = [
            {"type": "thinking", "thinking": "Let me reason about X..."},
            {"type": "text", "text": "Final answer."},
        ]
        text, think = self._split(content)
        assert text == "Final answer."
        assert think == "Let me reason about X..."

    def test_empty_thinking_block_skipped(self):
        """Signature-only deltas arrive as ``{type:thinking, thinking:""}``.
        We must skip them so we don't emit empty reasoning events."""
        content = [
            {"type": "thinking", "thinking": "", "signature": "abc"},
            {"type": "text", "text": "ok"},
        ]
        text, think = self._split(content)
        assert text == "ok"
        assert think == ""

    def test_redacted_thinking_uses_placeholder(self):
        """``redacted_thinking`` carries encrypted bytes we can't show.
        Substitute a short marker so the user still sees the model thought."""
        content = [
            {"type": "redacted_thinking"},
            {"type": "text", "text": "Answer"},
        ]
        text, think = self._split(content)
        assert text == "Answer"
        assert think == "[thinking redacted]"

    def test_openai_responses_summary_blocks(self):
        """OpenAI Responses API: ``{type:reasoning, summary:[{text:...}]}``.
        Not routed to today, but keep coverage so a future SDK bump that
        starts emitting it doesn't break the splitter silently."""
        content = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking step"}],
            },
            {"type": "text", "text": "answer"},
        ]
        text, think = self._split(content)
        assert text == "answer"
        assert think == "thinking step"

    def test_tool_use_blocks_dropped(self):
        """Tool-call JSON from router / rewriter / grader nodes must
        never leak into either field."""
        content = [
            {"type": "tool_use", "name": "search", "input": {}},
            {"type": "text", "text": "visible"},
        ]
        text, think = self._split(content)
        assert text == "visible"
        assert think == ""

    def test_multiple_text_blocks_concatenated(self):
        """Anthropic may emit multiple text blocks (e.g. after a tool
        use). Concatenate them in order."""
        content = [
            {"type": "text", "text": "Part 1. "},
            {"type": "text", "text": "Part 2."},
        ]
        text, think = self._split(content)
        assert text == "Part 1. Part 2."
        assert think == ""

    def test_unrecognized_block_type_dropped(self):
        """Unknown future block types should be silently skipped rather
        than crashing. Don't surface raw dicts."""
        content = [{"type": "future_block", "data": "???"}, {"type": "text", "text": "ok"}]
        text, think = self._split(content)
        assert text == "ok"
        assert think == ""

    def test_non_list_non_string_returns_empty(self):
        """Defensive: ``content=None`` or ``content=42`` should yield
        empty strings, not crash."""
        text, think = self._split(None)
        assert text == ""
        assert think == ""
        text, think = self._split(42)
        assert text == ""
        assert think == ""


# ---------------------------------------------------------------------------
# _serialize_messages — extracts reasoning from additional_kwargs on replay
# ---------------------------------------------------------------------------


class TestSerializeMessagesExtractsReasoning:
    """The replay path: AIMessage.additional_kwargs["reasoning"] →
    MessageRecord.reasoning. Without this, reopening an old conversation
    would show an empty thinking drawer even though the data is in the
    checkpoint."""

    def test_assistant_with_reasoning_round_trips(self):
        from src.api.routes.sessions import _serialize_messages

        msg = AIMessage(
            content="The capital is Paris.",
            additional_kwargs={"reasoning": "The user asked about France."},
        )
        records = _serialize_messages([msg])
        assert len(records) == 1
        assert records[0].content == "The capital is Paris."
        assert records[0].reasoning == "The user asked about France."

    def test_assistant_without_reasoning_is_none(self):
        """Old checkpoints (predating this field) or providers without
        thinking come back with ``reasoning=None``. The frontend drawer
        already handles this by not rendering."""
        from src.api.routes.sessions import _serialize_messages

        msg = AIMessage(content="Hi there.")
        records = _serialize_messages([msg])
        assert len(records) == 1
        assert records[0].reasoning is None

    def test_empty_reasoning_string_treated_as_none(self):
        """Defensive: a present-but-empty string shouldn't leak through
        as a non-null field. The frontend would render an empty drawer
        in that case, which looks like a bug."""
        from src.api.routes.sessions import _serialize_messages

        msg = AIMessage(content="Hi.", additional_kwargs={"reasoning": ""})
        records = _serialize_messages([msg])
        assert records[0].reasoning is None

    def test_non_string_reasoning_dropped(self):
        """If a provider or migration puts a non-string into the
        ``reasoning`` slot (dict, list, int), we must drop it rather
        than crash or leak a wrong-typed value."""
        from src.api.routes.sessions import _serialize_messages

        msg = AIMessage(
            content="Hi.",
            additional_kwargs={"reasoning": {"nested": "dict"}},
        )
        records = _serialize_messages([msg])
        assert records[0].reasoning is None

    def test_reasoning_and_sources_coexist(self):
        """Same trick used for ``sources``. Both fields round-trip
        together — reasoning for the thinking drawer, sources for
        the citation chips."""
        from src.api.routes.sessions import _serialize_messages

        msg = AIMessage(
            content="See [1].",
            additional_kwargs={
                "reasoning": "Used the first doc.",
                "sources": [{"index": 1, "filename": "doc.pdf"}],
            },
        )
        records = _serialize_messages([msg])
        assert records[0].reasoning == "Used the first doc."
        assert records[0].sources is not None
        # v2.0.7 SoT-1: ``Source`` is Pydantic; attribute access.
        assert records[0].sources[0].filename == "doc.pdf"

    def test_user_message_has_no_reasoning(self):
        """User messages don't get reasoning — only the assistant turn
        does. Replay should leave the field None for human messages."""
        from langchain_core.messages import HumanMessage

        from src.api.routes.sessions import _serialize_messages

        records = _serialize_messages([HumanMessage(content="Hello?")])
        assert len(records) == 1
        assert records[0].role == "user"
        assert records[0].reasoning is None