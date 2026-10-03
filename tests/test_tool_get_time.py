"""Regression tests for the ``get_current_time`` tool.

v2.0 — ReAct rewrite introduces this tool to kill the
"hallucinated date" failure mode (LLM guesses "today is March 5"
without a real clock). These tests pin the surface so a future
refactor can't silently drop the time hint or change its shape.
"""
from __future__ import annotations

import asyncio
import re

import pytest


# ---------------------------------------------------------------------------
# Pure-function tests (no LLM)
# ---------------------------------------------------------------------------


def test_format_clock_default_timezone_is_asia_shanghai():
    """Default tz = Asia/Shanghai — China's primary user base."""
    from src.agent.tools.get_current_time import _format_clock

    out = _format_clock("Asia/Shanghai")
    assert "Asia/Shanghai" in out
    # ISO 8601 with timezone offset.
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", out)


def test_format_clock_explicit_timezone():
    from src.agent.tools.get_current_time import _format_clock

    out = _format_clock("UTC")
    assert "UTC" in out
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", out)


def test_format_clock_unknown_timezone_falls_back_to_utc():
    """Bogus tz (e.g. 'Mars/Olympus_Mons') must NOT raise — fall back
    to UTC so the ReAct loop doesn't crash on a malformed tool call.
    """
    from src.agent.tools.get_current_time import _format_clock

    out = _format_clock("Mars/Olympus_Mons")
    assert "UTC" in out


def test_format_clock_includes_human_readable_line():
    """ISO line + Chinese human-readable line, newline-separated."""
    from src.agent.tools.get_current_time import _format_clock

    out = _format_clock("Asia/Shanghai")
    lines = out.splitlines()
    assert len(lines) == 2
    # Chinese human-readable line has the year/month/day characters.
    assert "年" in lines[1]
    assert "月" in lines[1]
    assert "日" in lines[1]


# ---------------------------------------------------------------------------
# Async tool invocation — exercises the LangChain @tool surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_current_time_async_returns_string():
    """The @tool decorator wraps the function so .ainvoke() returns
    the formatted clock string. Pin the type."""
    from src.agent.tools import get_current_time

    result = await get_current_time.ainvoke({"timezone": "UTC"})
    assert isinstance(result, str)
    assert "UTC" in result


@pytest.mark.asyncio
async def test_get_current_time_uses_default_timezone_when_omitted():
    """When the LLM emits tool_call without args, default kicks in."""
    from src.agent.tools import get_current_time

    result = await get_current_time.ainvoke({})
    assert "Asia/Shanghai" in result


@pytest.mark.asyncio
async def test_get_current_time_tool_metadata():
    """The tool registers with the LangChain @tool contract — name,
    description, args_schema all exposed. Pin so the agent binding
    doesn't break if someone renames the tool."""
    from src.agent.tools import get_current_time
    from src.agent.tools.get_current_time import GetCurrentTimeArgs

    assert get_current_time.name == "get_current_time"
    assert "current date" in get_current_time.description.lower()
    # args_schema is the Pydantic model class.
    assert get_current_time.args_schema is GetCurrentTimeArgs


# ---------------------------------------------------------------------------
# Pydantic validation — bad inputs are caught before the tool runs
# ---------------------------------------------------------------------------


def test_get_time_args_rejects_extreme_timezone_length():
    """Pydantic max_length=64 keeps user-supplied tz strings bounded."""
    from src.agent.tools.get_current_time import GetCurrentTimeArgs

    with pytest.raises(Exception):
        GetCurrentTimeArgs(timezone="A" * 200)


def test_get_time_args_accepts_normal_timezone():
    from src.agent.tools.get_current_time import GetCurrentTimeArgs

    args = GetCurrentTimeArgs(timezone="America/New_York")
    assert args.timezone == "America/New_York"
