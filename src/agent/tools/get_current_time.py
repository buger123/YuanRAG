"""get_current_time tool — wall-clock lookup in IANA timezones.

Closes the "hallucinated date" failure mode for questions like
"今天天气怎么样" / "昨天上证指数收盘多少" / "what day is today" /
"latest news this week" — without an authoritative clock, the LLM
either guesses a date or invents plausible-sounding recent events.
Returning ``datetime.now(ZoneInfo(tz)).isoformat()`` from the server
gives the LLM a single canonical timestamp per tool call (no
ambiguity about whose "now" it is — the user's local clock may
drift, the server's clock is NTP-synced).

Why a string return (not a Document)
------------------------------------
The time tool returns plain text. Unlike ``retrieve_docs`` /
``web_search`` (which return documents the LLM cites via ``[n]``
markers), time is reference data — there's no notion of "source
[1]" for the current second. Returning a string keeps the
``ToolMessage.content`` short and the LLM context minimal.

Output format
-------------
* ISO 8601 with timezone offset (the canonical, machine-parseable
  form).
* Human-readable Chinese line ("2026年9月6日 周日 14:23") so the LLM
  can refer to the date naturally in its reply.
* Both forms on one newline so the LLM picks whichever fits the
  surrounding context.

Error handling
--------------
Unknown timezone → fall back to UTC rather than raising. A botched
``get_current_time(tz="Mars/Olympus_Mons")`` would otherwise kill
the ReAct loop. ``ZoneInfo("UTC")`` is always available; we wrap
the user's request in try/except for any other IANA zone.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.core.logging import logger


class GetCurrentTimeArgs(BaseModel):
    """Pydantic schema for ``get_current_time`` arguments."""

    timezone: str = Field(
        default="Asia/Shanghai",
        max_length=64,
        description=(
            "IANA timezone name, e.g. 'Asia/Shanghai', 'America/"
            "New_York', 'Europe/London', 'UTC'. Default 'Asia/Shanghai' "
            "(China; the project's primary user base). Falls back to "
            "UTC on unknown timezones."
        ),
    )


# Chinese weekday names — used for the human-readable output line.
# Index by ``datetime.weekday()`` (Monday=0, Sunday=6).
_WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _format_clock(tz_name: str) -> str:
    """Return ``"YYYY-MM-DDTHH:MM:SS+TZ\n<chinese human-readable>"``."""
    try:
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
    except (ZoneInfoNotFoundError, ValueError):
        logger.debug(
            f"get_current_time: unknown timezone {tz_name!r}, falling back to UTC"
        )
        tz = ZoneInfo("UTC")
        now = datetime.now(tz)
        tz_name = "UTC"

    iso = now.isoformat()
    zh = f"{now.year}年{now.month}月{now.day}日 {_WEEKDAY_ZH[now.weekday()]} {now.hour:02d}:{now.minute:02d}"
    return f"{iso} ({tz_name})\n{zh}"


@tool("get_current_time", args_schema=GetCurrentTimeArgs)
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """Return the current date and time in the specified timezone.

    Use this when the question asks about 'today', 'yesterday',
    'tomorrow', 'now', 'this week', or any time-sensitive reference
    that needs the actual current date — don't guess or compute the
    date yourself; the server's clock is authoritative.

    Returns ISO 8601 with timezone offset + a Chinese human-readable
    line, e.g.:
      2026-09-06T14:23:11+08:00 (Asia/Shanghai)
      2026年9月6日 周日 14:23

    The tool's server clock is NTP-synced — do NOT rely on
    ``datetime.now()`` in your own reasoning; trust this tool's
    output as the single source of truth for "now".
    """
    return _format_clock(timezone)


__all__ = ["get_current_time", "GetCurrentTimeArgs"]