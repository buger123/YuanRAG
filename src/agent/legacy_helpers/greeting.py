"""Greeting pre-filter — single source of truth for "this is a
greeting, skip the LLM router".

Originally lived in ``nodes/route.py`` (DEPRECATED). Moved here
during the Phase-1 cleanup so the live ``intent_analysis`` import
points at a contract file rather than a module that also contains
a half-dead ``route_query_async`` entrypoint.

Why a regex pattern, not an LLM call
------------------------------------
This runs on every turn; an LLM here would add ~1 s of latency AND a
possible failure point. The regex is good enough for the high-signal
patterns the user actually writes ("hi" / "你好" / "你是谁" /
"thanks").

Edge cases
----------
* Empty / whitespace-only / punctuation-only → True.
* ``None`` → False (``None`` usually means upstream forgot to pull
  the query off the state — pretend-don't-pretend a None is a
  greeting would mask the upstream bug).
* A greeting followed by a real question
  (``你好,帮我把这份文档翻译成中文``) → False: the second half
  contains tokens the regex can't absorb.
"""
from __future__ import annotations

import re


_GREETING_PHRASES: tuple[str, ...] = (
    # Chinese greetings / identity / capability
    "你好", "您好", "嗨", "哈喽", "哈囉", "在吗",
    "你是谁", "您是谁", "你叫什么", "您叫什么",
    "你能做什么", "您能做什么", "你能帮我什么", "您能帮我什么",
    "你能干嘛", "您能干嘛", "你是干什么的", "您是干什么的",
    "谢谢", "感谢", "多谢", "辛苦了",
    "再见", "拜拜", "好的", "收到", "明白了", "ok", "okay",
    "早", "早安", "晚上好", "下午好",
    # English / mixed
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "thanks", "thank you", "cheers", "bye", "goodbye",
    "good morning", "good evening", "good afternoon", "see you",
    "who are you", "what can you do", "what are you",
)

# Cap the total message length we'll auto-classify as a greeting.
# "你好" alone is 2 chars; "你是谁 你是谁" is 7; even a generous
# emoji-laden "你好👋,在吗👀?" comes in under 30. Anything past 50
# chars is overwhelmingly NOT a pure greeting, and we'd rather let
# the LLM router see it than risk misrouting a real question.
_GREETING_MAX_LEN = 50

# Punctuation we treat as a delimiter. ASCII + fullwidth Chinese both
# included so ``who are you?`` and ``你好,`` split the same way.
_GREETING_PUNCT_CHARS = "。!?！？.,;；,、"

# Unicode codepoint ranges that cover the symbols we want to accept as
# greeting "gestures" — a thumbs-up or wave (``你好 👋``) is still a
# greeting even with emoji attached. The ranges are deliberately
# narrow-ish:
#
# * ``⌀-➿`` — Misc Technical (⌨⏰), Misc Symbols (☀⚡),
#   Dingbats (✂✈) and a chunk of "miscellaneous" codepages that
#   contain the visually-emoji-ish glyphs from the BMP.
# * ``\U0001F000-\U0001FFFF`` — the entire supplementary-plane range
#   that holds the modern emoji blocks (emoticons, transport, food,
#   activities, supplemental symbols & pictographs, etc.).
# * ``︀-️`` — variation selectors (skin-tone modifiers,
#   text-vs-emoji selectors) that ride along with the base emoji.
# * ``‍`` — zero-width joiner, used for compound emoji.
_GREETING_EMOJI_CLASS = (
    r"⌀-➿"
    r"\U0001F000-\U0001FFFF"
    r"︀-️"
    r"‍"
)


def _build_greeting_pattern() -> re.Pattern[str]:
    # Longest-first alternation so multi-word phrases ("who are you")
    # are tried before their sub-words ("who") would otherwise match.
    phrases_sorted = sorted(_GREETING_PHRASES, key=len, reverse=True)
    alts = "|".join(re.escape(p) for p in phrases_sorted)
    pattern = (
        r"^[\s" + re.escape(_GREETING_PUNCT_CHARS) + r"]*"
        r"(?:"
        + alts
        + r"|[" + _GREETING_EMOJI_CLASS + r"]+"
        + r"|\s+|[" + re.escape(_GREETING_PUNCT_CHARS) + r"]+"
        + r")+"
        r"[\s" + re.escape(_GREETING_PUNCT_CHARS) + r"]*$"
    )
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


_GREETING_MATCH_RE = _build_greeting_pattern()


def _is_obvious_greeting(query: str) -> bool:
    """Cheap pre-filter: does this query obviously not reference a doc?

    Returns True iff the message is short (``<= 50`` chars after
    strip) AND can be parsed as a sequence of greeting phrases /
    whitespace / punctuation — i.e. it contains nothing that looks
    like a real question, instruction, or reference to a document.

    Multi-word phrases (``who are you`` / ``good morning``) are
    matched as units — the regex tries the longest greeting
    phrase first so ``who are you?`` matches as
    ``[greeting] + [?]`` rather than ``[who] + [are] + [you?]``.
    """
    if query is None:
        return False
    q = query.strip()
    if not q:
        return True  # whitespace-only
    if len(q) > _GREETING_MAX_LEN:
        return False
    return _GREETING_MATCH_RE.match(q) is not None