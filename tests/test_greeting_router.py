"""Tests for ``_is_obvious_greeting`` pre-filter (v1.1.3).

The router pre-filter exists so a greeting query ("hi" / "你是谁" /
"thanks") skips the per-turn ``list_documents`` LanceDB scan AND the
LLM router call entirely, returning ``route_decision="direct"``
immediately. A false negative here is the bug that surfaces as
spurious citations on a doc-bearing thread (the LLM router sees
the greeting, sometimes misclassifies it as "retrieve" because
there's an uploaded PDF, and 8 chunks get attached to the answer).

These tests pin both halves of the contract:

1. Obvious greetings ARE classified as direct (positive tests).
2. Mixed messages (greeting + real question) ARE NOT (negative
   tests) — we'd rather over-fire on a real question than under-fire
   on a greeting.
"""
from __future__ import annotations


def _is_greeting(query: str) -> bool:
    from src.agent.legacy_helpers.greeting import _is_obvious_greeting

    return _is_obvious_greeting(query)


# ============================================================
# Positive — pure greeting phrases (the obvious fast path)
# ============================================================


def test_greeting_chinese_you_are_who():
    """``你是谁`` is the canonical identity question — direct."""
    assert _is_greeting("你是谁") is True


def test_greeting_chinese_what_can_you_do():
    assert _is_greeting("你能做什么") is True


def test_greeting_chinese_hi():
    assert _is_greeting("你好") is True


def test_greeting_english_hi():
    assert _is_greeting("hi") is True


def test_greeting_english_who_are_you():
    assert _is_greeting("who are you") is True


def test_greeting_thanks_english():
    assert _is_greeting("thanks") is True


def test_greeting_thanks_chinese():
    assert _is_greeting("谢谢") is True


def test_greeting_with_trailing_punctuation():
    """``?`` / ``!`` / ``.`` / ``。`` etc. — all allowed at the tail."""
    assert _is_greeting("你好?") is True
    assert _is_greeting("hi!") is True
    assert _is_greeting("你是谁?") is True
    assert _is_greeting("你是谁。") is True
    assert _is_greeting("who are you?") is True


def test_greeting_with_leading_trailing_whitespace():
    assert _is_greeting("  你好  ") is True


def test_greeting_case_insensitive():
    """English greetings are case-insensitive."""
    assert _is_greeting("HI") is True
    assert _is_greeting("Who Are You") is True
    assert _is_greeting("HELLO") is True


def test_greeting_empty_string():
    """An empty / whitespace-only message trivially counts as direct —
    the LLM router would just bounce on it anyway, but classifying
    here lets us skip the wasted LanceDB scan."""
    assert _is_greeting("") is True
    assert _is_greeting("   ") is True
    assert _is_greeting("。") is True


# ============================================================
# Positive — REPEATED / concatenated greetings (v1.1.3 regression)
# ============================================================


def test_greeting_chinese_you_are_who_repeated():
    """Regression for the v1.1.3 bug: the previous full-match regex
    rejected ``你是谁 你是谁`` because of the middle space + second
    greeting. Users do type this (curious / frustrated / hitting
    enter twice), and the resulting misroute surfaced 8 spurious
    citations on a doc-bearing thread."""
    assert _is_greeting("你是谁 你是谁") is True


def test_greeting_english_who_are_you_repeated():
    assert _is_greeting("who are you? who are you?") is True


def test_greeting_english_hi_repeated():
    assert _is_greeting("hi hi") is True


def test_greeting_chinese_hi_and_who():
    """A greeting + identity question in one message is still a
    greeting — the second segment is a recognized phrase."""
    assert _is_greeting("你好,你是谁") is True


def test_greeting_with_emoji_segment():
    """A pure emoji segment between greetings counts as a follow-up
    gesture, not a real question."""
    assert _is_greeting("你好 👋") is True
    assert _is_greeting("hi 👋") is True


# ============================================================
# Negative — must NOT be classified as direct (real questions)
# ============================================================


def test_real_question_with_you_in_it():
    """``你`` appears in many real questions — must not over-fire."""
    assert _is_greeting("你能帮我翻译这段话吗") is False
    assert _is_greeting("你的名字怎么读") is False


def test_real_question_after_greeting():
    """Greeting followed by an actual question: the question part
    is not a recognized phrase, so the whole thing falls through."""
    assert _is_greeting("你好,帮我把这个文档翻译成中文") is False
    assert _is_greeting("hi, can you summarize this PDF?") is False


def test_long_message_not_classified_as_greeting():
    """The 50-char length cap protects against weird over-firing
    on a long message that happens to mention a greeting phrase."""
    assert _is_greeting("你好" * 30) is False
    assert _is_greeting(
        "你好,我想问一下,关于这份合同里第三页的违约条款,如果对方在 30 天内未付款,我们可以采取哪些合法的催收手段?"
    ) is False


def test_document_reference_not_classified_as_greeting():
    """References to documents/files are explicitly NOT greetings."""
    assert _is_greeting("总结一下这个文档") is False
    assert _is_greeting("summarize this PDF") is False


def test_greeting_only_segment_mixed_with_code_like_token():
    """``你好 + 帮我写个python脚本`` — the second segment is not a
    greeting phrase, so the whole thing falls through to the LLM
    router (which will pick 'direct' or 'retrieve' on its merits)."""
    assert _is_greeting("你好,帮我写个 python 脚本") is False


def test_none_query():
    """Defensive: a None query must not crash the pre-filter."""
    assert _is_greeting(None) is False  # type: ignore[arg-type]


# ============================================================
# Boundary — punctuation-heavy inputs
# ============================================================


def test_greeting_only_punctuation_and_emoji():
    """Pure punctuation / emoji messages (no text) classify as
    greetings — better to silently treat them as conversational
    than to pay the LLM router cost for a one-keystroke input."""
    assert _is_greeting("???") is True
    assert _is_greeting("👋") is True
    assert _is_greeting("！！！") is True
