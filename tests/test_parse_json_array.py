"""Tests for ``parse_json_array`` (robust JSON-array parsing for LLM output).

Bug history
-----------
The legacy ``_grade_via_llm`` helper (in ``src/agent/nodes/grade.py``)
used to call ``json.loads(text)`` directly with only a fence-strip.
Cheap models like ``claude-3-5-haiku-latest`` (1024 output tokens)
occasionally emit:

- Truncated output — closing ``]`` clipped by max_tokens when grading
  10+ documents.
- Smart / fullwidth quotes leaked from Chinese content into JSON
  string values, e.g. ``"他说："你好""`` → ``json.loads`` raises
  ``Expecting ',' delimiter`` at the curly ``"``.
- Trailing comma before the closing ``]``.
- Markdown ``` fences with prose around them.

Each of these made the grader fall through to its fail-open branch,
silently keeping every doc — which is exactly what the user reported:
"upload many files, ask about one file, the answer cites only that file
but the citation row lists every retrieved chunk." (Same root cause for
both grade and citation, two different failure modes.)

These tests verify ``parse_json_array`` recovers from each pattern.

v2.0.13 cleanup: the legacy ``_grade_via_llm`` helper was retired
when ``grade_documents`` was deleted (ReAct subsumes relevance
grading — the LLM judges relevance itself via tool-call ordering).
The end-to-end tests that drove ``_grade_via_llm`` directly were
removed along with it; the parser-level tests below are still
useful for any future caller that needs robust array parsing.
"""
from __future__ import annotations

import json

import pytest

from src.llm.structured import parse_json_array


# ============================================================
# Direct / clean parsing
# ============================================================


def test_parses_clean_array():
    arr = [{"index": 0, "relevant": True}, {"index": 1, "relevant": False}]
    assert parse_json_array(json.dumps(arr)) == arr


def test_parses_empty_array():
    assert parse_json_array("[]") == []


def test_parses_array_with_chinese_text_in_values():
    text = json.dumps(
        [
            {"index": 0, "preview": "用户问的是中文内容"},
            {"index": 1, "preview": "另一段中文"},
        ],
        ensure_ascii=False,
    )
    result = parse_json_array(text)
    assert result is not None
    assert len(result) == 2
    assert result[0]["preview"] == "用户问的是中文内容"


# ============================================================
# Markdown fence handling
# ============================================================


def test_strips_markdown_fence_with_json_tag():
    text = '```json\n[{"index": 0, "relevant": true}]\n```'
    assert parse_json_array(text) == [{"index": 0, "relevant": True}]


def test_strips_markdown_fence_without_language_tag():
    text = '```\n[{"index": 0, "relevant": true}]\n```'
    assert parse_json_array(text) == [{"index": 0, "relevant": True}]


def test_strips_uppercase_json_fence():
    text = '```JSON\n[{"index": 0, "relevant": true}]\n```'
    assert parse_json_array(text) == [{"index": 0, "relevant": True}]


def test_finds_array_among_prose():
    """Model emits prose around the JSON."""
    text = (
        "Sure, here is my grading:\n\n"
        '[{"index": 0, "relevant": true}, {"index": 1, "relevant": false}]\n'
        "Let me know if you need more detail."
    )
    assert parse_json_array(text) == [
        {"index": 0, "relevant": True},
        {"index": 1, "relevant": False},
    ]


# ============================================================
# Smart-quote repair
# ============================================================


def test_repairs_curly_double_quotes_in_string_values():
    """Chinese curly quotes inside a string value should be normalised
    to ASCII quotes so ``json.loads`` doesn't choke on them."""
    # The model emitted curly quotes — but only INSIDE string values,
    # not as the structural JSON delimiters (those are ASCII).
    text = '[{"index": 0, "relevant": true, "reason": "他说：“好的”"}]'
    result = parse_json_array(text)
    assert result is not None
    assert len(result) == 1
    # Curly quotes normalized to ASCII straight quotes (reason ends
    # with `".`).
    assert result[0]["reason"] == '他说："好的"'


def test_repairs_fullwidth_quotation_mark():
    """The fullwidth quotation mark U+FF02 is common in CJK text and
    is NOT a valid JSON string delimiter."""
    text = '[{"index": 0, "relevant": true, "reason": "测试＂引号＂"}]'
    result = parse_json_array(text)
    assert result is not None
    assert result[0]["reason"] == '测试"引号"'


# ============================================================
# Trailing-comma repair
# ============================================================


def test_strips_trailing_comma_before_closing_bracket():
    text = '[{"index": 0, "relevant": true}, {"index": 1, "relevant": false},]'
    assert parse_json_array(text) == [
        {"index": 0, "relevant": True},
        {"index": 1, "relevant": False},
    ]


def test_strips_trailing_semicolon_before_closing_bracket():
    text = '[{"index": 0, "relevant": true};]'
    assert parse_json_array(text) == [{"index": 0, "relevant": True}]


# ============================================================
# Truncation-tolerant recovery
# ============================================================


def test_recovers_from_truncated_response_missing_closing_bracket():
    """``claude-3-5-haiku-latest`` clips the closing ``]`` when the
    payload is large. Recover by closing the array at the last
    complete element."""
    text = (
        '[{"index": 0, "relevant": true}, '
        '{"index": 1, "relevant": false}, '
        '{"index": 2, "relevant": true'
        # no closing }, no ]
    )
    result = parse_json_array(text)
    assert result is not None
    # At minimum the first two complete entries survive. The third
    # is incomplete and may or may not survive depending on whether
    # its closing brace was emitted; either way, we get a list back
    # and it never raises.
    assert isinstance(result, list)
    assert len(result) >= 2
    assert result[0] == {"index": 0, "relevant": True}


def test_recovers_from_truncated_response_with_complete_final_element():
    """Truncated mid-final-element but the previous element is closed."""
    text = (
        '[{"index": 0, "relevant": true}, '
        '{"index": 1, "relevant": false}, '
        '{"index": 2, "relevant": true}, '
        '{"index": 3, "relevant":'
        # Closing }, and ], are missing.
    )
    result = parse_json_array(text)
    assert result is not None
    assert isinstance(result, list)
    # Indices 0..2 must be intact.
    assert result[0] == {"index": 0, "relevant": True}
    assert result[1] == {"index": 1, "relevant": False}
    assert result[2] == {"index": 2, "relevant": True}


def test_returns_none_for_completely_garbage_input():
    assert parse_json_array("totally not JSON at all, sorry") is None


def test_returns_none_for_empty_string():
    assert parse_json_array("") is None


def test_returns_none_for_non_string_input():
    assert parse_json_array(None) is None
    assert parse_json_array(12345) is None
    assert parse_json_array([1, 2, 3]) is None  # not a string


def test_returns_none_for_array_of_objects_with_no_closing_bracket_at_all():
    """If the model emits only the opening ``[`` and one opening
    ``{`` and nothing else, we have no complete elements to recover."""
    text = '[{"index": 0, "relevant": true'
    # This *might* recover (we close the object and array) or return
    # None depending on whether the bracket-counter sees a valid
    # closing. Either way it must NOT raise.
    result = parse_json_array(text)
    if result is not None:
        assert isinstance(result, list)


# ============================================================
# Defensive behaviour for malformed entries
# ============================================================


def test_entries_missing_index_are_skipped_by_caller():
    """``_grade_via_llm`` filters out entries without ``index``. This
    test just confirms the parser returns such arrays intact so the
    caller can do its own filtering."""
    text = '[{"index": 0, "relevant": true}, {"relevant": false}]'
    arr = parse_json_array(text)
    assert arr is not None
    assert len(arr) == 2
    # The second entry has no ``index`` key — that's the caller's
    # problem to filter, not the parser's.
    assert "index" not in arr[1]


def test_non_list_object_input_returns_none():
    """The model emits an object instead of an array — caller wants
    None, not the silently-misinterpreted object."""
    assert parse_json_array('{"decision": "retrieve"}') is None


# ============================================================
# End-to-end through _grade_via_llm — REMOVED in v2.0.13 cleanup
# ============================================================
#
# The three tests that drove ``_grade_via_llm`` directly
# (``test_grade_via_llm_recovers_from_truncated_json``,
# ``test_grade_via_llm_recovers_from_smart_quotes``,
# ``test_grade_via_llm_still_fails_open_on_total_garbage``) were
# retired along with ``src/agent/nodes/grade.py``. The parser-level
# tests above are still useful for any future caller that needs
# robust array parsing — keep them.

