"""Shared utility for splitting an AIMessage.content into (text, reasoning).

Lives at the agent package root (not under ``nodes/``) because BOTH
``src.agent.runner`` and ``src.agent.nodes.generate`` need it, and the
two modules cannot import from each other without creating a circular
import (the runner imports the graph, which imports the generate node).

LangChain's ``AIMessageChunk.content`` is normally a string, but the
Anthropic provider emits **structured content blocks** for tool-use
inputs (e.g. ``[{"text": "...", "type": "text"}]``) AND for extended
thinking (e.g. ``[{"type": "thinking", "thinking": "..."}]``). When a
raw list leaks through to the React frontend as a WebSocket ``token``
event, the page tries to render the object directly and React throws
``Minified React error #31: Objects are not valid as a React child``.

The two return values are split so callers can yield them as distinct
events (``token`` vs ``reasoning``) and persist them on the message
separately. Block-by-block policy:

* ``{"type": "text"}``                       → ``text``
* ``{"type": "thinking", "thinking": ...}``  → ``reasoning`` (only when the
  delta carries non-empty content; signature-only deltas are skipped)
* ``{"type": "redacted_thinking"}``          → ``reasoning`` (placeholder
  text — the underlying bytes are encrypted and not recoverable)
* ``{"type": "reasoning", "summary": [...]}``→ ``reasoning`` (OpenAI Responses
  API path — future-proofing; not currently routed to by ``ChatOpenAI``
  in this app)
* ``tool_use`` / ``input_json_delta``        → dropped (route-decision JSON
  is parsed server-side via ``parse_structured``)
"""
from __future__ import annotations

import re
from typing import Any


# v1.1.11 — defense-in-depth against leaked tool_call / tool_result
# XML literals. Today no path in this project produces these strings
# (the LLM is never given ``bind_tools``, see ``src/llm/factory.py``;
# the runner emits no ``tool_call`` / ``tool_result`` event types per
# ``src/agent/events.py``) — so this strip is purely a safety net for
# future code paths, upstream provider regressions, or middleware
# injections that emit these tags as ordinary text content.
#
# The regex matches BOTH forms:
#   * self-closing:  ``<tool_call ... />``
#   * paired:        ``<tool_call ...>...</tool_call>``
# It uses a backreference to ensure the opening and closing tags have
# the same name (``tool_call`` or ``tool_result``). A lone opening tag
# without a matching close does NOT match — we do NOT want to mangle
# user-visible text that happens to contain the literal string
# ``<tool_call>`` (e.g. a code example mentioning it).
_RE_TOOL_BLOCK = re.compile(
    r"<\s*(?P<tag>tool_call|tool_result)\b[^>]*?(?:/>|>.*?<\s*/\s*(?P=tag)\s*>)",
    re.DOTALL | re.IGNORECASE,
)


# v2.0.4 — defense-in-depth against MiniMax-style leaked tool-call
# wire syntax. The MiniMax / DeepSeek / OpenAI o1 tool-call wire format
# emits tokens like ``]<minimax[>[<invoke name="web_search">``,
# ``<invoke name="...">``, ``<parameter name="query">``,
# ``</parameter>``, ``</invoke>``, ``<function_calls>`` etc. into the
# ``text`` channel of the answer stream. These are NOT paired XML
# (so ``_RE_TOOL_BLOCK`` above misses them), and they often arrive
# split across many 1-5 char deltas (so a single regex sweep on the
# final string misses them too — a 20-char leak delivered as 4 chunks
# looks like 4 innocuous strings individually).
#
# We do TWO things here:
#
#   1. ``_RE_LEAKED_TOOL_SYNTAX`` — a handful of regexes that match
#      the OBSERVED leak shapes. Each regex is run on the assembled
#      text after stripping paired tool_call/tool_result blocks.
#
#   2. ``stream_safe_chunk(text, buffer)`` — a stateful helper that
#      the runner uses on every streaming chunk. It maintains a small
#      trailing buffer (default 64 chars) and emits a *cleaned* chunk
#      string for the wire, holding back the final ~buffer_len chars
#      so a leak that arrives split across two chunks still gets
#      caught. Returns ``(cleaned_chunk, new_buffer)`` so the caller
#      can thread the buffer between calls.
#
# Why the stateful buffer is load-bearing
# ---------------------------------------
# Without it, a leak like ``]<minimax[>[<invoke name="web_se`` arriving
# as one chunk followed by ``arch">`` arriving as the next will see
# the first chunk regex'd clean (the full keyword isn't present yet)
# and the second chunk regex'd clean (the leading bracket is gone).
# Together they reconstruct the full leak on the client. The buffer
# holds back the trailing portion so the regex always sees a window
# that contains the complete keyword.
_RE_INVOKE_FRAGMENT = re.compile(
    # ``]<minimax[>[<invoke name="...">`` — the leaked prefix is
    # ``]<[A-Za-z0-9_-]+[?>`` then ``[<invoke ...>``. Note the inner
    # ``[`` (the original shape is ``]<minimax[>[<invoke`` — the
    # ``[`` comes BEFORE the ``>``), so we can't use a simple
    # ``<...>`` non-greedy match; we explicitly allow ``[`` to
    # appear in the tag body.
    #
    # v2.0.10.2 — make the closing ``>`` of the ``<invoke...>``
    # block OPTIONAL via ``(?:[^>]*>)?``. The original required a
    # complete ``<invoke...>`` tag, but a leak that's split across
    # two streaming chunks at exactly that boundary (chunk 1 =
    # ``<invoke name="web_se``, chunk 2 = ``arch">more text
    # after``) survived the per-chunk strip — the regex on chunk 1
    # didn't see the closing ``>``, the regex on chunk 2 didn't see
    # the opening ``<invoke``. With the optional closing ``>``, the
    # partial ``<invoke`` keyword + opening quote
    # (``<invoke name="...``) is enough to identify the leak on
    # whichever chunk it arrives in. The full leak shape
    # ``<invoke name="web_search">`` is still consumed in full
    # (the optional group matches the trailing ``>``).
    r"\]\s*<\s*[^>]{1,30}\s*>\s*\[\s*<\s*invoke\b(?:[^>]*>)?",
    re.DOTALL,
)
# v2.0.10.2 — truncated-prefix catch. v2.0.4's
# ``_RE_INVOKE_FRAGMENT`` only fires when the leak arrives as a
# FULL shape including ``<invoke`` after the prefix. When the
# stream is truncated mid-leak (e.g. the LLM cuts off right
# after the prefix) or when the prefix arrives split across
# chunks without ``<invoke`` ever being emitted in the same
# match window, the prefix survives cleanup and leaks to the
# user as visible text — the production-reported shape is
# literally ``]<]minimax>[`` (zero-width space between ``]``
# and ``minimax``, ``>`` before ``[``). Observed patterns:
#
#   * ``]<minimax[>``        (original documented shape)
#   * ``]<minimax]>``        (variant with ``]`` before ``>``)
#   * ``]<]minimax>[``       (production 2026-09-13 — extra
#                              ``]`` between ``<`` and identifier,
#                              ``>`` before ``[``)
#
# Body requirements (chosen to limit false positives in
# legitimate text):
#   * MUST start with a letter (rules out ``]<5>`` etc.)
#   * MUST contain at least 2 letters total (rules out
#     ``]<__>`` / ``]<-_>`` etc. — identifier names are
#     always alphabetic-leading)
#   * body is 1-30 chars total (model/provider identifiers
#     are short — `minimax` is 7)
#   * zero-width space ``​`` allowed in body to absorb
#     tokenizer artifacts that have been observed
#     interleaved between the closing ``]`` and the
#     identifier (the production case)
#   * optional trailing ``[`` (matches the trailing ``[``
#     of the full leak shape — some truncations include
#     it, some don't)
_RE_LEAKED_WIRE_PREFIX = re.compile(
    r"\]\s*<\s*\]?[\s​]*"
    r"[a-zA-Z]"
    r"(?:[\w​-]*[a-zA-Z])?"
    r"[\s​]*"
    r"[\]\[]?"
    r"[\s​]*"
    r">"
    r"[\s​]*"
    r"\[?",
    re.DOTALL,
)
_RE_INVOKE_TAG = re.compile(
    # ``<invoke name="...">`` or ``</invoke>``
    r"<\s*/?\s*invoke\b[^>]*>",
    re.DOTALL,
)
_RE_PARAMETER_TAG = re.compile(
    # ``<parameter name="...">`` or ``</parameter>``
    r"<\s*/?\s*parameter\b[^>]*>",
    re.DOTALL,
)
_RE_FUNCTION_CALLS = re.compile(
    # ``<function_calls>`` / ``<function_calls ...>`` / ``</function_calls>``
    r"<\s*/?\s*function_calls?\b[^>]*>",
    re.DOTALL,
)
_RE_ANTML_FENCE = re.compile(
    # ``<antml:invoke ...>...`` (Anthropic XML-mode variants)
    r"<\s*antml\s*:\s*[a-zA-Z]+[^>]*>",
    re.DOTALL,
)

_LEAK_REGEXES = (
    _RE_INVOKE_FRAGMENT,
    _RE_LEAKED_WIRE_PREFIX,
    _RE_INVOKE_TAG,
    _RE_PARAMETER_TAG,
    _RE_FUNCTION_CALLS,
    _RE_ANTML_FENCE,
)

# How much trailing text the streaming buffer holds back. Must be
# larger than the longest single leak keyword
# (``"function_calls"`` = 15 chars) so the regex on the held-back
# slice has a chance to match. 32 chars gives 2x headroom over the
# longest keyword AND is small enough that the user-perceived "first
# character delay" is <50 ms even at the slowest token rates we've
# seen (~5 tok/s). v2.0.10.2 — was 64; reduced because the old
# "hold ALL back if combined <= _STREAM_BUFFER_CHARS" rule broke
# incremental emission of short answers (test_runner_stream_dedup
# expects 4 separate tokens for a 17-char greeting; with 64 the
# buffer held the entire greeting back and emitted it as one
# chunk on end-of-stream flush).
_STREAM_BUFFER_CHARS = 32


def _strip_leaked_tool_syntax(text: str) -> str:
    """Strip MiniMax-style leaked tool-call wire syntax from ``text``.

    Removes the OBSERVED leak shapes
    (``]<minimax[>[<invoke name="...">``, ``<invoke ...>``,
    ``<parameter ...>``, ``<function_calls>``, ``<antml:...>``). Does
    NOT touch ordinary ``<b>`` / ``<i>`` / ``<think>`` style markup
    — those are part of the legitimate response surface.

    Whitespace-collapsing is intentionally NOT done here (unlike
    ``_strip_tool_blocks``) because the leaked fragments tend to be
    space-less; collapsing would change nothing visible to the user.
    """
    if not text:
        return text
    for rx in _LEAK_REGEXES:
        text = rx.sub(" ", text)
    return text


def stream_safe_chunk(text: str, buffer: str) -> tuple[str, str]:
    """Strip leaked tool-call syntax in a streaming-friendly way.

    The runner calls this on every text chunk before yielding it to
    the WebSocket. ``buffer`` carries the trailing text from the
    previous call so a leak split across chunk boundaries can still
    be caught by the regex.

    Returns ``(cleaned_yield, new_buffer)``. The caller emits
    ``cleaned_yield`` immediately and threads ``new_buffer`` into
    the next call. ``new_buffer`` is at most ``_STREAM_BUFFER_CHARS``
    long (the held-back tail that hasn't been regex-checked yet).

    State machine:
      * Concatenate ``buffer + text`` → ``combined``.
      * Run ``_strip_leaked_tool_syntax`` on ``combined``.
      * Emit the first ``len(combined) - _STREAM_BUFFER_CHARS``
        characters of the cleaned text (the "safe" prefix).
      * Hold back the last ``_STREAM_BUFFER_CHARS`` characters as
        the new buffer.

    Edge cases:
      * ``text`` empty → return ``("", buffer)`` unchanged so the
        caller's loop stays simple.
      * ``combined`` shorter than ``_STREAM_BUFFER_CHARS`` → no
        emission (all of it goes into ``new_buffer``). This is the
        initial-state behavior; the first real chunk will emit
        whatever's been accumulated.
      * Cleanup removes the entire combined string → emit ``""`` and
        reset buffer to ``""``.
    """
    if not text:
        return "", buffer
    combined = buffer + text
    cleaned = _strip_leaked_tool_syntax(combined)
    if len(cleaned) <= _STREAM_BUFFER_CHARS:
        # v2.0.10.2 — short enough that the buffer isn't needed:
        # emit everything and clear the buffer. The regex has had
        # a chance to match on the full slice and there's no
        # trailing context to wait for. The v2.0.4 "hold all back"
        # rule broke incremental emission of short answers
        # (test_runner_stream_dedup expects 4 separate tokens for
        # a 17-char greeting; with the old 64-char threshold, the
        # entire greeting was held back and emitted as one chunk
        # at end-of-stream flush).
        return cleaned, ""
    emit_len = len(cleaned) - _STREAM_BUFFER_CHARS
    emit = cleaned[:emit_len]
    new_buffer = cleaned[emit_len:]
    return emit, new_buffer


def _strip_tool_blocks(text: str) -> str:
    """Strip any ``<tool_call ...>...</tool_call>`` /
    ``<tool_result ...>...</tool_result>`` blocks from ``text``.

    Returns ``text`` unchanged when no paired block matches. The
    strip is whitespace-collapsing around the removed block so
    adjacent text doesn't end up with a double-space artifact.

    v2.0.4 — also strips MiniMax-style leaked tool-call wire
    fragments (see ``_strip_leaked_tool_syntax`` above) BEFORE the
    paired-block pass, so a chunk containing both ``<tool_call>``
    blocks AND ``<invoke>`` fragments loses everything harmful.
    """
    if not text:
        return text
    cleaned = _strip_leaked_tool_syntax(text)
    cleaned = _RE_TOOL_BLOCK.sub(" ", cleaned)
    # Collapse runs of whitespace introduced by the substitution so
    # ``"hello <tool_call>...</tool_call> world"`` becomes
    # ``"hello world"``, not ``"hello   world"``.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip() if cleaned == text.strip() else cleaned


def split_text_and_thinking(content: Any) -> tuple[str, str]:
    """Normalize an LLM chunk's ``content`` into ``(text, reasoning)``.

    Inputs:

    * ``str``   → returned verbatim as ``text``; ``reasoning`` empty.
    * ``list``  → walked block-by-block using the policy in the
      module docstring.
    * anything else (None, dict, …) → ``("", "")``.

    v1.1.11 — defense-in-depth strip applied to the returned ``text``
    on both paths (string passthrough AND assembled list blocks). The
    strip removes any ``<tool_call ...>...</tool_call>`` /
    ``<tool_result ...>...</tool_result>`` XML literal that a future
    provider / middleware might leak as ordinary text content. The
    regex requires paired tags (self-closing or open+close), so a
    lone ``<tool_call>`` substring in legitimate prose is preserved.
    """
    if isinstance(content, str):
        return _strip_tool_blocks(content), ""
    if not isinstance(content, list):
        return "", ""
    text_parts: list[str] = []
    think_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            # Skip signature-only deltas ({"type": "thinking", "thinking": "",
            # "signature": "..."}) so we don't emit empty reasoning events.
            t = block.get("thinking")
            if t:
                think_parts.append(t)
        elif btype == "redacted_thinking":
            # The raw bytes are encrypted and only Anthropic can decode them.
            # Surface a short marker so the user still sees that the model
            # thought, even if it can't show what.
            think_parts.append("[thinking redacted]")
        elif btype == "reasoning":
            # OpenAI Responses API: {"type": "reasoning", "summary": [{"type":
            # "summary_text", "text": "..."}, ...]}. Today this path isn't
            # routed to by ChatOpenAI in this app, but the SDK may start
            # routing here in future versions — keep us covered.
            for s in block.get("summary") or []:
                if isinstance(s, dict):
                    st = s.get("text")
                    if st:
                        think_parts.append(st)
        elif (btype in ("text", None)) and "text" in block:
            text_parts.append(str(block["text"]))
        # tool_use / input_json_delta — silently dropped (route-decision JSON)
    text = "".join(text_parts)
    return _strip_tool_blocks(text), "".join(think_parts)


def extract_text(content: Any) -> str:
    """Text-only convenience wrapper around :func:`split_text_and_thinking`."""
    text, _ = split_text_and_thinking(content)
    return text


__all__ = ["split_text_and_thinking", "extract_text", "_strip_tool_blocks"]
