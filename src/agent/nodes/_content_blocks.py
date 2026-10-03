"""AIMessage content-block preservation for Anthropic extended thinking.

Item 7 Step 3 (v2.0.22 FSM cleanup) — centralizes the block-preservation
logic that decides whether the final ``AIMessage.content`` should be:

* the raw blocks list (when the LLM streamed a content-block list with
  a ``type == "text"`` block present);
* the blocks list with a synthetic trailing text block appended
  (thinking-only blocks — Anthropic requires ``thinking`` blocks to
  come BEFORE any ``text`` blocks, so we append at the end);
* the plain answer string (when the LLM streamed a string chunk, not
  a blocks list).

Pre-Step-3 this 8-line ``if/elif/else`` block was copy-pasted into
both ``fsm.react_generate_direct`` and ``nodes.react_generate``.
Drift was guaranteed the moment Anthropic added a new block type.

Why this matters:

* History replay reconstructs the AIMessage from the saved state and
  passes it back to the LLM on the next turn. If we drop the
  thinking blocks here, the model loses its extended-thinking state
  on the next call — degrades reasoning quality.
* If we append a text block BEFORE the thinking blocks instead of
  AFTER, Anthropic rejects the request at validation time.

The function is pure (no side effects, no I/O) — call sites just
pass in ``last_chunk`` (the final streaming chunk from
``model.astream``) and ``answer_text`` (the accumulated visible text).
"""
from __future__ import annotations

from typing import Any, Union


def preserve_content_blocks(
    last_chunk: Any,
    answer_text: str,
) -> Union[list, str]:
    """Return the ``content`` payload for the final AIMessage.

    Args:
        last_chunk: The final chunk from ``model.astream(...)``. May be
            ``None`` (e.g. the LLM raised before yielding any chunk).
            If present, its ``.content`` attribute may be a ``list`` of
            blocks (Anthropic extended-thinking format) or a plain
            string.
        answer_text: The accumulated visible answer text from earlier
            chunks. Used as the synthetic text block when ``last_chunk``
            has thinking-only blocks.

    Returns:
        Either the raw ``list[dict]`` of blocks, or a plain ``str``
        — exactly what ``AIMessage(content=...)`` accepts.
    """
    if last_chunk is not None and isinstance(getattr(last_chunk, "content", None), list):
        blocks = last_chunk.content
        has_text = any(
            isinstance(b, dict) and b.get("type") == "text" for b in blocks
        )
        if has_text:
            return blocks
        # Thinking-only blocks → append a synthetic text block at the
        # END so history replay can extract visible text AND
        # Anthropic's ordering requirement (thinking before text) is
        # preserved.
        return list(blocks) + [{"type": "text", "text": answer_text}]
    return answer_text


__all__ = ["preserve_content_blocks"]
