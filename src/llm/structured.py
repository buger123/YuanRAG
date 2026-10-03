"""Helpers for parsing structured outputs from chat models.

Why this module exists
----------------------
LangChain's ``model.with_structured_output(Schema)`` uses **tool calling**
under the hood: the schema is converted to a tool the model is expected
to invoke. If the model refuses to call the tool (some proxies — e.g.
the MiniMax Anthropic-compatible endpoint — return ``None`` in this case
even when the model emitted valid JSON in its text), the wrapper yields
``None`` rather than the parsed object.

We saw this in production: ``route_query`` would log
``structured output returned None`` and fall back to a hard-coded
``"retrieve"`` decision, even for greetings like "你能做什么?". That's
wrong — a "what can you do?" question must take the direct
conversational path.

This module offers:

1. ``parse_json_object`` — pure string parser that strips markdown
   fences, finds the first JSON object in the response, and returns a
   dict. Useful when we want to call the model with a plain-JSON
   system prompt instead of relying on tool calling.

2. ``parse_json_array`` — same idea for JSON arrays. Used by the
   grader (``grade_documents``) where the model emits ``[{...}, ...]``
   with one entry per doc. Includes a JSON-repair step that handles
   the cheap model emitting smart quotes, trailing commas, or
   truncated output (e.g. ``haiku``-class models occasionally clip
   the closing ``]`` when many docs are graded at once).

3. ``parse_structured`` — convenience wrapper that validates the
   parsed dict against a Pydantic schema.

4. ``invoke_structured_with_fallback`` — runs both strategies back
   to back (``with_structured_output`` first, then plain-text +
   ``parse_structured``). Used by every node that needs a parsed
   ``BaseModel`` and would otherwise duplicate the same
   try/except boilerplate.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, TypeVar, Union

import anthropic
import httpx
import openai
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.logging import logger
from src.security.prompt_safety import redact_secrets


T = TypeVar("T", bound=BaseModel)


# Common markdown-fence pattern. We try the strictest patterns first
# and fall back to "find first brace pair" as a last resort.
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL
)

# Smart / curly quotes that the cheap model occasionally emits when
# writing JSON containing Chinese text. These are valid in JavaScript
# string literals but are NOT valid JSON — stdlib ``json.loads``
# rejects them with ``Expecting value`` / ``Expecting ',' delimiter``
# at the offending character. Normalize them before parsing.
_SMART_QUOTES = {
    "“": '"',  # left double curly
    "”": '"',  # right double curly
    "‘": "'",  # left single curly
    "’": "'",  # right single curly
    "＂": '"',  # fullwidth quotation mark (U+FF02 — common in CJK text)
}


def _strip_fences(text: str) -> str:
    """Strip a single leading ```` ```json ```` (or ````` ``` ````)
    fence from the response if the whole response is one fenced block."""
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def _repair_json(text: str) -> str:
    """Best-effort JSON repair for the cheap model's typical mistakes.

    1. Smart / fullwidth quotes → ASCII straight quotes. Cheap models
       occasionally emit JSON where curly quotes are used as the
       structural delimiters (``[{“index”: 0}]``); we substitute
       those in-place. Where curly quotes appear INSIDE an
       ASCII-delimited string value, we instead escape them as
       ``\\"`` so the structural quoting is preserved.
    2. ``,`` or ``;`` immediately before ``]`` or ``}`` (trailing
       punctuation) → dropped. Cheap models do this regularly when
       the LLM completes an array element then immediately closes.

    The string-aware step is the reason a naïve ``str.replace`` is
    not enough: the same character (curly ``"``) can be structural
    in one position and content in another, and treating them
    uniformly would over-quote the input. The scan walks the string
    in lockstep with the JSON parser's string-state machine.
    """
    # First pass: if every curly quote is at a structural position,
    # straight substitution is enough. If any curly quote appears
    # INSIDE an ASCII-delimited string, that string needs escaping.
    text = _escape_smart_quotes_in_strings(text)
    # Trailing comma / semicolon before closing bracket.
    text = re.sub(r",(\s*[\]}])", r"\1", text)
    text = re.sub(r";(\s*[\]}])", r"\1", text)
    return text


def _escape_smart_quotes_in_strings(text: str) -> str:
    """Walk the text in JSON-parser-string-state; whenever we're
    inside an ASCII-``"``-delimited string, replace any curly / fullwidth
    quote with an escaped ASCII quote (``\\"``).

    Outside of strings (structural positions), curly quotes are left
    alone so the parser can try them as ASCII delimiters later — most
    cases that come through here either contain zero curly quotes
    (the substitute-then-parse path handles them) or all curly quotes
    are at structural positions (the same path handles them by
    substituting ``"`` for ``"`` wholesale in the caller).

    The mixed case — ASCII structural, curly content — is the one
    this function fixes. The walking algorithm mirrors the parser's
    own state machine: enter string at unescaped ``"``, exit at the
    next unescaped ``"``, honour ``\\`` escapes.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            # Possible string-open or string-close. We don't actually
            # know which without proper parsing; do a best-effort
            # walk and assume ``"`` always opens (the cheap model
            # rarely writes structural JSON with unescaped ``"`` at
            # the wrong depth, and if it does, the trailing parse
            # will simply fail and we'll surface ``None``).
            out.append('"')
            i += 1
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    # Honour existing escape.
                    out.append(text[i : i + 2])
                    i += 2
                    continue
                if c == '"':
                    # String close.
                    out.append('"')
                    i += 1
                    break
                if c in _SMART_QUOTES:
                    # Curly quote INSIDE an ASCII-delimited string —
                    # escape it so the parser doesn't terminate the
                    # string prematurely.
                    out.append('\\"')
                    i += 1
                    continue
                out.append(c)
                i += 1
        elif ch in _SMART_QUOTES:
            # Curly quote at a structural position (no surrounding
            # ASCII string). Substitute to ASCII straight quote so
            # it acts as a delimiter.
            out.append('"')
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_json_object(content: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object from model output. Returns ``None`` on failure.

    Handles:
      * Markdown ```json ... ``` fences (with or without ``json`` tag)
      * Leading prose ("Sure, here's the answer: {...}")
      * Whitespace and newlines around the JSON
      * Smart / fullwidth quotes via ``_repair_json``

    Does NOT require the response to be a bare JSON object — the
    function locates the first top-level ``{...}`` pair, so a response
    like ``Reasoning: ... {"decision": "direct"}`` still parses.
    """
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None

    text = _strip_fences(text)
    text = _repair_json(text)

    # 1) Direct parse — strictest, fastest path
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) Find the first top-level JSON object in the response. We
    # walk the string with a brace-depth counter that respects string
    # literals (and backslash escapes) so we don't get confused by
    # braces inside strings.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                candidate = _repair_json(candidate)
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def parse_json_array(content: str) -> Optional[list[Any]]:
    """Parse a JSON array from model output. Returns ``None`` on failure.

    Same robustness as :func:`parse_json_object` but targets a JSON
    array — the shape used by ``GRADE_SYSTEM`` (``[{index, relevant,
    reason}, ...]``) and ``WEB_GRADE_SYSTEM``.

    Failure modes seen in production:
    - The cheap ``claude-3-5-haiku-latest`` (1024 output tokens) can
      truncate the closing ``]`` when grading 10+ documents. We
      attempt a best-effort truncation-tolerant recovery: locate the
      last complete element (``}`` at depth 1) and close the array
      there.
    - Smart quotes / fullwidth quotes inside Chinese content.
    - Trailing comma before the closing ``]``.
    - Model wraps the response in markdown ``` fences with a language
      tag we don't always recognise (e.g. ```JSONC or ```json5).
    - Leading prose (``Here is the grading: [...]``).
    """
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None

    text = _strip_fences(text)
    text = _repair_json(text)

    # 1) Direct parse — strictest, fastest path
    try:
        arr = json.loads(text)
        return arr if isinstance(arr, list) else None
    except Exception:
        pass

    # 2) Find the first top-level JSON array via bracket-depth counter.
    # Same string-escape discipline as parse_json_object so we don't
    # get confused by ``[`` inside string values.
    start = text.find("[")
    if start == -1:
        return None
    end_index = _find_array_end(text, start)
    if end_index is not None:
        candidate = text[start : end_index + 1]
        candidate = _repair_json(candidate)
        try:
            arr = json.loads(candidate)
            return arr if isinstance(arr, list) else None
        except Exception:
            pass
        # Last-ditch: if the array is well-balanced but ``json.loads``
        # still fails (smart-quote / unterminated-string in the middle),
        # try truncating at the last cleanly-closed element.
        truncated = _truncate_array_at_last_complete_element(candidate)
        if truncated is not None:
            return truncated

    # 3) No closing bracket — likely truncated by max_tokens. Try to
    # recover by closing the array at the last complete element.
    truncated = _truncate_array_at_last_complete_element(text[start:])
    if truncated is not None:
        return truncated

    return None


def _find_array_end(text: str, start: int) -> Optional[int]:
    """Return the index of the ``]`` that closes the array starting at
    ``start``. ``None`` if the array is unterminated (truncation)."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
    return None


def _truncate_array_at_last_complete_element(text: str) -> Optional[list]:
    """Best-effort recovery when the JSON array is malformed mid-stream.

    Walks the text looking for object boundaries at depth 1 (each
    element is an object), and tries to parse ``[<up to that point>]``
    with each successive cutoff. The first parse that succeeds is
    the recovered array. Returns ``None`` if no prefix parses.

    Why bother: ``claude-3-5-haiku-latest`` has a 1024-token output
    budget and the grade prompt can grow linearly with the number of
    docs. If 12 docs × 30 tokens of grading = 360 tokens for the
    payload, plus ~150 tokens for the array scaffolding, the response
    can clip the closing ``]`` and any partial final element. Without
    this fallback we'd lose ALL the grading work; with it we keep the
    first N complete entries.
    """
    if not text or text[0] != "[":
        return None
    depth = 0
    in_str = False
    escape = False
    last_good_end: Optional[int] = None
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "{":
            if depth == 1:
                # Track the end of the most recent depth-1 object so we
                # can try cutting there if the rest of the array is
                # malformed.
                pass
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1:
                last_good_end = i
        elif ch == "]":
            depth -= 1
            if depth == 0:
                # Got a clean closing — already covered by the direct
                # parse path, but record anyway.
                last_good_end = i
                break
    if last_good_end is None:
        return None
    # Try cutoffs from largest to smallest so we keep as much as
    # possible. Every successful parse returns the longest valid
    # prefix.
    for end in range(last_good_end, 0, -1):
        candidate = text[: end + 1] + "]"
        candidate = _repair_json(candidate)
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return arr
        except Exception:
            continue
    return None


def parse_structured(content: str, schema: type[T]) -> Optional[T]:
    """Parse model text into a Pydantic schema. Returns ``None`` on
    parse / validation failure.

    Use after calling ``model.invoke()`` (NOT ``with_structured_output``).
    """
    obj = parse_json_object(content)
    if obj is None:
        return None
    try:
        return schema.model_validate(obj)
    except Exception as exc:
        # v2.0.28.4 P1-2 — DEBUG→WARNING + redact secrets. Pre-PR-2
        # this was DEBUG, which meant operators chasing flaky
        # schema-validation failures saw NOTHING at the default INFO
        # log level. Promotion to WARNING makes the noise visible while
        # still being suppressible via ``RAG_LOG_LEVEL=WARNING`` filter.
        # ``redact_secrets`` masks any sk-/Bearer-/URL-credential-shaped
        # substring in the Pydantic error (which echoes the offending
        # field value verbatim — those values sometimes contain API
        # keys or proxy URLs when the model echoed them back).
        logger.warning(
            f"op=parse_structured schema={schema.__name__} "
            f"validation_failed type={type(exc).__name__}: "
            f"{redact_secrets(str(exc))[:200]}"
        )
        return None


def invoke_structured_with_fallback(
    *,
    model,
    schema: type[T],
    structured_messages: list[BaseMessage],
    plain_messages: list[BaseMessage],
    op_name: str = "structured",
) -> Optional[T]:
    """Try ``with_structured_output`` first, then plain-JSON + manual parse.

    Several LangChain call sites (``route_query``, ``rewrite_query``,
    ``check_hallucination``) need a parsed ``BaseModel`` from a cheap
    model. They all used to ship the same 20-line try/except block:
    one path for ``with_structured_output`` (preferred), a fallback
    where the model is asked to emit a JSON object in plain text and
    we parse the response. The fallback exists because some proxies —
    notably the MiniMax Anthropic-compatible endpoint used in this
    deployment — silently return ``None`` from ``with_structured_output``
    even when the underlying model emitted valid JSON.

    This helper consolidates both strategies and the surrounding
    logging. Returns the parsed model on success, or ``None`` if both
    strategies fail; the caller decides what the safe default is.

    Sync variant: blocks the calling thread. Use
    :func:`ainvoke_structured_with_fallback` from async node code to
    avoid holding up the event loop.

    Args:
        model: The chat model to call (``build_cheap_model(...)`` etc.).
        schema: The Pydantic model class the response should validate against.
        structured_messages: Message list for the ``with_structured_output``
            path. Caller assembles prompt + user content as it sees fit.
        plain_messages: Message list for the plain-text fallback. Usually
            the same content but with a JSON-only ``SYSTEM`` prompt.
        op_name: Short tag used in debug logs so it's easy to grep which
            node triggered a fallback.
    """
    # ----- Strategy 1: tool-calling structured output (preferred) -----
    try:
        result = model.with_structured_output(schema).invoke(structured_messages)
        if result is not None:
            return result
        logger.debug(f"{op_name}: structured output returned None; retrying plain")
    except Exception as exc:
        logger.debug(f"{op_name}: structured raised ({exc}); retrying plain")

    # ----- Strategy 2: plain JSON + parse (with retry on transient errors) -----
    try:
        # v2.0.28.4 P1-1 — wrap model.invoke in retry/backoff so a
        # single Anthropic 529 / httpx.ConnectError mid-fallback doesn't
        # blow up the user turn. Strategy 1 stays as-is (it has its own
        # "structured output returned None" retry path via plain-JSON,
        # and the wrapper around with_structured_output is itself a
        # LangChain retry boundary on some providers).
        resp = invoke_with_retry(model, plain_messages)
        content = (
            resp.content if isinstance(resp.content, str) else str(resp.content)
        )
        return parse_structured(content, schema)
    except Exception as exc:
        logger.debug(f"{op_name}: plain-JSON fallback failed: {exc}")
        return None


async def ainvoke_structured_with_fallback(
    *,
    model,
    schema: type[T],
    structured_messages: list[BaseMessage],
    plain_messages: list[BaseMessage],
    op_name: str = "structured",
) -> Optional[T]:
    """Async twin of :func:`invoke_structured_with_fallback`.

    Uses the LangChain async surface (``model.ainvoke``,
    ``model.with_structured_output(...).ainvoke``) so the event loop stays
    responsive while waiting for the upstream LLM. The structured-output
    wrapper for some LangChain providers actually proxies ``ainvoke``
    through ``asyncio.to_thread`` internally, so this is mostly the
    LangChain-compatible surface — the real win is composability:
    call sites can ``await asyncio.gather(...)`` several such calls
    without ever blocking the loop.

    Mirrors the sync helper's two-strategy structure: try
    ``with_structured_output`` first; on ``None`` or exception, retry
    with the plain-JSON prompt and parse the response text.
    """
    # ----- Strategy 1: tool-calling structured output (preferred) -----
    try:
        result = await model.with_structured_output(schema).ainvoke(
            structured_messages
        )
        if result is not None:
            return result
        logger.debug(f"{op_name}: structured output returned None; retrying plain")
    except Exception as exc:
        logger.debug(f"{op_name}: structured raised ({exc}); retrying plain")

    # ----- Strategy 2: plain JSON + parse (with retry on transient errors) -----
    try:
        # v2.0.28.4 P1-1 — async twin of sync retry wrapper. Uses
        # ``ainvoke_with_retry`` so the event loop stays responsive
        # during the exponential backoff (the wait is awaited, not
        # blocking the loop with time.sleep).
        resp = await ainvoke_with_retry(model, plain_messages)
        content = (
            resp.content if isinstance(resp.content, str) else str(resp.content)
        )
        return parse_structured(content, schema)
    except Exception as exc:
        logger.debug(f"{op_name}: plain-JSON fallback failed: {exc}")
        return None


__all__ = [
    "invoke_structured_with_fallback",
    "ainvoke_structured_with_fallback",
    "parse_json_object",
    "parse_json_array",
    "parse_structured",
    "invoke_with_retry",
    "ainvoke_with_retry",
]


# v2.0.28.4 P1-1 — retry/backoff on transient LLM provider errors.
#
# Why this exists:
#   Pre-PR-2 a single ``anthropic.APIStatusError(status_code=529)`` (Anthropic
#   overloaded) or ``httpx.ConnectError`` mid-turn meant the user lost the
#   whole turn with no retry. The web_search path already retries
#   (constants.py:103-108 max_attempts=3, backoff_seconds=(1.5, 3.0, 6.0)),
#   but the LLM path did not — the audit flagged this as the largest
#   single source of "user sees an error" reports.
#
# Why 3 attempts × exponential backoff (1, 2, 4s, capped at 4s):
#   3 attempts balances resilience vs latency. Worst-case latency
#   before user sees error = 1 + 2 + 4 + model-time-of-3rd-attempt ≈ 12s
#   + model latency. Compare to pre-PR-2 = 0 retries + immediate error.
#   Operators can monitor via the ``op=llm_retry`` log line.
#
# Why ONLY these exception types:
#   429 (rate-limit) and 529 (overloaded) on Anthropic; 429 / 500 / 503
#   on OpenAI; transport errors (ConnectError / ReadTimeout) on either.
#   Everything else (ValidationError, JSONDecodeError, our own
#   redaction errors) is a real bug or user error — retrying would
#   just delay the same failure.
_RETRYABLE_HTTP_STATUSES = {429, 500, 503, 529}

# Tuple form so tenacity's ``retry_if_exception_type`` can filter by
# the exact classes (not the function — tenacity needs class-based
# predicates). The instance check happens inside the retry loop body
# so we still distinguish "transient with retryable status code" from
# "real bug".
_RETRYABLE_EXC_TYPES = (
    anthropic.APIStatusError,
    openai.APIStatusError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` is a transient LLM provider / transport
    error worth retrying. See module-level retry rationale above."""
    if isinstance(exc, (anthropic.APIStatusError, openai.APIStatusError)):
        return exc.status_code in _RETRYABLE_HTTP_STATUSES
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)):
        return True
    return False


def invoke_with_retry(model, messages, *, max_attempts: int = 3) -> Any:
    """Sync ``model.invoke(messages)`` with retry on transient errors.

    Same shape as the bare ``model.invoke`` call — caller can swap one
    for the other without rewriting prompt handling. Logs each retry
    attempt with ``op=llm_retry`` so operators can grep the noise.
    """
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(_RETRYABLE_EXC_TYPES),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        ):
            with attempt:
                try:
                    return model.invoke(messages)
                except Exception as exc:
                    if _is_retryable(exc):
                        logger.warning(
                            f"op=llm_retry attempt={attempt.retry_state.attempt_number} "
                            f"type={type(exc).__name__}: {redact_secrets(str(exc))[:200]}"
                        )
                        raise
                    raise
    except Exception:
        # Re-raise — caller decides whether to swallow (current callers
        # wrap the whole helper in try/except and return ``None``).
        raise


async def ainvoke_with_retry(model, messages, *, max_attempts: int = 3) -> Any:
    """Async twin of :func:`invoke_with_retry` — uses
    ``model.ainvoke(messages)`` and ``AsyncRetrying`` so the event loop
    stays responsive between retries (the exponential wait is awaited,
    not blocking)."""
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_RETRYABLE_EXC_TYPES),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        ):
            with attempt:
                try:
                    return await model.ainvoke(messages)
                except Exception as exc:
                    if _is_retryable(exc):
                        logger.warning(
                            f"op=llm_retry attempt={attempt.retry_state.attempt_number} "
                            f"type={type(exc).__name__}: {redact_secrets(str(exc))[:200]}"
                        )
                        raise
                    raise
    except Exception:
        raise