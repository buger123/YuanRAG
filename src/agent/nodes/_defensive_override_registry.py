"""v2.0.29.3 (Phase 3) — defensive-override registry.

Per [[yuanrag-hallucination-optimization]] Phase 3: the defensive
override system that catches the synthesis LLM ignoring a tool's
returned data (e.g. v2.0.28.16's "now-bug3" — user asked time, LLM
answered "早点休息,晚安" instead of using the time TM data) was
hardcoded to ``get_current_time`` in three places:

  * ``_build_get_current_time_directive`` (post-script nudge)
  * ``_extract_time_marker_from_history`` (fingerprint extractor)
  * inline override block in ``react_generate`` (fallback builder)

Adding a new tool meant mirroring all of those by hand, in three
spots, AND adding a new ``inc(tool=...)`` call in the inline block
— a 30+ line change that's trivially wrong (you can update 2 of 3
and not notice the third is still hardcoded for the old tool).

This module replaces those three call-site touchpoints with three
plain Python dictionaries keyed by ``tool_name``. Adding a new
tool is now ONE-LINE-PER-REGISTRY (3 lines total) and a future
registry miss is detected by the unit tests in
``tests/test_defensive_override_registry.py`` (every registered
tool must have a fingerprint + fallback — pre-registration
contract, same as Phase 2's Counter registry).

The 3 registries
----------------

``DIRECTIVE_BUILDERS``
    Maps ``tool_name`` to a function that walks the sanitized
    history and returns a **post-script directive string** to be
    appended (inserted at index=2 per v2.0.28.17 consecutiveness)
    before the LLM streams the answer. Or ``None`` when the tool
    wasn't called. The directive nudges the LLM to use the tool's
    data instead of continuing prior conversational mode.

``FINGERPRINT_EXTRACTORS``
    Maps ``tool_name`` to a function that takes the raw TM
    ``content`` string and returns the **most-distinctive marker**
    (e.g. ISO date ``2026-09-27``), or ``None`` if extraction
    fails. The caller substring-checks ``answer_text`` for the
    marker; missing → LLM didn't incorporate the tool's data.

``FALLBACK_BUILDERS``
    Maps ``tool_name`` to a function that takes the
    :class:`ToolMessage` and returns the **override answer** the
    synthesis LLM should have produced. Used to REPLACE the LLM's
    wrong output with a deterministic answer based on the tool's
    data. Returns ``None`` when the tool's content is unparseable
    (caller falls through to keep the LLM answer).

Pre-registration contract
-------------------------

The three registries must be **aligned**: every tool listed in
``DIRECTIVE_BUILDERS`` should also have a fingerprint + fallback
in the corresponding registry, and vice versa. The
``_aligned_tools()`` helper exposes the intersection for diagnostics
+ the unit tests.

Adding a new tool
-----------------

  1. Write 3 functions: ``_build_<tool>_directive``,
     ``_extract_<tool>_marker``, ``_build_<tool>_fallback``.
  2. Add one line to each of the 3 registries.
  3. Add 3 unit tests (one per registry + one for
     ``apply_defensive_overrides``).
  4. Done. Zero call-site changes in ``react_generate``.

The ``apply_defensive_overrides`` helper
----------------------------------------

The single entry point for the synthesis-LLM safety net. Iterates
the (already-sanitized) history **in reverse** (most-recent TM wins
— multi-turn safety: in a thread where the user asked "现在几点"
twice, the LATEST TM is the one the synthesis LLM is answering,
not the first). For each TM with both fingerprint + fallback
registered, checks if the answer contains the marker; if not,
bumps ``SYNTHESIS_TM_IGNORED{tool=...}`` (Phase 2 metric) and
returns the fallback answer. Idempotent — fires at most once per
call (returns immediately on first miss).
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, ToolMessage

from src.core.logging import logger


# ---- Type aliases ----------------------------------------------------------

#: ``(sanitized_history) -> directive_string | None``. Walks the history
#: looking for the tool's TM and returns a post-script nudge. ``None``
#: when the tool wasn't called this turn.
DirectiveBuilder = Callable[[List[BaseMessage]], Optional[str]]

#: ``(raw_tm_content) -> marker_string | None``. Pure string → string.
#: Returns the most-distinctive marker the LLM should have cited
#: verbatim (e.g. ``"2026-09-27"`` for a date TM). ``None`` when no
#: usable marker is in the content (caller treats as "no override").
FingerprintExtractor = Callable[[str], Optional[str]]

#: ``(tool_message) -> override_answer | None``. Builds the answer
#: that should REPLACE the LLM's wrong output. ``None`` when the
#: tool's content is unparseable (caller keeps the LLM answer).
FallbackBuilder = Callable[[ToolMessage], Optional[str]]


# ---- Pre-registered builders for get_current_time ---------------------------
# V2.0.28.16 (now-bug3 fix) — these three functions together caught
# the 40% pre-fix failure mode where the synthesis LLM emitted
# greetings / farewells instead of stating the time. The directive
# is preventive (preempts the wrong-text emission); the fingerprint
# + fallback are defensive (catches the runs that slip through).


def _build_get_current_time_directive(messages: list) -> str | None:
    """If a ``get_current_time`` TM is in ``messages``, return a strong
    directive prompting the LLM to use the time data; else ``None``.

    v2.0.28.16 — preventive layer. The post-script is appended to
    ``msgs`` AFTER the sanitized history so the LLM sees it as the
    LAST instruction before generating — empirically biases the LLM
    toward using the tool data for factual answers, vs. continuing
    the prior turn's conversational mode (which produced 40% chat
    replies pre-fix).

    Returns ``None`` when no ``get_current_time`` TM is present
    (caller should NOT append a post-script in that case). Skips
    empty content too — no point telling the LLM "use this empty
    data".
    """
    for m in messages:
        if (
            isinstance(m, ToolMessage)
            and (getattr(m, "name", "") or "") == "get_current_time"
        ):
            raw = (getattr(m, "content", "") or "").strip()
            if not raw:
                continue
            return (
                "[CRITICAL — get_current_time tool result] "
                "A get_current_time tool call was just executed and "
                "returned the following time data:\n\n"
                f"{raw}\n\n"
                "You MUST use this exact data to answer the user's "
                "literal question. Do NOT respond with greetings, "
                "farewells, or casual chat replies — the user asked "
                "a factual question and expects a direct factual "
                "answer that includes the time."
            )
    return None


def _extract_time_marker(raw_content: str) -> str | None:
    """Return the most-distinctive time marker from a ``get_current_time``
    TM's raw content, or ``None``.

    V2.0.28.16 — defensive layer. Used by :func:`apply_defensive_overrides`
    to detect when the LLM emitted text but didn't incorporate the
    tool data (the 40% pre-fix failure mode). Prefer ISO date
    ``YYYY-MM-DD`` (most distinctive), else ``HH:MM[:SS]``, else
    Chinese ``M月D日``.

    Pure string function so the registry stays composable: the
    caller iterates the history and passes each TM's content here.
    Most-recent-wins iteration order lives in the registry caller
    (``apply_defensive_overrides``), not here.
    """
    iso = re.search(r"\d{4}-\d{2}-\d{2}", raw_content)
    if iso:
        return iso.group(0)
    t = re.search(r"\d{1,2}:\d{2}(?::\d{2})?", raw_content)
    if t:
        return t.group(0)
    cn = re.search(r"\d{1,2}月\d{1,2}日", raw_content)
    if cn:
        return cn.group(0)
    return None


def _build_time_fallback(tm: ToolMessage) -> str | None:
    """Build the override answer for a ``get_current_time`` TM.

    Mirrors the v2.0.28.12.1 thinking-only fallback (which itself
    mirrors the constructed answer the synthesis LLM was supposed
    to produce). Belt-and-suspenders with the post-script directive
    above: the directive tries to PREVENT the wrong-text emission;
    this fallback CATCHES the runs that slip through.

    Returns ``None`` when the TM content is empty (caller keeps the
    LLM answer — we have nothing to fall back to).
    """
    raw = (getattr(tm, "content", "") or "").strip()
    if not raw:
        return None
    return (
        f"根据刚刚查询,当前时间是 {raw.splitlines()[0] if raw else '未知'}。"
        + (f"\n详细信息:{raw}" if "\n" in raw else "")
    )


# ---- The 3 registries -----------------------------------------------------


DIRECTIVE_BUILDERS: Dict[str, DirectiveBuilder] = {
    # V2.0.28.16 (now-bug3 preventive layer).
    "get_current_time": _build_get_current_time_directive,
}


FINGERPRINT_EXTRACTORS: Dict[str, FingerprintExtractor] = {
    # V2.0.28.16 (now-bug3 defensive layer). ISO date preferred.
    "get_current_time": _extract_time_marker,
}


FALLBACK_BUILDERS: Dict[str, FallbackBuilder] = {
    # V2.0.28.16 (now-bug3 fallback). Constructed Chinese answer.
    "get_current_time": _build_time_fallback,
}


def _aligned_tools() -> set[str]:
    """Tools registered in ALL THREE registries (the intersection).

    The defensive override only fires for tools in the intersection —
    a tool with only a directive but no fingerprint (or vice versa)
    silently skips. ``_aligned_tools()`` exposes the aligned set for
    diagnostics + a unit-test sanity check that the three registries
    stay in sync as new tools are added.
    """
    return (
        set(DIRECTIVE_BUILDERS.keys())
        & set(FINGERPRINT_EXTRACTORS.keys())
        & set(FALLBACK_BUILDERS.keys())
    )


# ---- The single entry point -----------------------------------------------


def apply_defensive_overrides(
    answer_text: str,
    history: list,
    route_decision: str,
) -> str:
    """Run the defensive override check against ``answer_text``.

    Iterates ``history`` REVERSED so the most-recent ``get_current_time``
    TM (or any future tool's TM) wins — multi-turn safety: in a
    thread where the user asked "现在几点" twice, the LATEST TM is
    the one the synthesis LLM is answering, not the first. This
    avoids the v2.0.28.16 multi-turn bug where the defensive fallback
    overrode the answer with stale data.

    For each TM with both fingerprint + fallback registered, checks
    if the answer contains the marker; if not, bumps
    ``SYNTHESIS_TM_IGNORED{tool=tool_name}`` (Phase 2 metric) and
    returns the fallback answer. Idempotent — fires at most once
    per call (returns immediately on first miss).

    Guards (preserved from v2.0.28.16 / v2.0.28.12.1):

      * ``route_decision == "direct"`` — direct path has no TMs;
        skip.
      * ``not answer_text`` — empty answer means the synthesis LLM
        emitted thinking-only; the v2.0.28.12.1 thinking-only
        fallback (caller's responsibility, NOT this function)
        already handled that. We skip to avoid double-override.

    Returns the (possibly-overridden) ``answer_text``.
    """
    if route_decision == "direct" or not answer_text:
        return answer_text
    for tm in reversed(history):
        if not isinstance(tm, ToolMessage):
            continue
        tool_name = (getattr(tm, "name", "") or "")
        fp_ext = FINGERPRINT_EXTRACTORS.get(tool_name)
        fb_build = FALLBACK_BUILDERS.get(tool_name)
        if not (fp_ext and fb_build):
            # Tool not registered for defensive override (no fingerprint
            # AND no fallback). Either the tool doesn't need defensive
            # coverage (most tools — the LLM ignoring one is a known
            # failure mode only for ``get_current_time`` today), or the
            # registries are out of sync (test will catch that).
            continue
        marker = fp_ext((getattr(tm, "content", "") or ""))
        if marker and marker not in answer_text:
            # v2.0.29.2 (Phase 2) — observability hook. The synthesis
            # LLM produced output but didn't cite the TM data.
            # Defensive override catches the miss; we count how often
            # it fires. Dashboards monitor this rate to detect model
            # regressions (sudden spike = synthesis LLM ignoring TM
            # data systematically).
            from src.agent.metrics import SYNTHESIS_TM_IGNORED

            SYNTHESIS_TM_IGNORED.inc(tool=tool_name)
            fallback = fb_build(tm)
            if fallback is not None:
                logger.warning(
                    "react_generate: synthesis LLM emitted text but did "
                    "NOT incorporate %s TM data; overriding with "
                    "constructed fallback answer. LLM output: %r",
                    tool_name,
                    answer_text[:200],
                )
                return fallback
    return answer_text


__all__ = [
    "DirectiveBuilder",
    "FingerprintExtractor",
    "FallbackBuilder",
    "DIRECTIVE_BUILDERS",
    "FINGERPRINT_EXTRACTORS",
    "FALLBACK_BUILDERS",
    "_aligned_tools",
    "apply_defensive_overrides",
    # Re-export the per-tool builders so tests + future diagnostics
    # can introspect the registry's content.
    "_build_get_current_time_directive",
    "_extract_time_marker",
    "_build_time_fallback",
]