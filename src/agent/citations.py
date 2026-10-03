"""Citation-marker parsing helpers.

Pulled out of :mod:`src.agent.runner` so the runner can focus on the
WebSocket event-stream contract and the parser can be unit-tested
without spinning up the full graph. The rule lives here: anything that
only the runner used to share with no one else moved out when the
hotspot was refactored.
"""
from __future__ import annotations

import re
from typing import Optional, get_args

# v2.0.22 (Item 7 Step 10) — closed vocabulary for ``route_decision``.
# Used in the entry assertion of both helpers below so a typo like
# ``"retreive"`` fails loudly instead of silently taking the
# ``direct`` short-circuit (``route_decision != "retrieve"`` is the
# pre-Step-10 guard and treats *any* non-``"retrieve"`` value as
# ``direct``, including typos that should have been ``"retrieve"``).
from src.agent.types import RouteDecision
from src.core.logging import logger


# Pre-compiled once at import time — the runner hits this regex on every
# ``answer_complete`` event, so doing it in a module-level constant saves
# the per-call ``re.compile`` cost (small but non-zero on busy streams).
_CITATION_GROUP_RE = re.compile(r"\[([0-9,\-\s]+)\]")
_PART_SPLIT_RE = re.compile(r"[,\s]+")


def _to_source_dict(s) -> dict:
    """Normalize a ``Source`` (Pydantic) or ``dict`` to a plain dict.

    v2.0.7 SoT-1 introduced a Pydantic ``Source`` model with
    ``ConfigDict(extra="ignore")`` for forward-compat wire stability.
    The citation helpers below pre-date that change and used to assume
    ``dict`` access everywhere (``s.get("index")``, ``dict(s)``).
    Both still accept raw dicts (legacy callers, test fixtures, or
    checkpointed sessions that survived an upgrade) so we normalize at
    the function boundary rather than touch every internal loop.
    """
    if hasattr(s, "model_dump"):
        return s.model_dump()
    return dict(s)


def extract_cited_indices(text: str) -> set[int]:
    """Pull every citation index ``n`` from an answer's ``[n]`` markers.

    Handles the common shapes the LLM emits:

    =================  ====================================
    input              result
    =================  ====================================
    ``[1]``            {1}
    ``[1, 3]``         {1, 3}
    ``[1, 3, 5]``      {1, 3, 5}
    ``[1-3]``          {1, 2, 3}
    ``[1, 3-5, 7]``    {1, 3, 4, 5, 7}
    =================  ====================================

    Returns an empty set when no ``[n]`` markers are found (the model
    decided not to cite — common on the ``direct`` greeting path). On
    that empty result the caller must skip the filter and surface
    every source as-is, since ``[n]`` absence likely means the LLM
    answered without retrieval rather than that no sources apply.

    Non-citation brackets (``[link]``, ``[foo]``) are silently dropped
    by the character-class filter — we only enumerate groups whose
    content is digits, commas, whitespace, and dashes.
    """
    cited: set[int] = set()
    if not text:
        return cited
    for match in _CITATION_GROUP_RE.finditer(text):
        content = match.group(1)
        for part in _PART_SPLIT_RE.split(content):
            if not part:
                continue
            if "-" in part:
                try:
                    start_s, end_s = part.split("-", 1)
                    start_i, end_i = int(start_s), int(end_s)
                except (ValueError, TypeError):
                    continue
                if start_i <= end_i:
                    cited.update(range(start_i, end_i + 1))
                else:
                    cited.update(range(end_i, start_i + 1))
            else:
                try:
                    cited.add(int(part))
                except (ValueError, TypeError):
                    continue
    return cited


def filter_sources_to_cited(
    sources,
    answer: str,
    *,
    route_decision: Optional[RouteDecision] = None,
) -> list[dict]:
    """Drop sources that the answer didn't actually cite.

    ``generate._to_sources`` hands us every retrieved / graded document,
    but the model typically only writes ``[n]`` for the chunks it
    actually used — every other retrieved doc would still show up as
    a clickable chip in the sidebar, which is both noisy and
    misleading ("is this also a source? did the model use it?").

    ``sources`` may be a list of Pydantic ``Source`` instances (the
    v2.0.7 default) or plain dicts (legacy / test fixtures). We
    normalize at the boundary via :func:`_to_source_dict` so the
    internal loops can use ``.get()`` uniformly. Without this
    normalization a single Pydantic instance would raise
    ``AttributeError: 'Source' object has no attribute 'get'`` and
    filter out every chip in the sidebar — exactly the
    "``[1]`` markers but no chips" bug the original ``.get()`` was
    introduced to prevent.

    Direct-path leak (v1.1.3)
    -------------------------
    ``route_decision`` is an optional kwarg the runner passes from the
    LangGraph state. When it is ``"direct"`` we MUST return ``[]``
    regardless of ``[n]`` markers — the conversational / greeting
    path doesn't carry documents, so any sources that survived to
    this point are a state-leak from a previous turn or an upstream
    bug. Showing them as clickable chips on a "who are you?" answer
    is the bug that surfaces as 8 spurious citations of the same
    uploaded PDF on a no-doc question.

    Why the runner is the right place for this guard
    ------------------------------------------------
    The generate node already knows ``route_decision`` and is the
    proper place to skip attaching ``additional_kwargs["sources"]``
    on the direct path (see ``generate.py`` — direct branch returns
    ``sources=[]``). This kwarg is a belt-and-suspenders second
    check at the runner boundary, in case a future code path
    forgets to set sources=[] (e.g. a new conversational node that
    happens to share the same downstream wire payload).
    """
    # v2.0.22 (Item 7 Step 10) — runtime assertion. The type hint
    # already constrains ``route_decision`` to ``None | "direct" |
    # "retrieve"``, but the helper is on the hot path of every WS
    # ``answer_complete`` frame; a typo from a future caller
    # (``"retreive"`` / ``"direct"`` typo with stray spaces / ``None``
    # accidentally stringified) would otherwise silently take the
    # ``direct`` short-circuit below. ``get_args(RouteDecision)``
    # gives ``("direct", "retrieve")`` so the assertion stays in sync
    # with the Literal definition in :mod:`src.agent.types`.
    assert route_decision is None or route_decision in get_args(RouteDecision), (
        f"filter_sources_to_cited: route_decision must be None or one of "
        f"{get_args(RouteDecision)!r}, got {route_decision!r}"
    )
    # Direct-path guard: never surface sources on a conversational
    # turn. A non-None ``route_decision`` is explicit — the caller
    # has read it off the state and is asserting "this turn is
    # direct, no matter what". ``None`` means the runner didn't
    # bother to look (legacy / test path), and we fall back to the
    # legacy [n]-based filter so existing tests don't break.
    if route_decision is not None and route_decision != "retrieve":
        return []
    cited = extract_cited_indices(answer)
    if not cited:
        # No ``[n]`` markers and we're on the retrieve path → the
        # model decided to answer without naming any source. Surface
        # every source as-is rather than guessing which to drop — the
        # user might still want to see what was retrieved even though
        # the model chose not to cite. (Direct path returns [] above
        # so this branch only fires when ``route_decision == "retrieve"``.)
        return [_to_source_dict(s) for s in sources]
    return [_to_source_dict(s) for s in sources if _to_source_dict(s).get("index") in cited]


def renumber_citations_and_sources(
    answer: str,
    sources,
    *,
    route_decision: Optional[RouteDecision] = None,
) -> tuple[str, list[dict]]:
    """Renumber ``[n]`` markers + sources to a contiguous 1..K range.

    v2.0.5 — closes "回答 [1]–[23] 但侧栏只显示 3 条" (P1-3).

    The LLM sometimes cites high indices when only a few documents
    were actually retrieved (``[1] [3] [7] [23]`` against a 3-doc
    result set). Pre-v2.0.5 the runner only filtered uncited
    sources; the answer text was forwarded as-is, leaving dangling
    markers that don't correspond to any chip. This helper:

      1. Reads every distinct cited index from ``answer`` via
         :func:`extract_cited_indices`.
      2. Builds an ``old -> new`` remap where ``new`` is the
         1-based position of ``old`` in the sorted distinct set.
      3. Rewrites every ``[n]`` / ``[a,b]`` / ``[a-b]`` marker in
         ``answer`` to use the new indices (preserving the comma /
         range / mix shape — see ``_RENUMBER_REPL``).
      4. Re-stamps every source dict's ``index`` to its new value
         AND drops any source whose old index wasn't cited.
      5. Sorts the surviving sources by new index so the chip
         order matches the answer's marker order.

    ``sources`` may be a list of Pydantic ``Source`` instances (the
    v2.0.7 default) or plain dicts (legacy / test fixtures) — the
    helper normalizes via :func:`_to_source_dict` at the entry.
    Returns ``(answer, sources)`` where ``sources`` is always a list
    of plain dicts (the wire format), in ``[1]..[K]`` order.

    Direct-path safety (same as :func:`filter_sources_to_cited`):

    * ``route_decision="direct"`` returns ``(answer, [])`` — a
      greeting path has no sources and shouldn't be touched.
    * If no ``[n]`` markers are found on a retrieve path, returns
      ``(answer, sources)`` unchanged (legacy behavior — caller may
      choose to surface every source or ``[]``).

    Marker rewriting is range-aware so ``[1-3]`` becomes
    ``[1-3]`` (or whatever the renumber produces for each element).
    A range whose elements don't survive the citation set — e.g.
    ``[5-7]`` when only ``{5}`` is cited — collapses to ``[1]``
    rather than being dropped, on the theory that the LLM emitted
    the range deliberately.
    """
    # v2.0.22 (Item 7 Step 10) — runtime assertion, mirrors the one
    # in :func:`filter_sources_to_cited`. See that helper for the
    # rationale (``get_args(RouteDecision)`` keeps the assertion in
    # sync with the Literal definition).
    assert route_decision is None or route_decision in get_args(RouteDecision), (
        f"renumber_citations_and_sources: route_decision must be None or one of "
        f"{get_args(RouteDecision)!r}, got {route_decision!r}"
    )
    # Direct-path leak guard (v1.1.3 belt-and-suspenders).
    if route_decision is not None and route_decision != "retrieve":
        return answer or "", []
    cited = extract_cited_indices(answer or "")
    if not cited:
        return answer or "", list(sources)
    sorted_cited = sorted(cited)
    old_to_new = {old: new for new, old in enumerate(sorted_cited, start=1)}
    # Cap: indices that map to "K+1" or beyond are dropped. The
    # filtered-survives invariant is that ``new`` in [1..K].
    new_to_old = {new: old for old, new in old_to_new.items()}

    def _remap_single(token: str) -> Optional[str]:
        """Remap a single numeric token (no comma/dash). Returns
        ``None`` when the token isn't a clean integer so the
        caller can leave it unchanged."""
        token = token.strip()
        if not token:
            return None
        try:
            n = int(token)
        except ValueError:
            return None
        if n not in old_to_new:
            # The LLM cited an index that has no source (e.g. [23]
            # when only 3 docs survived). Drop the token silently —
            # leaving it would just produce the same dangling
            # marker we started with.
            return None
        return str(old_to_new[n])

    def _remap_part(part: str) -> str:
        """Remap one comma-or-range part (e.g. ``"1"`` / ``"1-3"``)."""
        if not part:
            return ""
        if "-" in part:
            bits = part.split("-", 1)
            if len(bits) != 2:
                return part
            try:
                a, b = int(bits[0]), int(bits[1])
            except ValueError:
                return part
            lo, hi = (a, b) if a <= b else (b, a)
            # v2.0.29.4 (Phase 4 PR-1) — source-presence filter.
            # Pre-PR-1 the collapse branch below was effectively dead
            # code: ``old_to_new`` is built from ``cited`` (every
            # integer the LLM wrote), so for any cited range every
            # element was in ``old_to_new`` and ``mapped`` always
            # equaled ``range(lo, hi+1)`` in length. The collapse
            # never fired, contradicting the docstring's intent of
            # ``[5-7] -> [5]`` when only source 5 survived.
            #
            # We filter by source-presence so a range where some
            # sources are missing collapses to the surviving
            # element(s) AND emits the WARNING we just added. The
            # filter is the source list, NOT the renumbered
            # ``remapped`` list below — that list is built AFTER
            # this function returns, and the original ``index`` on
            # each source is what we compare against.
            #
            # NB: ``sources`` may carry Pydantic ``Source`` instances
            # (the v2.0.7 default) — ``Source`` has no ``.get`` so
            # normalize via :func:`_to_source_dict` like the rest of
            # this module. ``raw.get("index")`` was the bug behind
            # ``test_citation_filter::test_answer_complete_keeps_range_cited``
            # (``AttributeError: 'Source' object has no attribute 'get'``
            # from pydantic/main.py:1042 — the test fixture uses the
            # real ``_to_sources`` path that emits Pydantic Sources).
            source_indices: set[int] = set()
            for raw in sources or []:
                d = _to_source_dict(raw)
                idx = d.get("index")
                if idx is not None:
                    source_indices.add(idx)
            mapped: list[int] = []
            for i in range(lo, hi + 1):
                if i in old_to_new and i in source_indices:
                    mapped.append(old_to_new[i])
            if not mapped:
                # Nothing in the range survived. Collapse to the
                # nearest surviving new index if any, else drop.
                return ""
            # Try to preserve the range shape when possible. A
            # single element can't be a range, so emit it bare.
            if len(mapped) == 1:
                # v2.0.29.4 (Phase 4 PR-1) — log the collapse. Pre-PR-1
                # this branch was silent (and effectively dead — see
                # the source-presence filter above), so a live→reload
                # drift where ``[5-7]`` became ``[5]`` (because only
                # source 5 survived the cited-set filter) was
                # invisible to dashboards. The collapse is benign —
                # the LLM emitted a range deliberately and we kept it
                # as the lowest survivor — but if the rate climbs,
                # that's a signal the renumber step is producing
                # fragments users can see flicker between sessions.
                #
                # NB: ``part`` is the original range string (e.g.
                # ``"5-7"``); ``mapped[0]`` is the lone survivor's new
                # index (already renumbered).
                #
                # Use an f-string (NOT loguru's lazy ``%s`` /
                # ``%r`` placeholders) so the captured ``{message}``
                # in a sink shows the substituted values — loguru's
                # lazy placeholders don't render until the
                # ``record.extra`` formatter runs, and a sink with
                # ``format="{message}"`` captures the raw format
                # string. This is the same f-string discipline used
                # in the inline defensive-override log in
                # ``react_generate.py`` (Phase 3 #4 debugging坑).
                logger.warning(
                    f"citations: collapsed range [{part}] -> [{mapped[0]}] "
                    f"(filtered {(hi - lo + 1) - len(mapped)} sources mid-range)"
                )
                return str(mapped[0])
            if mapped == list(range(mapped[0], mapped[-1] + 1)):
                return f"{mapped[0]}-{mapped[-1]}"
            return ",".join(str(m) for m in mapped)
        return _remap_single(part) or ""

    def _rewrite(m: re.Match) -> str:
        content = m.group(1) or ""
        parts = _PART_SPLIT_RE.split(content)
        rewritten_parts: list[str] = []
        for p in parts:
            mapped = _remap_part(p)
            if mapped:
                rewritten_parts.append(mapped)
        if not rewritten_parts:
            # Whole group collapsed to nothing — drop the marker.
            return ""
        return "[" + ",".join(rewritten_parts) + "]"

    rewritten = _CITATION_GROUP_RE.sub(_rewrite, answer or "")

    # Reindex sources; drop uncited. v2.0.7 SoT-1: ``Source`` is
    # now a Pydantic model, so ``s.get()`` and ``dict(s)`` no longer
    # work directly — normalize via :func:`_to_source_dict`.
    remapped: list[dict] = []
    for raw in sources or []:
        d = _to_source_dict(raw)
        old_idx = d.get("index")
        if old_idx is None or old_idx not in old_to_new:
            continue
        d["index"] = old_to_new[old_idx]
        remapped.append(d)
    remapped.sort(key=lambda s: s.get("index", 0))

    return rewritten, remapped


__all__ = [
    "extract_cited_indices",
    "filter_sources_to_cited",
    "renumber_citations_and_sources",
]