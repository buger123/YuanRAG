"""Tests for v2.0.22 (Item 7 Step 10) — ``route_decision`` Literal.

Pre-Step-10, three modules each declared ``route_decision`` with a
different (loose) type:

* ``react_generate._intent_to_route_decision(intent) -> str`` —
  closed vocabulary in the function body but untyped signature.
* ``FSMEvent`` (in ``fsm.py``) — didn't declare the field at all
  (it rode as a stray kwarg through ``yield FSMEvent(...)``).
* ``citations.filter_sources_to_cited`` /
  ``citations.renumber_citations_and_sources`` — typed ``str | None``
  / ``Optional[str]``, with a guard ``if route_decision !=
  "retrieve"`` that silently treated **any** other value
  (including typos like ``"retreive"``) as the direct short-circuit.

Step 10 fixes this by centralizing the closed vocabulary as
``RouteDecision = Literal["direct", "retrieve"]`` in
:mod:`src.agent.types`, tightening every signature to use it, adding
the field to ``FSMEvent`` as ``NotRequired[RouteDecision]``, and
adding a runtime assertion in the two citation helpers that fails
loudly on a typo instead of silently short-circuiting.

v2.0.32.5 (Stage 5.7, 2026-10-07) — eval harness caught Phase 8
(v2.0.29.9) verbatim extraction path that added 2 new values
(``"extractive"`` + ``"extractive_refusal"``) in
``src/agent/nodes/react_generate_extractive.py`` without registering
them in this Literal. ``renumber_citations_and_sources``
(``src/agent/citations.py:229``) asserts ``route_decision in
get_args(RouteDecision)`` — verbatim cases hit AssertionError and
got empty answers. The Literal extended to 4 values; this test
updated to pin the new vocabulary.

These tests pin the new contract:

1. ``RouteDecision`` is the closed Literal exported from
   :mod:`src.agent.types` (4 values as of v2.0.32.5).
2. ``FSMEvent`` declares ``route_decision`` (typed).
3. ``_intent_to_route_decision`` return annotation is the Literal.
4. ``filter_sources_to_cited`` / ``renumber_citations_and_sources``
   raise ``AssertionError`` on a typo (``"retreive"``) and accept
   ``None`` / ``"direct"`` / ``"retrieve"`` / ``"extractive"`` /
   ``"extractive_refusal"`` without raising.
"""
from __future__ import annotations

import inspect
from typing import Optional, Union, get_args, get_type_hints

import pytest


# ---------------------------------------------------------------------------
# 1. RouteDecision export from src.agent.types
# ---------------------------------------------------------------------------


def test_route_decision_is_literal_4_values():
    """``RouteDecision`` is a closed Literal of exactly the four
    route values the citation guards recognize.

    v2.0.22 (Item 7 Step 10) — original 2 values: ``"direct"``,
    ``"retrieve"``.

    v2.0.32.5 (Stage 5.7, 2026-10-07) — Phase 8 verbatim extraction
    (v2.0.29.9) added 2 more values (``"extractive"`` +
    ``"extractive_refusal"``) that the citation guards must
    recognize too. New values must be added here first so the
    assertion in ``citations.py`` stays in sync — the eval harness
    caught the gap when verbatim cases (golden-008-verbatim-zh /
    golden-008-verbatim-en) hit AssertionError and returned empty
    answers.
    """
    from src.agent.types import RouteDecision

    assert get_args(RouteDecision) == (
        "direct",
        "retrieve",
        "extractive",
        "extractive_refusal",
    )


def test_route_decision_is_in_types_dunder_all():
    """``__all__`` exposes ``RouteDecision`` so callers can
    ``from src.agent.types import RouteDecision`` without reaching
    into private module state.
    """
    from src.agent import types as types_mod

    assert "RouteDecision" in types_mod.__all__


def test_route_decision_is_a_string_at_runtime():
    """``Literal["direct", "retrieve"]`` is a typing alias for
    ``str`` at runtime — values compare equal to the literal
    strings. This pins the contract existing callers (FSM event
    consumers, runner, citation helpers) rely on.
    """
    from src.agent.types import RouteDecision

    # Runtime type of a Literal-typed value is ``str``.
    sample: RouteDecision = "direct"
    assert isinstance(sample, str)


# ---------------------------------------------------------------------------
# 2. FSMEvent.route_decision field declaration
# ---------------------------------------------------------------------------


def test_fsm_event_declares_route_decision_field():
    """``FSMEvent`` is a ``TypedDict(total=False)`` and now declares
    ``route_decision`` so the runner's ``fsm_evt.get("route_decision")``
    has a typed in-process contract (the value is dropped at the wire
    — see :mod:`src.agent.events` — but the in-process FSM ↔ runner
    boundary is still typed).
    """
    from src.agent.fsm import FSMEvent
    from src.agent.types import RouteDecision

    hints = get_type_hints(FSMEvent, include_extras=False)
    assert "route_decision" in hints
    # The annotation is the RouteDecision Literal (NotRequired is a
    # typing-extension-only marker — for ``total=False`` TypedDict the
    # Optionality comes from the ``total=False`` itself, so the raw
    # annotation should be the Literal).
    assert hints["route_decision"] is RouteDecision


def test_fsm_event_route_decision_is_optional_via_total_false():
    """``FSMEvent(total=False)`` makes every field optional; the
    runner reads other event kinds without ever touching
    ``route_decision``. The Optionality comes from the
    ``total=False`` declaration, not from a ``NotRequired`` wrapper
    on the annotation itself.
    """
    from src.agent.fsm import FSMEvent

    # ``__optional_keys__`` is TypedDict's introspection for
    # total=False (vs ``__required_keys__`` for total=True).
    assert hasattr(FSMEvent, "__optional_keys__")
    assert "route_decision" in FSMEvent.__optional_keys__


# ---------------------------------------------------------------------------
# 3. _intent_to_route_decision return type
# ---------------------------------------------------------------------------


def test_intent_to_route_decision_return_type_is_route_decision():
    """``_intent_to_route_decision``'s return annotation is
    ``RouteDecision`` so mypy / pyright can catch any future
    branch that returns a value outside the closed vocabulary.
    """
    from src.agent.nodes.react_generate import _intent_to_route_decision
    from src.agent.types import RouteDecision

    hints = get_type_hints(_intent_to_route_decision)
    assert hints["return"] is RouteDecision


def test_intent_to_route_decision_runtime_values_unchanged():
    """Pin the runtime mapping: the typing tightening does NOT
    change behavior — every pre-Step-10 test value still maps to
    the same literal.
    """
    from src.agent.nodes.react_generate import _intent_to_route_decision

    assert _intent_to_route_decision("greeting") == "direct"
    assert _intent_to_route_decision("simple_fact") == "direct"
    assert _intent_to_route_decision("direct") == "direct"
    assert _intent_to_route_decision("summary") == "retrieve"
    assert _intent_to_route_decision("qa_complex") == "retrieve"
    assert _intent_to_route_decision(None) == "retrieve"


# ---------------------------------------------------------------------------
# 4. Citation helpers — runtime assertion catches typos
# ---------------------------------------------------------------------------


def _build_sources() -> list[dict]:
    from src.agent.legacy_helpers.doc_formatting import _to_sources

    sources = _to_sources([])
    # ``_to_sources([])`` returns ``[]``; we need a non-empty list to
    # exercise the filter / renumber branches that the assertion
    # sits in front of. Build minimal hand-rolled dicts.
    return [
        {"index": 1, "filename": "a.txt", "chunk_id": "c0", "doc_id": "d0"},
        {"index": 2, "filename": "b.txt", "chunk_id": "c1", "doc_id": "d1"},
    ]


@pytest.mark.parametrize(
    "valid_value",
    ["direct", "retrieve", "extractive", "extractive_refusal"],
)
def test_filter_sources_to_cited_accepts_valid_literal_members(valid_value):
    """The four closed-vocabulary members pass the entry assertion
    and reach the existing guard (``!= "retrieve"`` short-circuit).

    v2.0.32.5 — Phase 8 verbatim extraction values added. Pre-fix,
    ``extractive`` + ``extractive_refusal`` raised AssertionError
    here too (the same bug the eval harness caught end-to-end).
    """
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources()
    # Either short-circuit (direct) or run the legacy filter
    # (retrieve). Either way, no AssertionError.
    out = filter_sources_to_cited(sources, "见[1]", route_decision=valid_value)
    assert isinstance(out, list)


def test_filter_sources_to_cited_accepts_none():
    """``None`` keeps the legacy behavior: caller didn't pass the
    kwarg (older test paths). The assertion allows ``None``.
    """
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources()
    out = filter_sources_to_cited(sources, "见[1]", route_decision=None)
    assert isinstance(out, list)


@pytest.mark.parametrize(
    "typo",
    [
        "retreive",   # the canonical typo — easy to mistype
        "Direct",     # case-sensitive
        " direct",    # leading whitespace
        "retrieve ",  # trailing whitespace
        "hybrid",     # a hypothetical new value not added to the Literal
        "DIRECT",     # all-caps
    ],
)
def test_filter_sources_to_cited_asserts_on_typo(typo):
    """Pre-Step-10, ``route_decision="retreive"`` would silently
    take the ``direct`` short-circuit (the guard was ``!=
    "retrieve"``). Post-Step-10 it raises ``AssertionError``
    immediately so the bug surfaces in the WS stream rather than
    in a downstream "8 spurious citations" user complaint.
    """
    from src.agent.citations import filter_sources_to_cited

    sources = _build_sources()
    with pytest.raises(AssertionError) as excinfo:
        filter_sources_to_cited(sources, "见[1]", route_decision=typo)
    # Error message names the helper so the stack is debuggable.
    assert "filter_sources_to_cited" in str(excinfo.value)
    # Error message includes the bad value (helps a future
    # operator trace which FSM / runner path emitted the typo).
    assert repr(typo) in str(excinfo.value)


@pytest.mark.parametrize(
    "valid_value",
    ["direct", "retrieve", "extractive", "extractive_refusal"],
)
def test_renumber_citations_and_sources_accepts_valid_literal_members(valid_value):
    """v2.0.32.5 — Phase 8 verbatim extraction values added.

    The eval harness caught this bug end-to-end (golden-008 cases
    returned empty answers because ``extractive_refusal`` hit the
    renumber helper's AssertionError). The mock-level test pins the
    contract so a future operator doesn't have to wait for an end-to-end
    eval run to discover the gap.
    """
    from src.agent.citations import renumber_citations_and_sources

    sources = _build_sources()
    out_ans, out_sources = renumber_citations_and_sources(
        "见[1]", sources, route_decision=valid_value,
    )
    assert isinstance(out_ans, str)
    assert isinstance(out_sources, list)


def test_renumber_citations_and_sources_accepts_none():
    from src.agent.citations import renumber_citations_and_sources

    sources = _build_sources()
    out_ans, out_sources = renumber_citations_and_sources(
        "见[1]", sources, route_decision=None,
    )
    assert isinstance(out_ans, str)
    assert isinstance(out_sources, list)


@pytest.mark.parametrize(
    "typo",
    [
        "retreive",
        "Direct",
        " direct",
        "retrieve ",
        "hybrid",
        "DIRECT",
    ],
)
def test_renumber_citations_and_sources_asserts_on_typo(typo):
    """Same typo-protection contract as
    ``filter_sources_to_cited`` — the two helpers run in the same
    runner code path on every ``answer_complete`` frame, so both
    need the guard.
    """
    from src.agent.citations import renumber_citations_and_sources

    sources = _build_sources()
    with pytest.raises(AssertionError) as excinfo:
        renumber_citations_and_sources("见[1]", sources, route_decision=typo)
    assert "renumber_citations_and_sources" in str(excinfo.value)
    assert repr(typo) in str(excinfo.value)


# ---------------------------------------------------------------------------
# 5. Signature tightening — the helper type hints are RouteDecision
# ---------------------------------------------------------------------------


def test_filter_sources_to_cited_signature_is_typed():
    """The signature ``route_decision: Optional[RouteDecision] = None``
    is visible to ``inspect.signature``. We don't pin the exact
    form (``Optional[RouteDecision]`` vs ``RouteDecision | None``)
    because that's a syntactic detail of the source — only that the
    annotation resolves to ``Optional[RouteDecision]`` (or its PEP
    604 ``RouteDecision | None`` equivalent) so a future caller
    can't silently widen it back to ``str``.
    """
    from src.agent.citations import filter_sources_to_cited
    from src.agent.types import RouteDecision

    hints = get_type_hints(filter_sources_to_cited)
    ann = hints["route_decision"]
    # Acceptable shapes: ``Optional[RouteDecision]``,
    # ``Union[RouteDecision, None]``, ``RouteDecision | None``.
    # All collapse to ``Union[RouteDecision, None]`` under
    # ``get_type_hints``; the non-``None`` member must be the
    # Literal we just defined.
    non_none_members = tuple(a for a in get_args(ann) if a is not type(None))
    assert non_none_members == (RouteDecision,), (
        f"filter_sources_to_cited.route_decision annotation must "
        f"contain RouteDecision, got {ann!r}"
    )


def test_renumber_citations_and_sources_signature_is_typed():
    from src.agent.citations import renumber_citations_and_sources
    from src.agent.types import RouteDecision

    hints = get_type_hints(renumber_citations_and_sources)
    ann = hints["route_decision"]
    non_none_members = tuple(a for a in get_args(ann) if a is not type(None))
    assert non_none_members == (RouteDecision,), (
        f"renumber_citations_and_sources.route_decision annotation "
        f"must contain RouteDecision, got {ann!r}"
    )


# ---------------------------------------------------------------------------
# 6. Wire boundary — runner drops route_decision (it is NOT in events.py)
# ---------------------------------------------------------------------------


def test_answer_complete_event_builder_does_not_take_route_decision():
    """The runner does NOT forward ``route_decision`` on the WS —
    ``events.answer_complete(...)`` has no such kwarg (and Pydantic
    ``extra="ignore"`` would silently drop it anyway). The field
    stays in-process only: typed on ``FSMEvent``, asserted on the
    citation-helper entry, but invisible on the wire. This pins
    the boundary so a future runner refactor doesn't accidentally
    leak the internal routing decision to the frontend.
    """
    from inspect import signature

    from src.agent.events import answer_complete

    sig = signature(answer_complete)
    assert "route_decision" not in sig.parameters
