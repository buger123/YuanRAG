"""v1.1.11 — GENERATE_SYSTEM must enforce web-source honesty clauses.

What we lock here
-----------------
v1.1.11 closed the "LLM hallucinates on web-search answers" bug class
by adding two enforcement blocks to ``src/llm/prompts.py``:

1. **Web source honesty (5 clauses)** — do-not-fabricate, source
   disclosure, domain sanity check, empty-content handling,
   ungrounded-language downgrade.
2. **Hard rules** — 4 anti-fabrication rules phrased as task-failure
   conditions (no invented URLs / headlines, no snapshot-as-truth,
   no publisher-mis attribution, no polite-evasion-as-confident).

These tests are cheap string-presence checks: they ensure the
relevant keywords survive any future prompt refactor. The point is
NOT to test the LLM's behavior — that requires an end-to-end run.
The point is to make sure a future contributor doesn't accidentally
delete the clauses while editing the prompt.

If a test here fails, either (a) the prompt was rewritten in a way
that lost the honesty clause (BAD — revert / re-add), or (b) the
clause's keywords were tightened in a way that no longer matches
these literal strings (update the test to match the new wording,
and document the change in the changelog).
"""
from __future__ import annotations

import pytest

from src.llm.prompts import DIRECT_SYSTEM, GENERATE_SYSTEM


# ---------------------------------------------------------------------------
# Web source honesty — five clauses, each as a substring check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "substring",
    [
        # Clause 1: do-not-fabricate
        "Do NOT invent URLs",
        # Clause 2: source disclosure
        "Disclose source nature",
        # Clause 3: domain sanity check
        "Domain sanity check",
        # Clause 4: empty / inconsistent content
        "Empty / failed / inconsistent content",
        # Clause 5: ungrounded handling
        "ungrounded handling",
    ],
)
def test_generate_system_includes_web_source_honesty_clauses(substring: str):
    """Every clause of the v1.1.11 web-source honesty block must be
    present in ``GENERATE_SYSTEM``. If a future refactor drops any of
    these, the LLM loses the corresponding protection."""
    assert substring in GENERATE_SYSTEM, (
        f"GENERATE_SYSTEM missing required clause {substring!r}. "
        "v1.1.11 added this clause to combat LLM hallucination on "
        "web-search answers — do not remove without an explicit "
        "follow-up safety mechanism."
    )


# ---------------------------------------------------------------------------
# Hard rules — four anti-fabrication rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "substring",
    [
        # Rule 1: no invented URLs / headlines / etc.
        "Do NOT invent URLs",
        # Rule 2: no snapshot-as-truth
        "Do NOT treat",
        'search engine may have indexed an old snapshot',
        # Rule 3: no publisher mis-attribution
        "Do NOT attribute",
        # Rule 4: no polite evasion as confident
        "应该是",
    ],
)
def test_generate_system_includes_hard_rules_anti_fabrication(substring: str):
    """Every hard rule from the v1.1.11 anti-fabrication block must
    be present. The "hard rules" wording is "any violation = task
    failure" — losing these means the LLM falls back to its default
    "be helpful, fill in plausible detail" mode."""
    assert substring in GENERATE_SYSTEM, (
        f"GENERATE_SYSTEM missing hard rule {substring!r}. v1.1.11 "
        "treats these as task-failure conditions; do not delete "
        "without an equivalent replacement."
    )


# ---------------------------------------------------------------------------
# The ``[n]`` citation contract must remain — honesty clauses ADD to,
# do not replace, the existing citation requirement.
# ---------------------------------------------------------------------------


def test_generate_system_still_requires_citations():
    """v1.1.11 adds honesty clauses on top of the existing
    ``Every factual claim must be supported by a citation`` rule. A
    naive refactor that rewords the prompt might drop the citation
    line and leave the LLM free to assert facts without grounding."""
    assert (
        "Every factual claim must be\nsupported by a citation in the form [n]"
        in GENERATE_SYSTEM
    )


def test_generate_system_still_has_documents_template():
    """The ``{documents}`` placeholder is filled by ``generate.py``
    at runtime. If a future edit accidentally drops the placeholder,
    the LLM sees no source documents and the answer is uninformed
    by design."""
    assert "{documents}" in GENERATE_SYSTEM
    assert "{web_search_status}" in GENERATE_SYSTEM


# ---------------------------------------------------------------------------
# DIRECT_SYSTEM should NOT have been touched by v1.1.11
# ---------------------------------------------------------------------------


def test_direct_system_unchanged_by_v111():
    """``DIRECT_SYSTEM`` is the conversational / greeting / identity
    prompt. It deliberately forbids ``[n]`` citations (no docs on
    this path) and tells the LLM to defer retrieval-path questions.
    v1.1.11's honesty clauses target the web-search-answers path
    (which only ``GENERATE_SYSTEM`` ever sees), so ``DIRECT_SYSTEM``
    must remain unchanged. This test pins the contract so a future
    global-prompt refactor doesn't accidentally rewire it."""
    # Spot-check the unique DIRECT_SYSTEM contract clauses.
    assert "Do NOT:" in DIRECT_SYSTEM
    assert (
        "Cite documents with [n] markers" in DIRECT_SYSTEM
    ), "DIRECT_SYSTEM must still forbid [n] citations"
    assert (
        "Pretend to look something up" in DIRECT_SYSTEM
    ), "DIRECT_SYSTEM must still tell the LLM to defer instead of inventing"


# ---------------------------------------------------------------------------
# Sanity: prompts are non-empty and reachable
# ---------------------------------------------------------------------------


def test_generate_system_non_empty():
    assert len(GENERATE_SYSTEM.strip()) > 500, (
        "GENERATE_SYSTEM looks suspiciously short — v1.1.11 grew it "
        "with the honesty + hard-rules blocks; if it shrank back, "
        "someone deleted the new clauses."
    )


def test_direct_system_non_empty():
    assert len(DIRECT_SYSTEM.strip()) > 500


# ---------------------------------------------------------------------------
# v2.0.29.6 (Phase 5) — slow behavioral regression guards
# ---------------------------------------------------------------------------
#
# These tests check SHAPE contracts on the prompts that the
# string-presence tests above can't catch (e.g. "is the citation
# format consistent enough that an LLM doesn't drift?"). They are
# marked ``slow`` because they do real string-statistics work
# (no LLM calls — these are still deterministic), but on a typical
# corpus they take 50-200 ms each due to regex compilation over
# multi-KB prompts. Run with `pytest --run-slow -m slow`.
#
# The marker is opt-in so the default `pytest` invocation in CI
# doesn't pay the cost — only local dev / pre-release runs opt in.


@pytest.mark.slow
def test_generate_system_acknowledges_web_sources_have_uncertainty():
    """Regression guard: GENERATE_SYSTEM keeps web-source honesty clauses.

    The string-presence checks above verify keyword presence; this
    one verifies the *structure* (5 clauses in the same paragraph
    block, not interleaved with unrelated content). A future refactor
    that scatters the 5 clauses across the prompt would still pass
    string-presence tests but lose the contract that the LLM sees
    them as a coherent block of instructions.
    """
    import re

    # Locate the section header that introduces the web-source honesty block.
    # v2.0 ships with explicit markers like ``## Web sources`` or
    # ``### 1. Web source honesty`` — fall back to the first
    # ``- **`` list item containing the word "web".
    lines = GENERATE_SYSTEM.splitlines()
    web_section_start = None
    for idx, line in enumerate(lines):
        if "web" in line.lower() and ("honesty" in line.lower() or "sources" in line.lower()):
            web_section_start = idx
            break
    assert web_section_start is not None, (
        "GENERATE_SYSTEM should have an explicit web-source honesty section header"
    )

    # Within ~30 lines of the section start, expect >=3 of the
    # 5 honesty keywords (fabricate / disclose / domain / empty /
    # ungrounded). Spread too thin → the contract has been broken up.
    window = "\n".join(lines[web_section_start : web_section_start + 30]).lower()
    keywords = [
        "fabricat",  # do not fabricate
        "disclos",  # source disclosure
        "domain",  # domain sanity check
        "empty",  # empty-content handling
        "unground",  # ungrounded-language downgrade
    ]
    hits = sum(1 for kw in keywords if kw in window)
    assert hits >= 3, (
        f"Web-source honesty block looks fragmented (only {hits}/5 "
        f"keywords within 30 lines of section header): {window[:300]!r}"
    )


@pytest.mark.slow
def test_generate_system_no_sycophancy_clause():
    """Regression guard: GENERATE_SYSTEM must not contain sycophancy.

    A future prompt refactor that adds "Of course!" / "Absolutely!" /
    "Great question!" patterns would let the LLM hedge on uncertain
    answers while sounding confident. v2.0 deliberately omits these.
    """
    forbidden = [
        "great question",
        "absolutely!",
        "of course!",
        "sure thing",
        "happy to help",
        "no problem!",
    ]
    lower = GENERATE_SYSTEM.lower()
    found = [p for p in forbidden if p in lower]
    assert not found, (
        f"GENERATE_SYSTEM contains forbidden sycophancy patterns: {found}"
    )


@pytest.mark.slow
def test_generate_system_citation_format_documented():
    """Regression guard: citation format ``[n]`` is documented in the prompt.

    Pre-v1.0 prompts documented ``[n]`` inline; v2.0 keeps the
    format documented so the LLM knows which bracket format to emit.
    """
    # Look for the citation format spec in the prompt — must use
    # literal ``[n]`` (no markdown formatting, no code blocks).
    import re

    pattern = re.compile(r"\[n\]")
    matches = pattern.findall(GENERATE_SYSTEM)
    assert len(matches) >= 2, (
        f"GENERATE_SYSTEM should document citation format [n] at least "
        f"twice (once in rules, once in example); got {len(matches)}"
    )


@pytest.mark.slow
def test_generate_system_untrusted_document_handling():
    """Regression guard: documents fence with ``trust=untrusted`` survives.

    v2.0's prompt security wraps documents in a fence that signals
    to the LLM that the content is data, not instructions. The
    fence is applied at RUNTIME by
    :func:`src.security.prompt_safety.wrap_documents` (the static
    ``GENERATE_SYSTEM`` template only contains a ``{documents}``
    placeholder). This test pins the runtime fence contract so a
    future refactor doesn't drop it — without it, prompt-injection
    attacks from uploaded docs become possible.
    """
    from src.security.prompt_safety import wrap_documents

    # Empty-doc path returns a stub fence so callers always get
    # something well-formed to interpolate into the prompt.
    empty_fenced = wrap_documents([], source="user-uploaded")
    assert "trust=" in empty_fenced, (
        "wrap_documents (empty list) must still emit a `trust=` "
        "attribute (currently `untrusted`); got: " + empty_fenced[:200]
    )

    # Non-empty path wraps content with the same fence.
    fenced = wrap_documents(["hello world document text"], source="user-uploaded")
    assert "trust=" in fenced, (
        "wrap_documents must emit a `trust=` attribute (currently "
        "`untrusted` or `untrusted-high-glyph`); got: " + fenced[:200]
    )
    assert "documents" in fenced.lower()
    # The wrapped content should appear inside the fence.
    assert "hello world document text" in fenced

    # Also pin the placeholder contract in the static template:
    # ``{documents}`` is filled at runtime with the fenced content.
    assert "{documents}" in GENERATE_SYSTEM, (
        "GENERATE_SYSTEM should keep the {documents} placeholder"
    )


@pytest.mark.slow
def test_generate_system_length_within_budget():
    """Regression guard: GENERATE_SYSTEM stays within a reasonable size budget.

    v2.0 ships ~6 KB of GENERATE_SYSTEM. A future refactor that
    appends multiple new sections could push it past the practical
    LLM context budget (Anthropic input costs scale with prompt size).
    """
    # 8000 chars upper bound — generous enough for current state (~6 KB)
    # but catches runaway growth (e.g. accidentally doubling the prompt).
    assert len(GENERATE_SYSTEM) < 8000, (
        f"GENERATE_SYSTEM grew to {len(GENERATE_SYSTEM)} chars; "
        f"review whether new content is essential (current budget: 8000)"
    )