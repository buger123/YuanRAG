"""check_hallucination node: lightweight LLM judge of answer grounding.

The same two-strategy parsing as ``route_query`` — see that docstring
for the rationale. We prefer ``with_structured_output`` and fall back to
plain-text + ``parse_structured`` for the MiniMax-style proxies that
return ``None`` from the tool-calling wrapper.

The final verdict is one of:

* ``"grounded"`` — the answer is supported by the documents
* ``"ungrounded"`` — the answer invents claims not in the docs
* ``"skipped"`` — couldn't run the judge (no docs, no answer, or both
  parsing strategies failed)
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.nodes._docs_for_hallucination import docs_for_hallucination
from src.agent.state import AgentState
from src.core.logging import logger
from src.llm.factory import build_cheap_model
from src.llm.prompts import HALLUCINATION_SYSTEM, HALLUCINATION_SYSTEM_PLAIN
from src.llm.schemas import HallucinationVerdict
from src.llm.structured import ainvoke_structured_with_fallback


# Phase 2 (v2.0.29.2) — observability hook. Increment one of three
# verdict counters + one grounding event counter per run. Skipped
# paths increment both counters because "skipped" is a legitimate
# verdict AND a grounding event was emitted.
def _emit_hallucination_metrics(verdict: str) -> None:
    """Bump the verdict + grounding counters. Tests can spy on this."""
    from src.agent.metrics import HALLUCINATION_VERDICT, GROUNDING_EVENT_EMITTED

    HALLUCINATION_VERDICT.inc(verdict=verdict)
    GROUNDING_EVENT_EMITTED.inc(status=verdict)


# v2.0.29.1 (Phase 1) — bump snippet budget from 500 → 2000 chars per
# document for the grounding judge. Pre-Phase-1 the 500-char cap was
# truncating mid-sentence on long technical / legal / medical docs —
# the cheap-model judge saw only the first 3 paragraphs of doc 1 and
# nothing of docs 2-3, so it had no ground truth to detect fabricated
# claims that the synthesis LLM sourced from doc 2's later sections.
#
# 2000 chars covers roughly:
#   - 250-400 words (English prose)
#   - 200-350 tokens (cheap-model budget is 1024 input tokens, with
#     3 docs + answer + system prompt we land at ~700-900 tokens)
#
# This is the upper bound before we start hitting the cheap-model
# context window. If a future doc is so long that even 2000 chars is
# unrepresentative, we should switch to per-paragraph sampling or
# chunk-level grounding (Phase 6's verification agent territory).
JUDGE_SNIPPET_CHARS_PER_DOC = 2000
JUDGE_SNIPPET_DOC_CAP = 3


async def check_hallucination_async(
    state: AgentState, *, step: int = 0
) -> AsyncIterator[Tuple[str, Any]]:
    """Judge whether the generated answer is supported by the retrieved documents.

    v2.0.22 (Item 7 Step 6) — async-generator protocol: yields
    exactly one ``("__delta__", {"hallucination_check": ...})``
    terminator. ``step`` is accepted-and-ignored because this node
    never emits per-step wire events of its own.

    The verdict is one of:

    * ``"grounded"`` — the answer is supported by the documents
    * ``"ungrounded"`` — the answer invents claims not in the docs
    * ``"skipped"`` — couldn't run the judge (no docs, no answer, or
      both parsing strategies failed)
    """
    answer = state.get("answer") or ""
    # v2.0.22 (Item 7 Step 12) — single source for "what counts
    # as docs" via the centralized helper (P1-B6 关掉). Pre-Step-12
    # the inline chain ``graded_documents → documents → []`` lived
    # here while the FSM predicate at ``fsm.should_check_hallucination``
    # only checked ``documents`` — the two diverged. Now both sites
    # call :func:`docs_for_hallucination`.
    docs = docs_for_hallucination(state)
    if not docs or not answer:
        # v2.0.22 (Item 7 Step 7) — yield the ``grounding`` wire
        # event BEFORE ``__delta__`` so the FSM forwards it. The FSM
        # used to emit this from a hardcoded `if current ==
        # "check_hallucination"` branch that read state after merge
        # — gone now.
        _emit_hallucination_metrics("skipped")
        yield ("grounding", {"status": "skipped"})
        yield ("__delta__", {"hallucination_check": "skipped"})
        return

    snippet = "\n\n".join(
        doc.page_content[:JUDGE_SNIPPET_CHARS_PER_DOC]
        for doc in docs[:JUDGE_SNIPPET_DOC_CAP]
    )
    model = build_cheap_model(temperature=0.0)
    user_msg = f"ANSWER:\n{answer}\n\nDOCUMENTS (truncated):\n{snippet}"

    verdict = await ainvoke_structured_with_fallback(
        model=model,
        schema=HallucinationVerdict,
        structured_messages=[
            SystemMessage(content=HALLUCINATION_SYSTEM),
            HumanMessage(content=user_msg),
        ],
        plain_messages=[
            SystemMessage(content=HALLUCINATION_SYSTEM_PLAIN),
            HumanMessage(content=user_msg),
        ],
        op_name="check_hallucination_async",
    )

    if verdict is None:
        logger.warning(
            "check_hallucination: both strategies failed; marking as skipped"
        )
        _emit_hallucination_metrics("skipped")
        yield ("grounding", {"status": "skipped"})
        yield ("__delta__", {"hallucination_check": "skipped"})
        return

    # Map verdict string → the canonical 3-tuple. Anything unexpected
    # falls back to "skipped" so the caller knows the judge didn't
    # produce a usable verdict.
    raw = verdict.verdict.lower().strip()
    if raw == "ungrounded":
        _emit_hallucination_metrics("ungrounded")
        yield ("grounding", {"status": "ungrounded"})
        yield ("__delta__", {"hallucination_check": "ungrounded"})
        return
    if raw == "grounded":
        _emit_hallucination_metrics("grounded")
        yield ("grounding", {"status": "grounded"})
        yield ("__delta__", {"hallucination_check": "grounded"})
        return
    _emit_hallucination_metrics("skipped")
    yield ("grounding", {"status": "skipped"})
    yield ("__delta__", {"hallucination_check": "skipped"})


__all__ = ["check_hallucination_async"]
