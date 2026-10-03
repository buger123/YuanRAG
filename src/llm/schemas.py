"""Pydantic schemas for structured LLM outputs (route / rewrite / grade)."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    decision: Literal["retrieve", "direct"]
    reason: str = Field(default="")


class RewriteDecision(BaseModel):
    rewritten_query: str
    reason: str = Field(default="")


class DocumentGrade(BaseModel):
    index: int
    relevant: bool
    reason: str = Field(default="")


class GradeBatch(BaseModel):
    grades: List[DocumentGrade]


class HallucinationVerdict(BaseModel):
    verdict: Literal["grounded", "ungrounded", "skipped"]
    reason: str = Field(default="")


class IntentDecision(BaseModel):
    """v2.0 — pre-ReAct intent classification.

    Used by :mod:`src.agent.nodes.intent_analysis` to decide whether
    the turn takes the greeting fast-path, the summary bulk-chunks
    path, the simple-fact direct-path, or the full ReAct loop. The
    cheap-model classifier also surfaces a (best-effort) typo
    correction and a recency marker so the ReAct prompt can hint
    at the get_current_time tool without the LLM having to
    discover it organically.

    v2.0.5 — added ``"simple_fact"`` for definitional / lookup
    queries ("光速是多少?" / "水的沸点?" / "what is the capital
    of France?"). These don't need retrieval or web search, so
    they skip the ReAct loop and go straight to a direct-answer
    branch (the LLM answers from training knowledge without
    requiring a citation, the way the greeting path works).

    ``corrected_query`` is intentionally ``Optional``: returning
    ``None`` (rather than echoing the original) is the model's way
    of saying "no correction needed". Downstream code treats
    ``None`` and the original query as equivalent.

    ``needs_current_time`` combines a regex pre-filter
    (``_query_needs_time``) with the LLM's verdict via OR — false
    positives (the LLM hints at the time tool when not strictly
    needed) cost one extra tool call; false negatives (the LLM
    hallucinates a date) cost the user's trust.
    """

    intent: Literal["greeting", "summary", "simple_fact", "qa_complex"]
    corrected_query: Optional[str] = Field(default=None)
    needs_current_time: bool = Field(default=False)
    # v2.0.29.9 (Phase 8) — verbatim extraction auto-detect signal.
    # Cheap-LLM surface for "the user's query strongly suggests
    # verbatim extraction from a referenced source" (legal / medical /
    # financial / contractual / quoted-text requests). When True, the
    # FSM routes qa_complex intent to ``react_generate_extractive``
    # instead of the full ReAct loop.
    #
    # Layer 1 (zero-cost regex) in ``intent_analysis._query_is_high_precision``
    # catches obvious keywords (verbatim / 原话 / 法律规定 etc.) BEFORE the
    # LLM is invoked. This field only surfaces from Layer 2 (cheap-LLM
    # fallback) for paraphrased cases ("according to the contract, how
    # does clause 12 read?" doesn't match a regex but the LLM flags it).
    #
    # Default False — additive field, pre-Phase-8 schemas parse unchanged.
    verbatim_needed: bool = Field(default=False)
    reason: str = Field(default="")


__all__ = [
    "RouteDecision",
    "RewriteDecision",
    "DocumentGrade",
    "GradeBatch",
    "HallucinationVerdict",
    "IntentDecision",
]
