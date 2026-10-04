"""Pass 2: cheap-LLM judge (accuracy/completeness/relevance).

v2.0.32.0 — reuses ``build_cheap_model(temperature=0.0)`` and
``ainvoke_structured_with_fallback`` from production (Agent 3 finding:
these are the same primitives 5 production callers use).

Scoring: accuracy 0-2 / completeness 0-2 / relevance 0-2. Composite =
mean. Pass = composite ≥ 1.5.

The judge prompt follows src/llm/prompts.py convention:
  - ``_PLAIN`` companion for the structured → plain fallback path
  - "CRITICAL — ..." preamble
  - "Do NOT ..." forbidden-phrase list
  - JSON-only output directive
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from tests.eval.cost import TokenSnapshot


JUDGE_SYSTEM = """\
You are a strict RAG-output judge. Score the assistant's answer against the
ground-truth answer and the retrieved documents.

Score 0-2 on three dimensions:
  - accuracy:       0=wrong, 1=partial, 2=fully correct
  - completeness:   0=misses everything, 1=misses key facts, 2=covers all key facts
  - relevance:      0=off-topic, 1=tangential, 2=directly addresses question

CRITICAL — Do NOT paraphrase or generalize. The retrieved documents are
the source of truth; if the answer contradicts them, mark accuracy=0.

Do NOT add commentary outside the JSON object.
Do NOT use hedging words ("somewhat", "arguably", "might be").

Reply with a JSON object exactly matching:
{"accuracy": N, "completeness": N, "relevance": N, "rationale": "..."}
"""

# Plain-JSON variant — mirrors the with_structured_output → plain fallback
# pattern from src/llm/structured.py:ainvoke_structured_with_fallback.
JUDGE_SYSTEM_PLAIN = """\
You are a strict RAG-output judge. Score the assistant's answer against the
ground-truth answer and the retrieved documents.

Score 0-2 on three dimensions:
  - accuracy:       0=wrong, 1=partial, 2=fully correct
  - completeness:   0=misses everything, 1=misses key facts, 2=covers all key facts
  - relevance:      0=off-topic, 1=tangential, 2=directly addresses question

CRITICAL — Do NOT paraphrase or generalize. The retrieved documents are
the source of truth; if the answer contradicts them, mark accuracy=0.

Do NOT add commentary outside the JSON object.
Do NOT use hedging words ("somewhat", "arguably", "might be").

Reply with ONLY a JSON object (no prose, no markdown fences):
{"accuracy": N, "completeness": N, "relevance": N, "rationale": "..."}
"""


class JudgeScore(BaseModel):
    """Pydantic schema for the cheap-LLM judge's reply."""

    accuracy: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    relevance: int = Field(ge=0, le=2)
    rationale: str = ""


@dataclass(frozen=True)
class CheapLlmJudgment:
    score_accuracy: int
    score_completeness: int
    score_relevance: int
    composite: float
    rationale: str
    tokens: TokenSnapshot  # judge call's own token usage

    @property
    def passed(self) -> bool:
        return self.composite >= 1.5

    def to_dict(self) -> dict:
        return {
            "score_accuracy": self.score_accuracy,
            "score_completeness": self.score_completeness,
            "score_relevance": self.score_relevance,
            "composite": self.composite,
            "rationale": self.rationale,
            "tokens": {
                "input": self.tokens.input_tokens,
                "output": self.tokens.output_tokens,
            },
        }


def _build_user_message(
    *,
    question: str,
    expected_answer: str,
    actual_answer: str,
    retrieved_docs: str,
) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED ANSWER:\n{expected_answer}\n\n"
        f"ACTUAL ANSWER:\n{actual_answer}\n\n"
        f"RETRIEVED DOCUMENTS (truncated):\n{retrieved_docs[:3000]}"
    )


async def run_cheap_llm_judge(
    *,
    question: str,
    expected_answer: str,
    actual_answer: str,
    retrieved_docs: str,
    op_name: str = "eval_judge",
) -> Optional[CheapLlmJudgment]:
    """Call build_cheap_model + ainvoke_structured_with_fallback.

    Returns ``None`` if both structured and plain fallback fail. Caller
    decides how to surface that (default: mark pass_=False in the
    case-level aggregation).
    """
    try:
        # Lazy import — production paths not required for unit tests.
        from langchain_core.messages import SystemMessage, HumanMessage

        from src.llm.factory import build_cheap_model
        from src.llm.structured import ainvoke_structured_with_fallback
    except Exception:
        return None

    judge_llm = build_cheap_model(temperature=0.0)
    user_msg = _build_user_message(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer,
        retrieved_docs=retrieved_docs,
    )

    structured_messages = [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user_msg)]
    plain_messages = [SystemMessage(content=JUDGE_SYSTEM_PLAIN), HumanMessage(content=user_msg)]

    try:
        parsed = await ainvoke_structured_with_fallback(
            model=judge_llm,
            schema=JudgeScore,
            structured_messages=structured_messages,
            plain_messages=plain_messages,
            op_name=op_name,
        )
    except Exception:
        return None

    if parsed is None:
        return None

    composite = (parsed.accuracy + parsed.completeness + parsed.relevance) / 3.0
    # Token capture isn't wired here — the judge call sits inside the
    # test's own TokenCapture, so the snapshot is taken by the caller.
    return CheapLlmJudgment(
        score_accuracy=parsed.accuracy,
        score_completeness=parsed.completeness,
        score_relevance=parsed.relevance,
        composite=composite,
        rationale=parsed.rationale,
        tokens=TokenSnapshot(),
    )


def composite_passes(composite: float, *, threshold: float = 1.5) -> bool:
    """Public — for downstream report aggregation."""
    return composite >= threshold