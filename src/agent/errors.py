"""Phase 3 — custom exceptions replacing LangGraph-coupled ones.

Before Phase 3, the runner classified LangGraph's
``GraphRecursionError`` (raised when ``recursion_limit`` is hit) and
emitted an actionable user-facing message. After Phase 3, the FSM owns
the recursion limit and raises :class:`MaxStepsExceeded` instead.

Step 7 (v2.0.21) — LangGraph has been removed entirely
(``graph.py`` + ``history.py`` deleted). The only recursion-budget
exception now is :class:`MaxStepsExceeded`. The runner's ``except``
block matches it directly via ``isinstance(exc, MaxStepsExceeded)``
— no helper, no tuple form, no ``from langgraph.errors`` import.
"""
from __future__ import annotations


class MaxStepsExceeded(Exception):
    """Raised by the FSM when ``step_count`` exceeds the resolved
    ``max_steps`` budget.

    The runner catches this in its ``except`` block and emits an
    actionable user-facing message (split the question / simplify)
    instead of the generic ``redact_exception`` fallback. Same
    contract as the pre-Phase-3 GraphRecursionError handler.

    The exception carries the resolved ``max_steps`` value as
    ``self.max_steps`` so the runner can show the actual budget the
    LLM hit (which may differ from the default if the operator
    overrode ``RAG_REACT_MAX_STEPS``).
    """

    def __init__(self, max_steps: int) -> None:
        super().__init__(f"max_steps={max_steps} exceeded")
        self.max_steps = max_steps


__all__ = ["MaxStepsExceeded"]