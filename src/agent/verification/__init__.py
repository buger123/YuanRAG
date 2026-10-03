"""v2.0.29.7 (Phase 6) — Public API for the verification subsystem.

Per [[yuanrag-hallucination-optimization]] Phase 6: the second-pass
verification layer (rule engine + cheap LLM) lives under this
package. The FSM node ``nodes/verify_answer.py`` consumes the
public API to gate the assistant's final answer against the
retrieved documents.

Public surface
--------------
- :func:`verify_answer`            — async entry point; runs the
                                     rule engine + cheap LLM and
                                     returns a
                                     :class:`VerificationResult`.
- :func:`should_trigger_verification` — zero-cost gate that decides
                                       whether to run the verifier
                                       at all on a given turn.
- :class:`VerificationResult`      — outcome dataclass: consistent
                                     / mismatches / reason / attempts.
- :data:`VERIFICATION_MAX_ATTEMPTS`— regen budget before fallback
                                     to Phase 1 ``REFUSAL_TEMPLATES``.
- :class:`ExtractedField`          — dataclass for rule-engine output.
- :class:`Mismatch`                — dataclass for one inconsistent
                                     field.
- :func:`extract_fields`           — pull concrete-fact tokens from
                                     a string (date / currency / %
                                     / article number).
- :func:`check_consistency`        — cross-check extracted fields
                                     against a doc blob.

Why a package (not a single module)
-----------------------------------
``verifier.py`` depends on ``rule_engine.py``; both are imported
by ``nodes/verify_answer.py``. A package keeps the dependency
hierarchy clean (no circular imports) and lets a future Phase 9
swap in a GLiNER-based NER extractor without touching the call
sites.
"""
from __future__ import annotations

from src.agent.verification.rule_engine import (
    ExtractedField,
    Mismatch,
    check_consistency,
    extract_fields,
)
from src.agent.verification.verifier import (
    VERIFICATION_MAX_ATTEMPTS,
    VERIFICATION_SYSTEM,
    VerificationResult,
    should_trigger_verification,
    verify_answer,
)


__all__ = [
    "VERIFICATION_MAX_ATTEMPTS",
    "VERIFICATION_SYSTEM",
    "ExtractedField",
    "Mismatch",
    "VerificationResult",
    "check_consistency",
    "extract_fields",
    "should_trigger_verification",
    "verify_answer",
]