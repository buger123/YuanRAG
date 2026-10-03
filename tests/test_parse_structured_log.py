"""v2.0.28.4 Item 11 PR-2 P1-2 — ``parse_structured`` observability.

Pre-PR-2 ``parse_structured`` logged schema-validation failures at
``logger.debug`` level (structured.py:419). At the default INFO
log level operators chasing flaky schema validation saw NOTHING —
silent fallback to ``None`` and the caller would proceed with a
safe-default answer, masking the real issue.

This file pins:
    1. The level bump (DEBUG → WARNING).
    2. The ``op=parse_structured`` label so operators can grep
       specifically these failures (vs other WARNING sites).
    3. The secret redaction in the log message — the Pydantic error
       echoes the offending field value verbatim, and that value
       sometimes contains API keys or proxy URLs the model echoed
       back. ``redact_secrets`` masks those before they hit disk.
"""
from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel, Field


class _SampleSchema(BaseModel):
    """A minimal Pydantic model that we'll intentionally fail to
    validate against — the test content is a dict with the wrong
    type for ``count``."""
    name: str
    count: int = Field(gt=0)


# v2.0.28.4 P1-2 — Test 1
def test_parse_structured_warns_on_validation_failure():
    """Bad JSON → WARNING-level log with ``op=parse_structured`` label
    AND secret-redacted error message.

    We construct a JSON object where ``count`` is a string instead
    of int — Pydantic rejects with a ValidationError that includes
    the field value. We embed a fake API key in the field value
    and verify it gets redacted to ``[REDACTED]`` in the log.
    """
    import loguru

    from src.llm.structured import parse_structured

    captured: list[dict] = []

    def _sink(message) -> None:
        record = message.record
        captured.append(
            {"level": record["level"].name, "message": record["message"]}
        )

    sink_id = loguru.logger.add(_sink, level="DEBUG")
    try:
        # Embed a fake API key in the OFFENDING field value. Pydantic's
        # ValidationError echoes the field value verbatim in the error
        # message (and ``redact_secrets`` masks sk-/Bearer-shaped
        # substrings). Putting the secret in ``name`` (a valid field)
        # wouldn't get it echoed — the error only contains the
        # failing field's value, not the whole input.
        bad_content = '{"name": "ok", "count": "Bearer sk-abcdefghijklmnopqrstuv"}'
        result: Optional[_SampleSchema] = parse_structured(
            bad_content, _SampleSchema
        )
        assert result is None  # parse failure → None

        warnings = [c for c in captured if c["level"] == "WARNING"]
        assert len(warnings) >= 1, (
            f"expected ≥1 WARNING, got {len(warnings)}: {captured}"
        )

        # Find the parse_structured log specifically (other WARNINGs
        # might come from elsewhere — filter to ours).
        parse_logs = [w for w in warnings if "op=parse_structured" in w["message"]]
        assert len(parse_logs) == 1
        msg = parse_logs[0]["message"]

        # Op label is present.
        assert "op=parse_structured" in msg
        # Schema name is in the log for grep-ability.
        assert "_SampleSchema" in msg
        # Type name is there for diagnosis.
        assert "type=" in msg
        # Secret is redacted — the raw ``sk-...`` substring must NOT
        # appear in the log line.
        assert "sk-abcdefghijklmnopqrstuv" not in msg
        # The redaction marker is there instead.
        assert "[REDACTED]" in msg
    finally:
        loguru.logger.remove(sink_id)


# v2.0.28.4 P1-2 — Test 2
def test_parse_structured_returns_none_silently_on_missing_json():
    """When the content has NO JSON object at all, ``parse_structured``
    still returns ``None`` (existing behavior preserved). The new
    WARNING log fires only on SCHEMA validation failure — a parse
    failure (no JSON found) is its own existing ``logger.debug`` path.

    This is a regression guard: pre-PR-2 had a debug log here, and
    we don't want a WARNING flood when the LLM just returned prose
    (the route / rewrite / hallucination nodes do this all the time).
    """
    import loguru

    from src.llm.structured import parse_structured

    captured: list[dict] = []

    def _sink(message) -> None:
        record = message.record
        captured.append(
            {"level": record["level"].name, "message": record["message"]}
        )

    sink_id = loguru.logger.add(_sink, level="DEBUG")
    try:
        result = parse_structured(
            "Sorry, I cannot answer that.", _SampleSchema
        )
        assert result is None

        # No parse_structured WARNING — only the (existing) DEBUG
        # from parse_json_object might fire, but not the new WARNING.
        parse_logs = [
            c for c in captured
            if "op=parse_structured" in c["message"]
        ]
        assert parse_logs == [], (
            f"expected no op=parse_structured WARNING on missing JSON, "
            f"got: {parse_logs}"
        )
    finally:
        loguru.logger.remove(sink_id)
