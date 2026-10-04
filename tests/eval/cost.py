"""Static pricing table for token → USD cost estimation.

v2.0.32.0 — hardcoded for the 6 (provider, model) pairs that cover ~95% of
deployments. ``test_pricing_table_current`` in tests/test_eval_runner.py
catches missing entries so a new model requires an explicit update.

Per Risk §7 of the plan: drift between hardcoded prices and real billing
is accepted as part of the cost story. The point of cost tracking is to
catch *deltas* between runs (e.g. a regression that doubles token
usage), not to match invoices to the cent.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TokenSnapshot:
    """Per-model token usage captured by TokenCapture."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenSnapshot") -> "TokenSnapshot":
        return TokenSnapshot(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class ModelCost:
    """USD per 1k tokens for one (provider, model)."""

    input_per_1k: Decimal
    output_per_1k: Decimal


# Pricing table — USD per 1k tokens. Sourced from public provider pricing
# pages at ship time. ``test_pricing_table_current`` enforces that every
# model we ever log tokens against has an entry here.
PRICING_USD_PER_1K: dict[tuple[str, str], ModelCost] = {
    ("anthropic", "MiniMax-M3"):  ModelCost(Decimal("0.003"),  Decimal("0.015")),
    ("anthropic", "claude-3-5-sonnet-latest"): ModelCost(Decimal("0.003"),  Decimal("0.015")),
    ("anthropic", "claude-3-5-haiku-latest"):  ModelCost(Decimal("0.0008"), Decimal("0.004")),
    ("openai", "gpt-4o"):          ModelCost(Decimal("0.005"),  Decimal("0.015")),
    ("openai", "gpt-4o-mini"):     ModelCost(Decimal("0.00015"), Decimal("0.0006")),
    ("openai", "o1"):              ModelCost(Decimal("0.015"),  Decimal("0.06")),
}


def estimate_cost(
    tokens: TokenSnapshot,
    *,
    provider: str,
    model: str,
) -> Decimal:
    """Compute USD cost for a single model's token usage.

    Falls back to ``Decimal(0)`` if the (provider, model) pair is
    missing from the pricing table. Callers should log a warning in that
    case (handled in ``runner.py``).
    """
    cost = PRICING_USD_PER_1K.get((provider, model))
    if cost is None:
        return Decimal(0)
    return (
        Decimal(tokens.input_tokens) / Decimal(1000) * cost.input_per_1k
        + Decimal(tokens.output_tokens) / Decimal(1000) * cost.output_per_1k
    )


def known_pricing_keys() -> set[tuple[str, str]]:
    """Public — for tests that enforce "every model we log has a price"."""
    return set(PRICING_USD_PER_1K.keys())