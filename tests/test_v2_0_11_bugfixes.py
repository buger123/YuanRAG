"""v2.0.11 — 「Agent 思考步骤超过 max_steps=10」 GraphRecursionError 全治 regression tests.

Bug history
-----------
The v2.0 ReAct graph (rewrite from v1.1.13) caps ``recursion_limit`` via
``AGENT_BUDGETS["max_steps"]`` and passes it to ``graph.astream(inputs,
cfg, ...)`` in ``src/agent/runner.py``. v2.0 set this to **10**, which
covered most queries but blew up on legitimate multi-tool turns:

* ReAct topology: 1 (intent) + 2*N (agent + tools loop iterations) +
  1 (generate) + 1 (judge). A query that needs 3 retrieve + 1 web +
  1 final = 8 tool-loop steps + entry/exit = 10. With LangGraph's
  conditional edge overhead we landed at exactly the cap and a real
  multi-tool query hit ``GraphRecursionError``.
* The ``except Exception as exc`` block in ``stream_agent`` caught the
  error but emitted a **generic** "抱歉,生成回答时出现了问题" message
  via ``redact_exception``. No actionable signal — the user couldn't tell
  WHY the agent stopped or WHAT to do.
* The loop breaker in ``react_agent`` only fired on **same-tool
  hammer** (``_MAX_SAME_TOOL_CALLS = 3``). It missed the "alternating"
  pattern: LLM calls ``retrieve_docs`` → ``web_search`` →
  ``retrieve_docs`` → ``web_search`` → ... which racks up 6+ tool calls
  without any single tool hitting 3.

真根因 (root cause)
-------------------
Three layers of the same vulnerability:

1. ``AGENT_BUDGETS["max_steps"] = 10`` was a guess, not measured.
   LangGraph's default is 25; the legacy v1.x graph had separate
   per-stage caps that summed to fewer nodes per turn, so 10 used to
   feel comfortable. v2.0 ReAct has more nodes per turn.
2. The ``except Exception`` block treated ``GraphRecursionError`` as
   "any other exception" — no specific actionable message.
3. The loop breaker's "same-tool × 3" check missed the alternating
   pattern that ``recursion_limit=10`` used to mask (because the cap
   fired before the alternation could go far). With the cap raised
   to 25, the alternation pattern becomes the new tail.

修法 (fix)
----------
* Bump ``AGENT_BUDGETS["max_steps"]`` 10 → 25. Comment block in
  ``config/constants.py`` rewritten to document the topology math
  (1 + 2*N + 1 + 1) so a future maintainer doesn't naively lower
  the cap back to 10.
* Add explicit ``isinstance(exc, GraphRecursionError)`` branch at the
  top of the ``except Exception`` block. Emits an actionable message
  ("问题需要拆分为多个步骤" / "LLM 在某个工具上反复重试") with the
  resolved ``max_steps`` value. Falls through to the existing
  ``AnthropicAPIError`` / ``AnthropicInvalidRequestError`` /
  ``redact_exception`` chain only when the exception is NOT a
  ``GraphRecursionError``.
* Extend loop breaker with ``_MAX_TOTAL_TOOL_CALLS = 6``. Fires
  whenever ``sum(tool_counts.values()) >= 6``, regardless of per-tool
  distribution. Catches the alternating pattern that the same-tool
  hammer missed.

These tests pin all three layers so a future refactor that lowers
the budget, drops the specific handler, or skips the total-tool
breaker fails CI immediately.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = REPO_ROOT / "config" / "constants.py"
RUNNER = REPO_ROOT / "src" / "agent" / "runner.py"
REACT_AGENT = REPO_ROOT / "src" / "agent" / "nodes" / "react_agent.py"


# ============================================================================
# 1. AGENT_BUDGETS["max_steps"] is at least 25
# ============================================================================


class TestAgentBudgetsMaxSteps:
    """The ReAct recursion budget must cover legitimate multi-tool
    queries. v2.0 set this to 10 (a guess from the legacy graph);
    user hit ``GraphRecursionError`` on a real multi-corpus query
    that needed 3 retrieve + 1 web + 1 final. v2.0.11 bumps to 25
    (LangGraph default) to give margin."""

    def test_max_steps_is_at_least_25(self):
        """v2.0.11 — ``AGENT_BUDGETS["max_steps"]`` must be ``>= 25``.
        Pre-fix was 10, which recursed on legitimate multi-tool
        queries."""
        text = CONSTANTS.read_text(encoding="utf-8")
        # Match the dict literal `AGENT_BUDGETS = {...}` body. The
        # key order is stable but pin only the value to avoid
        # false-positive breakage on cosmetic refactors.
        m = re.search(
            r'AGENT_BUDGETS\s*=\s*\{[^}]*"max_steps"\s*:\s*(\d+)',
            text,
        )
        assert m, (
            "config/constants.py missing AGENT_BUDGETS dict with "
            "'max_steps' key — the budget config moved"
        )
        max_steps = int(m.group(1))
        assert max_steps >= 25, (
            f"v2.0.11: AGENT_BUDGETS['max_steps'] = {max_steps}; "
            f"must be >= 25 (LangGraph default) to cover legitimate "
            f"multi-tool queries. Pre-fix was 10, which recursed on "
            f"queries needing 3+ retrieve + web + final."
        )

    def test_resolve_max_steps_clamps_to_max_100(self):
        """v2.0 wire-compat — ``_resolve_max_steps`` must accept
        ``RAG_REACT_MAX_STEPS`` env var AND clamp to ``[1, 100]``.
        Pre-existing behavior, pinned here so a future refactor
        can't accidentally drop the upper bound (which guards
        against a misconfigured prod env sending max=10000 and
        exhausting the LLM budget)."""
        text = CONSTANTS.read_text(encoding="utf-8")
        # Locate the _resolve_max_steps function body. Walk from
        # the def to the next top-level def or end-of-file.
        idx = text.find("def _resolve_max_steps")
        assert idx != -1, "constants.py missing _resolve_max_steps"
        next_def = text.find("\ndef ", idx + 1)
        body = text[idx:next_def if next_def != -1 else len(text)]
        assert "RAG_REACT_MAX_STEPS" in body, (
            "_resolve_max_steps must read RAG_REACT_MAX_STEPS env var"
        )
        assert re.search(r"if\s+1\s*<=\s*v\s*<=\s*100", body), (
            "_resolve_max_steps must clamp to [1, 100] — without the "
            "upper bound a misconfigured env can blow up the LLM "
            "budget (e.g. typo setting RAG_REACT_MAX_STEPS=10000)"
        )

    def test_constants_documents_topology_math(self):
        """v2.0.11 — the constants.py comment block must explain
        WHY 25 is the right number (the topology math: 1 + 2*N + 1
        + 1 = node count per turn). Without this, a future
        maintainer might naively lower the cap back to 10 thinking
        "10 should be enough"."""
        text = CONSTANTS.read_text(encoding="utf-8")
        # The comment uses parenthesized node names between the
        # numbers — e.g. `1 (intent) + 2*N (react_agent + tools
        # loop iterations)`. Match the key formula `2*N` directly
        # (the part most likely to be removed by a sloppy refactor).
        assert re.search(r"\b2\s*\*\s*N\b", text), (
            "constants.py must document the ReAct topology math "
            "(2*N tool-loop iterations where N = number of rounds). "
            "Without this, a future maintainer might lower the cap "
            "back to 10 thinking it's 'enough'."
        )
        # Also check the four node names are all mentioned in the
        # topology section — the math is meaningless without naming
        # what each term counts.
        for node in ("intent", "react_agent", "react_generate",
                     "check_hallucination"):
            assert node in text, (
                f"constants.py topology comment must name the "
                f"`{node}` node — without naming each term the math "
                f"`1 + 2*N + 1 + 1` is meaningless to a future reader"
            )


# ============================================================================
# 2. GraphRecursionError has its own user-facing message
# ============================================================================


class TestGraphRecursionErrorHandler:
    """``stream_agent`` must catch ``GraphRecursionError`` and emit a
    SPECIFIC actionable message instead of the generic
    ``redact_exception`` fallback. The user needs to know WHY the
    agent stopped (recursion budget exhausted) and WHAT to do
    (simplify the question / split into multiple turns)."""

    def test_runner_does_not_import_graph_recursion_error(self):
        """v2.0.11 — ``runner.py`` must import ``GraphRecursionError``
        from ``langgraph.errors`` so ``isinstance`` can match the
        specific exception type.

        v2.0.21 (Phase 3 Step 7) — LangGraph has been removed
        entirely. The runner no longer imports ``GraphRecursionError``
        (either top-level or lazy inside a helper). The recursion-
        budget exception is now ``MaxStepsExceeded`` exclusively.
        This test is inverted: the import must NOT appear anywhere
        in ``runner.py``.
        """
        text = RUNNER.read_text(encoding="utf-8")
        assert not re.search(
            r"from\s+langgraph\.errors\s+import\s+GraphRecursionError",
            text,
        ), (
            "runner.py must NOT import GraphRecursionError — "
            "Step 7 removed LangGraph entirely. The recursion-"
            "budget exception is now MaxStepsExceeded."
        )
        # Also assert the helper function is gone (it was the only
        # place left where the lazy import could live).
        assert "_is_graph_recursion_error" not in text, (
            "runner.py must NOT define _is_graph_recursion_error — "
            "Step 7 removed the LangGraph helper."
        )

    def test_runner_has_explicit_recursion_error_branch(self):
        """v2.0.11 — ``stream_agent`` must check
        ``isinstance(exc, GraphRecursionError)`` BEFORE the
        ``AnthropicAPIError`` / ``redact_exception`` chain. The
        check must be inside the ``except Exception as exc:``
        block.

        v2.0.21 (Phase 3 Step 7) — the recursion-budget check is
        now ``isinstance(exc, MaxStepsExceeded)`` (no helper, no
        tuple). All four previously-accepted forms (singular
        ``GraphRecursionError``, Step-1 tuple, Phase-3 helper-based)
        have collapsed into the bare ``MaxStepsExceeded`` check.
        """
        text = RUNNER.read_text(encoding="utf-8")
        idx = text.find("except Exception as exc")
        assert idx != -1, "runner.py missing `except Exception as exc`"
        body = text[idx:idx + 1500]
        has_maxsteps = re.search(
            r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)",
            body,
        ) is not None
        # Backwards-compat: pre-Step-7 forms (singular/tuple/helper)
        # still appear in older snapshots; accept them too.
        has_singular = "isinstance(exc, GraphRecursionError)" in body
        has_tuple = bool(
            re.search(
                r"isinstance\(\s*exc\s*,\s*\(\s*GraphRecursionError\s*,\s*MaxStepsExceeded\s*\)\s*\)",
                body,
            )
        )
        has_helper = (
            "MaxStepsExceeded" in body
            and "_is_graph_recursion_error" in body
        )
        assert has_maxsteps or has_singular or has_tuple or has_helper, (
            "stream_agent must check isinstance(exc, "
            "MaxStepsExceeded) (or the legacy GraphRecursionError "
            "forms) inside the except block to give the user an "
            "actionable message "
            "instead of generic redact_exception."
        )

    def test_recursion_error_message_includes_max_steps(self):
        """v2.0.11 — the GraphRecursionError user-facing message
        must include the resolved ``max_steps`` value so the user
        can correlate the error with the operator's
        ``RAG_REACT_MAX_STEPS`` env var (or the default 25).

        v2.0.19 — also accept the Step-1 tuple form AND the
        Phase-3 helper-based form. The message body is identical
        between forms (same events.error yield); only the
        isinstance check text differs.

        Implementation note: the bare singular form
        ``isinstance(exc, GraphRecursionError)`` may appear inside
        the ``_is_graph_recursion_error`` helper. That location has
        NO user-facing message body, so we only consider it when it
        appears in the actual except block (look for the actionable
        phrase "events.error" or "max_steps=" in the next 800 chars).
        """
        text = RUNNER.read_text(encoding="utf-8")
        # Prefer the Phase-3 Step-7 bare form first
        # (``isinstance(exc, MaxStepsExceeded)``), then the helper /
        # tuple / singular forms for backwards-compat with older
        # snapshots.
        maxsteps_match = re.search(
            r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)",
            text,
        )
        helper_match = re.search(
            r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)\s*or\s*_is_graph_recursion_error",
            text,
        )
        tuple_match = re.search(
            r"isinstance\(\s*exc\s*,\s*\(\s*GraphRecursionError\s*,\s*MaxStepsExceeded\s*\)\s*\)",
            text,
        )
        # Singular form: only consider it if it has the actionable
        # body following (avoid the helper's internal isinstance).
        singular_idx = -1
        for m in re.finditer(
            r"isinstance\(\s*exc\s*,\s*GraphRecursionError\s*\)",
            text,
        ):
            body_window = text[m.start():m.start() + 800]
            if ("max_steps" in body_window
                    and ("events.error" in body_window
                         or "max_steps=" in body_window)):
                singular_idx = m.start()
                break
        candidates = [i for i in (
            maxsteps_match.start() if maxsteps_match else -1,
            helper_match.start() if helper_match else -1,
            tuple_match.start() if tuple_match else -1,
            singular_idx,
        ) if i != -1]
        assert candidates, (
            "runner.py missing the MaxStepsExceeded isinstance "
            "check (Step-7 bare form, helper-based form, or "
            "Step-1 tuple form)"
        )
        idx = min(candidates)
        body = text[idx:idx + 800]
        assert "max_steps" in body, (
            "The MaxStepsExceeded branch must reference "
            "`max_steps` in the user-facing message so the user "
            "knows what budget was hit."
        )
        has_resolve = "_resolve_max_steps" in body
        has_attr = 'getattr(exc, "max_steps", None)' in body or "getattr(exc, 'max_steps', None)" in body
        assert has_resolve or has_attr, (
            "The recursion-error branch must surface the actual "
            "budget: either via `_resolve_max_steps()` or via "
            "`getattr(exc, \"max_steps\", None)` (Phase 3 FSM path)."
        )

    def test_recursion_error_message_has_actionable_advice(self):
        """v2.0.11 — the user-facing message must tell the user
        WHAT to do (split the question, simplify). Without this
        the user is left guessing why their query stopped.

        v2.0.21 (Phase 3 Step 7) — the isinstance check is now
        ``isinstance(exc, MaxStepsExceeded)`` (no helper, no tuple).
        The actionable body that follows the check is unchanged
        from the Phase-2 / Step-3 forms."""
        text = RUNNER.read_text(encoding="utf-8")
        maxsteps_match = re.search(
            r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)",
            text,
        )
        helper_match = re.search(
            r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)\s*or\s*_is_graph_recursion_error",
            text,
        )
        tuple_match = re.search(
            r"isinstance\(\s*exc\s*,\s*\(\s*GraphRecursionError\s*,\s*MaxStepsExceeded\s*\)\s*\)",
            text,
        )
        singular_idx = -1
        for m in re.finditer(
            r"isinstance\(\s*exc\s*,\s*GraphRecursionError\s*\)",
            text,
        ):
            body_window = text[m.start():m.start() + 1200]
            if ("max_steps" in body_window
                    and ("拆分" in body_window
                         or "简化" in body_window
                         or "events.error" in body_window)):
                singular_idx = m.start()
                break
        candidates = [i for i in (
            maxsteps_match.start() if maxsteps_match else -1,
            helper_match.start() if helper_match else -1,
            tuple_match.start() if tuple_match else -1,
            singular_idx,
        ) if i != -1]
        assert candidates, (
            "runner.py missing the MaxStepsExceeded isinstance "
            "check (Step-7 bare form, helper-based form, or "
            "Step-1 tuple form)"
        )
        idx = min(candidates)
        body = text[idx:idx + 1200]
        actionable = [
            "拆分",  # 拆分为多个步骤 / 拆分为多个对话
            "简化",  # 简化问题
            "反复重试",  # LLM 在某个工具上反复重试
        ]
        assert any(p in body for p in actionable), (
            f"The MaxStepsExceeded user-facing message must "
            f"include actionable advice (one of {actionable!r}). "
            f"A bare error string like 'recursion limit reached' "
            f"doesn't tell the user what to do."
        )


# ============================================================================
# 3. Loop breaker extends to total tool calls
# ============================================================================


class TestLoopBreakerTotal:
    """The same-tool hammer breaker (``_MAX_SAME_TOOL_CALLS = 3``)
    catches "LLM keeps rephrasing an unprofitable query". It
    misses the alternating pattern: ``retrieve_docs`` →
    ``web_search`` → ``retrieve_docs`` → ``web_search`` → ... which
    racks up 6+ tool calls without any single tool hitting 3.

    v2.0.11 adds ``_MAX_TOTAL_TOOL_CALLS`` so the breaker fires on
    the sum, not just on per-tool max."""

    def test_max_total_tool_calls_constant_exists(self):
        """v2.0.11 — ``_MAX_TOTAL_TOOL_CALLS`` must be defined as a
        module-level constant (not inlined in react_agent) so it
        can be tuned without code edits and the test can pin it."""
        text = REACT_AGENT.read_text(encoding="utf-8")
        m = re.search(
            r"_MAX_TOTAL_TOOL_CALLS\s*=\s*(\d+)",
            text,
        )
        assert m, (
            "react_agent.py must define _MAX_TOTAL_TOOL_CALLS as a "
            "module-level constant. Pre-v2.0.11 the loop breaker "
            "only had _MAX_SAME_TOOL_CALLS (3) which missed the "
            "alternating retrieve/web pattern."
        )
        max_total = int(m.group(1))
        # Sanity: must be > _MAX_SAME_TOOL_CALLS (otherwise the
        # same-tool check dominates and the new constant is dead).
        same_match = re.search(
            r"_MAX_SAME_TOOL_CALLS\s*=\s*(\d+)",
            text,
        )
        assert same_match
        max_same = int(same_match.group(1))
        assert max_total > max_same, (
            f"_MAX_TOTAL_TOOL_CALLS ({max_total}) must be > "
            f"_MAX_SAME_TOOL_CALLS ({max_same}) — otherwise the "
            f"same-tool check dominates and the new constant is dead."
        )

    def test_loop_breaker_checks_total_tool_calls(self):
        """v2.0.11 — the ``loop_breaker_active`` computation must
        OR the same-tool check with a total-call check. Without
        the OR, the new constant is dead code."""
        text = REACT_AGENT.read_text(encoding="utf-8")
        # The `sum(tool_counts.values())` line is usually ABOVE the
        # loop_breaker_active assignment (extract a variable, then
        # use it). Walk back ~300 chars from `loop_breaker_active`
        # to catch it.
        idx = text.find("loop_breaker_active = ")
        assert idx != -1, (
            "react_agent.py missing `loop_breaker_active = ` "
            "assignment — the loop breaker was removed or renamed"
        )
        # Walk back to capture the sum(...) line AND walk forward
        # to capture the assignment body. Total window: ~600 chars
        # centered on the assignment.
        window_start = max(0, idx - 300)
        body = text[window_start:idx + 400]
        assert "sum(" in body, (
            "react_agent.py must compute `sum(tool_counts.values())` "
            "to track total tool calls (not just per-tool max). "
            "Pre-v2.0.11 only checked per-tool."
        )
        assert "total_tool_calls" in body, (
            "react_agent.py must bind `sum(...)` to a named variable "
            "(e.g. `total_tool_calls`) and reference it in "
            "`loop_breaker_active` — without this the total-cap "
            "constant is dead code."
        )
        assert "_MAX_TOTAL_TOOL_CALLS" in body, (
            "loop_breaker_active must reference "
            "_MAX_TOTAL_TOOL_CALLS — without this the total-cap "
            "constant is dead code."
        )

    def test_max_same_tool_calls_still_exists(self):
        """v2.0.11 wire-compat — the same-tool breaker must still
        exist. The total-cap is ADDITIVE, not REPLACEMENT — the
        same-tool hammer pattern still needs to fire on its own
        even when total count is below the total cap."""
        text = REACT_AGENT.read_text(encoding="utf-8")
        m = re.search(
            r"_MAX_SAME_TOOL_CALLS\s*=\s*(\d+)",
            text,
        )
        assert m, (
            "react_agent.py must still define _MAX_SAME_TOOL_CALLS. "
            "v2.0.11 only ADDS a total-cap; it does not replace the "
            "same-tool hammer (which catches the 'LLM keeps "
            "rephrasing web_search' pattern that total-cap misses "
            "when total count is still 1-2)."
        )
        max_same = int(m.group(1))
        assert max_same >= 1 and max_same <= 5, (
            f"_MAX_SAME_TOOL_CALLS = {max_same}; expected 1-5 "
            f"(empirically 3 catches the web_search hammer pattern "
            f"without false-positive breaks on normal 2-tool turns)"
        )


# ============================================================================
# 4. Runner passes resolved max_steps to LangGraph
# ============================================================================


class TestRunnerWiring:
    """The runner must pass the RESOLVED max_steps (env-overridable)
    to LangGraph, not a hardcoded literal. Without this, the env
    var override is silently ignored."""

    def test_runner_cfg_uses_resolve_max_steps(self):
        """v2.0 wire-compat — runner must call ``_resolve_max_steps()``
        so the resolved budget flows into the FSM. A hardcoded
        ``max_steps=10`` (or 25) would silently ignore
        ``RAG_REACT_MAX_STEPS`` env var overrides.

        v2.0.19 (Phase 3) — the runner no longer passes the budget
        through LangGraph's ``recursion_limit`` cfg key; it passes
        ``max_steps=max_steps`` to ``run_fsm(...)``. Accept either
        form during the migration window.
        """
        text = RUNNER.read_text(encoding="utf-8")
        # Accept both forms: legacy LangGraph cfg key OR Phase-3 kwarg.
        has_legacy = re.search(
            r'"recursion_limit"\s*:\s*_resolve_max_steps\(',
            text,
        )
        # Phase-3: ``_resolve_max_steps()`` must be called and its
        # result passed to ``run_fsm(... max_steps=...)``. We split
        # this into two assertions so a future refactor that breaks
        # the wiring (e.g. computes the value but never passes it)
        # still trips this test.
        calls_resolve = "_resolve_max_steps()" in text
        passes_to_run_fsm = bool(
            re.search(
                r"run_fsm\([^)]*max_steps\s*=\s*max_steps\b",
                text,
                re.DOTALL,
            )
        ) or bool(
            re.search(
                r"run_fsm\([^)]*max_steps\s*=\s*_resolve_max_steps\(",
                text,
                re.DOTALL,
            )
        )
        assert has_legacy or (calls_resolve and passes_to_run_fsm), (
            "runner must pass `_resolve_max_steps()` (not a "
            "hardcoded literal) into the FSM budget. Either "
            "via LangGraph's `recursion_limit` cfg key (legacy) "
            "or via `run_fsm(..., max_steps=...)` (Phase 3). "
            "Without this, RAG_REACT_MAX_STEPS env var is "
            "silently ignored."
        )

    def test_runner_no_hardcoded_recursion_limit_literal(self):
        """v2.0.11 — runner.py must not contain a hardcoded
        ``recursion_limit: 10`` or ``recursion_limit: 25`` literal.
        The fix is to call ``_resolve_max_steps()``; a hardcoded
        literal would freeze the cap at the value chosen at refactor
        time, bypassing the env var override.

        v2.0.19 (Phase 3) — LangGraph's ``recursion_limit`` cfg key
        no longer exists in runner.py. We accept either: (a) no
        ``recursion_limit: <int>`` literals remain, OR (b) the
        migration-window form where any ``recursion_limit`` key is
        sourced from ``_resolve_max_steps()``.
        """
        text = RUNNER.read_text(encoding="utf-8")
        # Search for "recursion_limit": <int> pattern. The valid
        # form is `recursion_limit: _resolve_max_steps()`. Any
        # integer literal RHS is a regression.
        for m in re.finditer(r"recursion_limit[\"']?\s*:\s*([^\n,}]+)", text):
            rhs = m.group(1).strip()
            assert rhs.startswith("_resolve_max_steps"), (
                f"runner.py has hardcoded recursion_limit value: "
                f"`{rhs}`. v2.0.11 fix: must call "
                f"`_resolve_max_steps()` so RAG_REACT_MAX_STEPS env "
                f"var override works."
            )
        # Phase-3: any ``max_steps`` kwarg to ``run_fsm`` must use
        # the resolved value (not a hardcoded int literal).
        for m in re.finditer(
            r"run_fsm\([^)]*?max_steps\s*=\s*([^\n,)\s]+)", text, re.DOTALL
        ):
            rhs = m.group(1).strip()
            assert rhs in ("max_steps", "_resolve_max_steps()"), (
                f"runner.py has hardcoded `run_fsm(..., max_steps={rhs})`. "
                f"v2.0.19 fix: must pass `max_steps=max_steps` (where "
                f"`max_steps = _resolve_max_steps()` was bound earlier) "
                f"so RAG_REACT_MAX_STEPS env var override works."
            )


# ============================================================================
# 5. Wire compatibility — v2.0.x invariants must survive
# ============================================================================


class TestWireCompatInvariants:
    """Sanity checks that the v2.0.11 fix didn't regress the
    earlier fixes in the same release line."""

    def test_loop_break_hint_still_exists(self):
        """v2.0 wire-compat — ``_LOOP_BREAK_HINT`` constant must
        still exist. v2.0.11 only adds a total-cap threshold; the
        same-tool hint text is unchanged."""
        text = REACT_AGENT.read_text(encoding="utf-8")
        assert "_LOOP_BREAK_HINT" in text, (
            "react_agent.py missing _LOOP_BREAK_HINT — the same-tool "
            "loop break hint was removed (or renamed). The LLM "
            "needs this text to understand why it should finalize."
        )

    def test_add_chunks_atomic_flip_unchanged(self):
        """v2.0 wire-compat — ``add_chunks`` must still flip
        ``status='indexed'`` AFTER ``table.add(``."""
        store_path = REPO_ROOT / "src" / "storage" / "lancedb_store.py"
        text = store_path.read_text(encoding="utf-8")
        start = text.find("def add_chunks(")
        assert start != -1
        next_def = text.find("\ndef ", start + 1)
        body = text[start:next_def if next_def != -1 else len(text)]
        indexed_idx = body.find('status="indexed"')
        table_add_idx = body.find("table.add(")
        assert indexed_idx != -1 and table_add_idx != -1
        assert indexed_idx > table_add_idx, (
            "v2.0.11 regressed: add_chunks no longer flips "
            "status='indexed' AFTER table.add( — the v2.0 atomic "
            "flip invariant must survive"
        )

    def test_resolve_max_steps_returns_int(self):
        """v2.0 wire-compat — ``_resolve_max_steps()`` must return
        an int (not None, not str). LangGraph's recursion_limit
        requires an int."""
        from config.constants import _resolve_max_steps
        result = _resolve_max_steps()
        assert isinstance(result, int), (
            f"_resolve_max_steps() returned {type(result).__name__} "
            f"({result!r}); must be int for LangGraph's "
            f"recursion_limit config key"
        )
        assert result >= 25, (
            f"_resolve_max_steps() returned {result}; expected "
            f">= 25 (v2.0.11 default). If you set "
            f"RAG_REACT_MAX_STEPS below 25 for testing, that's "
            f"fine — but unset it for prod runs."
        )


# ============================================================================
# 6. Env var override (RAG_REACT_MAX_STEPS) wire-compat
# ============================================================================


class TestEnvVarOverride:
    """Operators can override the default via RAG_REACT_MAX_STEPS
    env var. Tests pin that:
      1. unset → returns AGENT_BUDGETS["max_steps"] (now 25)
      2. set → returns parsed int
      3. invalid (non-int / out of range) → falls back to default"""

    def test_unset_returns_default(self, monkeypatch):
        """No env var → returns the constants dict value."""
        monkeypatch.delenv("RAG_REACT_MAX_STEPS", raising=False)
        from config.constants import _resolve_max_steps, AGENT_BUDGETS
        assert _resolve_max_steps() == AGENT_BUDGETS["max_steps"]

    def test_valid_value_overrides(self, monkeypatch):
        """Valid int in [1, 100] → returns that value."""
        monkeypatch.setenv("RAG_REACT_MAX_STEPS", "15")
        from config.constants import _resolve_max_steps
        assert _resolve_max_steps() == 15

    def test_invalid_value_falls_back(self, monkeypatch):
        """Non-int or out-of-range → returns default (no exception)."""
        from config.constants import _resolve_max_steps, AGENT_BUDGETS
        default = AGENT_BUDGETS["max_steps"]
        for bad in ("not-a-number", "0", "-1", "1000", "  "):
            monkeypatch.setenv("RAG_REACT_MAX_STEPS", bad)
            assert _resolve_max_steps() == default, (
                f"_resolve_max_steps() returned non-default for "
                f"invalid env value {bad!r}; must fall back to "
                f"AGENT_BUDGETS['max_steps']"
            )


# ============================================================================
# 7. Phase 3 — MaxStepsExceeded parallel handler (v2.0.19+)
# ============================================================================


class TestMaxStepsExceededHandler:
    """v2.0.19 (Phase 3) — the FSM owns the recursion limit going
    forward and raises :class:`src.agent.errors.MaxStepsExceeded` (a
    project-local exception) instead of LangGraph's
    ``GraphRecursionError``.

    During the Steps 1-7 migration window the runner's ``isinstance``
    branch classifies BOTH exception types so the actionable
    user-facing message survives the LangGraph → FSM transition.
    Step 7 deletes the LangGraph import.

    These tests pin the new exception's import + parallel structure
    so a future refactor that breaks either side fails CI.
    """

    def test_runner_imports_max_steps_exceeded(self):
        """v2.0.19 — ``runner.py`` must import ``MaxStepsExceeded``
        from ``src.agent.errors`` so the FSM-raised exception is
        matched by ``isinstance``. Without the import, the
        actionable-message branch is silently skipped."""
        text = RUNNER.read_text(encoding="utf-8")
        assert re.search(
            r"from\s+src\.agent\.errors\s+import\s+MaxStepsExceeded",
            text,
        ), (
            "runner.py must `from src.agent.errors import "
            "MaxStepsExceeded` so the FSM's exception type can be "
            "matched in the except block. Without it the "
            "MaxStepsExceeded branch is dead."
        )

    def test_runner_classifies_only_maxsteps_after_step7(self):
        """v2.0.19 — during the migration window (Steps 1-6) the
        ``isinstance`` check accepted BOTH ``GraphRecursionError``
        (LangGraph runtime) AND ``MaxStepsExceeded`` (FSM).

        v2.0.21 (Phase 3 Step 7) — LangGraph has been removed. The
        runner's recursion-budget check is now bare
        ``isinstance(exc, MaxStepsExceeded)`` — no tuple form, no
        helper, no dual branch. The pre-Step-7 tuple / dual-branch
        forms are obsolete and must NOT appear in ``runner.py``.
        """
        text = RUNNER.read_text(encoding="utf-8")
        # The migration-window tuple form must be gone.
        has_tuple = bool(
            re.search(
                r"isinstance\(\s*exc\s*,\s*\(\s*GraphRecursionError\s*,\s*MaxStepsExceeded\s*\)\s*\)",
                text,
            )
        )
        # The dual-branch form (separate isinstance for each) must
        # also be gone.
        has_dual_branch = (
            "isinstance(exc, GraphRecursionError)" in text
            and "isinstance(exc, MaxStepsExceeded)" in text
        )
        assert not has_tuple and not has_dual_branch, (
            "runner.py must NOT classify GraphRecursionError — "
            "Step 7 removed LangGraph. The recursion-budget "
            "exception is now MaxStepsExceeded only."
        )
        # And the Step-7 bare form must be present.
        assert re.search(
            r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)",
            text,
        ), (
            "runner.py must check isinstance(exc, MaxStepsExceeded) "
            "to give the user an actionable message when the FSM "
            "hits the recursion budget."
        )

    def test_max_steps_exceeded_carries_resolved_budget(self):
        """v2.0.19 — the actionable-message branch must surface the
        resolved ``max_steps`` value (carried on
        ``MaxStepsExceeded.max_steps``) so the user sees the actual
        budget the LLM hit, not a hardcoded "25".

        v2.0.21 (Phase 3 Step 7) — the isinstance check is now the
        bare ``isinstance(exc, MaxStepsExceeded)`` form. Older
        snapshots may still have the Step-1 tuple / Phase-3 helper
        forms; accept them for backwards-compat.
        """
        text = RUNNER.read_text(encoding="utf-8")
        # Locate the actionable-message block via whichever isinstance
        # check the file currently uses. NOTE: the bare singular form
        # ``isinstance(exc, GraphRecursionError)`` may appear inside
        # the ``_is_graph_recursion_error`` helper itself — that
        # location has NO user-facing message body. To avoid
        # false-positives, prefer the Phase-3 Step-7 bare form
        # first, then the helper-based / tuple / singular forms.
        idx = -1
        # 1. Phase-3 Step-7 bare form.
        maxsteps_match = re.search(
            r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)",
            text,
        )
        if maxsteps_match:
            idx = maxsteps_match.start()
        # 2. Phase-3 helper-based form (active in Step 3-6).
        if idx == -1:
            helper_match = re.search(
                r"isinstance\(\s*exc\s*,\s*MaxStepsExceeded\s*\)\s*or\s*_is_graph_recursion_error",
                text,
            )
            if helper_match:
                idx = helper_match.start()
        # 3. Step-1 tuple form.
        if idx == -1:
            idx = text.find(
                "isinstance(exc, (GraphRecursionError, MaxStepsExceeded))"
            )
        if idx == -1:
            idx = text.find("GraphRecursionError, MaxStepsExceeded")
        # 4. Pre-Phase-3 singular form (only if it's in stream_agent's
        #    except block — look for the actionable message that
        #    always follows).
        if idx == -1:
            for m in re.finditer(
                r"isinstance\(\s*exc\s*,\s*GraphRecursionError\s*\)",
                text,
            ):
                body_window = text[m.start():m.start() + 1000]
                if "max_steps" in body_window and (
                    "events.error" in body_window
                    or "拆分" in body_window
                ):
                    idx = m.start()
                    break
        assert idx != -1, (
            "runner.py missing the MaxStepsExceeded isinstance "
            "check (Step-7 bare form, helper-based form, or "
            "Step-1 tuple form)"
        )
        body = text[idx:idx + 1000]
        assert (
            "getattr(exc, \"max_steps\", None)" in body
            or "getattr(exc, 'max_steps', None)" in body
        ), (
            "The MaxStepsExceeded branch must read `exc.max_steps` "
            "so the user-facing message shows the actual budget."
        )


class TestMaxStepsExceededException:
    """v2.0.19 — :class:`src.agent.errors.MaxStepsExceeded` itself."""

    def test_max_steps_exceeded_carries_max_steps(self):
        """The exception must carry the resolved ``max_steps`` value
        so the runner can show the actual budget in the actionable
        message."""
        from src.agent.errors import MaxStepsExceeded

        exc = MaxStepsExceeded(max_steps=42)
        assert exc.max_steps == 42
        assert "42" in str(exc)

    def test_max_steps_exceeded_is_an_exception(self):
        """``MaxStepsExceeded`` must subclass ``Exception`` so the
        runner's ``except Exception`` block catches it (and a more
        specific ``except (GraphRecursionError, MaxStepsExceeded)``
        also matches it)."""
        from src.agent.errors import MaxStepsExceeded

        assert issubclass(MaxStepsExceeded, Exception)
        with __import__("pytest").raises(MaxStepsExceeded):
            raise MaxStepsExceeded(max_steps=10)
