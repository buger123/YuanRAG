"""Tests for v2.0.22 (Item 7 Step 11) — error-recovery helpers (P2-B4 关掉).

Pre-Step-11, two patterns were duplicated across FSM-node except
sites:

1. **Byte-identical LLM-failure apology text** + token yield — in
   ``fsm.react_generate_direct`` and
   ``nodes.react_generate.react_generate``. Any future i18n pass
   had to touch both call sites.

2. **Inconsistent ``logger.exception`` formats** — 8 sites used
   5+ different message styles
   (``"react_generate LLM call failed"`` /
   ``"summary_path: list_chunks_by_thread failed"`` /
   ``"Retrieval failed"`` / etc.), and ``react_agent.react_agent``
   had a bare ``except Exception`` that swallowed
   ``list_documents_cached`` failures with **no log at all**
   (the silent-failure bug).

Step 11 collapses the two patterns that actually generalize:

* :func:`llm_call_failure` — single source for the apology text
  + token-yield tuple. Used by both LLM-stream answer generators.
* :func:`log_node_failure` — uniform
  ``{node_name} failed: {exc}`` format for every defensive site.

These tests pin the helpers' contracts so future edits don't
silently break i18n, log-grep, or wire-event shape.
"""
from __future__ import annotations

import ast
import re

import pytest


# ---------------------------------------------------------------------------
# 1. llm_call_failure — return shape
# ---------------------------------------------------------------------------


def test_llm_call_failure_returns_answer_text_and_token_event():
    """The helper returns ``(answer_text, token_event)`` so the
    caller can assign both in one line and yield ``token_event``
    then proceed to its normal delta path with ``answer_text`` as
    the answer.
    """
    from src.agent.nodes._recovery import llm_call_failure

    exc = RuntimeError("anthropic 503")
    answer_text, token_event = llm_call_failure("test_node", exc)

    assert isinstance(answer_text, str)
    assert token_event == ("token", {"text": answer_text})


def test_llm_call_failure_apology_byte_identical_to_pre_step11():
    """Pin the apology string byte-for-byte against the pre-Step-11
    inline copy. Any future i18n change has to land in
    ``_recovery.py``; this test catches accidental string drift in
    the helper.
    """
    from src.agent.nodes._recovery import llm_call_failure

    exc = RuntimeError("anthropic 503")
    answer_text, _ = llm_call_failure("react_generate", exc)

    expected = (
        "抱歉,生成回答时出现了问题(底层语言模型服务暂时不可用)。"
        "\n\n技术细节:anthropic 503\n\n请稍后重试。"
    )
    assert answer_text == expected


def test_llm_call_failure_apology_includes_exception_repr():
    """The apology surfaces the original ``str(exc)`` so the user
    (and the support team reading the answer stream) can see what
    actually went wrong.
    """
    from src.agent.nodes._recovery import llm_call_failure

    exc = ValueError("tool input malformed: missing 'query' field")
    answer_text, _ = llm_call_failure("react_generate", exc)

    assert "tool input malformed: missing 'query' field" in answer_text


def test_llm_call_failure_detail_override_uses_custom_string():
    """The middle clause between ``"出现了问题("`` and ``")"`` is
    parameterizable for callers that want a different failure
    reason (e.g. a token-overflow error). Default is
    ``"底层语言模型服务暂时不可用"``.
    """
    from src.agent.nodes._recovery import llm_call_failure

    exc = RuntimeError("output exceeds 8192 token limit")
    answer_text, _ = llm_call_failure(
        "react_generate", exc, detail="输出超过 token 上限",
    )
    assert "输出超过 token 上限" in answer_text
    assert "底层语言模型服务暂时不可用" not in answer_text


# ---------------------------------------------------------------------------
# 2. llm_call_failure — logging behavior
# ---------------------------------------------------------------------------


def test_llm_call_failure_logs_exception_with_node_name():
    """The helper logs at ``logger.exception`` level so the full
    traceback reaches production logs (matches the pre-Step-11
    ``logger.exception`` semantics — not ``logger.warning`` /
    ``logger.error`` which lose the traceback in JSON-formatted
    pipelines).

    Uses loguru's native capture (a sink that appends to a list)
    rather than ``caplog`` because ``src.core.logging`` uses
    loguru, which doesn't propagate to stdlib ``logging`` by
    default.
    """
    import loguru

    from src.agent.nodes._recovery import llm_call_failure

    captured: list[str] = []
    sink_id = loguru.logger.add(captured.append, level="ERROR")
    try:
        exc = RuntimeError("anthropic 503")
        llm_call_failure("react_generate_direct", exc)
    finally:
        loguru.logger.remove(sink_id)

    assert any(
        "react_generate_direct failed: anthropic 503" in line
        for line in captured
    ), f"expected log line, got {captured!r}"


def test_llm_call_failure_accepts_base_exception():
    """``except Exception`` would not catch ``BaseException``; the
    helper signature widens to ``BaseException`` so a caller that
    wraps ``except BaseException`` (e.g. for cleanup ordering)
    can still delegate without TypeError.
    """
    from src.agent.nodes._recovery import llm_call_failure

    exc = BaseException("synthetic BaseException")
    answer_text, token_event = llm_call_failure("test", exc)
    assert "synthetic BaseException" in answer_text
    assert token_event == ("token", {"text": answer_text})


# ---------------------------------------------------------------------------
# 3. log_node_failure — logging behavior
# ---------------------------------------------------------------------------


def test_log_node_failure_logs_uniform_format():
    """The helper logs at ``logger.exception`` level with the
    ``{node_name} failed: {exc}`` format — replacing the 5+ bespoke
    pre-Step-11 formats
    (``"react_generate LLM call failed"`` /
    ``"summary_path: list_chunks_by_thread failed"`` /
    ``"Retrieval failed"`` / etc.). The uniform format means a
    single ``grep "failed:" logs/`` reliably identifies any FSM-node
    exception.
    """
    import loguru

    from src.agent.nodes._recovery import log_node_failure

    captured: list[str] = []
    sink_id = loguru.logger.add(captured.append, level="ERROR")
    try:
        exc = RuntimeError("bm25 future exploded")
        log_node_failure("retrieve.bm25_search", exc)
    finally:
        loguru.logger.remove(sink_id)

    assert any(
        "retrieve.bm25_search failed: bm25 future exploded" in line
        for line in captured
    ), f"expected log line, got {captured!r}"


def test_log_node_failure_returns_none():
    """The helper is a logging side-effect only; it returns
    ``None``. Callers handle the recovery decision inline.
    """
    from src.agent.nodes._recovery import log_node_failure

    assert log_node_failure("any", RuntimeError("x")) is None


def test_log_node_failure_accepts_base_exception():
    """Signature mirrors ``llm_call_failure`` — accepts the full
    ``BaseException`` hierarchy, not just ``Exception``.
    """
    from src.agent.nodes._recovery import log_node_failure

    # Must not raise.
    log_node_failure("any", BaseException("synthetic"))


def test_log_node_failure_includes_traceback():
    """``logger.exception`` (vs ``logger.error``) attaches the
    current exception's traceback to the log record. Pre-Step-11
    the storage-failure sites already used ``logger.exception``;
    this test pins that we don't regress to ``logger.warning``
    which would silently drop the traceback in production.

    Uses loguru with the ``format=``-aware sink so we capture the
    rendered string (loguru inlines the traceback into the message
    when ``diagnose=True`` / default sink).
    """
    import loguru

    from src.agent.nodes._recovery import log_node_failure

    captured: list[str] = []
    sink_id = loguru.logger.add(captured.append, level="ERROR")
    try:
        try:
            raise ValueError("traceback carrier")
        except ValueError as exc:
            log_node_failure("test_node", exc)
    finally:
        loguru.logger.remove(sink_id)

    matching = [
        line for line in captured
        if "test_node failed: traceback carrier" in line
    ]
    assert matching, (
        f"no log record with expected message, got {captured!r}"
    )
    # loguru's default sink inlines the traceback into the message
    # string, so the captured line itself contains the frames —
    # there's no separate ``exc_info`` attribute to check.
    assert "Traceback" in matching[0], (
        "logger.exception must include the traceback; "
        "logger.warning/error would drop it"
    )


# ---------------------------------------------------------------------------
# 4. Module surface
# ---------------------------------------------------------------------------


def test_recovery_module_exports():
    """``__all__`` exposes both helpers so callers can
    ``from src.agent.nodes._recovery import llm_call_failure,
    log_node_failure`` without reaching into private module state.
    """
    from src.agent.nodes import _recovery

    assert "llm_call_failure" in _recovery.__all__
    assert "log_node_failure" in _recovery.__all__


# ---------------------------------------------------------------------------
# 5. AST guard — apology text is centralized
# ---------------------------------------------------------------------------


def test_apology_text_only_lives_in_recovery_module():
    """Pre-Step-11, the apology string was duplicated byte-for-byte
    in two places (``fsm.py:react_generate_direct`` and
    ``react_generate.py:react_generate``). Post-Step-11 the string
    lives in ``_recovery.py`` only — any future reintroduction of
    the literal in those two nodes is a regression (Step 11's
    centralization guarantee).

    Scoped to ``fsm.py`` and ``react_generate.py`` specifically
    because other modules (notably ``runner.py``) have their own
    "抱歉,生成回答时出现了问题" string in a *different* context
    (``events.error(...)`` for the WS-level error frame on a
    general exception, not the LLM-stream fallback) — that's not
    the apology text Step 11 centralizes.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    needle = "抱歉,生成回答时出现了问题"
    # Scoped — see docstring.
    target_files = [
        repo / "src" / "agent" / "fsm.py",
        repo / "src" / "agent" / "nodes" / "react_generate.py",
    ]

    offenders: list[str] = []
    for path in target_files:
        if needle in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(repo)))

    assert not offenders, (
        f"Step 11 centralizes the LLM-failure apology text in "
        f"src/agent/nodes/_recovery.py. Offending files (must use "
        f"llm_call_failure instead): {offenders}"
    )


def test_react_generate_direct_uses_helper():
    """AST-level guard: ``react_generate_direct`` in fsm.py must
    invoke ``llm_call_failure`` (no inline apology). Catches
    accidental regressions where a future contributor copies the
    apology block back into the LLM stream's except clause.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fsm_src = (repo_root / "src" / "agent" / "fsm.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(fsm_src)

    func = next(
        n for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "react_generate_direct"
    )
    # Walk ONLY the function body (statements), not the args/kwargs
    # descriptors which don't have lineno.
    body_src_lines: list[int] = []
    for stmt in func.body:
        body_src_lines.extend(range(stmt.lineno, stmt.end_lineno + 1))
    src_lines = fsm_src.splitlines()
    body_src = "\njoin".join(src_lines[i - 1] for i in body_src_lines)
    assert "llm_call_failure" in body_src, (
        "react_generate_direct must call llm_call_failure in its "
        "except clause; the inline apology block has been moved to "
        "_recovery.llm_call_failure (P2-B4)."
    )


def test_react_generate_uses_helper():
    """Mirror guard for ``nodes/react_generate.py::react_generate``."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    rg_path = (
        repo_root / "src" / "agent" / "nodes" / "react_generate.py"
    )
    rg_src = rg_path.read_text(encoding="utf-8")
    tree = ast.parse(rg_src)

    func = next(
        n for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "react_generate"
    )
    body_src_lines: list[int] = []
    for stmt in func.body:
        body_src_lines.extend(range(stmt.lineno, stmt.end_lineno + 1))
    src_lines = rg_src.splitlines()
    body_src = "\n".join(src_lines[i - 1] for i in body_src_lines)
    assert "llm_call_failure" in body_src, (
        "react_generate must call llm_call_failure in its except "
        "clause (P2-B4)."
    )


# ---------------------------------------------------------------------------
# 6. Logging format uniformity across call sites
# ---------------------------------------------------------------------------


def _all_log_failure_callers():
    """AST scan: every ``log_node_failure`` call site in src/."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    src = repo / "src" / "agent"
    callers: list[tuple[str, int, str]] = []
    for path in src.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "log_node_failure"
            ):
                # Pull the node_name arg if it's a string literal.
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    callers.append((
                        str(path.relative_to(repo)),
                        node.lineno,
                        node.args[0].value,
                    ))
    return callers


def test_all_log_node_failure_callers_use_string_node_name():
    """Every ``log_node_failure`` call passes a string literal as
    ``node_name`` (so the log message is grep-uniform). Catches
    accidental ``log_node_failure(some_variable, exc)`` calls that
    would break ``grep "<node_name> failed:"``.
    """
    callers = _all_log_failure_callers()
    assert callers, "expected multiple log_node_failure call sites"
    for path, lineno, name in callers:
        assert re.match(r"^[a-z][a-z0-9_.]*$", name), (
            f"{path}:{lineno} — node_name {name!r} should be "
            f"lowercase dotted identifier (got non-string-arg call?)"
        )


def test_log_node_failure_callers_cover_all_pre_step11_loggers():
    """Pre-Step-11 sites that used ``logger.exception`` for
    recoverable failures must now route through the helper. This
    test asserts the centralized sites cover what was there before
    (so the audit gap "what nodes failed?" stays grep-able).

    Note: ``react_generate_direct`` is NOT in this set — it uses
    ``llm_call_failure`` (the LLM-failure-specific helper), not
    ``log_node_failure``. The two helpers are siblings, not
    duplicates.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    src = repo / "src" / "agent"
    # Collect unique node_name arguments across all log_node_failure
    # callers.
    names = {name for _, _, name in _all_log_failure_callers()}

    # Pre-Step-11 exception-logging sites (5+ distinct message
    # formats). Each must have a matching log_node_failure caller
    # that subsumes it (same conceptual site, possibly renamed).
    expected_subsumed = {
        "summary_path",            # fsm.py:386 (was "...list_chunks_by_thread failed")
        "retrieve",                # retrieve.py:217 (was "Retrieval failed")
        "fsm.save_state",          # fsm.py:760 (was "...end-of-turn snapshot...")
        "react_agent.list_documents_cached",  # react_agent.py:244 (was SILENT)
    }
    missing = expected_subsumed - names
    assert not missing, (
        f"these pre-Step-11 exception-logging sites must route "
        f"through log_node_failure: {missing}"
    )


# ---------------------------------------------------------------------------
# 7. Apology text i18n readiness (forward-compat sanity)
# ---------------------------------------------------------------------------


def test_apology_constants_are_module_level_not_class():
    """The apology prefix / detail / tail are module-level string
    constants so a future i18n loader (e.g. reading from
    ``settings.i18n.apology_zh``) can rebind them without
    monkeypatching class instances. This is a forward-compat
    sanity check, not a behavior assertion.
    """
    import src.agent.nodes._recovery as rec

    assert isinstance(rec._LLM_APOLOGY_PREFIX, str)
    assert isinstance(rec._LLM_APOLOGY_DETAIL_DEFAULT, str)
    assert isinstance(rec._LLM_APOLOGY_TAIL, str)
    assert rec._LLM_APOLOGY_PREFIX.startswith("抱歉")


# ---------------------------------------------------------------------------
# 8. smoke — helpers compose without monkeypatching
# ---------------------------------------------------------------------------


def test_llm_call_failure_returns_node_event_tuple_consumer_ready():
    """The returned ``token_event`` tuple can be yielded directly
    by an async-generator node — confirms the shape matches the
    FSMEvent forwarding contract
    (kind, payload_dict). The runner will pass it through as
    ``events.token(text)`` on the WS.
    """
    from src.agent.nodes._recovery import llm_call_failure

    exc = ConnectionError("upstream timeout")
    answer_text, token_event = llm_call_failure("react_generate", exc)

    # tuple[Literal["token"], dict[str, str]] — matches FSMEvent
    # forwarding contract.
    assert isinstance(token_event, tuple)
    assert len(token_event) == 2
    kind, payload = token_event
    assert kind == "token"
    assert isinstance(payload, dict)
    assert payload["text"] == answer_text
