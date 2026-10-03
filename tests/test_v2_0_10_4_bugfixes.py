"""v2.0.10.4 — 「v2.0.10.3 fix was insufficient — race condition in pollUntilIndexed
premature-exit still strands the chip at 向量化中」regression tests.

Bug history
-----------
The v2.0.10.3 fix (``deadline 60s → 180s`` + ``scheduleFallbackRefresh``)
closed the case where ``pollUntilIndexed`` *timed out* and the chip
stayed stale. But there was a second staleness path that v2.0.10.3
made worse, not better:

The polling could exit *prematurely* (returning ``reachedTerminal=true``)
on the FIRST tick when ``mine === undefined`` — the race where the
backend's background task hadn't finished its ``doc_registry.register``
call yet. The pre-fix code defaulted:

    const status = mine?.status ?? "indexed";

— which silently turned "doc not in list yet" into "indexed", exiting
the loop before the doc ever appeared. The post-polling
``listDocuments`` then caught the doc mid-``"embedding"`` state, the
chip rendered "向量化中(N 块)", and because ``reachedTerminal===true`` the
fallback refresh was NEVER scheduled. The user saw a stuck chip until
manual F5.

真根因 (root cause)
-------------------
1. ``status = mine?.status ?? "indexed"`` defaults an
   *unobserved* doc to "indexed", turning the register-race window
   into a false-positive terminal exit.
2. The fallback refresh was only scheduled on ``!reachedTerminal``,
   so this specific exit path got zero follow-up.

修法 (fix)
----------
* Drop the "indexed" default — when ``mine`` is undefined, just
  keep polling. The first ``listDocuments`` that finds the doc will
  see its real status and the loop will exit normally.
* Surface every poll's fresh ``listDocuments`` response to the
  caller via an optional ``onDocuments`` callback so the sidebar's
  ``documents`` state mirrors the backend view in real time —
  not just on the post-polling snapshot. This means even if some
  future edge case exits polling early, the chip will already
  reflect whatever status the backend reported at that tick.
* Always schedule a fast (5 s tick / 60 s cap) fallback refresh
  after upload — the post-polling-exit case is now a near-zero
  cost (one ``listDocuments`` then self-stop) so there's no reason
  to skip it. The slow (15 s / 5 min) path is retained for the
  genuine ``!reachedTerminal`` budget-expiration case.
* Generalize ``scheduleFallbackRefresh`` to take ``tickMs`` /
  ``capMs`` parameters with the legacy values as defaults so
  the existing call sites keep working unchanged.

These tests pin all three layers (default dropped, real-time
docs mirror, always-scheduled refresh) so a future refactor that
re-introduces any of them fails CI immediately.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SIDEBAR = REPO_ROOT / "src" / "frontend" / "src" / "components" / "Sidebar.tsx"


def _sidebar_text() -> str:
    return SIDEBAR.read_text(encoding="utf-8")


def _function_body(name: str, max_bytes: int = 2500) -> str:
    """Locate ``function <name>`` (or ``async function <name>``) by
    name and return the body between the opening brace and the next
    2500 chars. Signatures are multi-line in this file so a regex
    bracket-match doesn't work."""
    text = _sidebar_text()
    idx = text.find(f"function {name}")
    assert idx != -1, f"Sidebar.tsx missing function {name}"
    body_start = text.find("{", idx) + 1
    return text[body_start:body_start + max_bytes]


# ============================================================================
# 1. status default is no longer "indexed" — the register-race fix
# ============================================================================


class TestPollStatusNoDefault:
    """The single-line cause of v2.0.10.3's hole. ``mine?.status ?? "indexed"``
    turned "doc not in list yet" into a false-positive terminal exit.
    The v2.0.10.4 fix removes the default — when ``mine`` is undefined
    we keep polling until the doc appears or the budget expires."""

    def test_status_does_not_default_to_indexed(self):
        """v2.0.10.4 — ``pollUntilIndexed`` must NOT default the status
        to ``"indexed"`` when ``mine`` is undefined. The default
        created a register-race hole where the polling exited before
        the doc ever appeared in the list, stranding the chip at
        "向量化中" because the fallback refresh was never scheduled."""
        body = _function_body("pollUntilIndexed")
        # The pre-fix bug. Multiple variants could trigger it —
        # match any of them with a flexible regex so we cover
        # `?? "indexed"`, `|| "indexed"`, ternary defaults, etc.
        forbidden = re.findall(
            r'(?:mine\s*\?\s*\.[A-Za-z_]+\s*)?(?:mine\s*\?\s*\.[A-Za-z_]+\s*)?'
            r'\?\?\s*[\'"]indexed[\'"]',
            body,
        )
        assert not forbidden, (
            "v2.0.10.4 regression: pollUntilIndexed still defaults "
            "status to 'indexed' when the doc isn't in the list — "
            f"matches: {forbidden}. This is the register-race hole "
            "that strands the chip at '向量化中' until F5."
        )

    def test_status_assignment_uses_optional_chaining(self):
        """v2.0.10.4 — ``const status = mine?.status;`` is the new
        pattern. The post-fix code uses optional chaining to leave
        //status undefined when mine is undefined; the terminal-status
        //check below then ignores the undefined value naturally.
        """
        body = _function_body("pollUntilIndexed")
        assert re.search(
            r"const\s+status\s*=\s*mine\s*\?\.\s*status\s*;",
            body,
        ), (
            "v2.0.10.4 regression: pollUntilIndexed's status "
            "assignment is not `const status = mine?.status;` — "
            "the optional-chaining no-default pattern that the "
            "fix relies on"
        )

    def test_terminal_check_excludes_undefined(self):
        """v2.0.10.4 — the terminal check must NOT match an undefined
        status. With ``mine?.status;``, status is ``undefined`` when
        the doc isn't in the list yet; the only way to avoid a
        false-positive exit is for the check to specifically compare
        against the three literal strings (``"indexed"`` /
        ``"empty"`` / ``"failed"``). A bare ``if (status)`` or
        ``if (status !== "parsing")`` would re-introduce the hole.
        """
        body = _function_body("pollUntilIndexed")
        # The expected terminal check is the literal comparison pattern
        # we landed on. Pin it tightly so a future refactor that
        # reverts to a truthy check fails CI immediately.
        assert re.search(
            r'status\s*===\s*"indexed"\s*\|\|'
            r'\s*status\s*===\s*"empty"\s*\|\|'
            r'\s*status\s*===\s*"failed"',
            body,
        ), (
            "v2.0.10.4 regression: pollUntilIndexed no longer uses "
            "the strict `status === 'indexed' || status === 'empty' || "
            "status === 'failed'` terminal check. A looser check "
            "(e.g. `if (status)` or `if (status !== 'pending')`) "
            "would let an undefined status sneak through as terminal."
        )

    def test_returns_boolean_after_no_default(self):
        """v2.0.10.4 — the Promise<boolean> contract from v2.0.10.3
        must survive: ``return true;`` on terminal observed and
        ``return false;`` on budget expiration."""
        body = _function_body("pollUntilIndexed")
        assert "return true;" in body, (
            "pollUntilIndexed must `return true` when a terminal "
            "status is observed"
        )
        assert "return false;" in body, (
            "pollUntilIndexed must `return false` when the budget "
            "expires — without this, the fallback refresh path "
            "in handleFile never fires"
        )


# ============================================================================
# 2. handleFile mirrors backend view in real time via onDocuments
# ============================================================================


class TestRealtimeDocsMirror:
    """The polling must surface every fresh ``listDocuments`` response
    to the caller so the sidebar's ``documents`` state mirrors the
    backend view in real time — closing the stale-chip window between
    upload and the eventual flip."""

    def test_poll_signature_has_on_documents_parameter(self):
        """v2.0.10.4 — ``pollUntilIndexed`` must declare the fourth
        optional ``onDocuments`` parameter so ``handleFile`` can wire
        ``setDocuments(docs)`` on every poll tick."""
        text = _sidebar_text()
        idx = text.find("async function pollUntilIndexed")
        assert idx != -1
        # Read 800 chars past the function name to span the multi-line
        # signature (param list spans 4-5 lines for this function).
        signature = text[idx:idx + 800]
        m = re.search(
            r"onDocuments\s*\?\s*:\s*\(\s*docs\s*:\s*Document\s*\[\s*\]\s*\)",
            signature,
        )
        assert m, (
            "v2.0.10.4 regression: pollUntilIndexed's signature "
            "no longer has the `onDocuments?: (docs: Document[]) => void` "
            "parameter. Without this the polling's fresh docs "
            "responses are stranded in the closure and the sidebar's "
            "documents state is stale until the post-polling snapshot."
        )

    def test_poll_calls_on_documents_per_tick(self):
        """v2.0.10.4 — every successful poll tick must call
        ``onDocuments(docs)`` so the docs state stays current. A
        single call at the bottom of the function (or only inside
        the terminal branch) would re-introduce the stale-chip
        window we just closed."""
        body = _function_body("pollUntilIndexed")
        assert re.search(
            r"onDocuments\s*\?\.\s*\(\s*docs\s*\)",
            body,
        ), (
            "v2.0.10.4 regression: pollUntilIndexed no longer calls "
            "`onDocuments?.(docs)` per tick. The docs state will "
            "stay stale during polling and re-introduce the chip "
            "stuck at '向量化中' regression."
        )

    def test_handle_file_wires_set_documents(self):
        """v2.0.10.4 — ``handleFile`` must pass an ``onDocuments``
        callback that calls ``setDocuments(docs)``. Without this
        wire-up the new parameter is dead code."""
        text = _sidebar_text()
        # Locate handleFile body.
        m = re.search(
            r"async\s+function\s+handleFile\s*\([^)]*\)\s*\{",
            text,
        )
        assert m, "Sidebar.tsx missing handleFile"
        body = text[m.end():m.end() + 3500]
        # Inside the `await pollUntilIndexed(threadId, result.doc_id,
        # (chunkCount) => {...}, <here>)` call the caller must pass
        # the setDocuments wrapper. Locate the closing paren of the
        # pollUntilIndexed call and check the trailing arg.
        start = body.find("await pollUntilIndexed(")
        assert start != -1, "handleFile missing pollUntilIndexed call"
        # Walk to the matching `)` — naive bracket matching is fine
        # since there are no nested parens deeper than 1 level in
        # this call (the chunkCount arrow has braces but no parens).
        depth = 0
        end = start
        for i, ch in enumerate(body[start:]):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break
        call = body[start:end]
        assert "setDocuments(docs)" in call, (
            "v2.0.10.4 regression: handleFile's pollUntilIndexed call "
            "no longer passes an `(docs) => setDocuments(docs)` "
            "callback as the fourth argument. Without this, the new "
            "real-time docs mirror feature is unwired and the "
            "register-race chip-stuck regression is back."
        )


# ============================================================================
# 3. fallback refresh is ALWAYS scheduled, not just on timeout
# ============================================================================


class TestFallbackAlwaysScheduled:
    """The v2.0.10.3 fix scheduled the fallback only when polling
    returned ``!reachedTerminal``. v2.0.10.3's premature-exit bug
    (``mine === undefined`` → default to "indexed" → ``reachedTerminal=true``)
    went through that hole. v2.0.10.4 unconditionally schedules a
    fast (5 s / 60 s) fallback so any future regression — even if
    the polling exits early for some reason — still gets caught."""

    def test_handle_file_unconditionally_schedules_fast_fallback(self):
        """v2.0.10.4 — ``handleFile`` must call
        ``scheduleFallbackRefresh(threadId, doc_id, 5_000, 60_000)``
        unconditionally (no ``if (!reachedTerminal)`` gate)."""
        text = _sidebar_text()
        m = re.search(
            r"async\s+function\s+handleFile\s*\([^)]*\)\s*\{",
            text,
        )
        assert m
        body = text[m.end():m.end() + 4000]
        # Find the scheduleFallbackRefresh call with the fast-path
        # arguments (5_000 tick, 60_000 cap).
        assert re.search(
            r"scheduleFallbackRefresh\s*\(\s*threadId\s*,\s*"
            r"result\.doc_id\s*,\s*5_000\s*,\s*60_000\s*\)",
            body,
        ), (
            "v2.0.10.4 regression: handleFile no longer schedules "
            "the unconditional fast-path fallback "
            "`scheduleFallbackRefresh(threadId, result.doc_id, 5_000, 60_000)`. "
            "Without this, the premature-exit bug (polling exits via "
            "the register-race hole) has no follow-up refresh and the "
            "chip stays at '向量化中' until F5."
        )

    def test_handle_file_keeps_slow_fallback_on_timeout(self):
        """v2.0.10.4 — the legacy ``!reachedTerminal`` branch must
        still schedule the slow (15 s / 5 min) fallback so a truly
        hung backend (>180 s budget) has a chance to recover."""
        text = _sidebar_text()
        m = re.search(
            r"async\s+function\s+handleFile\s*\([^)]*\)\s*\{",
            text,
        )
        assert m
        # Read a large body — handleFile's last fallback call may sit
        # 5000+ chars into the body because of all the explanatory
        # comments above it.
        body = text[m.end():m.end() + 8000]
        # The slow-path branch lives behind `if (!reachedTerminal)`.
        # Use a wide-open match that allows any whitespace / newlines
        # between the brace and the scheduleFallbackRefresh call so
        # the explanatory comment block doesn't break the regex.
        assert re.search(
            r"if\s*\(\s*!reachedTerminal\s*\)\s*\{[\s\S]{0,2000}?"
            r"scheduleFallbackRefresh\s*\(\s*threadId\s*,\s*"
            r"result\.doc_id\s*,\s*15_000\s*,\s*300_000\s*\)",
            body,
        ), (
            "v2.0.10.4 regression: handleFile no longer schedules the "
            "slow-path fallback "
            "`scheduleFallbackRefresh(threadId, result.doc_id, 15_000, 300_000)` "
            "behind `if (!reachedTerminal)`. A genuinely slow backend "
            "(>180 s) would have no recovery path."
        )


# ============================================================================
# 4. scheduleFallbackRefresh accepts tickMs / capMs parameters
# ============================================================================


class TestScheduleFallbackRefreshGeneralized:
    """The helper must accept tickMs / capMs parameters so callers
    can pick the right cadence (fast vs slow path)."""

    def test_helper_signature_has_tick_and_cap_params(self):
        """v2.0.10.4 — ``scheduleFallbackRefresh(threadId, docId,
        tickMs?: number, capMs?: number)`` with the legacy defaults."""
        text = _sidebar_text()
        idx = text.find("function scheduleFallbackRefresh")
        assert idx != -1
        # Read enough to span the multi-line signature — it sits
        # 5 lines deep into the body to fit the docstring.
        signature = text[idx:idx + 1500]
        m = re.search(
            r"tickMs\s*\??\s*:\s*number\s*=\s*15_000",
            signature,
        )
        assert m, (
            "v2.0.10.4 regression: scheduleFallbackRefresh no longer "
            "declares `tickMs?: number = 15_000` (or `tickMs: number = 15_000`) "
            "— the parameter that lets callers (handleFile) pick a faster "
            "cadence for the register-race window."
        )
        m = re.search(
            r"capMs\s*\??\s*:\s*number\s*=\s*300_000",
            signature,
        )
        assert m, (
            "v2.0.10.4 regression: scheduleFallbackRefresh no longer "
            "declares `capMs?: number = 300_000` (or `capMs: number = 300_000`) "
            "— the parameter that caps how long the fallback refresh can run."
        )

    def test_helper_uses_tick_and_cap_internally(self):
        """v2.0.10.4 — the helper body must reference ``tickMs`` and
        ``capMs`` (NOT the hard-coded ``15_000`` / ``300_000``
        constants from v2.0.10.3)."""
        body = _function_body("scheduleFallbackRefresh", max_bytes=2000)
        assert "tickMs" in body, (
            "scheduleFallbackRefresh body must reference `tickMs` "
            "(was hard-coded `15_000` in v2.0.10.3)"
        )
        assert "capMs" in body, (
            "scheduleFallbackRefresh body must reference `capMs` "
            "(was hard-coded `300_000` in v2.0.10.3)"
        )

    def test_helper_still_self_stops_on_terminal(self):
        """v2.0.10.4 — terminal-status self-stop must survive the
        generalization. Otherwise a 5-min fallback could keep
        firing forever after the doc is already indexed."""
        body = _function_body("scheduleFallbackRefresh", max_bytes=2000)
        assert re.search(
            r'status\s*===\s*"indexed"\s*\|\|'
            r'\s*status\s*===\s*"empty"\s*\|\|'
            r'\s*status\s*===\s*"failed"',
            body,
        ), (
            "v2.0.10.4 regression: scheduleFallbackRefresh no longer "
            "self-stops on terminal status. The interval would fire "
            "forever after a doc reaches indexed/empty/failed."
        )


# ============================================================================
# 5. Wire compat — v2.0.10.x invariants must survive
# ============================================================================


class TestWireCompatInvariants:
    """Sanity checks that v2.0.10.4 didn't regress the earlier
    fixes in the same release line."""

    def test_add_chunks_still_flips_indexed_atomically(self):
        """v2.0.10 invariant — ``add_chunks`` must still flip
        ``status="indexed"`` AFTER ``table.add(``. v2.0.10.4 only
        changed the frontend (Sidebar.tsx); the backend flip must
        remain intact."""
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
            "v2.0.10.4 (frontend fix) regressed the v2.0.10 backend "
            "atomic flip — `status='indexed'` must stay AFTER "
            "`table.add(` in add_chunks"
        )

    def test_documents_route_no_redundant_indexed_after_add_chunks(self):
        """v2.0.10 invariant — ``documents.py`` must NOT re-add a
        ``doc_registry.update("indexed")`` call after
        ``add_chunks(records)``; the flip lives in ``add_chunks``
        only. v2.0.10.4 should not touch this."""
        route_path = REPO_ROOT / "src" / "api" / "routes" / "documents.py"
        text = route_path.read_text(encoding="utf-8")
        add_chunks_idx = text.find("add_chunks(records)")
        assert add_chunks_idx != -1
        tail = text[add_chunks_idx: add_chunks_idx + 800]
        code_lines = [
            line for line in tail.splitlines()
            if not line.lstrip().startswith("#")
        ]
        code_tail = "\n".join(code_lines)
        assert 'doc_registry.update(' not in code_tail or \
               'status="indexed"' not in code_tail, (
            "v2.0.10.4 regressed: documents.py has a redundant "
            "doc_registry.update(status='indexed') after add_chunks. "
            "The atomic flip lives in add_chunks only."
        )

    def test_poll_budget_still_at_least_180_seconds(self):
        """v2.0.10.3 invariant — polling budget must stay >= 180 s.
        A future "let's tighten this back to 60 s" refactor would
        re-introduce the >60s budget-exit chip-stuck bug."""
        body = _function_body("pollUntilIndexed")
        m = re.search(
            r"deadline\s*=\s*Date\.now\(\)\s*\+\s*(\d[\d_]*)",
            body,
        )
        assert m, "pollUntilIndexed missing `deadline = Date.now() + N`"
        deadline_ms = int(m.group(1).replace("_", ""))
        assert deadline_ms >= 180_000, (
            f"v2.0.10.3 invariant broken: pollUntilIndexed budget "
            f"is {deadline_ms} ms; must be >= 180_000 ms. The "
            "60 s budget races any large PDF + BGE-M3 end-to-end."
        )

    def test_unmount_cleanup_still_drains_fallback_intervals(self):
        """v2.0.10.3 invariant — the unmount cleanup must still
        drain the fallback-refresh intervals. A user navigating
        away mid-ingest would leak timers otherwise."""
        text = _sidebar_text()
        idx = text.find("fallbackRefreshRef.current.forEach")
        assert idx != -1, (
            "Sidebar.tsx missing the `fallbackRefreshRef.current.forEach` "
            "unmount cleanup — navigating away mid-ingest would leak timers"
        )
        before = text[max(0, idx - 400):idx]
        after = text[idx:idx + 400]
        assert "useEffect(" in before, (
            "the cleanup must live inside a useEffect (not a bare "
            "function) so it fires on unmount"
        )
        assert "}, [])" in after or "},[])" in after, (
            "the unmount useEffect must have an EMPTY deps array "
            "`[]` so the cleanup only fires on unmount"
        )