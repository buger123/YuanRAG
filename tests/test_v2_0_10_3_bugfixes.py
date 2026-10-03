"""v2.0.10.3 — 「上传文档之后,后台已经显示分析完成,但前端仍然显示向量化中,刷新之后才正常显示」regression tests.

Bug history
-----------
The original v2.0.10 fix closed a race window where the Sidebar's
``pollUntilIndexed`` exited on ``chunk_count > 0`` before the
backend flipped ``doc_registry`` to ``"indexed"``, leaving the doc
chip stuck on "向量化中(N 块)" indefinitely. That fix added the
atomic flip inside ``add_chunks`` and taught the polling to wait
for terminal status (``indexed`` / ``empty`` / ``failed``).

What v2.0.10.3 actually fixes is a different staleness path:

1. ``pollUntilIndexed`` had a 60-second budget — fine when Docling
   was the only slow stage. After ``run_ingestion`` (Docling,
   30-90 s for a typical PDF) the pipeline then runs
   ``embed_documents`` (BGE-M3, 10-30 s for a few hundred chunks).
   End-to-end realistic budget: 90-180 s. The e2e suite uses 180 s.

2. When the budget expired, ``pollUntilIndexed`` returned
   unconditionally. ``handleFile`` then removed the uploadProgress
   row, fetched ``listDocuments`` one last time, and called it a
   day. No further refresh.

3. If the backend finished AFTER the 60 s budget expired (common
   for any non-trivial file), the chip stayed at "向量化中(N 块)"
   until the user pressed F5. Backend state was already
   ``"indexed"`` — the user could query the doc in chat — but the
   sidebar UI never re-fetched.

真根因 (root cause)
-------------------
``pollUntilIndexed`` (a) had too-short a budget AND (b) had no
fallback to catch the eventual terminal flip. The two together
guaranteed a visible staleness for any upload that crossed the
60-second threshold.

修法 (fix)
----------
* Extend the polling budget from 60 s → 180 s to match realistic
  ingestion time on a large PDF + BGE-M3 batch. The e2e suite
  already used 180 s, so we're matching the existing test budget
  rather than picking a magic number.
* Make ``pollUntilIndexed`` return ``true`` when it saw terminal
  status and ``false`` when the budget expired.
* When ``false``, ``handleFile`` schedules a slow-rate
  (15-second) fallback refresh that updates the documents state
  every tick and clears itself the moment the doc reaches a
  terminal status. The fallback has a 5-minute cap so we don't
  spin forever if ingestion actually hangs.
* Track active fallback intervals in a ref + cleanup useEffect so
  unmounting mid-ingest (e.g., the user clicks a different thread)
  doesn't leave ``setInterval`` firing forever.

These tests pin all four layers so a future refactor that
shortens the budget, drops the fallback, or skips the cleanup
fails CI immediately.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SIDEBAR = REPO_ROOT / "src" / "frontend" / "src" / "components" / "Sidebar.tsx"


def _sidebar_text() -> str:
    return SIDEBAR.read_text(encoding="utf-8")


# ============================================================================
# 1. pollUntilIndexed budget is at least 180 s
# ============================================================================


class TestPollBudgetExtended:
    """The polling budget must cover realistic Docling + BGE-M3
    end-to-end latency. Anything < 180 s recreates the v2.0.10.3
    bug: upload > budget → chip stuck → user F5."""

    def test_deadline_is_at_least_180_seconds(self):
        """v2.0.10.3 — the ``deadline`` constant inside
        ``pollUntilIndexed`` must be ``>= 180_000`` ms. Pre-fix was
        ``60_000``, which races large PDFs (Docling routinely
        takes 60-90 s; + BGE-M3 10-30 s on top = 70-120 s)."""
        text = _sidebar_text()
        # Find pollUntilIndexed by name then walk to the first
        # opening brace on the next line (signature is multi-line,
        # so we can't match through to ``{`` in one regex).
        name_idx = text.find("async function pollUntilIndexed")
        assert name_idx != -1, "Sidebar.tsx missing pollUntilIndexed"
        body_start = text.find("{", name_idx)
        assert body_start != -1
        body_start += 1
        # Take the next ~1.5 KB of body — well past the deadline line.
        body = text[body_start:body_start + 1500]
        # Look for the literal "deadline = Date.now() + <number>".
        m = re.search(r"deadline\s*=\s*Date\.now\(\)\s*\+\s*(\d[\d_]*)", body)
        assert m, (
            "pollUntilIndexed missing `deadline = Date.now() + N` — "
            "could not pin the timeout value"
        )
        deadline_ms = int(m.group(1).replace("_", ""))
        assert deadline_ms >= 180_000, (
            f"v2.0.10.3: pollUntilIndexed budget is {deadline_ms} ms; "
            f"must be >= 180_000 ms (3 min) to cover Docling + BGE-M3 "
            f"end-to-end on a large PDF. Pre-fix was 60_000, which "
            f"raced any file past the 60-s mark."
        )

    def test_returns_boolean(self):
        """v2.0.10.3 — ``pollUntilIndexed`` must return ``true`` when
        it observed terminal status and ``false`` when the budget
        expired. The caller uses the boolean to decide whether to
        schedule a fallback refresh."""
        text = _sidebar_text()
        name_idx = text.find("async function pollUntilIndexed")
        assert name_idx != -1
        body_start = text.find("{", name_idx) + 1
        # The return type is on the signature line(s) BEFORE the body.
        # Search backwards from name_idx for the closing paren of
        # the param list, then the return type.
        signature = text[name_idx:body_start]
        m = re.search(r":\s*Promise<(\w+)>\s*\{", signature)
        assert m, (
            "pollUntilIndexed must declare Promise<boolean> return "
            "type so the caller can branch on terminal-vs-timeout"
        )
        assert m.group(1) == "boolean", (
            f"pollUntilIndexed return type is Promise<{m.group(1)}>; "
            f"expected Promise<boolean> for the v2.0.10.3 contract"
        )
        body = text[body_start:body_start + 4000]
        # Must have a `return true;` (terminal observed) AND
        # a `return false;` (budget expired) inside the body.
        # Use a large body window because v2.0.10.4 added a long
        # explanatory comment block above the terminal check.
        assert "return true;" in body, (
            "pollUntilIndexed must `return true` when terminal "
            "status is observed"
        )
        assert "return false;" in body, (
            "pollUntilIndexed must `return false` when budget "
            "expires — without this, the fallback refresh path "
            "in handleFile never fires"
        )


# ============================================================================
# 2. handleFile schedules fallback refresh on poll timeout
# ============================================================================


class TestFallbackRefreshOnTimeout:
    """When ``pollUntilIndexed`` times out (returns ``false``),
    ``handleFile`` must schedule a fallback refresh so the user
    sees the terminal status without a manual F5."""

    def test_handle_file_branches_on_poll_result(self):
        """v2.0.10.3 — ``handleFile`` must inspect the boolean return
        of ``pollUntilIndexed`` and call the fallback-refresh
        helper when ``false``. Without this branch the polling
        timeout is silent and the chip stays stale."""
        text = _sidebar_text()
        # Locate handleFile body.
        m = re.search(
            r"async\s+function\s+handleFile\s*\([^)]*\)\s*\{",
            text,
        )
        assert m, "Sidebar.tsx missing handleFile"
        body_start = m.end()
        body = text[body_start:body_start + 6000]
        # The pollUntilIndexed call's result must be captured in
        # a variable so we can branch on it. A bare ``await ...;``
        # would silently drop the boolean.
        assert re.search(
            r"const\s+reachedTerminal\s*=\s*await\s+pollUntilIndexed",
            body,
        ), (
            "handleFile must capture pollUntilIndexed's boolean "
            "result in a variable (e.g., `const reachedTerminal = "
            "await pollUntilIndexed(...)`). A bare `await ...;` "
            "silently drops the return value and breaks the "
            "fallback-refresh branch."
        )
        # And the !reachedTerminal branch must schedule a fallback.
        # Use a wide-open match that allows any whitespace / newlines
        # between the brace and the scheduleFallbackRefresh call so
        # v2.0.10.4's comment block doesn't break the regex.
        assert re.search(
            r"if\s*\(\s*!reachedTerminal\s*\)\s*\{[\s\S]{0,2000}?"
            r"scheduleFallbackRefresh",
            body,
        ), (
            "handleFile must call scheduleFallbackRefresh() when "
            "pollUntilIndexed returns false. Without this, a poll "
            "timeout leaves the chip stuck at 向量化中(N 块) until "
            "the user manually F5s."
        )

    def test_fallback_helper_exists_and_is_bounded(self):
        """v2.0.10.3 — the fallback-refresh helper must exist, fire
        on a slow interval (>= 10 s so the cost is negligible), and
        have a bounded duration (≤ 10 min) so we don't spin forever
        if ingestion actually hangs."""
        text = _sidebar_text()
        # Helper definition.
        m = re.search(
            r"function\s+scheduleFallbackRefresh\s*\([^)]*\)\s*\{",
            text,
        )
        assert m, (
            "Sidebar.tsx missing scheduleFallbackRefresh helper — "
            "the v2.0.10.3 fallback path is gone"
        )
        body_start = m.end()
        body = text[body_start:body_start + 2000]
        # Must use setInterval (not just setTimeout chains).
        assert "setInterval" in body, (
            "scheduleFallbackRefresh must use setInterval — a "
            "setTimeout chain is fragile (a thrown tick breaks "
            "the chain permanently)"
        )
        # Must bound the duration. Look for `>= <ms>` against an
        # elapsed counter (startedAt / Date.now()). Accept either a
        # literal (e.g. `>= 300_000`) or the v2.0.10.4 parameterized
        # form (`>= capMs`).
        assert re.search(
            r">=\s*(?:\d{3,}_?\d{3}|capMs)",
            body,
        ), (
            "scheduleFallbackRefresh must have a bounded duration "
            "(e.g., `if (elapsed >= 300_000) return` or `>= capMs` "
            "in the v2.0.10.4 generalized form). Without a bound, "
            "a hung backend leaves setInterval firing forever."
        )
        # Must stop on terminal status.
        assert re.search(
            r'status\s*===\s*"indexed"\s*\|\|'
            r'\s*status\s*===\s*"empty"\s*\|\|'
            r'\s*status\s*===\s*"failed"',
            body,
        ), (
            "scheduleFallbackRefresh must stop when the doc "
            "reaches a terminal status (indexed / empty / "
            "failed). Otherwise the interval keeps firing on a "
            "doc that's already done."
        )
        # Must call clearInterval at least once for cleanup.
        assert body.count("clearInterval") >= 2, (
            "scheduleFallbackRefresh must clearInterval on BOTH "
            "the budget-exhausted branch AND the terminal-status "
            "branch — one clearInterval means a hung backend path "
            "leaks the timer"
        )


# ============================================================================
# 3. Cleanup on unmount so a navigation away doesn't leak the timer
# ============================================================================


class TestUnmountCleanup:
    """The fallback-refresh intervals must be cleared when Sidebar
    unmounts. Otherwise navigating to a different thread mid-ingest
    leaves setInterval firing forever — and React will warn about
    state updates on an unmounted component."""

    def test_unmount_clears_fallback_intervals(self):
        """v2.0.10.3 — Sidebar must have a useEffect with an empty
        dep array whose cleanup clears every interval in the
        fallback-refresh ref. Without it, navigating away from a
        thread mid-ingest leaves a leaked timer."""
        text = _sidebar_text()
        # Locate the unmount cleanup by anchor: it must reference
        # both ``fallbackRefreshRef`` and ``clearInterval`` and
        # live inside a useEffect with empty deps. We allow the
        # cleanup fn to span multiple lines (the real code does).
        idx = text.find("fallbackRefreshRef.current.forEach")
        assert idx != -1, (
            "Sidebar.tsx must have a cleanup that calls "
            "`fallbackRefreshRef.current.forEach((id) => clearInterval(id))` "
            "to drain leaked fallback timers on unmount"
        )
        # Walk back to find the enclosing useEffect — there should
        # be a `useEffect(` within ~400 chars before the cleanup.
        before = text[max(0, idx - 400):idx]
        assert "useEffect(" in before, (
            "the `clearInterval` / `fallbackRefreshRef` cleanup "
            "must live inside a useEffect (not a bare function)"
        )
        # Walk forward to confirm empty deps.
        after = text[idx:idx + 400]
        assert "}, [])" in after or "},[])" in after, (
            "the unmount useEffect must have an EMPTY deps array "
            "`[]` so the cleanup only fires on unmount, not on "
            "every re-render"
        )

    def test_ref_tracks_active_intervals(self):
        """v2.0.10.3 — the fallback intervals must be tracked in a
        useRef so the unmount cleanup can iterate them. A plain
        const / let set would not work (closure capture stale,
        cleanup would clear the wrong ids)."""
        text = _sidebar_text()
        # Must use useRef with a Set type. Anchor on `useRef<Set<`
        # (the type parameter is the strongest signature; a `useRef([])`
        # of an array would also work but loses membership semantics
        # — pin the Set contract).
        assert re.search(
            r"useRef\s*<\s*Set\s*<",
            text,
        ), (
            "Sidebar.tsx must use `useRef<Set<...>>` to track active "
            "fallback intervals — a const / let would lose the "
            "handles across renders and the unmount cleanup would "
            "silently no-op."
        )
        # And scheduleFallbackRefresh must ADD to the set. Locate by
        # name (signature is multi-line so we can't bracket-match).
        name_idx = text.find("function scheduleFallbackRefresh")
        assert name_idx != -1, (
            "Sidebar.tsx missing scheduleFallbackRefresh helper"
        )
        body_start = text.find("{", name_idx) + 1
        body = text[body_start:body_start + 2000]
        assert "fallbackRefreshRef.current.add" in body, (
            "scheduleFallbackRefresh must add the interval id to "
            "fallbackRefreshRef.current so the unmount cleanup "
            "can find it"
        )
        # And every clearInterval branch must also delete from the set,
        # otherwise the Set grows unbounded across many uploads.
        # Invariant: each ``clearInterval(intervalId)`` is paired with
        # a ``fallbackRefreshRef.current.delete(intervalId)`` so the
        # Set stays in sync with the live timers (NOT a strict
        # adds == deletes — the budget-exhausted branch and the
        # terminal-status branch each contain one clear+delete pair,
        # and only ONE of the two will fire per interval).
        clear_count = body.count("clearInterval(intervalId)")
        delete_count = body.count(".delete(intervalId)")
        assert clear_count >= 1 and clear_count == delete_count, (
            f"scheduleFallbackRefresh has {clear_count} "
            f"`clearInterval(intervalId)` and {delete_count} "
            f"`.delete(intervalId)` — they must be 1:1 paired so "
            f"the Set doesn't leak handles when timers are cleared"
        )


# ============================================================================
# 4. Wire compatibility — v2.0.10.x invariants must survive
# ============================================================================


class TestWireCompatInvariants:
    """Sanity checks that the v2.0.10.3 fix didn't regress the
    earlier fixes in the same release line."""

    def test_add_chunks_still_flips_indexed_atomically(self):
        """v2.0.10 invariant — ``add_chunks`` must still flip
        ``status="indexed"`` AFTER ``table.add(``. v2.0.10.3 only
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
            "v2.0.10.3 (frontend fix) regressed the v2.0.10 backend "
            "atomic flip — `status='indexed'` must stay AFTER "
            "`table.add(` in add_chunks"
        )

    def test_documents_route_no_redundant_indexed_after_add_chunks(self):
        """v2.0.10 invariant — ``documents.py`` must NOT re-add a
        ``doc_registry.update("indexed")`` call after
        ``add_chunks(records)``; the flip lives in ``add_chunks``
        only. v2.0.10.3 should not touch this."""
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
            "v2.0.10.3 regressed: documents.py has a redundant "
            "doc_registry.update(status='indexed') after add_chunks. "
            "The atomic flip lives in add_chunks only."
        )

    def test_sidebar_polling_still_waits_for_terminal_status(self):
        """v2.0.10 invariant — Sidebar's ``pollUntilIndexed`` must
        STILL wait for terminal status (``indexed`` / ``empty`` /
        ``failed``) before returning true. v2.0.10.3 changed the
        budget from 60 s → 180 s and made it return a boolean; the
        terminal-status check must remain."""
        text = _sidebar_text()
        for terminal in ("indexed", "empty", "failed"):
            assert f'"{terminal}"' in text and 'status ===' in text, (
                f"v2.0.10.3 regressed: pollUntilIndexed no longer "
                f"checks status={terminal!r} as a terminal state"
            )