"""v2.0.13 regression tests — pin the fix for the
"answer_complete overwrites streamed content with whitespace" bug.

Bug summary
-----------
For tool-using queries, the LLM sometimes streams only
``<​tool_call>...</​tool_call>`` XML (the wire-syntax the backend's
``_strip_tool_blocks`` collapses to whitespace), OR emits only thinking
blocks (no visible text reaches ``answer_text`` on the backend). The
frontend's ``answer_complete`` handler used:

    canonicalContent = answerText.length > 0 ? answerText : last?.content ?? ""

The bare ``length > 0`` check accepts whitespace-only canonicals
(``" "``, ``"\\n"``) as "real content" and clobbers the accumulated
streamed text with empty space. ``markdown-body`` collapses to
``height: 0`` and the user sees thinking drawer + tool-call card with
NO answer text — until F5 loads the persisted AIMessage (whose
``content`` survived the streaming cycle intact).

Fix
---
Use ``answerText.trim().length > 0`` so whitespace-only canonicals fall
through to ``last?.content ?? ""`` — preserving the streamed
accumulation. Real canonicals still overwrite for the dedup purpose
(v1.1.3 contract).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP_TSX = REPO / "src" / "frontend" / "src" / "App.tsx"
ANSWER_COMPLETE_HANDLER = (
    REPO / "src" / "frontend" / "src" / "chat" / "eventHandlers"
    / "answerCompleteHandler.ts"
)
FRONTEND_DIST_DIR = REPO / "src" / "frontend" / "dist" / "assets"


def _read_answer_complete_source() -> str:
    """Read the answer_complete handler source.

    PR-3 (v2.0.26) split ``handleAgentEvent`` into per-event modules
    under ``src/frontend/src/chat/eventHandlers/``. The
    ``answer_complete`` branch lives in
    ``answerCompleteHandler.ts`` now, not ``App.tsx``. This helper
    reads whichever file currently holds it — App.tsx pre-PR-3,
    answerCompleteHandler.ts post-PR-3.
    """
    if ANSWER_COMPLETE_HANDLER.exists():
        return ANSWER_COMPLETE_HANDLER.read_text(encoding="utf-8")
    return APP_TSX.read_text(encoding="utf-8")


class TestAnswerCompleteCanonicalGuard(unittest.TestCase):
    """Pin the v2.0.13 fix at the source level: the ``answer_complete``
    handler must guard the canonical overwrite with ``trim()`` so a
    whitespace-only server answer doesn't blank out the streamed
    content. The bare ``length > 0`` check from v1.1.3 had this hole."""

    def _answer_complete_block(self) -> str:
        src = _read_answer_complete_source()
        # Find the answer_complete branch in handleAgentEvent.
        # Walk from `evt.type === "answer_complete"` to the end of
        # the outer setThreads((all) => { ... }) block. We anchor
        # on the inner canonicalContent assignment specifically.
        match = re.search(
            r"const\s+canonicalContent\s*=\s*([^;]+);",
            src,
        )
        self.assertIsNotNone(
            match,
            "Could not locate `const canonicalContent = ...;` in "
            "the answer_complete handler — it was rewritten "
            "or removed. v2.0.13 fix relies on this binding.",
        )
        return match.group(0)

    def test_canonical_uses_trim_length_not_bare_length(self):
        """The canonical-overwrite guard must use
        ``answerText.trim().length > 0``, not ``answerText.length > 0``.
        The latter accepts ``" "`` (a single space) as non-empty,
        which is exactly the bug."""
        block = self._answer_complete_block()
        self.assertIn(
            ".trim().length",
            block,
            f"answer_complete canonicalContent assignment must use "
            f".trim().length — bare .length accepts whitespace as "
            f"'real content' and blanks out the streamed answer. "
            f"Found: {block!r}",
        )
        self.assertNotIn(
            "answerText.length",
            block,
            f"answer_complete canonicalContent assignment still uses "
            f"bare `answerText.length` — whitespace-only canonicals "
            f"will overwrite the streamed content with empty space. "
            f"Found: {block!r}",
        )

    def test_fallback_preserves_streamed_content(self):
        """When canonical is whitespace-only, the fallback must use
        ``last?.content ?? ""`` so the streamed accumulation is kept."""
        block = self._answer_complete_block()
        # The fallback chain must reference last?.content — that's
        # how the streamed accumulation gets preserved.
        self.assertIn(
            "last?.content",
            block,
            f"When canonical is empty/whitespace, must fall back to "
            f"last?.content (preserves streamed accumulation). "
            f"Found: {block!r}",
        )

    def test_docstring_marks_v2_0_13(self):
        """The rationale comment must mark v2.0.13 so future readers
        can git-blame the change. Same convention as v2.0.12.

        PR-3 (v2.0.26) moved the handler into its own module. The
        v2.0.13 marker is on the module-level docstring at the top
        of ``answerCompleteHandler.ts``, well above the
        ``canonicalContent`` binding. Scan the whole file (not just
        forward from the binding) so this test keeps passing.
        """
        src = _read_answer_complete_source()
        self.assertIn(
            "v2.0.13",
            src,
            "answer_complete handler source must mark the v2.0.13 "
            "fix so future contributors can find the rationale via "
            "git blame.",
        )
        self.assertIn(
            "trim",
            src,
            "answer_complete handler source must reference "
            ".trim() so the why-not-bare-length rationale is preserved.",
        )


class TestBuildArtifactTrimsCanonicalCheck(unittest.TestCase):
    """Verify the rebuilt JS bundle carries the .trim() guard. This
    catches the case where source is fixed but dist/ wasn't rebuilt."""

    def _latest_js(self) -> Path | None:
        if not FRONTEND_DIST_DIR.exists():
            return None
        candidates = sorted(
            FRONTEND_DIST_DIR.glob("index-*.js"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def test_bundle_has_trim_guard_in_answer_complete_path(self):
        """The bundle must carry the .trim() guard. We can't grep for
        ``answerText.trim`` directly (the variable is minified to a
        single letter), so we look for the canonical pattern: a
        trim-then-length-comparison gating a ternary. The minified
        shape is ``(t=t.trim()).length>0`` or equivalent.

        If this test fails post-rebuild, the dist/ still has the
        v1.1.3 bug — run ``npm run build`` in src/frontend/."""
        js = self._latest_js()
        if js is None:
            self.skipTest(
                "No built JS in dist/assets/ — run `npm run build` "
                "in src/frontend/ to produce it."
            )
        text = js.read_text(encoding="utf-8")
        # Look for .trim() being called and the result compared to 0
        # in a boolean context. We require BOTH that .trim() appears
        # AND that there's a .trim().length > 0 shape somewhere — the
        # second is a tighter pattern that distinguishes our guard
        # from any incidental .trim() usage elsewhere in the bundle.
        trim_len_zero_pattern = re.compile(r"\.trim\(\)\.length")
        matches = trim_len_zero_pattern.findall(text)
        self.assertGreaterEqual(
            len(matches),
            1,
            f"Built bundle {js.name} does not contain any "
            f"`.trim().length` expression — the v2.0.13 canonical "
            f"guard didn't make it into the dist. Run `npm run build`.",
        )


if __name__ == "__main__":
    unittest.main()
