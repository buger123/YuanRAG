"""v2.0.12 regression tests — pin the fix for the
"tool-using query: answer doesn't render during streaming" bug.

Bug summary
-----------
For tool-using queries (which produce long answers, typically >= a few
hundred chars), the assistant content rendered during streaming was
invisible. Tool calls rendered normally, the answer text showed only
after F5 (which loads from history DB and bypasses the streaming
renderer).

Root cause
----------
``src/frontend/src/StreamingMarkdown.tsx`` wrapped ``split.prefix`` in
``useDeferredValue`` to defer the markdown re-parse to lower priority.
That deferral caused a regression on tool-using queries:

* The freeze-point split engages for content >= 24 chars, which is
  virtually every tool-using-query answer.
* The split prefix keeps advancing as the LLM streams new sentences.
* ``useDeferredValue`` returns the STALE prefix every render while
  the actual prefix keeps moving.
* React never has a quiet moment to commit the deferred value
  (token emissions are back-to-back).
* The user only sees the live ``<Tail>`` — the markdown-rendered
  historical sentences are invisible until ``done`` flips
  ``isStreaming`` to ``false`` and the renderer switches to
  ``<Markdown>``.

Fix
---
Drop ``useDeferredValue`` from ``StreamingMarkdown``. ``FrozenPrefix``
is already ``React.memo``-wrapped, so when the prefix string is
unchanged across renders (the common mid-stream case) the memo
short-circuits the markdown re-parse anyway — the deferral was both
unnecessary AND the bug.

Why this test class
-------------------
Static / structural tests that pin the post-fix source shape so future
refactors don't accidentally re-introduce the deferral.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STREAMING_MD = REPO / "src" / "frontend" / "src" / "StreamingMarkdown.tsx"
FRONTEND_DIST_DIR = REPO / "src" / "frontend" / "dist" / "assets"


class TestStreamingMarkdownDeferralRemoved(unittest.TestCase):
    """Pin that ``useDeferredValue`` is no longer applied to the
    streaming-answer prefix. The deferral was the root cause of the
    v2.0.11-and-earlier bug where tool-using-query answers stayed
    invisible during streaming (until F5 loaded from history)."""

    def test_no_use_deferred_value_import(self):
        """The React hook ``useDeferredValue`` must not be imported
        into StreamingMarkdown.tsx anymore — the prefix-render path
        must be synchronous so the markdown-rendered sentences appear
        as they stream in."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        # `import { ... useDeferredValue ... } from "react"` is the
        # only place this hook enters the module. Strip comments /
        # whitespace first so docstring mentions don't false-positive.
        code_only = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        code_only = re.sub(r"//[^\n]*", "", code_only)
        self.assertNotIn(
            "useDeferredValue",
            code_only,
            "StreamingMarkdown.tsx still references useDeferredValue "
            "outside of comments — the v2.0.12 fix must drop it. "
            "The deferral caused tool-using-query answers to stay "
            "invisible during streaming until the page was reloaded.",
        )

    def test_no_use_deferred_value_call_site(self):
        """No call site should wrap the prefix in ``useDeferredValue``
        anymore. Even if the import were re-added under a different
        name, the call pattern would re-introduce the bug."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        code_only = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        code_only = re.sub(r"//[^\n]*", "", code_only)
        self.assertNotRegex(
            code_only,
            r"useDeferredValue\s*\(",
            "StreamingMarkdown.tsx still calls useDeferredValue(...) — "
            "the prefix must be passed to <FrozenPrefix> synchronously.",
        )

    def test_prefix_passed_synchronously_to_frozen_prefix(self):
        """The JSX that renders the frozen prefix must pass
        ``split.prefix`` directly, not a deferred wrapper. This is
        the actual render path — if a refactor accidentally re-wraps
        it, this test fails immediately rather than letting the bug
        ship again."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        # Find the FrozenPrefix JSX site. The comment in v2.0.12
        # says `split.prefix` must be passed directly.
        match = re.search(
            r"<FrozenPrefix[^/]*?/>",
            src,
        )
        self.assertIsNotNone(
            match,
            "Could not find <FrozenPrefix ... /> JSX in StreamingMarkdown",
        )
        jsx = match.group(0)
        self.assertIn(
            "split.prefix",
            jsx,
            f"<FrozenPrefix> JSX must receive split.prefix directly. "
            f"Found: {jsx!r}",
        )
        self.assertNotIn(
            "deferredPrefix",
            jsx,
            f"<FrozenPrefix> JSX still references deferredPrefix — the "
            f"v2.0.12 fix passes split.prefix synchronously. Found: {jsx!r}",
        )

    def test_no_deferred_prefix_binding(self):
        """No local binding named ``deferredPrefix`` (or
        ``deferred_prefix``) should exist anymore. The naming is a
        strong signal the deferral is back."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        code_only = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        code_only = re.sub(r"//[^\n]*", "", code_only)
        self.assertNotRegex(
            code_only,
            r"\bdeferredPrefix\b|\bdeferred_prefix\b",
            "StreamingMarkdown.tsx still binds a 'deferredPrefix' "
            "variable — the v2.0.12 fix removes it.",
        )


class TestFrozenPrefixMemoIntact(unittest.TestCase):
    """Pin that ``FrozenPrefix`` is still ``React.memo``-wrapped. This
    is the optimization that makes dropping ``useDeferredValue``
    safe: when the prefix string is unchanged, the memo short-circuits
    the markdown re-parse. Without the memo, every token would force
    a full markdown re-parse of the entire historical content — the
    exact cost the original deferral was trying to avoid."""

    def test_frozen_prefix_is_memo_wrapped(self):
        src = STREAMING_MD.read_text(encoding="utf-8")
        # The pattern: `const FrozenPrefix = memo(function ...`
        match = re.search(
            r"FrozenPrefix\s*=\s*memo\s*\(",
            src,
        )
        self.assertIsNotNone(
            match,
            "FrozenPrefix must be React.memo-wrapped — without it, "
            "dropping useDeferredValue would re-parse the prefix "
            "markdown on every token (the cost the deferral was "
            "trying to avoid). The memo IS the optimization now.",
        )

    def test_frozen_prefix_renders_markdown(self):
        """FrozenPrefix must render <Markdown content={content}> so
        the historical sentences get full markdown structure
        (lists, tables, bold) rather than plain text."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        # Locate the FrozenPrefix memo definition.
        idx = src.find("const FrozenPrefix = memo(")
        self.assertGreater(
            idx, -1,
            "Could not locate `const FrozenPrefix = memo(...)`",
        )
        # Slice forward from the memo definition; assert the return
        # statement uses <Markdown> with both `content` and `sources`.
        snippet = src[idx: idx + 600]
        self.assertIn(
            "return <Markdown",
            snippet,
            "FrozenPrefix must `return <Markdown ... />`",
        )
        self.assertIn(
            "content={content}",
            snippet,
            "FrozenPrefix must forward the `content` prop to <Markdown>",
        )
        self.assertIn(
            "sources={sources}",
            snippet,
            "FrozenPrefix must forward the `sources` prop to <Markdown>",
        )


class TestSplitLogicIntact(unittest.TestCase):
    """Pin the split / freeze-point heuristic — it's still what we
    want; we only removed the deferral. These tests catch accidental
    deletions of the split when someone tries to "simplify" the
    component (which would re-introduce the markdown re-parse cost)."""

    def test_find_freeze_point_function_exists(self):
        src = STREAMING_MD.read_text(encoding="utf-8")
        self.assertRegex(
            src,
            r"function\s+findFreezePoint\s*\(",
            "findFreezePoint helper must still exist — it's the "
            "freeze-point heuristic.",
        )

    def test_min_prefix_len_constant_is_24(self):
        """The 24-char minimum is the empirical threshold below which
        the split prefix would be too short to bother freezing.
        Pinning the constant prevents accidental tuning that would
        either always-split (expensive) or never-split (no memo
        benefit)."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        match = re.search(
            r"const\s+MIN_PREFIX_LEN\s*=\s*(\d+)\s*;",
            src,
        )
        self.assertIsNotNone(match, "MIN_PREFIX_LEN constant missing")
        self.assertEqual(
            int(match.group(1)),
            24,
            f"MIN_PREFIX_LEN must stay at 24 — tuned to the smallest "
            f"prefix worth memoizing. Found {match.group(1)}.",
        )

    def test_use_memo_wraps_split_computation(self):
        """The split computation must be memoized — without it the
        findFreezePoint regex would run on every render."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        # Look for the useMemo wrapping the split
        match = re.search(
            r"const\s+split\s*=\s*useMemo\s*\(\s*\(\)\s*=>\s*\{[^}]*findFreezePoint",
            src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "split computation must be useMemo-wrapped around "
            "findFreezePoint — otherwise the regex runs on every render.",
        )

    def test_split_returns_prefix_and_tail(self):
        """The split object must have `prefix` and `tail` keys so the
        JSX can reference them."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        # Look for the literal `prefix: content.slice(...)` + tail
        # pattern. We can't use a simple `[^,]+` because the slice
        # argument contains a comma (`content.slice(0, idx)`).
        has_prefix = re.search(
            r"prefix\s*:\s*content\.slice\(\s*0\s*,\s*idx\s*\)",
            src,
        )
        has_tail = re.search(
            r"tail\s*:\s*content\.slice\(\s*idx\s*\)",
            src,
        )
        self.assertIsNotNone(
            has_prefix,
            "split must compute prefix = content.slice(0, idx)",
        )
        self.assertIsNotNone(
            has_tail,
            "split must compute tail = content.slice(idx)",
        )

    def test_short_content_falls_through_to_tail(self):
        """When split.prefix is empty (content too short), the
        component must render Tail(content) — the full content as a
        live tail. This is the path that short direct answers
        (tool-using query summaries, greetings) take."""
        src = STREAMING_MD.read_text(encoding="utf-8")
        match = re.search(
            r"if\s*\(\s*!split\.prefix\s*\)\s*\{[^}]*<Tail\s+content=\{content\}",
            src,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "When split.prefix is empty, the component must fall "
            "through to <Tail content={content} ... /> (full content "
            "as live tail). Missing fallback would cause short answers "
            "to render nothing.",
        )


class TestBuildArtifactNoDefiniteDeferral(unittest.TestCase):
    """Verify the rebuilt JS bundle no longer defers the prefix
    render. This catches the case where source is fixed but the
    dist/ wasn't rebuilt (a common slip in shipping pipelines)."""

    def _latest_js(self) -> Path | None:
        if not FRONTEND_DIST_DIR.exists():
            return None
        candidates = sorted(
            FRONTEND_DIST_DIR.glob("index-*.js"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def test_built_js_omits_deferred_prefix_binding(self):
        """The minified bundle must not contain a deferred-prefix
        binding wired to <FrozenPrefix>. Look for the pattern: a
        variable assigned the deferred value, then passed to a
        memoized renderer. If this test fails post-rebuild, the
        dist/ still has the bug."""
        js = self._latest_js()
        if js is None:
            self.skipTest(
                "No built JS in dist/assets/ — run `npm run build` "
                "in src/frontend/ to produce it."
            )
        text = js.read_text(encoding="utf-8")
        # The minified VS function previously had:
        #   const r = F.useDeferredValue(n.prefix);
        #   return n.prefix ? v.jsxs(v.Fragment, {children: [
        #     v.jsx(HS, {content: r, sources: t}),  <-- deferred value
        #     v.jsx(vf, {content: n.tail, sources: t}),
        #   ]}) : ...
        # The post-fix shape passes n.prefix directly:
        #   v.jsx(HS, {content: n.prefix, sources: t})
        # We assert the deferred assignment is gone.
        # `F.useDeferredValue` is the minified React-hook call.
        self.assertNotIn(
            "F.useDeferredValue",
            text,
            f"Built bundle {js.name} still contains F.useDeferredValue "
            "— the v2.0.12 source fix didn't make it into the dist. "
            "Run `npm run build` in src/frontend/ and re-ship.",
        )


class TestReactivityPriorityDoc(unittest.TestCase):
    """Pin the v2.0.12 docstring rationale so future contributors
    understand WHY useDeferredValue was removed (and don't put it
    back "for performance")."""

    def test_docstring_explains_why_no_deferral(self):
        src = STREAMING_MD.read_text(encoding="utf-8")
        # The rationale comment must contain the key terms so
        # future readers see the why-not.
        self.assertIn(
            "useDeferredValue",
            src,
            "Docstring must reference useDeferredValue so future "
            "readers see why it was removed.",
        )
        self.assertIn(
            "v2.0.12",
            src,
            "Docstring must mark the v2.0.12 change so git blame "
            "points here.",
        )
        self.assertIn(
            "FrozenPrefix",
            src,
            "Docstring must explain FrozenPrefix's role as the "
            "memoized optimization that replaces the deferral.",
        )


class TestFindFreezePointBehavior(unittest.TestCase):
    """Pin the findFreezePoint heuristic numerically. This is the
    function that decides whether to split at all (and where). The
    v2.0.12 fix doesn't touch this logic, but pinning the output
    prevents accidental drift that could re-introduce the bug for
    differently-shaped content."""

    def _run_freeze_point(self, content: str) -> int:
        # Reimplement the freeze-point heuristic verbatim from the
        # source. Keep in sync with StreamingMarkdown.tsx.
        MIN_PREFIX_LEN = 24
        SENTENCE_END_RE = re.compile(r"([.!?。！？])(\s+|$)")

        if len(content) < MIN_PREFIX_LEN:
            return 0
        last_blank = content.rfind("\n\n")
        if last_blank >= MIN_PREFIX_LEN:
            return last_blank + 2
        best = -1
        for m in SENTENCE_END_RE.finditer(content):
            end_idx = m.end()
            if end_idx >= MIN_PREFIX_LEN:
                best = end_idx
        return max(best, 0)

    def test_short_content_returns_zero(self):
        """Content under MIN_PREFIX_LEN must NOT split — falls through
        to the Tail-only render path (no deferral involved)."""
        for n in (0, 1, 12, 23):
            self.assertEqual(
                self._run_freeze_point("a" * n),
                0,
                f"content length {n} should return 0",
            )

    def test_paragraph_break_freeze(self):
        """A double-newline past position 24 freezes at the blank line."""
        prefix = "a" * 30
        content = prefix + "rest of paragraph\n\nmore content"
        # lastBlank = len(prefix + "rest of paragraph") = 48, +2 = 50
        self.assertEqual(
            self._run_freeze_point(content),
            prefix.__len__() + len("rest of paragraph") + 2,
        )

    def test_sentence_end_freeze(self):
        """Sentence-ending punctuation past position 24 freezes
        at the punctuation+whitespace boundary."""
        # 24 chars of prefix + "Hello. World" → freeze at position 31 (after ". ")
        content = "a" * 24 + "Hello. World"
        # period at position 29, then ". " match ends at 31 (period + space)
        self.assertEqual(
            self._run_freeze_point(content),
            31,
        )

    def test_cjk_sentence_end_freeze(self):
        """CJK 。 should also trigger freeze — the LAST sentence-end
        wins (so 多文段答案的 freeze 在最后一个句号)."""
        content = "a" * 24 + "你好。世界！"
        # Char positions: 你(24) 好(25) 。(26) 世(27) 界(28) ！(29)
        # "。" matches at 26, end at 27 (just the period).
        # "！" matches at 29, end at 30 (just the bang).
        # Last sentence-end >= 24 is "！" → freeze_idx = 30.
        self.assertEqual(
            self._run_freeze_point(content),
            30,
        )

    def test_no_freeze_when_mid_sentence(self):
        """If content is >= 24 chars but no sentence-end exists yet,
        freeze_idx must be 0 (whole content rendered as Tail)."""
        content = "a" * 100  # 100 'a' chars, no sentence-end
        self.assertEqual(
            self._run_freeze_point(content),
            0,
        )

    def test_long_answer_splits_with_deferred_prefix_bug_shape(self):
        """Document the SHAPE that triggered the v2.0.11 bug:
        a long tool-using-query answer that splits into a large
        prefix + small tail. With useDeferredValue, the prefix
        never commits during streaming."""
        # Build a realistic long answer: 5 paragraphs, each ending
        # with a double newline. Total ~500 chars (typical for a
        # tool-using query answer that synthesizes retrieved docs).
        paragraphs = [
            "根据文档库检索到的内容,我为您整理了以下信息。"
            "本次查询覆盖了 RAG 系统的核心模块,包括 query rewriting "
            "embedding retrieval reranking 等关键环节。"
            "\n\n",
            "1. Query Rewriting: 用户原始 query 经过改写后能更精准地 "
            "匹配文档库的语义空间,改写策略包括同义词扩展、query "
            "decomposition 等。"
            "\n\n",
            "2. Embedding: BGE-M3 是当前使用的 embedding 模型,"
            "支持中英文双语,维度 1024,适合 dense retrieval。"
            "\n\n",
            "3. Retrieval: Hybrid search (BM25 + vector) 是默认模式,"
            "召回 top_k 个候选文档,然后进入 reranking 阶段。"
            "\n\n",
            "4. Reranking: BGE-reranker-v2-m3 对候选文档重排序,"
            "输出最终的相关性分数,top_n 用于后续 answer generation。"
        ]
        content = "".join(paragraphs)
        # Confirm the split engages (prefix non-empty)
        freeze_idx = self._run_freeze_point(content)
        self.assertGreater(
            freeze_idx,
            24,
            f"Long tool-using-query answer should split, got freeze_idx={freeze_idx}",
        )
        # And confirm the prefix is SUBSTANTIAL — this is exactly
        # the shape that triggered the bug (large deferred prefix
        # that never committed).
        prefix_len = freeze_idx
        self.assertGreater(
            prefix_len,
            100,
            f"Long answer prefix must be substantial (>=100 chars) "
            f"to match the bug's repro shape, got {prefix_len}",
        )


if __name__ == "__main__":
    unittest.main()
