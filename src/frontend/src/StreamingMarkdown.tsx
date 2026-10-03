/**
 * Streaming-aware Markdown renderer.
 *
 * Splits a single in-flight ``content`` string at a freeze point so that
 * only the *tail* (the unsent / still-being-written text) re-renders on
 * every token. The *prefix* is fully rendered via ``<Markdown>`` once
 * and never re-parses until the freeze point advances.
 *
 * Why split instead of buffering / debouncing
 * ------------------------------------------
 * The naive fix would be to debounce the ReactMarkdown re-render so it
 * only fires N times per second. But debouncing only delays the heavy
 * work; the work itself still runs over the *entire* accumulated
 * content (often several KB by mid-answer). For a 3-sentence answer,
 * that's ~3× the parse cost per debounce tick vs. only-parsing-the-tail.
 *
 * Splitting at a freeze point is also visually cleaner: previous
 * sentences are already rendered with full markdown structure
 * (lists, tables, bold) the moment they complete, so the user sees
 * them "lock in" rather than seeing the whole answer shimmer in
 * lockstep.
 *
 * Freeze-point heuristic
 * ----------------------
 * Look for the LAST boundary in ``content`` that's at least
 * ``MIN_PREFIX_LEN`` chars from the start, in priority order:
 *
 *   1. A blank-line (``\n\n``) — visually a paragraph break.
 *   2. A sentence-ending punctuation followed by space or newline:
 *      ``. `` / ``! `` / ``? `` / ``。 `` / ``！ `` / ``？ ``
 *      (Chinese + Latin). These are the natural "thought complete"
 *      points — after a period the LLM is unlikely to back-fill the
 *      same sentence, so re-parsing it would just churn React's tree.
 *   3. Otherwise (mid-sentence), freeze nothing — the tail carries
 *      the whole content.
 *
 * No useDeferredValue on the prefix
 * ---------------------------------
 * v2.0.12 — we previously wrapped ``split.prefix`` in
 * ``useDeferredValue`` to keep the markdown re-parse off the critical
 * path. That deferral caused a regression on tool-using queries
 * (where the answer is long enough to engage the freeze-point split):
 * ``useDeferredValue`` keeps returning the STALE prefix every render
 * while the actual prefix keeps advancing on every token. React never
 * has a quiet moment to commit the deferred value, so the markdown-
 * rendered portion of the answer is INVISIBLE for the entire stream
 * — the user only sees the live ``<Tail>``. After ``done`` the
 * component switches to plain ``<Markdown>`` and the answer finally
 * appears, which looked like "answer doesn't render until F5".
 *
 * The fix is to NOT defer. ``<FrozenPrefix>`` is already
 * ``React.memo``-wrapped: when the prefix string is unchanged across
 * renders (the common case mid-stream), the memo short-circuits and
 * the expensive markdown re-parse is skipped anyway. When the freeze
 * point advances, the prefix is genuinely new content that MUST be
 * parsed synchronously to be visible — deferring it was hiding the
 * answer, not optimizing it.
 */
import { memo, useMemo } from "react";
import { Markdown } from "./Markdown";
import type { Source } from "./chat-utils";
import { CitationChip } from "./components/CitationChip";

interface Props {
  content: string;
  sources?: Source[];
}

/** Don't bother splitting before this many chars — the prefix would be
 *  too short for markdown to even start rendering. */
const MIN_PREFIX_LEN = 24;

const SENTENCE_END_RE = /([.!?。！？])(\s+|$)/g;

function findFreezePoint(content: string): number {
  if (content.length < MIN_PREFIX_LEN) return 0;

  // 1. Last blank line (paragraph break) past the minimum.
  const lastBlank = content.lastIndexOf("\n\n");
  if (lastBlank >= MIN_PREFIX_LEN) {
    // Include the trailing blank line itself so the prefix ends at a
    // visible boundary.
    return lastBlank + 2;
  }

  // 2. Last sentence-ending punctuation past the minimum.
  let best = -1;
  SENTENCE_END_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = SENTENCE_END_RE.exec(content)) !== null) {
    const endIdx = m.index + m[0].length;
    if (endIdx >= MIN_PREFIX_LEN) {
      best = endIdx;
    }
  }
  if (best > 0) return best;

  return 0;
}

/** Plain-text tail renderer. Does NOT re-run markdown; only replaces
 *  ``[n]`` citation markers with inline clickable chips so the tail
 *  looks visually identical to what ``<Markdown>`` would produce for
 *  a single-paragraph plain-text fragment. */
function Tail({ content, sources }: Props) {
  const parts = useMemo(() => splitWithCitations(content), [content]);
  return (
    <span className="streaming-tail">
      {parts.map((p, i) => {
        if (p.kind === "text") {
          // Preserve trailing newlines so the cursor / next-line spacing
          // is right when the prefix is followed by more content.
          return <span key={i}>{p.value}</span>;
        }
        // Citation chip — same click semantics as Markdown.tsx.
        // ``splitWithCitations`` always sets ``idx`` for cite parts;
        // the union marks it optional so text parts don't need a
        // dummy value. Narrow here for the chip / web-link
        // branches.
        if (p.idx === undefined) return null;
        const idx = p.idx;
        const source = sources?.find((s) => s.index === idx);
        const isWeb = source?.source_kind === "web" && !!source?.url;
        if (isWeb) {
          return (
            <a
              key={i}
              className="citation-chip-link"
              href={source!.url}
              target="_blank"
              rel="noopener noreferrer"
              title={source!.text}
            >
              [{idx}] {source!.domain || source!.url}
            </a>
          );
        }
        return (
          <CitationChip key={i} idx={idx} source={source} />
        );
      })}
    </span>
  );
}

interface CitationPart {
  kind: "text" | "cite";
  value: string;
  // Cite parts always carry a 1-based index (the regex
  // ``[1]`` match groups into a number); text parts don't have
  // one at all. Splitting the union keeps ``idx`` non-optional
  // for cite parts without forcing text parts to invent a value.
  idx?: number;
}

function splitWithCitations(content: string): CitationPart[] {
  const out: CitationPart[] = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(content)) !== null) {
    if (m.index > last) {
      out.push({ kind: "text", value: content.slice(last, m.index) });
    }
    out.push({ kind: "cite", value: m[0], idx: Number(m[1]) });
    last = m.index + m[0].length;
  }
  if (last < content.length) {
    out.push({ kind: "text", value: content.slice(last) });
  }
  return out;
}

const FrozenPrefix = memo(function FrozenPrefix({
  content,
  sources,
}: {
  content: string;
  sources?: Source[];
}) {
  return <Markdown content={content} sources={sources} />;
});

export function StreamingMarkdown({ content, sources }: Props) {
  // Compute the freeze point synchronously; if the content grew past
  // the previous freeze point by enough, advance. The ``<FrozenPrefix>``
  // component below is ``React.memo``-wrapped, so it skips re-render
  // when the prefix string is unchanged (the common case mid-stream).
  // v2.0.12 — DO NOT wrap in ``useDeferredValue``: it caused the
  // long-answer (tool-using query) regression where the markdown
  // prefix never became visible during streaming.
  const split = useMemo(() => {
    const idx = findFreezePoint(content);
    return {
      prefix: content.slice(0, idx),
      tail: content.slice(idx),
    };
  }, [content]);

  if (!split.prefix) {
    // Too short to split — render the whole thing as a live tail.
    return <Tail content={content} sources={sources} />;
  }
  return (
    <>
      <FrozenPrefix content={split.prefix} sources={sources} />
      <Tail content={split.tail} sources={sources} />
    </>
  );
}
