/**
 * Citation chip — the clickable `[n]` rendered inline in markdown
 * content. PR-4 (v2.0.26.1) extracted this from the 2-way-duplicated
 * block in ``Markdown.tsx`` + ``StreamingMarkdown.tsx`` so a future
 * tweak to the chip shape (locale, accessibility, click handler)
 * only happens in one place.
 *
 * Usage:
 *   <CitationChip idx={1} source={source} />
 *
 * The chip consumes ``useLocale()`` internally so callers don't have
 * to thread ``t``. If the ``source`` is missing (e.g. the LLM cited
 * an index that didn't make it through the backend filter) the chip
 * falls back to the index-only aria-label and title.
 */
import { useLocale } from "../i18n";
import type { Source } from "../chat-utils";

interface CitationChipProps {
  /** 1-based citation index (matches the wire payload). */
  idx: number;
  /** The resolved source — optional because the lookup can miss. */
  source?: Source;
}

export function CitationChip({ idx, source }: CitationChipProps) {
  const { t } = useLocale();

  // Build locale-aware suffix pieces. If the source carries neither
  // page nor sheet, the suffix is "" and the aria-label collapses to
  // ``citationAriaFallback`` for screen readers.
  const pageSuffix = source?.page
    ? t("markdown.pageSuffix", { page: source.page })
    : "";
  const sheetSuffix = source?.sheet
    ? t("markdown.sheetSuffix", { sheet: source.sheet })
    : "";

  // Snippet preview (first 120 chars of the source text). Empty
  // sources render the "no preview" copy so the tooltip still
  // carries SOME signal — better than an empty title that screen
  // readers skip.
  const previewLine =
    source?.text && source.text.length > 0
      ? source.text.slice(0, 120).replace(/\s+/g, " ") +
        (source.text.length > 120 ? "…" : "")
      : t("markdown.noPreview");

  return (
    <span
      className="citation-chip"
      data-citation-index={idx}
      onClick={() => {
        // v2.0.29.4 (Phase 4 PR-1) — the chip-level click event now
        // ships chunk_id / doc_id / source_kind alongside ``index``
        // so the App.tsx listener (``src/frontend/src/App.tsx``) and
        // any debugging tooling can identify the exact source this
        // chip points at. Pre-PR-1 the chip only carried ``{ index }``
        // — when the LLM cited the wrong source for a chip (e.g.
        // due to renumber drift), there was no chip-level data to
        // cross-check the LLM's reference against.
        //
        // Backward-compat: App.tsx still only reads ``detail.index``;
        // the new fields are additive. Older frontends consuming the
        // event from a post-PR-1 build will simply ignore the new
        // keys.
        //
        // ``source_kind`` (not ``source_type``) matches the wire
        // field name on ``src/api/schemas.py:Source.source_kind``.
        window.dispatchEvent(
          new CustomEvent("rag:citation-click", {
            detail: {
              index: idx,
              chunk_id: source?.chunk_id,
              doc_id: source?.doc_id,
              source_kind: source?.source_kind,
            },
          }),
        );
      }}
      // Focusable + announceable: the underlying element is the
      // caller's job (button / span). The aria-label here is what
      // screen readers say when focus lands on the chip.
      aria-label={
        source
          ? t("markdown.citationAria", {
              idx,
              filename: source.filename,
              pageSuffix,
              sheetSuffix,
            })
          : t("markdown.citationAriaFallback", { idx })
      }
      // Multi-line ``title`` joined with \n renders as a tooltip
      // with line breaks in Chrome / Edge / Firefox. Keep the order:
      // title → preview. ``title`` is what the browser shows on
      // hover; ``aria-label`` is what the screen reader announces.
      title={
        source
          ? [
              `${source.filename}${pageSuffix}`,
              previewLine,
            ].join("\n")
          : t("markdown.citationTitle", { idx })
      }
    >
      [{idx}]
    </span>
  );
}
