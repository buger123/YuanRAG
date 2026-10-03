/**
 * PR-4 (v2.0.26.1) — CitationChip tests. PR-3 split
 * ``handleAgentEvent`` and PR-4 extracted the citation chip from
 * ``Markdown.tsx`` + ``StreamingMarkdown.tsx`` so both call
 * sites now go through this single component.
 *
 * These tests verify the locale-aware shape:
 *   - aria-label is built from the catalog key (filename + page
 *     / sheet suffixes).
 *   - title carries the snippet preview (truncated to 120 chars).
 *   - Missing source falls back to the index-only aria-label.
 *   - Empty source text falls back to ``markdown.noPreview``.
 *   - Switching the active locale flips the aria-label text.
 *
 * The chip dispatches ``rag:citation-click`` CustomEvents on
 * click — covered here as a side-effect spy.
 */
import { describe, expect, it, vi, afterEach } from "vitest";
import { fireEvent, render, screen, cleanup } from "@testing-library/react";
import { CitationChip } from "../CitationChip";
import { LocaleProvider } from "../../i18n";
import type { Source } from "../../chat-utils";

const sampleSource: Source = {
  index: 1,
  filename: "annual-report.pdf",
  page: 42,
  text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
  score: 0.95,
  source_kind: "local",
};

const sheetSource: Source = {
  index: 2,
  filename: "sales-data.xlsx",
  sheet: "Q4",
  text: "Header row of the Q4 sheet",
  score: 0.81,
  source_kind: "local",
};

const emptyTextSource: Source = {
  index: 3,
  filename: "image-only.pdf",
  page: 1,
  text: "",
  score: 0.5,
  source_kind: "local",
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("CitationChip — locale-aware rendering", () => {
  it("renders the index in the visible label", () => {
    render(
      <LocaleProvider>
        <CitationChip idx={1} source={sampleSource} />
      </LocaleProvider>,
    );
    expect(screen.getByText("[1]")).toBeInTheDocument();
  });

  it("uses the Chinese aria-label by default (project default locale)", () => {
    render(
      <LocaleProvider>
        <CitationChip idx={1} source={sampleSource} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/跳转到来源 1/);
    expect(chip).toBeInTheDocument();
    // Should include the filename + page number in Chinese.
    expect(chip.getAttribute("aria-label")).toContain("annual-report.pdf");
    expect(chip.getAttribute("aria-label")).toContain("第 42 页");
  });

  it("uses the English aria-label when the locale is switched", () => {
    // Set the localStorage value before mounting so the
    // LocaleProvider reads it on init.
    localStorage.setItem("rag.locale", "en");
    render(
      <LocaleProvider>
        <CitationChip idx={1} source={sampleSource} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/Jump to source 1/);
    expect(chip.getAttribute("aria-label")).toContain(
      "annual-report.pdf",
    );
    expect(chip.getAttribute("aria-label")).toContain("Page 42");
    localStorage.removeItem("rag.locale");
  });

  it("falls back to the index-only aria when no source is provided", () => {
    render(
      <LocaleProvider>
        <CitationChip idx={5} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/跳转到来源 5/);
    // No filename / page references in the fallback.
    expect(chip.getAttribute("aria-label")).not.toContain("annual-report");
  });

  it("uses the sheet suffix (not page) when source.sheet is set", () => {
    render(
      <LocaleProvider>
        <CitationChip idx={2} source={sheetSource} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/跳转到来源 2/);
    expect(chip.getAttribute("aria-label")).toContain("Q4");
    // No page suffix — sheet wins in the suffix slot.
    expect(chip.getAttribute("aria-label")).not.toContain("页");
  });

  it("renders the snippet preview in the title attribute", () => {
    render(
      <LocaleProvider>
        <CitationChip idx={1} source={sampleSource} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/跳转到来源 1/);
    // Multi-line title: filename + preview.
    const title = chip.getAttribute("title") || "";
    expect(title).toContain("annual-report.pdf");
    expect(title).toContain("Lorem ipsum");
  });

  it("falls back to noPreview in the title when source text is empty", () => {
    render(
      <LocaleProvider>
        <CitationChip idx={3} source={emptyTextSource} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/跳转到来源 3/);
    const title = chip.getAttribute("title") || "";
    expect(title).toContain("(无内容预览)");
  });

  it("truncates long snippets to 120 chars + ellipsis", () => {
    const longText = "a".repeat(500);
    const longSource: Source = {
      ...sampleSource,
      text: longText,
    };
    render(
      <LocaleProvider>
        <CitationChip idx={1} source={longSource} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/跳转到来源 1/);
    const title = chip.getAttribute("title") || "";
    // The preview line in the title is 120 'a's followed by …
    expect(title).toMatch(/a{120}…/);
    expect(title.length).toBeLessThan(200);
  });

  it("dispatches rag:citation-click CustomEvent on click", () => {
    const spy = vi.fn();
    window.addEventListener("rag:citation-click", spy);
    render(
      <LocaleProvider>
        <CitationChip idx={1} source={sampleSource} />
      </LocaleProvider>,
    );
    const chip = screen.getByLabelText(/跳转到来源 1/);
    fireEvent.click(chip);
    expect(spy).toHaveBeenCalledTimes(1);
    const ev = spy.mock.calls[0][0] as CustomEvent;
    // v2.0.29.4 (Phase 4 PR-1) — the chip-level click event now
    // ships chunk_id / doc_id / source_kind alongside ``index`` so
    // the App.tsx listener and any debugging tooling can identify
    // the exact source this chip points at. The wire payload
    // (``src/api/schemas.py:Source``) always carries chunk_id + doc_id;
    // sampleSource here is the test fixture which doesn't set them,
    // so they surface as ``undefined`` (the optional TS fields).
    expect(ev.detail).toEqual({
      index: 1,
      chunk_id: undefined,  // sampleSource doesn't set chunk_id
      doc_id: undefined,    // sampleSource doesn't set doc_id either
      source_kind: "local", // sampleSource.source_kind === "local"
    });
    window.removeEventListener("rag:citation-click", spy);
  });
});
