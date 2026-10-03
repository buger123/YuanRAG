/**
 * v2.0.26.2 (PR-5) — SettingsDialog theme section tests.
 *
 * Verifies the new 3-state segmented control (light / dark /
 * system) wired to ``useTheme()``:
 *   - Renders all 3 options with the correct localized labels.
 *   - Clicking an option flips ``document.documentElement.dataset.theme``
 *     (the side-effect the actual app depends on).
 *   - The active option carries ``aria-checked="true"``.
 *
 * Locale is forced to English to keep the assertions stable; the
 * Chinese labels are validated by the PR-4 i18n parity test.
 */
import { describe, expect, it, afterEach, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { SettingsDialog } from "../SettingsDialog";
import { LocaleProvider } from "../../i18n";
import { ThemeProvider } from "../../i18n/ThemeProvider";
import type { Settings } from "../../api/client";

const baseSettings: Settings = {
  llm_provider: "anthropic",
  llm_model: "MiniMax-M3",
  has_api_key: true,
  api_key_source: "env",
};

function renderDialog() {
  // LocaleProvider reads ``rag.locale`` from localStorage. Set it
  // before render so the segmented control shows English labels.
  localStorage.setItem("rag.locale", "en");
  return render(
    <LocaleProvider>
      <ThemeProvider>
        <SettingsDialog
          current={baseSettings}
          modelsReady={true}
          embedding={{ name: "BGE-M3", status: "ready" }}
          reranker={{ name: "BGE Reranker v2-M3", status: "ready" }}
          onClose={() => undefined}
          onSaved={() => undefined}
        />
      </ThemeProvider>
    </LocaleProvider>,
  );
}

/** Variant that exposes onModelsReady so we can verify the
 *  "no false-close on mount" fix from PR-5 — see the
 *  useModelsReadyRef / transition guard inside SettingsDialog. */
function renderDialogWithOnModelsReady(
  onModelsReady: () => void,
  modelsReady = true,
) {
  localStorage.setItem("rag.locale", "en");
  return render(
    <LocaleProvider>
      <ThemeProvider>
        <SettingsDialog
          current={baseSettings}
          modelsReady={modelsReady}
          embedding={{ name: "BGE-M3", status: "ready" }}
          reranker={{ name: "BGE Reranker v2-M3", status: "ready" }}
          onClose={() => undefined}
          onSaved={() => undefined}
          onModelsReady={onModelsReady}
        />
      </ThemeProvider>
    </LocaleProvider>,
  );
}

describe("SettingsDialog — theme section", () => {
  afterEach(() => {
    cleanup();
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
    vi.restoreAllMocks();
  });

  it("renders all 3 theme options with English labels", () => {
    renderDialog();
    const radiogroup = screen.getByRole("radiogroup", { name: /appearance/i });
    expect(radiogroup).toBeInTheDocument();
    // The 3 segmented-option buttons
    const buttons = radiogroup.querySelectorAll(".segmented-option");
    expect(buttons).toHaveLength(3);
    expect(buttons[0]).toHaveTextContent(/^Light$/);
    expect(buttons[1]).toHaveTextContent(/^Dark$/);
    expect(buttons[2]).toHaveTextContent(/^System$/);
  });

  it("clicking Dark flips data-theme and marks aria-checked", () => {
    renderDialog();
    const radiogroup = screen.getByRole("radiogroup", { name: /appearance/i });
    const darkBtn = radiogroup.querySelectorAll(".segmented-option")[1];
    fireEvent.click(darkBtn);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(darkBtn.getAttribute("aria-checked")).toBe("true");
  });

  it("clicking Light flips data-theme back to light", () => {
    renderDialog();
    const radiogroup = screen.getByRole("radiogroup", { name: /appearance/i });
    const buttons = radiogroup.querySelectorAll(".segmented-option");
    fireEvent.click(buttons[1]); // Dark
    expect(document.documentElement.dataset.theme).toBe("dark");
    fireEvent.click(buttons[0]); // Light
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(buttons[0].getAttribute("aria-checked")).toBe("true");
  });

  it("System option falls back to OS preference (light by default in test env)", () => {
    // Default stub from test-setup.ts returns matches: false.
    renderDialog();
    const radiogroup = screen.getByRole("radiogroup", { name: /appearance/i });
    const systemBtn = radiogroup.querySelectorAll(".segmented-option")[2];
    fireEvent.click(systemBtn);
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(systemBtn.getAttribute("aria-checked")).toBe("true");
  });

  it("does NOT fire onModelsReady on mount when modelsReady is already true", () => {
    // PR-5 regression guard. Before the fix, SettingsDialog called
    // ``onModelsReady()`` immediately on mount whenever
    // ``modelsReady === true`` — which caused the parent to
    // ``setSettingsOpen(false)`` and the dialog closed itself the
    // instant the user opened it (mobile.spec.ts Assertion 5).
    const onModelsReady = vi.fn();
    renderDialogWithOnModelsReady(onModelsReady, true);
    expect(onModelsReady).not.toHaveBeenCalled();
  });

  it("does fire onModelsReady when modelsReady flips false → true", () => {
    const onModelsReady = vi.fn();
    // Start with modelsReady=false, then re-render with true.
    const { rerender } = render(
      <LocaleProvider>
        <ThemeProvider>
          <SettingsDialog
            current={baseSettings}
            modelsReady={false}
            embedding={{ name: "BGE-M3", status: "downloading" }}
            reranker={{ name: "BGE Reranker v2-M3", status: "pending" }}
            onClose={() => undefined}
            onSaved={() => undefined}
            onModelsReady={onModelsReady}
          />
        </ThemeProvider>
      </LocaleProvider>,
    );
    expect(onModelsReady).not.toHaveBeenCalled();
    rerender(
      <LocaleProvider>
        <ThemeProvider>
          <SettingsDialog
            current={baseSettings}
            modelsReady={true}
            embedding={{ name: "BGE-M3", status: "ready" }}
            reranker={{ name: "BGE Reranker v2-M3", status: "ready" }}
            onClose={() => undefined}
            onSaved={() => undefined}
            onModelsReady={onModelsReady}
          />
        </ThemeProvider>
      </LocaleProvider>,
    );
    expect(onModelsReady).toHaveBeenCalledTimes(1);
  });
});
