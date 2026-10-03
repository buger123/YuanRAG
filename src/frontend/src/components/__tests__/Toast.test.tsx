/**
 * v2.0.26.3 (PR-6) — Toast notification system tests.
 *
 * Covers:
 *   - showToast adds a toast that renders with the correct kind.
 *   - Auto-dismiss timer fires after ``duration`` ms.
 *   - Manual dismiss button removes the toast.
 *   - ``duration: 0`` keeps the toast sticky (no auto-dismiss).
 *   - Stack cap (MAX_VISIBLE = 3) drops the oldest entry silently
 *     when a 4th is pushed.
 *   - ``useToast`` throws when used outside a Provider.
 *   - ``getCurrentDispatcher`` returns null when no Provider is
 *     mounted, then returns a working dispatcher after one mounts.
 *   - ToastViewport subscribes via useSyncExternalStore — toasts
 *     pushed from outside React re-render the viewport.
 *   - aria-live / role reflect kind (error → role=alert, others → status).
 */
import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  renderHook,
  screen,
  act,
} from "@testing-library/react";
import {
  ToastProvider,
  ToastViewport,
  useToast,
  setToastDispatcher,
  getCurrentDispatcher,
  __resetToastsForTests,
  type ShowToast,
} from "../Toast";
import { LocaleProvider } from "../../i18n";

function renderWithProviders(ui: React.ReactNode) {
  localStorage.setItem("rag.locale", "en");
  __resetToastsForTests();
  return render(
    <LocaleProvider>
      <ToastProvider>
        {ui}
        <ToastViewport />
      </ToastProvider>
    </LocaleProvider>,
  );
}

describe("Toast — showToast + dismiss", () => {
  beforeEach(() => {
    localStorage.setItem("rag.locale", "en");
    __resetToastsForTests();
  });
  afterEach(() => {
    cleanup();
    localStorage.clear();
    setToastDispatcher(null);
    __resetToastsForTests();
    vi.restoreAllMocks();
  });

  it("renders an error toast with the correct message and role=alert", () => {
    function Harness() {
      const { showToast } = useToast();
      return (
        <button type="button" onClick={() => showToast({ kind: "error", message: "Boom" })}>
          fire
        </button>
      );
    }
    renderWithProviders(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "fire" }));
    const toast = screen.getByRole("alert");
    expect(toast).toHaveTextContent("Boom");
    // The kind class colors the toast
    expect(toast.className).toContain("toast-error");
  });

  it("renders a success toast with role=status", () => {
    function Harness() {
      const { showToast } = useToast();
      return (
        <button type="button" onClick={() => showToast({ kind: "success", message: "OK" })}>
          fire
        </button>
      );
    }
    renderWithProviders(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "fire" }));
    const toast = screen.getByRole("status");
    expect(toast.className).toContain("toast-success");
  });

  it("manual dismiss button removes the toast", () => {
    function Harness() {
      const { showToast } = useToast();
      return (
        <button type="button" onClick={() => showToast({ kind: "info", message: "Hello" })}>
          fire
        </button>
      );
    }
    renderWithProviders(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "fire" }));
    const dismiss = screen.getByRole("button", { name: /dismiss notification/i });
    expect(screen.getByText("Hello")).toBeInTheDocument();
    fireEvent.click(dismiss);
    expect(screen.queryByText("Hello")).not.toBeInTheDocument();
  });

  it("auto-dismisses after the duration", () => {
    vi.useFakeTimers();
    try {
      function Harness() {
        const { showToast } = useToast();
        return (
          <button type="button" onClick={() => showToast({ kind: "info", message: "Bye", duration: 1000 })}>
            fire
          </button>
        );
      }
      renderWithProviders(<Harness />);
      fireEvent.click(screen.getByRole("button", { name: "fire" }));
      expect(screen.getByText("Bye")).toBeInTheDocument();
      act(() => {
        vi.advanceTimersByTime(1100);
      });
      expect(screen.queryByText("Bye")).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("duration=0 keeps the toast visible after the default timer fires", () => {
    vi.useFakeTimers();
    try {
      function Harness() {
        const { showToast } = useToast();
        return (
          <button
            type="button"
            onClick={() => showToast({ kind: "warning", message: "Sticky", duration: 0 })}
          >
            fire
          </button>
        );
      }
      renderWithProviders(<Harness />);
      fireEvent.click(screen.getByRole("button", { name: "fire" }));
      expect(screen.getByText("Sticky")).toBeInTheDocument();
      act(() => {
        vi.advanceTimersByTime(60_000);
      });
      // Still visible — no auto-dismiss
      expect(screen.getByText("Sticky")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("stack cap of 3 drops the oldest when a 4th is pushed", () => {
    function Harness() {
      const { showToast } = useToast();
      return (
        <button
          type="button"
          onClick={() => {
            showToast({ kind: "info", message: "1st", duration: 0 });
            showToast({ kind: "info", message: "2nd", duration: 0 });
            showToast({ kind: "info", message: "3rd", duration: 0 });
            showToast({ kind: "info", message: "4th", duration: 0 });
          }}
        >
          fire
        </button>
      );
    }
    renderWithProviders(<Harness />);
    fireEvent.click(screen.getByRole("button", { name: "fire" }));
    // Oldest is gone, the other three remain
    expect(screen.queryByText("1st")).not.toBeInTheDocument();
    expect(screen.getByText("2nd")).toBeInTheDocument();
    expect(screen.getByText("3rd")).toBeInTheDocument();
    expect(screen.getByText("4th")).toBeInTheDocument();
  });

  it("useToast throws when used outside a Provider", () => {
    // The renderHook helper renders without a Provider wrapping the hook.
    expect(() => renderHook(() => useToast())).toThrow(/ToastProvider/);
  });
});

describe("Toast — module-level dispatcher", () => {
  beforeEach(() => {
    localStorage.setItem("rag.locale", "en");
    __resetToastsForTests();
  });
  afterEach(() => {
    cleanup();
    localStorage.clear();
    setToastDispatcher(null);
    __resetToastsForTests();
    vi.restoreAllMocks();
  });

  it("returns null when no Provider is mounted", () => {
    expect(getCurrentDispatcher()).toBeNull();
  });

  it("returns the provider's dispatcher once a Provider mounts, null again on unmount", () => {
    const ref: { current: ShowToast | null } = { current: null };
    function Probe() {
      const { showToast } = useToast();
      ref.current = showToast;
      return null;
    }
    const { unmount } = render(
      <LocaleProvider>
        <ToastProvider>
          <Probe />
          <ToastViewport />
        </ToastProvider>
      </LocaleProvider>,
    );
    expect(getCurrentDispatcher()).toBe(ref.current);
    unmount();
    expect(getCurrentDispatcher()).toBeNull();
  });

  it("a non-React caller can fire a toast via getCurrentDispatcher", () => {
    function Harness() {
      return (
        <button
          type="button"
          onClick={() => {
            // Simulate a non-React caller (e.g. useThreadSocket catch).
            getCurrentDispatcher()?.({ kind: "info", message: "From outside React" });
          }}
        >
          fire
        </button>
      );
    }
    render(
      <LocaleProvider>
        <ToastProvider>
          <Harness />
          <ToastViewport />
        </ToastProvider>
      </LocaleProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "fire" }));
    expect(screen.getByText("From outside React")).toBeInTheDocument();
  });
});