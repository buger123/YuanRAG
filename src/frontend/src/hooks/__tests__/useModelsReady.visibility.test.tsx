/**
 * v2.0.28.20 — useModelsReady visibility-aware polling tests.
 *
 * Verifies the new ``visibilitychange`` handler in useModelsReady:
 *   - Hook attaches a visibilitychange listener on mount.
 *   - Dispatching visibilitychange → "visible" forces an immediate
 *     re-poll that bypasses the throttled setTimeout (Chrome/Edge
 *     throttle background tabs to ~60 s, so without this the
 *     spinner would stay stuck for the entire interval after the
 *     user comes back from another app).
 *   - Listener is cleaned up on unmount (no leak across hot reloads
 *     or repeated mounts).
 *   - Listener does NOT re-poll when visibilitychange fires for
 *     "hidden" (we only care about coming back to the foreground).
 */
import { describe, expect, it, afterEach, beforeEach, vi } from "vitest";
import { cleanup, render, act } from "@testing-library/react";
import { useModelsReady } from "../useModelsReady";

// Mock the API client so we can drive the polling deterministically.
vi.mock("../../api/client", () => ({
  getModelsStatus: vi.fn(),
  getSettings: vi.fn(),
}));

import { getModelsStatus, getSettings } from "../../api/client";

const mockGetModelsStatus = getModelsStatus as unknown as ReturnType<typeof vi.fn>;
const mockGetSettings = getSettings as unknown as ReturnType<typeof vi.fn>;

function Probe() {
  const result = useModelsReady();
  // Render the spinner rule from ChatPane so we can assert DOM state.
  return (
    <div>
      {result.ready ? null : (
        <span className="chat-header-loading">模型加载中</span>
      )}
      <span data-testid="status">{result.status}</span>
      <span data-testid="checked">{String(result.checked)}</span>
    </div>
  );
}

function deferEmbedReady() {
  // First call returns not-ready; subsequent calls return ready. Used to
  // simulate the "models were loading when the tab was backgrounded"
  // scenario where the throttled timer would have missed the transition.
  let calls = 0;
  mockGetModelsStatus.mockImplementation(async () => {
    calls += 1;
    if (calls === 1) {
      return {
        embedding: { name: "BGE-M3", status: "downloading" },
        reranker: { name: "BGE Reranker v2-M3", status: "downloading" },
        ready: false,
      };
    }
    return {
      embedding: { name: "BGE-M3", status: "ready" },
      reranker: { name: "BGE Reranker v2-M3", status: "ready" },
      ready: true,
    };
  });
  mockGetSettings.mockResolvedValue({
    llm_provider: "anthropic",
    llm_model: "MiniMax-M3",
    has_api_key: true,
    api_key_source: "env",
  });
  return () => calls;
}

describe("useModelsReady visibility-aware polling (v2.0.28.20)", () => {
  beforeEach(() => {
    mockGetModelsStatus.mockReset();
    mockGetSettings.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("re-polls immediately when the tab returns to the foreground", async () => {
    const getCalls = deferEmbedReady();
    // First poll resolves to "downloading"; the spinner is visible.
    // The hook schedules another poll via setTimeout (fake timers here,
    // so it won't fire unless we advance them). The visibility listener
    // is what actually delivers the second poll in the throttled real
    // browser scenario.

    const renderResult = render(<Probe />);

    // Let the first poll resolve.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockGetModelsStatus).toHaveBeenCalledTimes(1);
    expect(renderResult.container.querySelector(".chat-header-loading")).toBeTruthy();

    // Simulate the user coming back to the tab. Without the
    // visibilitychange handler this would do nothing; with it, the
    // hook fires another poll that returns ready=true and clears the
    // spinner.
    await act(async () => {
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        configurable: true,
      });
      document.dispatchEvent(new Event("visibilitychange"));
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(getCalls()).toBe(2);
    expect(renderResult.container.querySelector(".chat-header-loading")).toBeNull();
    expect(renderResult.getByTestId("status").textContent).toBe("ready");
  });

  it("does NOT re-poll when visibilitychange fires for 'hidden'", async () => {
    deferEmbedReady();
    render(<Probe />);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(mockGetModelsStatus).toHaveBeenCalledTimes(1);

    // Going hidden shouldn't trigger an extra poll — only coming back
    // to visible should.
    await act(async () => {
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        configurable: true,
      });
      document.dispatchEvent(new Event("visibilitychange"));
      await Promise.resolve();
    });
    expect(mockGetModelsStatus).toHaveBeenCalledTimes(1);
  });

  it("cleans up the visibilitychange listener on unmount", async () => {
    deferEmbedReady();
    const addSpy = vi.spyOn(document, "addEventListener");
    const removeSpy = vi.spyOn(document, "removeEventListener");
    const renderResult = render(<Probe />);
    await act(async () => {
      await Promise.resolve();
    });

    const added = addSpy.mock.calls.filter((c) => c[0] === "visibilitychange");
    expect(added.length).toBeGreaterThanOrEqual(1);

    // Unmount and verify the matching removeEventListener call fired.
    const initialAddCount = added.length;
    const initialRemoveCount = removeSpy.mock.calls.filter(
      (c) => c[0] === "visibilitychange",
    ).length;

    renderResult.unmount();

    const finalRemoveCount = removeSpy.mock.calls.filter(
      (c) => c[0] === "visibilitychange",
    ).length;
    expect(finalRemoveCount).toBeGreaterThan(initialRemoveCount);
    expect(finalRemoveCount - initialRemoveCount).toBeGreaterThanOrEqual(initialAddCount);
  });

  it("respects cancelled flag — unmounting during a tick leaves no orphan poll", async () => {
    let resolveFirst: (v: unknown) => void = () => undefined;
    mockGetModelsStatus.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve;
        }),
    );
    mockGetSettings.mockResolvedValue({
      llm_provider: "anthropic",
      llm_model: "MiniMax-M3",
      has_api_key: true,
      api_key_source: "env",
    });

    const renderResult = render(<Probe />);

    // While the first poll is pending, unmount.
    renderResult.unmount();

    // Now resolve the first poll. The hook must NOT call setReady /
    // setChecked (the hook is unmounted), and the visibilitychange
    // handler it registered must NOT fire (it's been removed).
    resolveFirst({
      embedding: { name: "BGE-M3", status: "ready" },
      reranker: { name: "BGE Reranker v2-M3", status: "ready" },
      ready: true,
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    // No assertion needed beyond "no error thrown" — the test passes
    // if React doesn't warn about state updates on an unmounted
    // component.
  });
});