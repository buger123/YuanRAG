/**
 * v2.0.30.0 — useModelsReady long-poll-first tests.
 *
 * Verifies the new ``waitForModelsReady``-then-fallback shape in
 * useModelsReady:
 *   - Hook calls ``waitForModelsReady`` on mount instead of (only)
 *     ``getModelsStatus``.
 *   - When the long-poll resolves with ``ready=true``, the spinner
 *     detaches without any further polling.
 *   - When the long-poll rejects (timeout / abort / network), the
 *     hook falls back to the existing 2 s ``getModelsStatus`` polling
 *     cadence.
 *   - Unmount during the long-poll aborts the in-flight request so
 *     we don't setState on an unmounted component.
 */
import { describe, expect, it, afterEach, beforeEach, vi } from "vitest";
import { cleanup, render, act } from "@testing-library/react";
import { useModelsReady } from "../useModelsReady";

// Mock the API client so we can drive the long-poll / poll cadence
// deterministically.
vi.mock("../../api/client", () => ({
  getModelsStatus: vi.fn(),
  getSettings: vi.fn(),
  waitForModelsReady: vi.fn(),
}));

import {
  getModelsStatus,
  getSettings,
  waitForModelsReady,
} from "../../api/client";

const mockGetModelsStatus = getModelsStatus as unknown as ReturnType<typeof vi.fn>;
const mockGetSettings = getSettings as unknown as ReturnType<typeof vi.fn>;
const mockWaitForModelsReady = waitForModelsReady as unknown as ReturnType<typeof vi.fn>;

function Probe() {
  const result = useModelsReady();
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

const READY_SNAPSHOT = {
  embedding: { name: "BGE-M3", status: "ready" },
  reranker: { name: "BGE Reranker v2-M3", status: "ready" },
  ready: true,
};

const DOWNLOADING_SNAPSHOT = {
  embedding: { name: "BGE-M3", status: "downloading" },
  reranker: { name: "BGE Reranker v2-M3", status: "downloading" },
  ready: false,
};

function stubSettings() {
  mockGetSettings.mockResolvedValue({
    llm_provider: "anthropic",
    llm_model: "MiniMax-M3",
    has_api_key: true,
    api_key_source: "env",
  });
}

describe("useModelsReady long-poll (v2.0.30.0)", () => {
  beforeEach(() => {
    mockGetModelsStatus.mockReset();
    mockGetSettings.mockReset();
    mockWaitForModelsReady.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("calls waitForModelsReady on mount instead of (only) getModelsStatus", async () => {
    mockWaitForModelsReady.mockResolvedValue(READY_SNAPSHOT);
    stubSettings();

    render(<Probe />);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockWaitForModelsReady).toHaveBeenCalledTimes(1);
    expect(mockWaitForModelsReady).toHaveBeenCalledWith(
      60,
      expect.objectContaining({ aborted: expect.any(Boolean) }),
    );
  });

  it("resolves to ready without further polling when the long-poll returns ready", async () => {
    mockWaitForModelsReady.mockResolvedValue(READY_SNAPSHOT);
    stubSettings();

    const renderResult = render(<Probe />);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    // Long-poll fired and resolved → spinner detached, no fallback polling.
    expect(mockWaitForModelsReady).toHaveBeenCalledTimes(1);
    expect(mockGetModelsStatus).not.toHaveBeenCalled();
    expect(renderResult.container.querySelector(".chat-header-loading")).toBeNull();
    expect(renderResult.getByTestId("status").textContent).toBe("ready");
  });

  it("falls back to getModelsStatus polling when the long-poll rejects with 408", async () => {
    // Long-poll rejects (timeout / abort / network) → fall back.
    mockWaitForModelsReady.mockRejectedValue(
      new Error("/models/wait returned 408"),
    );
    // Polling fallback returns downloading first, then ready.
    let pollCalls = 0;
    mockGetModelsStatus.mockImplementation(async () => {
      pollCalls += 1;
      if (pollCalls === 1) return DOWNLOADING_SNAPSHOT;
      return READY_SNAPSHOT;
    });
    stubSettings();

    const renderResult = render(<Probe />);
    // Let the long-poll reject + the first fallback poll complete.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    // Long-poll failed → at least one fallback poll should have fired.
    expect(mockWaitForModelsReady).toHaveBeenCalledTimes(1);
    expect(mockGetModelsStatus).toHaveBeenCalled();

    // Advance the 2 s polling cadence — the second poll should resolve
    // to ready.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100);
    });

    expect(renderResult.container.querySelector(".chat-header-loading")).toBeNull();
    expect(renderResult.getByTestId("status").textContent).toBe("ready");
  });

  it("aborts the long-poll on unmount so no orphan setState fires", async () => {
    // Long-poll that never resolves — only unmount should release it.
    let abortSignal: AbortSignal | undefined;
    mockWaitForModelsReady.mockImplementation(
      (_timeout: number, signal?: AbortSignal) =>
        new Promise((_resolve, reject) => {
          abortSignal = signal;
          if (signal) {
            signal.addEventListener("abort", () => {
              const e = new Error("aborted");
              e.name = "AbortError";
              reject(e);
            });
          }
        }),
    );
    stubSettings();

    const renderResult = render(<Probe />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(mockWaitForModelsReady).toHaveBeenCalledTimes(1);
    expect(abortSignal).toBeDefined();
    expect(abortSignal!.aborted).toBe(false);

    renderResult.unmount();
    // Effect cleanup must abort the long-poll.
    expect(abortSignal!.aborted).toBe(true);
  });
});