# Layer 5 e2e — Playwright

This directory is the **Layer 5 verification harness** — the canonical
gate before shipping any fix or feature. See:

- [`../../memory/development-methodology.md`](../../memory/development-methodology.md)
  — the rule "fixes don't ship without Layer 5"
- [`../../memory/verification-protocol.md`](../../memory/verification-protocol.md)
  — the exact steps Layer 5 enforces

## What's here

| File | Purpose |
|---|---|
| `playwright.config.ts` | Spawns `vite preview` (which builds + serves `dist/`) and proxies `/api` + `/ws` to the FastAPI backend on `:8765`. |
| `smoke.spec.ts` | The four canonical DOM assertions: mid-stream height, final-text not whitespace, no leftover spinner, exactly one assistant bubble. |
| `run.sh` | One-shot runner: `npm install` if needed, `playwright install chromium`, `playwright test`. |
| `.gitignore` | Excludes `output/`, `test-results/`, `playwright-report/`. |

## Run

```bash
# from repo root, with backend already up + .env populated
bash tests/e2e/run.sh
```

The harness does NOT start the backend itself — it must be running on
`http://127.0.0.1:8765` with a real LLM key, otherwise `smoke.spec.ts`
SKIPs with a clear message (the harness is meant for humans on real
hardware, not blind CI).

## What Layer 5 catches

Past bugs only this layer could surface:

- **v2.0.12** — `useDeferredValue` starved mid-stream; `.markdown-body`
  height stayed at 0 until `done`. Caught by assertion 1.
- **v2.0.13** — `answer_complete` whitespace canonical clobbered streamed
  text. Caught by assertion 2.
- **v1.1.6** — LangGraph `messages` channel emits synthesized full
  chunk, doubling live content. Caught indirectly: token-event count
  would diverge from canonical length.
- **v1.1.8** — `events.web_search` `bool` class typo. Caught by the
  schema guard in `src/agent/wire_protocol.py` (added by Item 3 of the
  Phase 1 refactor).

## What Layer 5 does NOT replace

- Unit tests still own individual function correctness.
- Integration tests still own wire-format round-trip.
- The conftest's `_isolate_user_data_dir` autouse still owns per-test
  hermeticity.

Layer 5 is the *only* layer that exercises **React 19 runtime +
LangGraph astream + Anthropic streaming + WebSocket multiplexing + DOM
rendering + CSS layout + Vitest build artifacts** simultaneously. The
unit suite mocks out at least one of those.

## CI integration

This harness is **not** wired into CI today (no real LLM key in CI).
To add it: provision CI with a real key + pre-loaded BGE-M3 cache, then
add a `e2e` job that runs `bash tests/e2e/run.sh` after backend boots.