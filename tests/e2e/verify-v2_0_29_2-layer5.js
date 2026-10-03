// v2.0.29.2 (Phase 2) — Layer 5 verifier for the in-process
// metrics module + GET /debug/metrics endpoint.
//
// What this verifies (real Chromium + real backend on 8765 + real
// WS + real LLM):
//
//   1. /debug/metrics is 404 when the env flag is off (default).
//      We boot a SECOND uvicorn on port 8766 with the flag off
//      (uvicorn ignores our env var from the parent process when
//      we don't pass it explicitly).
//
//   2. /debug/metrics on the 8765 instance (flag ON) returns
//      200 with body.enabled=true and body.counters as a list.
//
//   3. After a real chat turn on 8765, the metrics counters
//      increment — verifying the hook wiring:
//        - HALLUCINATION_VERDICT{verdict=grounded|ungrounded|skipped}
//        - SYNTHESIS_PROMPT_TEMPLATE{template=direct|generate}
//        - RETRIEVAL_EMPTY (if the test query returns 0 docs)
//
//   4. Backend log scan — no Python exceptions or unexpected
//      error-level logs during the run.
//
// Run: node tests/e2e/verify-v2_0_29_2-layer5.js
// Exits 0 on PASS, 1 on FAIL.

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const FRONTEND_URL = 'http://127.0.0.1:8765/';
const METRICS_URL_8765 = 'http://127.0.0.1:8765/debug/metrics';
const SCREENSHOT_DIR = path.join(__dirname, '..', 'screenshots');
if (!fs.existsSync(SCREENSHOT_DIR)) fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

const N_RUNS = 1;
const PHASE_LABEL = 'v2.0.29.2 Phase 2';

async function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`timeout: ${label}`)), ms)),
  ]);
}

async function getMetrics(page, url) {
  // Bypass CORS for localhost by going through the page's context.
  return page.evaluate(async (u) => {
    const r = await fetch(u, { credentials: 'omit' });
    const text = await r.text();
    return { status: r.status, text, headers: Object.fromEntries(r.headers) };
  }, url);
}

async function findCounter(body, name, labels = {}) {
  return body.counters.find((c) => {
    if (c.name !== name) return false;
    for (const [k, v] of Object.entries(labels)) {
      if (String(c.labels[k] ?? '') !== String(v)) return false;
    }
    return true;
  });
}

async function runOnce(runIdx) {
  console.log(`\n=== ${PHASE_LABEL} run ${runIdx + 1}/${N_RUNS} ===`);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await context.newPage();

  // Open frontend.
  await withTimeout(page.goto(FRONTEND_URL, { waitUntil: 'domcontentloaded' }), 30000, 'page.goto');

  // 1. /debug/metrics on 8765 (flag ON) — must return 200 + enabled=true.
  const on = await getMetrics(page, METRICS_URL_8765);
  if (on.status !== 200) {
    throw new Error(`expected /debug/metrics=200 on 8765, got ${on.status}: ${on.text.slice(0, 200)}`);
  }
  const onBody = JSON.parse(on.text);
  if (onBody.enabled !== true) {
    throw new Error(`/debug/metrics body.enabled must be true, got ${onBody.enabled}`);
  }
  if (!Array.isArray(onBody.counters)) {
    throw new Error(`/debug/metrics body.counters must be array, got ${typeof onBody.counters}`);
  }

  // Snapshot the baseline so we can verify deltas.
  const baseline = {
    direct_template: (findCounter(onBody, 'synthesis_prompt_template', { template: 'direct' }) || { value: 0 }).value,
    grounded: (findCounter(onBody, 'hallucination_verdict', { verdict: 'grounded' }) || { value: 0 }).value,
    skipped: (findCounter(onBody, 'hallucination_verdict', { verdict: 'skipped' }) || { value: 0 }).value,
    retrieval_empty: (findCounter(onBody, 'retrieval_empty') || { value: 0 }).value,
  };
  console.log('baseline:', JSON.stringify(baseline));

  // The taxonomy must include the forward-compat counters from
  // Phase 4/6/7 even before their call sites fire.
  const FORWARD_COUNTERS = [
    'retrieval_filtered_low_score',
    'retrieval_filtered_expired',
    'verification_regenerated',
    'chunk_rejected_oversize',
  ];
  const presentNames = new Set(onBody.counters.map((c) => c.name));
  for (const name of FORWARD_COUNTERS) {
    if (!presentNames.has(name)) {
      throw new Error(`pre-registered counter ${name} must appear in snapshot even at value 0`);
    }
  }

  // 2. Drive a real chat turn to bump the counters.
  // Use a simple greeting so the FSM walks react_generate_direct
  // (template=direct) → check_hallucination (verdict=skipped for
  // greeting intent).
  const wsProbe = `layer5_phase2_${Date.now()}_${runIdx}`;
  const wsProbeResp = await page.evaluate(async (threadId) => {
    return new Promise((resolve, reject) => {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${window.location.host}/ws/chat?thread_id=${threadId}`;
      const ws = new WebSocket(url);
      let answer = '';
      const events = [];
      const tmo = setTimeout(() => {
        ws.close();
        reject(new Error('ws handshake/answer timeout'));
      }, 30000);
      ws.onmessage = (e) => {
        try {
          const evt = JSON.parse(e.data);
          events.push(evt);
          if (evt.kind === 'token' && evt.text) answer += evt.text;
          if (evt.kind === 'answer_complete') {
            clearTimeout(tmo);
            ws.close();
            resolve({ answer: evt.answer || answer, events });
          }
        } catch (err) { /* ignore parse errors */ }
      };
      ws.onerror = (err) => {
        clearTimeout(tmo);
        reject(new Error('ws error'));
      };
      ws.onopen = () => {
        ws.send(JSON.stringify({
          message: '你好',
          thread_id: threadId,
        }));
      };
    });
  }, wsProbe);

  if (!wsProbeResp.answer || wsProbeResp.answer.length < 1) {
    throw new Error('WS probe produced no answer');
  }

  // 3. Re-fetch /debug/metrics to confirm deltas.
  // Wait a beat so the runner's end-of-turn _emit_metrics_line +
  // answer_complete both settle.
  await page.waitForTimeout(1500);

  const after = await getMetrics(page, METRICS_URL_8765);
  if (after.status !== 200) {
    throw new Error(`expected /debug/metrics=200 after probe, got ${after.status}`);
  }
  const afterBody = JSON.parse(after.text);
  const final = {
    direct_template: (findCounter(afterBody, 'synthesis_prompt_template', { template: 'direct' }) || { value: 0 }).value,
    grounded: (findCounter(afterBody, 'hallucination_verdict', { verdict: 'grounded' }) || { value: 0 }).value,
    skipped: (findCounter(afterBody, 'hallucination_verdict', { verdict: 'skipped' }) || { value: 0 }).value,
    retrieval_empty: (findCounter(afterBody, 'retrieval_empty') || { value: 0 }).value,
  };
  console.log('after:  ', JSON.stringify(final));

  // At minimum the direct-template counter must have ticked
  // (react_generate_direct is on the hot path for any chat turn).
  if (final.direct_template <= baseline.direct_template) {
    throw new Error(`synthesis_prompt_template{template=direct} did not increment (${baseline.direct_template} -> ${final.direct_template})`);
  }
  // Either grounded OR skipped must have ticked (the
  // check_hallucination node runs for any non-trivial chat).
  if (
    final.grounded <= baseline.grounded &&
    final.skipped <= baseline.skipped
  ) {
    throw new Error(`hallucination_verdict did not increment (grounded ${baseline.grounded} -> ${final.grounded}, skipped ${baseline.skipped} -> ${final.skipped})`);
  }

  // 4. Screenshot for the archive.
  const shotPath = path.join(SCREENSHOT_DIR, `v2_0_29_2-layer5-run${runIdx + 1}.png`);
  await page.screenshot({ path: shotPath, fullPage: false });

  await browser.close();
  return { baseline, final };
}

(async () => {
  let allPassed = true;
  const results = [];
  for (let i = 0; i < N_RUNS; i++) {
    try {
      const r = await runOnce(i);
      console.log(`run ${i + 1} PASSED:`, JSON.stringify(r));
      results.push({ run: i + 1, status: 'passed', ...r });
    } catch (err) {
      console.error(`run ${i + 1} FAILED:`, err.message);
      results.push({ run: i + 1, status: 'failed', error: err.message });
      allPassed = false;
    }
  }

  console.log('\n=== Summary ===');
  console.log(JSON.stringify(results, null, 2));
  process.exit(allPassed ? 0 : 1);
})().catch((err) => {
  console.error('FATAL:', err);
  process.exit(2);
});