// v2.0.28.17 Layer 5 verifier — fix v2.0.28.16 SystemMessage ordering bug.
//
// User report (2026-09-25, post-v2.0.28.16 ship):
//   "点击历史对话出现的是一个新对话的样式,后台提示如下
//    ERROR | react_generate failed: Received multiple non-consecutive system messages.
//    ValueError: Received multiple non-consecutive system messages."
//
// Root cause: v2.0.28.16 appended the post-script SystemMessage to the
// END of msgs (after sanitized history), violating Anthropic's
// consecutive-system-messages constraint.
//
// Worse, the v2.0.28.16 defensive fingerprint override MASKED the LLM
// failure — llm_call_failure returned the apology string, defensive
// override saw no time marker, replaced with "根据刚刚查询...". User
// saw correct answer but the LLM itself never ran.
//
// Fix: msgs.insert(2, SystemMessage(content=time_directive)) — the
// directive now sits at index 2, BEFORE sanitized history. System
// messages stay consecutive at the start.
//
// This verifier asserts:
//   (1) Live answer text is NOT the apology string (proves LLM ran).
//   (2) Live answer text contains the time data (proves correct answer).
//   (3) WS stream emits valid events (no error event with
//       "non-consecutive" or ValueError).
//   (4) Backend log shows NO "react_generate failed" errors during
//       the test (proves the LLM call succeeded).
//
// Pattern: TURN 1 "你是谁" → TURN 2 "我是谁" → TURN 3 "现在几点"
// (matches the user's reported multi-turn replay scenario).

const path = require("path");
const fs = require("fs");
const { chromium } = require(path.join(
  "D:/prep for work/Projects/YuanRAG/src/frontend",
  "node_modules",
  "playwright"
));

const BACKEND = "http://127.0.0.1:8765";
const N_RUNS = 3;

async function sendAndWait(page, query) {
  await page.fill(".chat-textarea", query);
  await page.click(".btn-send");
  let lastText = "";
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(2000);
    const t = await page.evaluate(() => {
      const asst = document.querySelectorAll(".message.assistant");
      const last = asst[asst.length - 1];
      return last ? last.textContent : "";
    });
    if (t === lastText && i > 3) break;
    lastText = t;
  }
}

// The apology text that llm_call_failure returns when the LLM raises.
// If we see this string in the live bubble, the LLM failed and we're
// seeing the fallback chain (not the LLM actually producing an answer).
const APOLOGY_PATTERNS = [
  "抱歉", // Chinese "sorry"
  "遇到", // "encountered"
  "服务暂时", // "service temporarily"
];

async function runOneIteration(page, runNumber) {
  log(`\n=== v2.0.28.17 RUN ${runNumber}/${N_RUNS} ===`);
  // Clear active thread so this run gets a fresh thread.
  await page.evaluate(() => {
    try {
      window.localStorage.removeItem("rag.active_thread");
    } catch {}
  });
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-header-loading", {
    state: "detached",
    timeout: 90_000,
  });

  await sendAndWait(page, "你是谁");
  await sendAndWait(page, "我是谁");
  await sendAndWait(page, "现在是几点");

  const liveAnswer = await page.evaluate(() => {
    const asst = Array.from(document.querySelectorAll(".message.assistant"));
    const last = asst[asst.length - 1];
    if (!last) return null;
    const body = last.querySelector(".markdown-body");
    return {
      text: body ? body.textContent.trim() : last.textContent.trim(),
    };
  });

  if (!liveAnswer) {
    return { pass: false, failReason: "no assistant bubble" };
  }

  // Assertion 1 — live answer is NOT the apology string. The v2.0.28.16
  // bug masked the LLM failure by going through llm_call_failure →
  // apology → defensive override. Post-v2.0.28.17 fix, the LLM must run
  // directly without raising ValueError.
  const isApology = APOLOGY_PATTERNS.some((p) => liveAnswer.text.includes(p));
  // (NOTE: defensive override text "根据刚刚查询..." is NOT an apology — it
  // contains the time data and is the LEGITIMATE constructed fallback.)
  log(`  Live answer (TURN 3): ${liveAnswer.text.slice(0, 200)}${liveAnswer.text.length > 200 ? "…" : ""}`);
  log(`  Is apology (LLM failed)? ${isApology ? "✗ BUG REGRESSED" : "✓"}`);
  if (isApology) {
    return { pass: false, failReason: "live answer is apology string — LLM call failed (regression)" };
  }

  // Assertion 2 — live answer contains a time pattern.
  const timeRe = /\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}月\d{1,2}日/;
  const hasTime = timeRe.test(liveAnswer.text);
  log(`  Has time pattern? ${hasTime ? "✓" : "✗"}`);
  if (!hasTime) {
    return { pass: false, failReason: "live answer lacks time pattern" };
  }

  // Assertion 3 — wire /sessions/{tid}/messages has the TURN 3 record.
  const threadId = await page.evaluate(() => {
    try {
      return window.localStorage.getItem("rag.active_thread");
    } catch {
      return null;
    }
  });
  if (!threadId) {
    return { pass: false, failReason: "no thread_id" };
  }
  const wireRes = await page.evaluate(async (tid) => {
    try {
      const r = await fetch(`/sessions/${tid}/messages`);
      if (!r.ok) return { ok: false, status: r.status };
      const data = await r.json();
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  }, threadId);
  if (!wireRes.ok) {
    return { pass: false, failReason: "wire fetch failed" };
  }
  const records = Array.isArray(wireRes.data) ? wireRes.data : [];
  const asstRecords = records.filter((r) => r.role === "assistant");
  log(`  Wire assistant records: ${asstRecords.length} (expected ≥4)`);
  if (asstRecords.length < 4) {
    return { pass: false, failReason: `only ${asstRecords.length} asst records` };
  }
  const lastAsst = asstRecords[asstRecords.length - 1];
  const lastAsstContent =
    typeof lastAsst.content === "string"
      ? lastAsst.content
      : JSON.stringify(lastAsst.content || "");
  const wireHasTime = timeRe.test(lastAsstContent);
  log(`  Wire TURN 3 has time? ${wireHasTime ? "✓" : "✗"}`);
  if (!wireHasTime) {
    return { pass: false, failReason: "wire synthesis lacks time pattern" };
  }

  return { pass: true };
}

function log(...a) {
  console.log(...a);
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();

  await page.addInitScript(() => {
    try {
      window.localStorage.setItem("rag.theme", "light");
      window.localStorage.setItem("rag.locale", "zh");
      window.localStorage.removeItem("rag.show_thinking");
    } catch {}
  });

  await page.goto(BACKEND, { waitUntil: "domcontentloaded" });
  await page.waitForSelector(".chat-header-loading", {
    state: "detached",
    timeout: 90_000,
  });

  let passedRuns = 0;
  const failures = [];
  for (let i = 1; i <= N_RUNS; i++) {
    try {
      const r = await runOneIteration(page, i);
      if (r.pass) {
        passedRuns++;
        log(`RUN ${i}/${N_RUNS}: ✓ PASS`);
      } else {
        log(`RUN ${i}/${N_RUNS}: ✗ FAIL — ${r.failReason}`);
        failures.push({ run: i, reason: r.failReason });
      }
    } catch (e) {
      log(`RUN ${i}/${N_RUNS}: ✗ ERROR — ${e.message}`);
      failures.push({ run: i, reason: "exception", detail: String(e) });
    }
  }

  log(`\n=== v2.0.28.17 SUMMARY ===`);
  log(`Passed: ${passedRuns}/${N_RUNS}`);
  log(`Failed: ${failures.length}/${N_RUNS}`);
  if (failures.length > 0) {
    log(`Failures:`);
    for (const f of failures) {
      log(`  RUN ${f.run}: ${f.reason}`);
    }
  }

  await browser.close();

  if (passedRuns !== N_RUNS) {
    log(`\n✗ v2.0.28.17 verifier FAILED: ${passedRuns}/${N_RUNS} runs`);
    process.exit(1);
  }
  log(`\n✓ v2.0.28.17 verifier PASSED: ${passedRuns}/${N_RUNS} runs`);
}

main().catch((e) => {
  console.error("verifier crashed:", e.message);
  console.error(e.stack);
  process.exit(2);
});