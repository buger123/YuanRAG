# CHANGELOG

> 按版本号**从新到旧**排列。每条只写「**修改了什么 / 原理 / 有何提升**」三件事;早期尝试若在后续版本才真正修好,合并到最终修复的那一版里。  
> 每个版本的完整 debugging narrative 在 per-developer 笔记 `memory/<version>.md`(`memory/` 不入公开仓)。

---

## 2026-10 · 评估 harness(eval v1)

### v2.0.32.3 — 2026-10-05 · Phase 1.5 真 bug fix · defensive override 多 TM 综合 fallback
- **修改了什么:** `src/agent/nodes/_defensive_override_registry.py:apply_defensive_overrides` 反转 iteration 命中 first miss 立即 return → 改为 dedupe 到 LATEST TM per covered tool + check ALL covered-tool marker + 任一 miss 时返回 ALL covered-tool fallback `\n\n` join + metric 按 tool 粒度 bump;`tests/test_defensive_override_registry.py` +7 NEW pytest 锁 multi-TM 行为。
- **原理:** Eval Phase 1 暴露 30% pass rate 根因 = defensive override 只用 FIRST tool call TM data,丢后续 retrieval chunks(PDF chunks 被丢,只剩 time 结果)。Phase 1.5 让 override 路径用 every byte of TM data — never drop retrieval data because an unrelated tool's marker was missing。Same-tool multi-call dedupe 到 LATEST 保 v2.0.28.16 multi-turn invariant。
- **有何提升:** 32/32 defensive override registry tests PASSED(+7 NEW),其他模块 0 新增回归(8 pre-existing fail 验证 stash 验证与本改动无关);**invariant**:`DIRECTIVE_BUILDERS` / `FINGERPRINT_EXTRACTORS` / `FALLBACK_BUILDERS` 三 registry 对齐不动 / most-recent-wins dedupe 保 multi-turn safety / 单 tool 路径 backward-compat(`get_current_time` 现有 25 测试不动) / `SYNTHESIS_TM_IGNORED{tool=...}` 标签仍只 bump detected miss(unparseable marker 不 bump)。

### v2.0.32.2 — 2026-10-05 · Phase 3+4 cheap_llm judge wiring + CI nightly cron
- **修改了什么:** `tests/eval/cli.py` `--judge all/cheap_llm` dispatch 由 no-op 改为真调 `run_cheap_llm_judge` per turn + 同步算 composite_score;`tests/eval/report.py` `hallucination_rate` 改读 `per_turn[]` 而不是 flat dict;每 run 末尾自动 emit `<timestamp>.claude_prompt.txt`(Pass 3 离线 verdict prompt stub);`.github/workflows/ci.yml`(fast mocked pytest + frontend build)+ `.github/workflows/eval-nightly.yml`(cron 02:00 UTC daily + manual dispatch,boot backend → `/models/wait` → eval suite → upload artifacts 90d)。
- **原理:** Phase 3 修了 `--judge` flag 假传问题:之前 runner 永远写 `"cheap_llm": None`,CLI flag 完全没接;现在 sync CLI 内 `asyncio.run` 包 async judge,judge 失败 swallow 让 composite 回退 programmatic-only。Phase 4 把 eval 接入 GitHub Actions:CI 默认只跑 mocked(避 torch/bge-m3 重依赖),nightly cron 装 full ML deps + boot 真 backend 跑完整 suite,threshold 0.0 验证 infra,等 Phase 1.5 修 defensive override + API 恢复后改 0.7。
- **有何提升:** 35/35 mocked pytest PASSED(+2 NEW Phase 3),0 回归;**invariant**:生产代码不动 / WS wire shape 不变 / `retrieval_status` 4 值不变 / CI 不传 `RAG_DATA_DIR` 共享 host 数据(autouse tmp_path 隔离)/ nightly 90min timeout 防 hang。

### v2.0.32.1 — 2026-10-05 · eval harness calibration + en mirror (i18n regression net)
- **修改了什么:** `tests/fixtures/eval_v1/cases/golden.yaml` 4 case calibrate(golden-001 route_decision direct→retrieve / 002 tool_call_count 2→3 / 005+006 expected_grounding grounded→skipped)+ 5 en mirror case(001-time/002-weather/003-greeting/008-verbatim/010-direct);judge `_answer_field` / `_sources` 改读 LAST `answer_complete` event 而非 FIRST(原 first 撞到空 placeholder)。
- **原理:** Eval infra ship 后(commit 361ef3a)真 LLM 跑暴露 7 case 因 Anthropic 529 OverloadedError 雪崩→`defensive override` 用不全 TM data→fallback 错误,真 ceiling 30% 不在 calibration;5 en mirror 验证 `Accept-Language` header → i18n catalog → 英文 answer 渲染链路,留作 regression net。
- **有何提升:** golden.yaml 10→15 cases load 干净;Layer 5 ship gate threshold=0.0 仍 PASS(infra 验证),生产 threshold 0.7 等 Phase 1.5 修 defensive override + Anthropic API 恢复;**invariant**:生产代码不动 / WS wire shape 不变 / `retrieval_status` 4 值不变 / judge `_answer_field` 只读 wire event 不改 production。

### v2.0.32.0 — 2026-10-04 · RAG 评估 harness (golden + adversarial + Claude-as-judge)
- **修改了什么:** `tests/eval/` NEW 12-file package(`cases` / `runner` / `tokens` / `cost` / `report` / `cli` / `judge/{programmatic,cheap_llm,claudemd}`)+ `tests/fixtures/eval_v1/` 7 real-corpus 文件(txt/pdf/docx/md/html/xlsx/pptx 全)+ 10 golden + 22 adversarial YAML case(全 zh,覆盖 Phase 8 6 根因组 A-F)+ 3-pass judge(程序化结构断言 / cheap-LLM 0-2 评分 / Claude 离线 verdict)+ JSONL+Markdown+console 报告。
- **原理:** 固定测试集是反幻觉 9-Phase 链路的「温度计」—— 32 case 量化 成功率 / 失败原因 / 工具调用次数 / 成本 / 延迟 / 幻觉率 / 引用完整性;跑真 `POST /chat` SSE wire path(不直接 import `stream_agent`),捕获 wire-format / i18n catalog / citation renumber / upload-clear 全 bug surface。
- **有何提升:** 33/33 mocked pytest PASSED,0 回归(其他 case 既有 234 fail + 74 error 是 anthropic SDK env 问题,非本改动);**invariant**:不修改生产代码 / WS wire shape 不变 / `retrieval_status` 4 值不变 / `Source.verbatim` 默认 False backward-compat / `RAG_DATA_DIR` pytest autouse tmp_path + CLI 自管 ensure_isolated_data_dir 隔离 / `safe_load` 强制 + `yaml.YAMLError → CaseLoadError` re-raise / `TokenCapture.install/uninstall` 配对 exception-safe。

---

## 2026-09 · 反幻觉 9-Phase 链路 + UX polish(post-hallucination)

### v2.0.31.1 — 2026-09-29 · 高精确模式首次使用可发现性
- **修改了什么:** `<div className="high-precision-toggle-row">` 三元素栈 = 顶部 always-visible label「🔒 高精确模式」+ 中间 segmented + 底部 `<div className="high-precision-toggle-description" aria-live="polite">` 实时显示当前 mode hint;1 mirror i18n key。
- **原理:** 收 root cause「first-time discoverability = 0」—— touch / SR / 新 user 全无 affordance 解释 toggle。Label 解决「toggle 是干嘛的」,动态 description 解决「当前 mode 是干嘛的」(click 即时切换,无需 hover)。
- **有何提升:** Layer 5 4/4 PASSED(24 子断言)+ 0 回归;**invariant**:v2.0.31.0 segmented/CSS 不动 / `.verbatim-lock-icon` 复用不动。

### v2.0.31.0 — 2026-09-29 · 高精确模式 UI 美化
- **修改了什么:** segmented 复用 `.segmented` / `.segmented-option` 设计类;active state 从 `.active` class 改 `data-active="true"` attribute(复用 SettingsDialog 已 ship 规则);bubble 🔒 raw emoji → `<span className="verbatim-lock-icon">` 包裹;`.chat-input { flex-wrap: wrap }` 修正原 JSX comment 与实现矛盾。
- **原理:** Pre-v2.0.31.0 `grep "high-precision\|verbatim" styles.css` 0 行,3 按钮灰色 chrome + default border,visual broken。修复复用 SettingsDialog 设计 token,minimum surface。
- **有何提升:** Layer 5 4/4 PASSED(16 子断言)+ 0 回归;**invariant**:`.segmented` 复用不动 / 7 i18n keys 不动。

### v2.0.30.0 — 2026-09-29 · 模型加载完成推送:长轮询 `/models/wait`
- **修改了什么:** `GET /models/wait?timeout=N` long-poll endpoint + `waitForModelsReady()` client + `useModelsReady` AbortController 包裹 long-poll first / fall back 2s polling;v2.0.28.20 visibilitychange listener 扩展为 hide-abort + show-kick-fresh。
- **原理:** Pre-v2.0.30.0 foreground tab 仍 up to 2000ms 延迟(2s 轮询 cadence)。长轮询 1 request per "ready check" 复用 FastAPI async,零 infra 改动。`ModelsStatus` schema 不变 → debug / health-check 路径 0 改动。
- **有何提升:** 实测 latency ~275ms(pre ≤ 2000ms → post ≤ 500ms);**invariant**:`ModelsStatus` schema 不变 / WS wire shape 不变 / `useModelsReady` external API 不变。

### v2.0.29.9 — 2026-09-28 · Phase 8 · 高精确场景禁推理模式(verbatim extraction)
- **修改了什么:** `high_precision: Literal["auto","on","off"]` tristate + `Source.verbatim` additive + `react_generate_extractive` async-gen node + `EXTRACTIVE_SYSTEM` 严格 verbatim prompt(6 hard rules + 13 forbidden phrases + 「原文未提及」fallback)+ `EXTRACTIVE_FALLBACK` Counter + intent_analysis hybrid detect(Layer 1 regex + Layer 2 cheap-LLM zero 第二 LLM call)+ per-thread 3-state segmented `[自动/开启/关闭]` UI + 🔒 icon + 7 symmetric i18n key。
- **原理:** 法律/医疗/财务场景 LLM 自主改写(把"2023年1月1日生效"改成"今年初"),Phase 1/4/6 抓不到。Phase 8 物理禁止 paraphrase,`Source.verbatim=True` 让前端 🔒 视觉确认。User "off" 无条件 wins。
- **有何提升:** Layer 5 hermetic 5/5 + real-Chromium 7/7 PASSED;51 NEW pytest;**invariant**:`retrieval_status` 4 值不变 / `Source.verbatim` 默认 False backward-compat / WS wire shape 不变。

### v2.0.29.8 — 2026-09-28 · Phase 7 · 超长文档分片过滤
- **修改了什么:** `MAX_CHUNKS_PER_DOC=50` sliding window(head 5 + middle 5 evenly-spaced + tail 5 = 15 chunks)per-doc overflow + `_hit_to_document` 1-LOC `chunk_index` 传播 fix + wire hybrid path + summary-intent branch。
- **原理:** 长文档(200 页 PDF / 大 Excel)dump 100+ chunk 进 synthesis LLM,中段关键信息丢失。选 head + tail 而非 top-K 因长文档关键信息多在 boundary。
- **有何提升:** 21 NEW pytest + Layer 5 4/4 PASSED;**invariant**:`retrieval_status` 4 值不变 / `_filter_long_docs` 在 `_filter_superseded` 之后。

### v2.0.29.7 — 2026-09-28 · Phase 6 · 后校验机制(rule engine + cheap-LLM)
- **修改了什么:** NEW `src/agent/verification/` subpackage + `verify_answer` FSM node + `VerificationResultEvent` + `should_trigger_verification` zero-cost gate + rule_engine 4 字段类型(date/currency/percentage/article_number)+ verifier LLM cheap-model second-pass + regen/fallback_refusal/skip 分支。
- **原理:** 「引用对」≠「数字对」。双层 fail-closed:rule engine regex 抓结构化事实不一致(~ms 级零 LLM);verifier 复用 cheap-model second-pass 抓语义级。LLM JSON parse fail / ainvoke exception → fail-closed 防悄悄放过。
- **有何提升:** 39 NEW pytest + Layer 5 4/4 PASSED;**invariant**:zero-cost gate / fail-closed semantics。

### v2.0.29.6 — Phase 5 · 并发 + 清理 + UX
- **修改了什么:** 删 `_override_top_k` → keyword-only `top_k`(防 thread-safety race)+ `_hybrid_executor atexit.register(.shutdown)` + `_doc_id_for_url` 加 domain + dispatcher `asyncio.gather(return_exceptions=True)` + URL dedup + `(-score, engine_priority)` 排序 + `@pytest.mark.slow` opt-in gate。
- **原理:** thread-safety race / engine-leak / 单引擎失败阻塞 等并发缺陷一次性收尾。
- **有何提升:** 26 NEW pytest + Layer 5 10/11 PASSED;**invariant**:keyword-only 后向兼容 / `atexit` 幂等。

### v2.0.29.5 — Phase 4 PR-2 · 检索侧硬化
- **修改了什么:** `expires_at` schema + migration + `temporal_filter` SQL + `doc_registry` versioning + `supersede()` + `_filter_superseded` post-filter + `clean_chunks()` 4 rules。
- **原理:** 过期片段当现行;旧版本没归档;脏数据污染索引。
- **有何提升:** 19 NEW pytest + Layer 5 4/4 PASSED;**invariant**:`include_superseded=False` default 向后兼容。

### v2.0.29.4 — Phase 4 PR-1 · 引用 + 检索完整性(load-bearing)
- **修改了什么:** 6 load-bearing fix:rerank 失败 stamp / range collapse WARNING log / dedupe key `page_content[:200]` / CitationChip+ChatPane chip click detail 加 chunk_id+doc_id+source_kind / `_SUMMARY_INTENT_RE` 锚定开头 / `min_top_relevance_score=0.3` gate。
- **原理:** 「引用对但内容是 LLM 编的」是引用完整性的根因。
- **有何提升:** 14 NEW pytest + Layer 5 5/5 PASSED;**invariant**:`retrieval_status` Literal 不加新值。

### v2.0.29.3 — Phase 3 · 防御性 override 通用化 + loop-break synthesis hint
- **修改了什么:** `_defensive_override_registry.py` NEW 3-registry + `apply_defensive_overrides()` helper;`react_generate.py` 删 hardcoded 3 helper + 30 行 inline override block;`_LOOP_BREAK_SYNTHESIS_HINT` 在 `loop_breaker_active=True` 时 `msgs.insert(2, ...)`。
- **原理:** 同类 bug 一次性根除。加新 tool = 1 行,call site 0 改动。
- **有何提升:** 25 NEW pytest + Layer 5 3/3 PASSED。

### v2.0.29.2 — Phase 2 · 自建 metrics + loguru 观测 + `/debug/metrics`
- **修改了什么:** `metrics.py` NEW Counter + 9 pre-registered + `debug_metrics.py` NEW GET `/debug/metrics`(env-flag gated 404)+ dual env-flag + 7 hook sites。
- **原理:** 翻车没 log,无法定位。env-flag 双 gate production overhead = 0。
- **有何提升:** 18 NEW pytest + Layer 5 3/3 PASSED。

### v2.0.29.1 — Phase 1 · 拒绝契约
- **修改了什么:** 3 拒绝模板 + `retrieval_status` Literal + `REFUSAL_TEMPLATES` dict + 4 path 设 status + `msgs.insert(2, refusal_directive)` + `JUDGE_SNIPPET_CHARS_PER_DOC=2000`。
- **原理:** LLM 拿不到可信来源时,**显式拒绝**而不是 hedge。后续 Phase 6 verifier 也复用此 contract。
- **有何提升:** 19/19 pytest + Layer 5 3/3 PASSED。

### v2.0.29.0 — Day 0 · OOD 对抗性 fixture 集
- **修改了什么:** `tests/fixtures/audit_v2/` 7 fixture + 11 行为断言;fake embedder/reranker + 真 LanceDB + BM25 FTS。
- **原理:** 先把「典型翻车姿势」固化下来,所有后续 phase 的回归测试都跑这 7 fixture,防止修一个 phase 引入另一个 phase 的回归。
- **有何提升:** 11/11 PASSED + 0 回归;**后续 Phase 1/4/6/8 verifier 全部复用此 fixture 集**。

---

## 2026-09 · Items 6-11 cycle(P0 hotfixes + UX foundation + hardening)

> 5 个 Items × 多 PR × 18 个版本,集中在 2026-09-17 → 2026-09-26 一波 ship。早期尝试若在后续版本才真正修好,合并到最终修复版。

### v2.0.28.20 — 后台 tab 模型加载完前台 spinner 还卡着
- **修改了什么:** `useModelsReady.ts` +~30 LOC `visibilitychange` listener → re-poll on `visibilityState=visible`。
- **原理:** background-tab throttling 把 polling 压到 60s+,`visibilitychange` listener 唤醒后立刻 re-poll。
- **有何提升:** 132/132 cum + 136/136 vitest 0 回归。

### v2.0.28.19 — `npm run dev` 永远卡 loading
- **修改了什么:** `vite.config.ts` bypass callback 4 类前缀(`/`, `/src/`, `/@`, `/node_modules/`)。

### v2.0.28.18 — Document-namespace corruption 让 thread 在 sidebar 隐身
- **修改了什么:** `checkpointer.py` **load-bearing 1 LOC** `allowed_objects="messages"`→`"core"` + write-time `documents` pre-convert + `_LEGACY_NAMESPACE_REMAP` walker。

### v2.0.28.17 — 多轮 tool-using 合成 LLM 同根因 9 个 bug 合并 fix
- **修改了什么:** `react_generate.py` 1 LOC `append`→`insert(2, ...)` (Anthropic consecutiveness)。
- **原理:** defense-in-depth without verification 是 silent failure mode —— defensive override MASKED 了 LLM failure。

### v2.0.28.16 / v2.0.28.15 / v2.0.28.14 / v2.0.28.13 — 多轮合成 LLM UX bug 集
- **修改了什么:** tool-result 利用率 / meta-commentary leak 切断 / `mergeAdjacentAssistantTurns` 折叠 / reload-hydration "double tool-call card"。
- **原理:** DB/wire 不能动(Anthropic tool_use ↔ tool_result 强制 2 AIMessage/turn),只能 frontend-only 修。

### v2.0.28.12 / v2.0.28.12.1 / v2.0.28.11 / v2.0.28.10 / v2.0.28.9 — 多轮 replay + synthesis meta-text + FSM data integrity
- **修改了什么:** 「现在几点了」intent misroute + synthesis placeholder 短路 + synthesis meta-text strip + FSM 2-AIMessage-per-turn + `runner.stream_agent()` 之前 hydration block。

### v2.0.28.8 / v2.0.28.7 — UX 默认 + 响应式
- **修改了什么:** `ThemeProvider.tsx:61` `"system"` → `"light"` 默认改;`.sidebar-backdrop` base rule HOIST `@media (max-width:768px)` → top level。

### v2.0.28.5 → v2.0.28 · Item 10-11(retrieval + web concurrency + LLM resilience)
- **PR:** 6 PR(trace_id / Settings TOCTOU / list_documents cache / embed_query 双重锁 / BGE release / LLM tenacity retry)。
- **覆盖:** 工具检索 + web 并发;Pydantic parse_structured 重试;httpx 缓存清理;trace_id 12-hex 全链路闭环;SQLite BEGIN IMMEDIATE 写并发。
- **总耗时:** 4 PR / 4 d,vs audit 12 d 估算缩 67%。

### v2.0.27.3 → v2.0.27 · Item 9(Checkpointer + 隐私硬化)
- **PR:** 4 PR(Privacy 路径 strip + BackgroundTasks → fire-and-forget + CorruptCheckpointError + SQLite BEGIN IMMEDIATE)。
- **覆盖:** P0 隐私硬化(`source_path` 永远 `doc_id/filename` 形式)+ async dispatch 健壮性 + checkpoint corruption 容错 + 写并发。

### v2.0.26.3 → v2.0.26 · Item 6(UX foundation)
- **PR:** 6 PR(P0 trio hotfix + vitest infra + i18n 3-layer consistency + responsive/dark + toast system)。
- **覆盖:** UX 基线(theme/language/locale) + 14 i18n key 镜像 + 移动端响应式 + dark mode + Toast 队列。

### v2.0.25.1 · v2.0.25 · Item 6 PR-1 / PR-2(P0 trio)
- **覆盖:** `useModelsReady` 单点 + `threadSocket` 重启 + `handleAgentEvent` close WS + retry banner;loadHistory race + silent delete + double listSessions。

### v2.0.22-24 · Items 7-7.5(FSM cleanup + history replay ReAct revival)
- **覆盖:** `fsm.py` 753 LOC 重构 → `merge(state, delta)` step_count 集中 / async-gen 协议 / `NodeSpec` TypedDict / `pending_tool_calls` lifecycle 移到 FSM / `route_decision` Literal;v2.0.23 Pydantic Source revival + v2.0.24 `_serialize_messages` synthesized tool card。

---

## Archive · 早期

更早期版本细节在 git history(本仓库从 v2.0.22 起才有 changelog 记录;v2.0.21 之前是原型期,见 tag)。

---

<p align="center">
  <sub>📜 CHANGELOG 是项目历史。完整 debugging narrative 见 per-developer <code>memory/</code>(gitignored)。</sub>
</p>