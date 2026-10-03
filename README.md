# Yuan RAG · 源 RAG 智能问答

把资料丢给它,问什么都行。

PDF、Word、Excel、PPT、HTML、TXT,通吃。每条回答都自带出处,点一下回到原文。

不只是"搜资料"这么简单 —— 代理会自己判断问题该怎么查:BM25 关键词匹配、向量化语义检索、重排序精排,该用哪个就用哪个;检索结果不满意还能自动改写问题再试一次;本地资料实在不够,它会自己上网搜。支持 OpenAI 和 Anthropic 模型。多轮对话自动接上下文,多个会话互不干扰 —— 你可以一边让 A 线程跑长任务,一边在 B 线程问新问题,谁也不耽误谁。模型思考的过程会实时展开,**翻看历史对话时仍然可见**,而且现在会跟随你的提问语言输出(中文提问就中文思考)。Markdown 答案(标题 / 列表 / 表格 / 代码块 / 引用)正常渲染,引用编号在结构中仍然可点。数据完全留在本地,不出机器。

> **状态**:开发中。

---

## 更新日志

> **阅读约定:** 按版本号**从新到旧**排列。每条只写「**修改了什么 / 原理 / 有何提升**」三件事;迭代到后续版本才真正修好的早期尝试不再单列,合并到最终修复的那一版里。

---

### v2.0.31.1 — 2026-09-29 · 高精确模式首次使用可发现性:加 label + 动态 description,告别 hover-only 提示(post-hallucination 间隔期 UX polish)
- **修改了什么:** 1 个生产文件(jsx)+ 1 个 CSS 文件 + 2 i18n key + 1 Layer 5 verifier。`src/frontend/src/components/ChatPane.tsx` 把原 `<div className="high-precision-toggle segmented">` 包裹进新的 `<div className="high-precision-toggle-row">` 三元素栈:顶部 `<div className="high-precision-toggle-label">🔒 {t("chatInput.highPrecisionLabel")}</div>` 让用户**首次打开页面**就知道这是「高精确模式」toggle(无需 hover / 点击)+ 中间 segmented control(沿用 v2.0.31.0 `.segmented` + `.segmented-option`)+ 底部 `<div className="high-precision-toggle-description" aria-live="polite">` 显示**当前模式**的 hint 文案(`t("chatInput.mode{Auto,On,Off}Hint")`),用户点哪个按钮,描述就实时切换到哪个 hint;每个按钮的 `title` tooltip 仍保留(沿用),桌面 hover 用户仍可用,移动端 / 触屏 / screen-reader 用户**不再依赖 hover** 就能理解每个 mode 含义。`src/frontend/src/i18n/{zh,en}.ts` 加 `chatInput.highPrecisionLabel` 镜像 key(zh「高精确模式」/ en「Verbatim mode」)。`src/frontend/src/styles.css` 加 `.high-precision-toggle-row { flex: 0 0 100%; display: flex; flex-direction: column; gap: 6px; }`(沿用 v2.0.31.0 的「独占 textarea 上方一行」layout,只是把目标从 `.high-precision-toggle` 挪到 `.high-precision-toggle-row`,因为现在 row 包含 label + segmented + description 三个元素)+ `.high-precision-toggle-label { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 500; color: var(--text-dim); }`(用 v2.0.31.0 的 `.verbatim-lock-icon` 样式 + 复用 `--text-dim` token)+ `.high-precision-toggle-description { font-size: 11px; line-height: 1.5; color: var(--text-dim); min-height: 1.5em; max-width: 100%; word-wrap: break-word; }`(`min-height: 1.5em` 锁最长的 auto-mode hint,避免切换 mode 时 textarea 上下抖动)+ 移除原 `.high-precision-toggle { flex: 0 0 100% }`(职责挪给 row)。`tests/e2e/verify-v2_0_31_1-layer5.js` NEW 4 Layer 5 断言(24 子断言),含「label 可见 + 含 icon + 含文字 + 在 segmented 上方 + inline-flex」「description 可见 + 在 segmented 下方 + aria-live=polite + 11px + text-dim 颜色 + min-height 防 layout shift」「3 次 mode click 产生 3 个不同 description + 各自 >10 字 + re-click 同 mode 无变化」「无 hover / 无 click 时 SR / 移动用户看到 label + description + 3 button(role=radio + aria-checked + role=radiogroup + aria-label)」;`tests/e2e/verify-v2_0_31_0-layer5.js` 跟着改 1 处(`.high-precision-toggle` → `.high-precision-toggle-row` 测试 flex-basis 100%)。详细 file-by-file 见 `memory/v2.0.31.1.md`。
- **原理:** 收 root cause「first-time discoverability = 0」—— pre-v2.0.31.1 用户首次打开页面看到「自动 / 开启 / 关闭」3 按钮 + 顶部「模型加载中」spinner,**零视觉 affordance 解释 toggle 是干什么的**。每按钮的 `title` attribute 是 hover-only tooltip,触屏设备完全没用;`role=radiogroup` + `aria-label="Verbatim mode"` 只对 screen-reader 用户有意义。修复用「always-visible label + dynamic description」双管齐下:label 解决「toggle 是干嘛的」(self-documenting 入口);description 解决「当前 mode 是干嘛的」(实时反馈,切换时 aria-live=polite 朗读)。沿用 v2.0.31.0 的 `.verbatim-lock-icon` + `.segmented` + token-based CSS,**零新 design token**,不增加 design debt。
- **有何提升:** Layer 5 4/4 PASSED (24 子断言)+ v2.0.31.0 4/4 PASSED (更新后,row wrapper 替代直接 segmented 测 flex-basis)+ v2.0.30.0 5/5 PASSED (no regression)+ vitest 140 cum 0 回归 + tsc clean + npm run build clean (dist css 33.17 → 33.55 KB +0.38 KB 新增 .high-precision-toggle-row / -label / -description 规则)。2 个调试坑:**(1) React state 异步更新** —— Layer 5 第一版 Run C 「3 次 click 产生 3 个不同 description」失败,原因是 React `useState` setter 是 batched,click() 后立刻读 DOM 拿到的是上一 frame 的状态;改成 `page.waitForFunction` 等 `data-active` 切到目标 mode 后再读 description,4/4 PASSED。**(2) v2.0.31.0 verifier stale assertion** —— `.high-precision-toggle` 现在被 `.high-precision-toggle-row` 包裹,直接测它的 flex-basis 100% fail(实际是 auto,因为 row 才是 flex item);更新断言到 row wrapper。不变量:`.segmented` / `.segmented-option` CSS 不动(复用)/ `.highPrecision` state 不动(3-state auto/on/off)/ 7 i18n keys 不动 + 加 1 个 mirror key `highPrecisionLabel` / v2.0.30.0 long-poll 不动 / WS wire shape 不动 / v2.0.29.x 幻觉链路不动。

### v2.0.31.0 — 2026-09-29 · 高精确模式 UI 美化:segmented 与 🔒 提示器统一观感(post-hallucination 间隔期 UX polish)
- **修改了什么:** 2 个生产文件(jsx + css)+ 1 Layer 5 verifier。`src/frontend/src/components/ChatPane.tsx` 改 `<div className="high-precision-toggle">` → `<div className="high-precision-toggle segmented">`(复用 SettingsDialog theme/language 的 `.segmented` 设计语言)+ 改 `<button className="high-precision-toggle-btn${active ? " active" : ""}">` → `<button className="segmented-option high-precision-toggle-btn" data-active={active ? "true" : undefined}>`(从 `.active` class 改 `data-active` attribute,匹配 `.segmented-option[data-active="true"]` 的现成 active state 规则)+ bubble 顶部 `<div className="verbatim-lock-indicator">` 内 `🔒` raw emoji → `<span className="verbatim-lock-icon" aria-hidden="true">🔒</span>` 独立 span(便于单独控制 icon 尺寸 / baseline)+ 「开启」按钮前缀同 span 替换。`src/frontend/src/styles.css` 加 `.verbatim-lock-indicator`(inline-flex pill,`width: fit-content` + `border-radius: 999px` + `bg: var(--accent-soft)` + `color: var(--accent-strong)` + `font: 11px / 500` + label span `text-overflow: ellipsis` 防溢出)+ `.verbatim-lock-icon`(inline-block / 12px / `vertical-align: -1px` 让 emoji baseline 与 CJK / Latin 文字对齐)+ `.high-precision-toggle { flex: 0 0 100%; justify-content: flex-start; }`(segmented 独占一行,在 textarea 上方,实现原 JSX comment 的「above textarea」意图)+ `.high-precision-toggle-btn { font-size: 12px; padding: 5px 12px; display: inline-flex; align-items: center; gap: 4px; }` + `:focus-visible` 加 `var(--shadow-focus)` 焦点环 + `.chat-input { flex-wrap: wrap; }`(让 segmented 在窄屏自动 wrap 到 textarea 上方,避免 textarea 被 3 按钮挤瘪)。`tests/e2e/verify-v2_0_31_0-layer5.js` NEW 4 Layer 5 断言(segmented pill 渲染 / active state 用 data-active / bubble indicator pill / flex-wrap 布局)。详细 file-by-file 见 `memory/v2.0.31.0.md`。
- **原理:** 收 root cause「high-precision UI 视觉破功」—— Pre-v2.0.31.0 `grep "high-precision\|verbatim" styles.css` 返回 0 行,3 按钮渲染为默认灰色 chrome + default border + 无 pill 容器;bubble 🔒 是 raw emoji,与 PR-5 `.segmented` / PR-4 `.citation-chip` 的「blue + white + paper」设计语言脱节,用户在 chat input 区域看到一个「不属于这里」的元素。修复复用 `.segmented` 设计 token 是 minimum surface — 同一套 CSS 类被 3 处使用(theme / language / highPrecision),新增视觉一致性而不增加 design debt。原 JSX comment 已写「Placed ABOVE the textarea (not beside)」,pre-v2.0.31.0 CSS `display: flex` 把 segmented 放在 textarea 左边,comment 与实现矛盾,顺手一并修。
- **有何提升:** Layer 5 4/4 PASSED(16 子断言)+ v2.0.30.0 5/5 PASSED(no regression)+ vitest 140 cum 0 回归 + tsc -b --noEmit clean + npm run build clean(320 modules → 1.42s, dist css 32.35 → 33.17 KB +0.82 KB 新增 verbatim-lock-indicator + high-precision-toggle + verbatim-lock-icon 规则)。1 个调试坑:**CSS Display 3 §2.2 blockification** —— `display: inline-flex` 在 flex container 的 item 上会 blockify 成 `flex`,computed `.display` 是 `flex` 不是 `inline-flex`;test 断言第一版卡在这里,改成断言「is flex container 且 `flex-direction: row` + `align-items: center`」通过(spec-compliant 行为,不是 bug)。不变量:`.segmented` / `.segmented-option` CSS 不动(复用,不重定义)/ `.highPrecision` state 不动(3-state auto/on/off 不变)/ i18n 7 keys symmetric 不动(只是 icon 从 inline emoji 改成 `<span>` 包裹,文案照旧)/ v2.0.30.0 long-poll 不动 / Phase 8 verbatim extraction 不动 / `useModelsReady` 不动 / WS wire shape 不动。

### v2.0.30.0 — 2026-09-29 · 模型加载完成推送:长轮询 `/models/wait`(post-hallucination,spinner UX 收尾)
- **修改了什么:** 2 个生产(后端 + 前端 hook)+ 1 个前端 API client + 2 个新测试文件 + 1 Layer 5 verifier。`src/api/routes/models.py` NEW `GET /models/wait?timeout=N&poll_interval=N` long-poll endpoint(`bge_loaded()` / `bge_loading()` / `bge_load_error()` / `reranker_*` 全复用,async-sleep loop,error / missing state 立即返回,timeout 抛 408);`src/frontend/src/api/client.ts` NEW `waitForModelsReady(timeoutSeconds, AbortSignal?)`;`src/frontend/src/hooks/useModelsReady.ts` 重构:`AbortController` 包裹长轮询,`waitForReady()` → ready 立即 stamp `setReady` / 408 + abort + network error → fall back 2s 轮询(沿用 v2.0.28.20 `POLL_MS=2000` + `visibilitychange` listener),unmount 二次 abort;`tests/test_models_wait.py` NEW 7 pytest(立即 ready / poll-until-loaded / 408 timeout / load-error path / Query 参数 validation × 2 / missing-state 即返回);`src/frontend/src/hooks/__tests__/useModelsReady.wait.test.tsx` NEW 4 vitest(mount 调 wait / ready 路径 / 408 退到 polling / unmount abort);`tests/e2e/verify-v2_0_30_0-layer5.js` NEW 5 Layer 5 断言(长轮询 latency / 408 fallback / ready DOM / tab-hide abort / backend log scan)。详细 file-by-file 见 `memory/v2.0.30.0.md`。
- **原理:** 收 root cause「foreground spinner stale-up-to-2s」—— pre-v2.0.30.0 foreground tab 仍 up to 2000ms 延迟(`useModelsReady` 2s `setTimeout` 轮询,后端 `_loaded` flip 后还要等下一次 tick),background tab 由 v2.0.28.20 `visibilitychange` listener 修过。长轮询 vs 加速轮询 / SSE / WS-broadcast:长轮询是 1 request per "ready check" 复用 FastAPI async(无 infra change、无 connection manager、无 generator cleanup);SSE generator + cleanup 复杂度过剩;WS-broadcast surface 改动过大;200ms 轮询 4× backend load for 30s load window 无 signaling guarantee。`ModelsStatus` schema 不变 → `/wait` 与 `/status` 同 shape,debug / health-check 脚本无感。v2.0.28.20 visibilitychange listener 不动,扩展为「hide → abort long-poll,show → kick fresh long-poll」。
- **有何提升:** 7 NEW pytest + 4 NEW vitest + Layer 5 5/5 PASSED;**1629 + 5 skipped = 1634 cum pytest** + **140 cum vitest** 0 新增回归 + 8 pre-existing stale 不在本批;不变量:`ModelsStatus` schema 不变 / `/models/status` 行为不变(debug + health 路径 0 改动)/ v2.0.28.20 visibilitychange listener 保留并扩展 / v2.0.28.17 `insert(2, ...)` Anthropic consecutiveness 不动 / v2.0.29.x 幻觉链路 Phase 1-8 不动 / WS wire shape 不变 / `useModelsReady` external API(`App.tsx` destructures `{ready, modelsChecked, embedding, reranker, refresh}`)不变 / long-poll 是 mount-time one-shot,fallback polling 与现状同 cadence;实测 latency ~275ms(实测实测: pre-v2.0.30.0 ≤ 2000ms → post-v2.0.30.0 ≤ 500ms)。1 个调试坑(`page.route("**/models/wait")` glob 不带尾随 `**` 不匹配 query string — Playwright URL parser 把 `?timeout=60` 当 delimiter,改成 `**/models/wait**` 后 5/5 PASSED)见 `memory/v2.0.30.0.md`。

### v2.0.29.9 — 2026-09-28 · Phase 8 高精确场景禁推理模式(verbatim extraction)(幻觉优化项目)
- **修改了什么:** 11 个生产/前端 + 8 个新/扩展测试 + 2 个 Layer 5 verifier(hermetic + real-Chromium)。NEW `high_precision: Literal["auto","on","off"]` tristate + `Source.verbatim` additive + `react_generate_extractive` async-gen node + `EXTRACTIVE_SYSTEM` 严格 verbatim prompt(6 hard rules + 13 forbidden phrases + 「原文未提及」fallback)+ `EXTRACTIVE_FALLBACK` Counter + intent_analysis hybrid detect(Layer 1 regex + Layer 2 cheap-LLM)+ `_resolve_high_precision` helper + per-thread 3-state segmented `[自动/开启/关闭]` UI + 🔒 icon 渲染 + 7 个 symmetric i18n key。详细 file-by-file 见 `memory/v2.0.29.9.md`。
- **原理:** Phase 8 收 root cause「verbatim-faithfulness hallucination」—— 法律/医疗/财务场景 LLM 自主改写(把"2023年1月1日生效"改成"今年初")Phase 1/4/6 抓不到(LLM 引用 [n] → verdict=supported pass);Phase 8 设 invariant:tristate → FSM 路由 `react_generate_extractive` → EXTRACTIVE_SYSTEM 物理禁止 paraphrase → Source.verbatim=True 让前端 🔒 视觉确认。User "off" 无条件 wins(per user decision 2026-09-28)。
- **有何提升:** 51 NEW pytest + Layer 5 hermetic 5/5 + real-Chromium(Playwright + 真 dist/)7/7 PASSED;**1622 + 5 skipped = 1627 cum pytest 0 新增回归** + 8 pre-existing stale 不在本批;不变量:`retrieval_status` 4 值不变 / `Source.verbatim` 默认 False backward-compat / WS wire shape 不变 / Phase 6 verify_answer chain 仍 apply / `insert(2, ...)` Anthropic consecutiveness 不动 / i18n 7 keys symmetric zh↔en。7 个调试坑(write-then-load ThreadState 缺 highPrecision / TS1381 ChatPane stray `)}` / `getByText exact:true` fail on emoji-prefix label 等)见 `memory/v2.0.29.9.md`。

### v2.0.29.8 — 2026-09-28 · Phase 7 超长文档分片过滤(幻觉优化项目)
- **修改了什么:** 3 个生产 + 5 个测试文件 + 1 Layer 5 verifier。`config/constants.py` NEW `MAX_CHUNKS_PER_DOC=50` + env override;`src/agent/nodes/retrieve.py` 1-LOC `_hit_to_document` 传播 `chunk_index` 修复 + NEW `_filter_long_docs` helper(mirror `_filter_superseded` shape,head 5 + middle 5 evenly-spaced + tail 5 = 15)+ wire hybrid path + summary-intent branch + `CHUNK_REJECTED_OVERSIZE.inc(amount=N)` lazy import;`src/agent/fsm.py` wire `summary_path`;`tests/test_chunking_discipline.py` NEW 16 断言 + 4 已有 test 文件 extend。详细 file-by-file 见 `memory/v2.0.29.8.md`。
- **原理:** Phase 7 收 root cause「corpus-side overflow」—— Pre-Phase-7 长文档(200 页 PDF / 大 Excel)dump 100+ chunk 进 synthesis LLM context,Phase 6 cheap-model verifiers 在截断 context 上 fail-open(`retrieve.py:282-285` 已 pin),LLM 漏中段。Phase 7 sliding window invariant:任一 `doc_id` > 50 → head 5 + middle 5 evenly-spaced + tail 5(选 head+tail 而非 top-K 因长 doc 关键信息多在 boundary —— executive summary 顶部 / conclusions 底部)。`_hit_to_document` 漏传 `chunk_index` 是 latent bug(与 `_bulk_chunk_to_document:134` 不一致),1-LOC 修复让 window 有 ordering signal。
- **有何提升:** 21 NEW pytest + Layer 5 4/4 PASSED;**1567 + 5 skipped = 1572 cum pytest 0 新增回归** + 8 pre-existing stale 不在本批;不变量:`retrieval_status` Literal 4 值不变 / `_filter_long_docs` 在 `_filter_superseded` 之后跑 / `chunk_index is None` sort last / sliding window bounded at `MAX_CHUNKS_PER_DOC × N docs` / `inc(amount=N)` 单次多 drop 防 lock contention。3 个调试坑(middle-slice 边界算错 / `Edit` dangling line / `summary_path` monkeypatch module-level binding)见 `memory/v2.0.29.8.md`。

### v2.0.29.7 — 2026-09-28 · Phase 6 后校验机制(幻觉优化项目)
- **修改了什么:** 4 个新文件 + 5 处现有 + 39 NEW pytest + 1 Layer 5 verifier。NEW `src/agent/verification/` subpackage(`rule_engine.py` 4 字段类型 date/currency/percentage/article_number + extract_fields + check_consistency;`verifier.py` `VerificationResult` + `should_trigger_verification` zero-cost gate + `verify_answer()` async + JSON fence stripper + fail-closed)+ `verify_answer.py` FSM node(4 分支:skip / consistent / regen / fallback_refusal)+ `wire_protocol.py` `VerificationResultEvent` + `events.py` factory + `runner.py` emit branch;`fsm.py` `after_react_generate` 走 verify_answer + `NODES["verify_answer"]` 注册;`metrics.py` `VERIFICATION_REGENERATED` 加 `outcome` label。详细 file-by-file 见 `memory/v2.0.29.7.md`。
- **原理:** Phase 6 收 root cause「retrieval-attended-but-not-grounded」—— Pre-Phase-6 LLM 在「具体多少钱 / 第几条规定」上给 hedge 但仍给具体数字,引用对但数字是 LLM 推的(grounding verdict 抓不到因为有 cited source)。Phase 6 双层 fail-closed:(1) `rule_engine` regex 抓结构化事实不一致(date/currency/%/article)零 LLM 调用 ~ms 级;(2) `verifier.py` 复用 `build_cheap_model(temperature=0.0)` second-pass 抓语义级 hallucination。Zero-cost gate(`_TRIGGER_QUERY_RE` 命中 OR answer 含 extractable 字段)→ 否则跳过。重试 2 次,第三次 fallback Phase 1 `REFUSAL_TEMPLATES["low_relevance"]`。LLM JSON parse fail / ainvoke exception → fail-closed `consistent=False`(防 verifier 挂掉时悄悄放过)。
- **有何提升:** 39 NEW pytest + Layer 5 4/4 PASSED;**1546 + 5 skipped = 1551 cum pytest 0 新增回归** + 8 pre-existing stale 不在本批;不变量:`should_trigger_verification` zero-cost 大多数 turn 跳过 / `extract_fields` regex module-level compiled / `_metrics_mod.VERIFICATION_REGENERATED` lazy import 防 `importlib.reload` stale binding / fail-closed semantics。6 个调试坑(CN date regex overlap / Counter unknown label / lazy import monkeypatch site / operator precedence / `importlib.reload` stale / `EXPECTED_NODES` 漏更新)见 `memory/v2.0.29.7.md`。

### v2.0.29.6 — 2026-09-28 · Phase 5 并发 + 清理 + UX(幻觉优化项目)
- **修改了什么:** 5 个生产 + 4 个新测试 + 1 conftest + 1 Layer 5 verifier。`src/agent/tools/retrieve_docs.py` 删 `_override_top_k` context manager → keyword-only `top_k` 参数(防 thread-safety race);`src/retrieval/hybrid_search.py` `_hybrid_executor atexit.register(.shutdown)`(沿用 `src/llm/factory.py:274` precedent);`src/web_search/__init__.py` `_doc_id_for_url` 加 domain 段 + `_to_doc` populate `chunk_id` + dispatcher `asyncio.gather(return_exceptions=True)` + URL dedup + `(-score, engine_priority)` 排序 + `_engine` tag emit 前 strip;`pyproject.toml` + `conftest.py` 加 `@pytest.mark.slow` opt-in gate。详细 file-by-file 见 `memory/v2.0.29.6.md`。
- **原理:** Phase 5 收 root cause F(并发 / 状态泄漏)。`_override_top_k` 跨 coroutine 共享 module-level dict mutation → 两个并发 retrieve_docs 互相覆盖 top_k(A 用 3 / B 用 10 → A finally restore 把 B 推回 5);`_hybrid_executor` 无 shutdown hook → worker threads 在 uvicorn hot-reload / pytest teardown 泄漏(Windows 阻止 interpreter exit);`_doc_id_for_url` 旧 `web-{sha1[:12]}` 不同 site hash-collision;engine chain first-non-empty 抛弃高质结果(Bing mediocre / DDG excellent → 用户只见 Bing)。`@pytest.mark.slow` opt-in 让 5 个 behavioral regression guard 默认零成本,只在 pre-release 跑。
- **有何提升:** 26 NEW pytest + 5 NEW slow pytest(`--run-slow`)+ Layer 5 10/11 PASSED;**1503 + 5 skipped = 1508 cum pytest 0 新增回归** + 8 pre-existing stale 不在本批;不变量:keyword-only `top_k=None` 后向兼容 / `atexit.register` 幂等 / `gather(return_exceptions=True)` 捕获 per-engine failure + cancellation。6 个调试坑(`from X import Y` 让 Y 是 function / `@tool` wraps async / package re-export / Pydantic arg-schema / `wrap_documents` not `fence_documents` / URL dedup first-seen-wins)见 `memory/v2.0.29.6.md`。

---

### v2.0.29.5 — 2026-09-28 · Phase 4 PR-2 检索侧硬化:过期片段过滤 + 旧版本归档 + 脏数据清洗(幻觉优化项目)
- **修改了什么:** 4 个生产 + 3 个 schema/registry + 1 个 ingest pipeline + 3 个新测试 + 1 Layer 5 verifier。`src/storage/schema.py` 加 `expires_at` 字段 + migration;`src/retrieval/hybrid_search.py` NEW `_temporal_filter` + `_compose_where` + SQL injection guard via `'` doubling;`src/storage/doc_registry.py` `DocStatus` 加 `"superseded"` + `version` / `superseded_by` 字段 + `supersede()` idempotent helper + `list_for_thread` 默认过滤 superseded;`src/agent/nodes/retrieve.py` NEW `_filter_superseded` post-filter;`src/ingestion/cleaning.py` NEW `clean_chunks` 4 rules(empty / sub-noise-floor / SHA-256 exact-dup / UTF-8 round-trip)+ `pipeline.py` 8 个 return path wrap。详细 file-by-file 见 `memory/v2.0.29.4_p2.md`。
- **原理:** Phase 4 PR-2 把 user-added 检索侧优化(过期 / 旧版本 / 脏数据)落地。temporal_filter 让 `expires_at < now()` chunk 在 SQL 阶段剔除(retrieve 不浪费 context);superseded_by 让 retrieval 跳过 v1(避免 LLM 拿到 stale 数据 + 旧版挤掉新版 top-K);cleaning 让 OCR/PDF 乱码入库前过滤(避免 embedding 噪声)。Pre-registration 复用:Phase 2 `RETRIEVAL_FILTERED_LOW_SCORE` / `RETRIEVAL_FILTERED_EXPIRED` 已 pre-register,PR-2 wired-through,仅当 `expires_at` 列存在时 fire。
- **有何提升:** 19 NEW pytest + Layer 5 4/4 PASSED;**1458 + 19 = 1477 cum pytest 0 新增回归** + 8 pre-existing stale 不在本批;不变量:`expires_at` schema migration 幂等 / `include_superseded=False` default 向后兼容 / temporal SQL injection guard via `'` doubling / cleaning 4 rule 顺序确定。4 个调试坑(bootstrap dummy row 漏 `expires_at` / `_compose_where` parens SQL / legacy migration strip / Windows GBK UnicodeEncodeError)见 `memory/v2.0.29.4_p2.md`。

### v2.0.29.4 — 2026-09-28 · Phase 4 PR-1 引用 + 检索完整性 load-bearing(幻觉优化项目)
- **修改了什么:** `src/retrieval/rerank.py` rerank 失败 stamp `rerank_failed=True` + `rerank_score=0.0`;`src/agent/citations.py` range collapse 加 WARNING log;`src/agent/nodes/react_generate.py` dedupe key 加 `page_content[:200]` 防 stale upload 静默覆盖;`src/frontend/src/components/CitationChip.tsx` + `ChatPane.tsx` chip click detail 加 `chunk_id` / `doc_id` / `source_kind`(backward-compat);`src/agent/nodes/retrieve.py` `_SUMMARY_INTENT_RE` 锚定开头 + `min_top_relevance_score=0.3` gate(`rerank_failed` carve-out)+ `_hit_to_document` propagate;`tests/test_retrieval_filters.py` NEW 14 断言 + 4 已有 test 文件更新 + 1 Layer 5 verifier。
- **原理:** Phase 4 PR-1 修 root cause C(检索门松)+ D(引用完整性)两组。min-score gate 拒 low-relevance → Phase 1 `REFUSAL_TEMPLATES["low_relevance"]` 自动 fire(死代码激活);rerank 失败不冒充低分(避免 OOM 当 low_relevance 误拒绝);chip click 携带 `chunk_id/doc_id` 让 App.tsx 能定位 chip 跳错目标 source;range collapse WARNING log 让 live/reload 不一致可见;dedupe key 含 `page_content[:200]` 防 stale upload;summary-intent 锚定开头排除 definitional QA(`X 是什么` 不走全 chunk 投喂)。
- **有何提升:** 14 NEW pytest + Layer 5 5/5 PASSED;**1457 + 14 - 5 fail = 1466 cum pytest + 8 pre-existing stale 不在本批**;不变量:`retrieval_status` Literal 不加新值(只新增 `"low_relevance"` path)/ `_SUMMARY_INTENT_RE` 锚定 Chinese-only + start-anchored / CitationChip event detail backward-compat / `rerank_failed` carve-out 让 gate 区分「rerank 失败」vs「rerank 0 分」。5 个调试坑(Pydantic `.get()` 不存在 / loguru `%s` lazy 占位符 / `importlib.reload` stale / Counter CamelCase vs JSON lowercase / `ChatPane.tsx` 侧栏 chip 漏改)见 `memory/v2.0.29.4_p1.md`。

### v2.0.29.3 — 2026-09-28 · Phase 3 defensive override 通用化 + loop-breaker post-script(幻觉优化项目)

- **修改了什么:** 新建 `src/agent/nodes/_defensive_override_registry.py`(3 个 registry `DIRECTIVE_BUILDERS` / `FINGERPRINT_EXTRACTORS` / `FALLBACK_BUILDERS` + `apply_defensive_overrides()` helper + `_aligned_tools()` 交集查询);`src/agent/nodes/react_generate.py` 删 hardcoded `_build_get_current_time_directive` / `_extract_time_marker_from_history` / `_answer_has_marker` 三个 helper,改成 iterate `DIRECTIVE_BUILDERS` registry + 单行 `answer_text = apply_defensive_overrides(...)` 调用 + 新增 `_LOOP_BREAK_SYNTHESIS_HINT` 在 `state["loop_breaker_active"]` 为 True 时 `msgs.insert(2, ...)`;`src/agent/nodes/react_agent.py` 在 loop breaker fire 时 `loop_breaker_delta = {"loop_breaker_active": True/False}` 写入 `__delta__`(FSM `merge()` 自动持久化任意 state key);`tests/test_v2_0_28_16_bugfixes.py` 把 `_extract_time_marker_from_history` 测试改用 registry 的 `_extract_time_marker`(纯 string → string)+ `TestAnswerHasMarker` 改测 `apply_defensive_overrides` 集成(行为等价);新增 `tests/test_defensive_override_registry.py` 25 断言;`tests/e2e/verify-v2_0_29_3-layer5.py` NEW Layer 5 verifier。
- **原理:** Pre-Phase-3 defensive override 只能 cover `get_current_time` 一个 tool,加新 tool 要重写 30+ 行(3 个 helper + 1 个 inline 30 行 override block + 1 处 metric bump)且必须 mirror 现有 helper 的形态 / 时序 → 极易漂移;3 个 registry 让「加新 tool」变 1 行 / registry(共 3 行),`apply_defensive_overrides` 是 single entry point,intersection 强制 3 registry 对齐(不全是 silent skip)。Loop breaker fire 时合成 LLM 没信号知道 research 被切 → 倾向于续查;新 hint 在 `insert(2, ...)` 告诉 LLM「资料不足时按 REFUSAL_TEMPLATES 拒绝」,refusal > loop-break > time 顺序让 Anthropic recency bias 把最强 contract 放最右。
- **有何提升:** 25 NEW pytest PASSED(`tests/test_defensive_override_registry.py`) + `tests/test_v2_0_28_16_bugfixes.py` 旧 test 改测新 API 后仍全绿;Layer 5 3/3 PASSED(真 Chromium + 真 WS + 真 LLM + 真 dist/);`apply_defensive_overrides` 11/3 次 fire 在 multi-turn time probe,`SYNTHESIS_TM_IGNORED{tool=get_current_time}` 正确从 baseline 0/4/8 → after 1/5/9;1433+25 = 1458 pytest + 8 pre-existing v2.0.6/7/11/28.18 stale 不在本批;关键不变量:`insert(2, ...)` Anthropic consecutiveness(v2.0.28.17)保留、`apply_defensive_overrides` direct / empty / marker-present 三重守卫、`SYNTHESIS_TM_IGNORED` per-tool label 复用 Phase 2 dashboard、loop_breaker_active state key 不污染 wire shape(server-internal only)。**原则:通用化优于修特定 bug**——加 tool 是 1 行不是 30 行,defensive override 不能只 cover 已知 case,要可扩展。
### v2.0.29.2 — 2026-09-27 · Phase 2 自建 metrics + loguru 观测 + `/debug/metrics` 端点(幻觉优化项目)

- **修改了什么:** 新建 `src/agent/metrics.py`(Counter 原语 + 9 个 pre-registered counter + `snapshot()` + `_emit_metrics_line()` loguru JSON-line emitter,env-flag gated);`src/api/routes/debug_metrics.py` 新建 `GET /debug/metrics`(env-flag gated 404);`src/app.py` mount 新 router;`src/agent/nodes/hallucination.py` 5 个 verdict 路径 + `src/agent/nodes/retrieve.py` 2 个 empty 路径 + `src/agent/nodes/react_generate.py` 3 个 prompt-template + 1 个 tm-ignored + `src/agent/fsm.py` 1 个 direct-template 都加 `inc()` hook;`src/agent/runner.py` `answer_complete` 调 `_emit_metrics_line()` + 显式 hydration boundary 注释(durable vs per-turn);`retrieve.py` 删 3 处 dead `retrieval_count` / `iteration_count` writes;新增 `tests/test_metrics_aggregation.py` 18 断言;`Counter.snapshot()` label-less counter 在 0 也露面(让 dashboard 启动就看到完整 taxonomy)。
- **原理:** Phase 1, 3, 4, 6, 7, 8 修了任何东西没 metric 就没法验证是否回归 → 不能用 Prometheus / OpenTelemetry(用户 IN/OUT 明确不要)→ 用 loguru 自带 JSON-line + in-process Counter。Pre-registered contract 让 Phase 4/6/7 的 dashboard 在 call site 落地前就能 query;label-less 在 0 也露面让「计数器没工作」和「计数器工作但没触发」可区分;`_EMIT_ENABLED` env flag 让 production overhead = 0,debug 时 opt-in。
- **有何提升:** 18 NEW pytest PASSED + Layer 5 3/3 PASSED(真 Chromium + 真 WS + 真 LLM,baseline→final 每次 direct_template +1 / skipped +1);1415+18 = 1433 pytest + 8 pre-existing v2.0.6/7/11/28.18 stale 不在本批;关键不变量:env-flag 双 gate(emit 0 cost / endpoint 0 leak)、labeled counter 不预 fill(避免 dashboard 误报全 0)、pre-registration 让 forward-compat 计数器不破坏未来 Phase。**原则:observability 不引入新基础设施,只复用 loguru + endpoint**——Python 自带 Counter 类 + loguru JSON-line 比 Prometheus + Grafana + node_exporter 三件套少 0 个部署。
### v2.0.29.1 — 2026-09-27 · Phase 1 拒绝契约(幻觉优化项目)

- **修改了什么:** 6 个文件:`src/llm/prompts.py` 加 3 个结构化拒绝模板 + 禁词列表 + `REFUSAL_TEMPLATES` 字典;`src/agent/state.py` 加 `retrieval_status` Literal 字段(empty/empty_bulk/low_relevance/success);`src/agent/nodes/retrieve.py` + `src/agent/fsm.py` 在 4 个 code path 设置 status;`src/agent/nodes/react_generate.py` 在 `msgs.insert(2, ...)` 注入 refusal SystemMessage(v2.0.28.17 consecutiveness 规则保留);`src/agent/nodes/hallucination.py` snippet 从 `[:500]` 改可配 `JUDGE_SNIPPET_CHARS_PER_DOC=2000`;新增 `tests/test_refusal_contract.py` 19 个行为断言。
- **原理:** LLM 想拒绝时没模板可锚定 → 「应该是 / 可能是 / I think」这种 hedge 表达会落到零上下文但自信的输出。Phase 1 把「say so explicitly」换成结构化契约:Acknowledge 段 + 「what is missing」段 + 「next steps」段 + 禁词列表;`retrieval_status` 跨 retrieve / summary_path 4 个 path 收集信号,react_generate 据此在 msgs[2] 插对应模板;snippet 2000 字让 cheap-model judge 看到 doc 1 后半段 + docs 2-3 才不会被 fabrication 蒙混过关。
- **有何提升:** 19 NEW pytest PASSED,1416 cum pytest 0 新增回归(8 pre-existing v2.0.6 / 7 / 11 / 28.18 不在本批);关键不变量:`insert(2, ...)` Anthropic consecutiveness、pre-Phase-1 checkpoint `retrieval_status` 缺 key 自动 fallback 到 success、direct (greeting / simple_fact) path 完全不动。**原则:拒绝是契约不是礼貌**——禁词列表必须有,且必须在 prompt 中靠近指令(Anthropic prompt 结构偏好近距锚定)。
### v2.0.29.0 — 2026-09-27 · Day 0 OOD 对抗性 fixture 集(幻觉优化项目前置)

- **修改了什么:** 新建 `tests/fixtures/audit_v2/` 7 个对抗 fixture + `tests/test_audit_fixture_behavior.py` 11 个行为断言;每个 fixture 设计成「触发特定失败模式」。
- **原理:** 32 findings 收敛为 6 个根因组后,Phase 1-8 各自针对一组根因修,但没有 fixture 集就没有办法「修完一个,验证它真的修了」。fixture 提前建好,每 Phase ship 时直接拿对应 fixture 验证。
- **有何提升:** 后续 9 个 Phase 的 verifier 全部复用这套 fixture,「修对」vs「修了一个相邻回归」在测试里可观测。Day 0 本体 11/11 PASSED,1389 pytest 0 新增回归。
### v2.0.28.20 — 2026-09-26 · 后台 tab 模型加载完前台 spinner 还卡着

- **修改了什么:** 切到后台 tab 等几分钟回来,前端 spinner 仍卡着不动,刷新才能恢复。
- **原理:** Chrome/Edge 把隐藏 tab 的 `setTimeout` 节流到 ~60s,`useModelsReady` 轮询被冻住。
- **有何提升:** 加 `visibilitychange` 监听,前台恢复立刻 re-poll;4 NEW vitest + 3 NEW Layer 5 PASSED,132/132 零回归。
### v2.0.28.19 — 2026-09-26 · `npm run dev` 永远卡 loading(后端 ready=true)

- **修改了什么:** dev 模式启动后永远卡在「正在加载模型中」,只有 hard-refresh 才能恢复。
- **原理:** vite proxy 把 `/`、`/src/...`、`/@...`、`/node_modules/...` 全转发到 FastAPI,bundle 404,前端跑不起来。
- **有何提升:** vite.config.ts 加 bypass callback 白名单 4 类前缀,3 NEW Layer 5 + 129/129 cum PASS + HMR 恢复。
### v2.0.28.18 — 2026-09-26 · Document-namespace corruption 让 thread 在 sidebar 隐身

- **修改了什么:** 1 条 thread 在 sidebar 看不到,history 不可达,后端 ERROR log 持续刷。
- **原理:** `_serialize_state` 写 `Document` 时用 legacy ID,`_deserialize_state` 的 `allowed_objects="messages"` 不认 Document → ValueError。
- **有何提升:** 1 LOC `allowed_objects="messages"` → `"core"` + write-time `documents` 预 convert + legacy-namespace walker;12 NEW pytest + Layer 5 3/3 + 126/126 cum PASS + 后端 log delta=0。
### v2.0.28.17 — 2026-09-23~25 · 多轮 tool-using 合成 LLM 同根因 9 个 bug 合并 fix

- **修改了什么:** `_sanitize_history_for_generate` 删 trailing AIMessage (v2.0.28.11) + intent greeting guard + react_agent retry+synthetic-AIMessage + thinking-only fallback (v2.0.28.12) + placeholder reorder after fallback (v2.0.28.12.1) + reload 时 `_find_following_tool_result` 填 completed (v2.0.28.13) + `mergeAdjacentAssistantTurns` 折叠相邻 turn (v2.0.28.14) + 3 处 reasoning leak drop + chat-utils no-fallback (v2.0.28.15) + post-script SystemMessage + defensive override (v2.0.28.16) + `msgs.insert(2, ...)` 防 Anthropic non-consecutive raise (v2.0.28.17) + checkpointer `loads(..., allowed_objects="messages")` (v2.0.28.11 polish)。
- **原理:** 同一根因 = 多轮 tool-using 时 synthesis LLM 看不到 tool 数据 / 把 planning-step AIMessage 当 prior reply / 不遵循 directive,9 种症状(0 reasoning / "no answer text" / greeting fallback / DIRECT_SYSTEM denial verbatim / 工具卡 ⏳ infinite / 2 个 bubble / 元评论 leak / silent LLM fail / LangChain pending-deprecation)在不同 path 暴露,v2.0.28.9 hydration fix un-mask 全部后续 latent bugs。
- **有何提升:** **117+ NEW tests 全部 green**,Layer 5 累计 132/132 PASSED,user 多轮对话 reload 干净、合成不再 leak 元评论、Anthropic 不再 silent raise;**关键原则**:**当 LLM ignore directive 不 add more directives,加 post-hoc override REPLACE LLM output**(v2.0.28.16 load-bearing);**defense-in-depth without verification 是 silent failure mode**(v2.0.28.17 加 `is_apology` 断言防 defensive override future-mask);**frontend-only fix 不可改 DB/wire**(v2.0.28.14 因 Anthropic tool_use ↔ tool_result 配对强制 2 AIMessage/turn)。
---

### v2.0.28.10 — 2026-09-23 · Live 助手数 ≠ reload 助手数

- **修改了什么:** 第二次提问后 live UI 显示 2 个助手 bubble,DB 里却是 3 条 / 4 条记录。
- **原理:** FSM 每轮产 2 条 AIMessage(planning + synthesis),`merge()` 用 `state` 快照读 → 用 stale 值 → 多写一条。
- **有何提升:** `replace_last_message` key 从 `out`(in-flight accumulator)读,15 NEW pytest + 16 NEW Layer 5 PASSED;**原则:merge() 读 out 不 state 快照**。
### v2.0.28.9 — 2026-09-22 · 多轮 reload 只显示最后一轮

- **修改了什么:** reload 多轮 thread 只看到最后一对 Q&A,前面全消失。
- **原理:** runner 启动时没加载 prior 轮 messages,新 turn 直接 append,历史不可见。
- **有何提升:** `runner.stream_agent()` 之前 `_initial_state()` call 加 hydration block 尝试 `await load_latest(thread_id)`;6 NEW pytest + Layer 5 13/13 + 23/23 零回归;**原则:lazy import 防 cycle + step_count 重置 per turn + graceful try/except**。
---

### v2.0.28.8 — 2026-09-22 · 默认主题 = light(was system)

- **修改了什么:** OS 暗色模式下用户首次打开被强制进 dark 主题,要手动去 Settings 切。
- **原理:** `ThemeProvider` 默认值 `"system"` 在 OS dark 下走 dark,违反 user 期望。
- **有何提升:** 1 LOC 默认改 `"light"`;2 NEW vitest + 2 旧测试 opt-in + Layer 5 23/23 + 11/11 baseline;**原则:code default ≠ migration,OS dark + light default 让 user 必须开 Settings 是 UX 反模式**。
---

### v2.0.28.7 — 2026-09-22 · 桌面端 sidebar 和 chat-pane 上下堆叠

- **修改了什么:** 桌面端 sidebar 和 chat-pane 上下堆叠(应该左右并排)。
- **原理:** `.sidebar-backdrop` base rule 写在 `@media (max-width:768px)` 内,桌面端不生效 → 变 grid child 偷 row 1 col 1。
- **有何提升:** base rule HOIST 到顶层;21/21 viewports + PR-4 11/11 PASSED;**原则:`grid-template-rows: 1fr` 在 item overflow 时 ≠ 1 row,base layout rule 永远不进 breakpoint @media**。
---

### v2.0.28.5 — 2026-09-21 · Item 11 PR-3(retrieval+web 并发)

- **修改了什么:** `embed_query` lock-across-inference 去重同 key N 次推理,`_TTLCache` 加 `threading.Lock`,Bing 429 short-circuit chain,`asyncio.gather` 加 `return_exceptions=True` 防 MemoryError cancel siblings。
- **原理:** DCL 错(N 同 key 同时 miss → 全跑推理);CPython 3.13+ free-threaded 下 GIL 不再 serialize bytecode;Bing rate-limit 会让 chain 浪费 16s timeout。
- **有何提升:** 3 sites DEBUG→WARNING;Layer 5 59.3s 16/16;13/16 pytest green;**原则:lock-across-inference > DCL;monkeypatch USE site 不 definition site**。
---

### v2.0.28.3 — 2026-09-21 · Item 11 PR-1(trace_id LLM 闭环)

- **修改了什么:** `redact_exception` body 改 `trace_id = get_trace_id() or new_trace_id()`,LLM 异常 log 永远带 12-hex trace_id 跟 TraceIdMiddleware + checkpointer 一致。
- **原理:** pre-PR-1 用 `uuid.uuid4().hex[:8]` 产 8-hex,operator grep 跟 12-hex 不匹配,跨文件 trace 断链。
- **有何提升:** 3 NEW pytest;Layer 5 1.2m 16/16;计划 1.5d → 0.5d 完成(audit P0-2 stale finding);**原则:audit findings 进 PR 前 verify by reading code 5 分钟省 1 天**。
### v2.0.28.2 — 2026-09-21 · Item 10 PR-3(Settings TOCTOU)

- **修改了什么:** `get_settings` 加 30s TTL(`SETTINGS_CACHE_TTL_SEC=30.0` + `_settings_cache_ts` monotonic),`force_reload: bool=False` semantic A = refresh + stick(不 peek-only)。
- **原理:** TOCTOU race + operator 改 .env 后 cache 不刷新,看不出新值。
- **有何提升:** 6 NEW pytest;Layer 5 41.6s 16/16;1205 pytest;**原则:`time.monotonic()` 必须(DST/NTP-safe);`config/__init__.py:2` shadow 把 `import config.settings as foo` 拿成 `_SettingsProxy` instance,monkeypatch 必须在 module 上**。

---

### v2.0.28.1 — 2026-09-21 · Item 10 PR-2(trace_id middleware)

- **修改了什么:** NEW `src/core/trace_id.py` ContextVar module + pure ASGI `TraceIdMiddleware`(非 `BaseHTTPMiddleware`,后者 WS 会 bypass)+ loguru `configure(patcher=_inject_trace_id)` + format `{extra[trace_id]}` + middleware 加在 CORS 之后。
- **原理:** 跨 HTTP + WS scope 一致带 trace_id,operator 一行 grep 追 bug。
- **有何提升:** 11 NEW pytest + 2 Layer 5;Layer 5 37.7s 16/16;**原则:`BaseHTTPMiddleware` 不覆盖 WS → 必须 pure ASGI `__call__(scope, receive, send)`;Starlette middleware order = reverse**。
---

### v2.0.28 — 2026-09-21 · Item 10 PR-1(list_documents 缓存)

- **修改了什么:** `retrieve.py:143` 字面 1 行 flip `list_documents` → `list_documents_cached`(5s TTL wrapper),`GET /documents` 故意不动(用户期望 delete 后立即 invisible)。
- **原理:** summary-intent hot path 之前每 turn 全 LanceDB scan 200-500ms。
- **有何提升:** 2 NEW pytest + 1 Layer 5 assertion;Layer 5 50.1s;**原则:monkeypatch USE site 不 definition site;raising=False for new-attr spy**。
---

### v2.0.27.3 — 2026-09-21 · Item 9 PR-4(SQLite 并发写)

- **修改了什么:** `_open_db` 加 `isolation_level=None` + `PRAGMA busy_timeout=5000` + `_db_write_lock=threading.Lock()`,`_save_sync` / `_delete_thread_sync` 包 `BEGIN IMMEDIATE` + try/except ROLLBACK/COMMIT。
- **原理:** Python sqlite3 default 隐式 BEGIN 同 connection 多 coroutine 冲突;跨 connection writer 需要 DB-level RESERVED lock 串行化。
- **有何提升:** Layer 5 44.4s;**原则:`isolation_level=None` 必须(否则 `cannot start a transaction within a transaction`);threading.Lock 必须(BEGIN IMMEDIATE 不保护同 connection 多 coroutine)**。
---

### v2.0.27.2 — 2026-09-21 · Item 9 PR-3(CorruptCheckpointError 契约)

- **修改了什么:** NEW `CorruptCheckpointError` 带 thread_id/original_error/trace_id,`_deserialize_state` narrow except → raise,`load_latest` catch → return None,`list_threads` signature `→ tuple[list, int]`,`list_sessions` 只在 `corrupt_count>0` 时设 `X-Corrupt-Thread-Count` header。
- **原理:** silent swallow 让 corrupt thread 让整个 `/sessions` 500;operator 没 anchor 找不到根因。
- **有何提升:** Layer 5 44.9s;**原则:monkeypatch USE site(patch `src.storage.checkpointer.history_db_path`,不 `src.core.paths`)**。
---

### v2.0.27.1 — 2026-09-20 · Item 9 PR-2(BackgroundTasks 换 asyncio)

- **修改了什么:** upload endpoint 从 FastAPI `BackgroundTasks` 迁到 `asyncio.create_task(asyncio.to_thread(...))` + tracked set + done callbacks,lifespan teardown 加 `cancel_ingest_tasks()`。
- **原理:** FastAPI `BackgroundTasks` 在 shutdown 时不 cancel,50 MB PDF mid-ingest 跟 doc_registry flusher race。
- **有何提升:** 镜像 `models.py` 已 ship 的 canonical pattern;Layer 5 56.7s;**原则:任何 `asyncio.create_task` 必须配 `cancel_*_tasks()`**。
---

### v2.0.27 — 2026-09-20 · Item 9 PR-1(隐私硬化)

- **修改了什么:** strip absolute filesystem path from `_registry.json` + LanceDB chunk row + `/documents` wire;write-time `_serialize_source_path` + read-time `_coerce_legacy_source_path` + `_migrate_absolute_paths` 一次性 rewrite。
- **原理:** 路径泄露让 wire payload 反推 user 设备目录结构。
- **有何提升:** Layer 5 1.1m;**原则:stale uvicorn PID 持 pre-PR code 内存 → 必须 taskkill + `python -m uvicorn src.app:app`**。
---

### v2.0.26.3 — 2026-09-20 · Item 6 PR-6(Toast 系统)

- **修改了什么:** 新建 `Toast.tsx`(ToastProvider + useToast + module singleton + useSyncExternalStore + stack cap MAX_VISIBLE=3 + 4 kind + sticky),wire 4 高 + 2 低优先级 site,删 PR-2 inline sidebar-error-banner。
- **原理:** inline banner 复用冲突 + 集中通知中心更可观测。
- **有何提升:** 10 NEW vitest;Layer 5 50.7s;**原则:Hook return shape 必须是 object destructuring 跟 useTheme/useLocale 一致**。
---

### v2.0.26.2 — 2026-09-20 · Item 6 PR-5(响应式 + 移动端 + 暗色模式)

- **修改了什么:** `< 768px` sidebar off-canvas drawer + 汉堡 + backdrop + 44×44 tap target;3-state theme toggle light/dark/system 含 `prefers-color-scheme`;5 token + dark palette + `:root[data-theme]`。
- **原理:** 移动端不能用桌面 grid 布局;OS auto-detect 满足「跟 OS 走」用户习惯。
- **有何提升:** 6 NEW mobile.spec.ts iPhone 13 viewport;**原则:CSS pipeline @media flatten bug(lightningcss inline @media into main cascade);SettingsDialog mount 时 `onModelsReady` 立刻 fire → useRef 捕获 prev, only false→true transition fire**。
---

### v2.0.26.1 — 2026-09-19 · Item 6 PR-4(i18n 3-layer 一致性)

- **修改了什么:** backend catalog-driven i18n.py + `parse_accept_language` RFC 7231 + websocket/documents/sessions/chat refactor;frontend 31 i18n key + status.ts/errors.ts function form + CitationChip shared component。
- **原理:** backend/frontend/wire 三层 locale 状态漂移导致用户看到混合语言。
- **有何提升:** 84 NEW tests;Layer 5 50.5s;**原则:`'zh' cannot be used as value because imported using 'import type'` → 改 import type → value import**。
---

### v2.0.26 — 2026-09-19 · Item 6 PR-3(vitest + handleAgentEvent 分裂)

- **修改了什么:** vitest@1.6.0 + @testing-library/react@16 + jsdom@24 infra;`handleAgentEvent` 438-LOC → dispatch.ts switch 12 类型 + 12 per-event 模块,App.tsx 1344→1049 LOC。
- **原理:** 单文件大 switch 不可测 + 难维护;vitest 给后续 PR 回归保护层。
- **有何提升:** 49 vitest;Layer 5 40.6s。
---

### v2.0.25.1 — 2026-09-19 · Item 6 PR-2(扩展 silent delete + race)

- **修改了什么:** `loadHistory` AbortController + monotonic seq 双 guard,`getSessionMessages` 加 AbortSignal,`deleteError` state + UI banner。
- **原理:** loadHistory race + silent delete 错误让 user 看到「删除成功」其实没删。
- **有何提升:** Layer 5 39.3s。
---

### v2.0.25 — 2026-09-19 · Item 6 PR-1(P0 三件套)

- **修改了什么:** `useModelsReady` 单点持有(3 → 1),`threadSocket` 重启用 `onReconnected`/`onReconnectAttempt`/`onReconnectExhausted`,`handleAgentEvent` error 分支显式 close WS,banner 加重试按钮。
- **原理:** 3 处轮询互相 race + WS 断线 banner 不恢复。
- **有何提升:** Layer 5 53.6s。
---

### v2.0.24 — 2026-09-19 · history-replay 缺中间 ReAct rounds

- **修改了什么:** `_serialize_messages` 删 drop intermediate ReAct AIMessage 的 guard,抽 thinking block 进 reasoning + 抽 native `AIMessage.tool_calls` 进 cards。
- **原理:** `_serialize_messages` 之前 drop intermediate AIMessage → reload 只看到最后一轮。
- **有何提升:** 21 NEW pytest;1146 pytest。
---

### v2.0.23 — 2026-09-19 · checkpointer Pydantic Source revival 失败

- **修改了什么:** `/sessions` 40/40 thread 全部显示空 messages 的 NotImplementedError,3 层 defense:写入侧 `model_dump()` 预转换 + 读取侧 `_revive_not_implemented` walker + `ast.literal_eval` 安全解析 Pydantic repr。
- **原理:** Source Pydantic 没注册 LangChain serializer → `loads()` raise NotImplementedError。
- **有何提升:** 15 NEW pytest;1125 pytest。
---

### v2.0.22 (PR-I) — 2026-09-17 · Item 7 Step 13: final cleanup + SHIPPED

- **修改了什么:** 10/10 grep hard-gate 全过 + pytest 1110 全绿 + Layer 5 PASSED smoke 8.4s + web_search 25.7s,backend-agent 13/13 findings 关闭 P0 3/3 + P1 6/6 + P2 4/4。
- **原理:** Item 7 = step_count 中央化 / FSM cleanup,Step 13 收尾做 grep 硬 gate 确保 no residual hard-code。
- **有何提升:** Item 7 SHIPPED;LangGraph 0 imports。
---

### v2.0.22 (PR-H) — 2026-09-17 · Item 7 Step 12: 删死字段 + 集中 helper

- **修改了什么:** 删 `tool_call_log` 死字段(0 写入 site)+ 加 `_docs_for_hallucination` helper 集中「什么算 docs」定义。
- **原理:** 死字段误导下次重构;predicate/node 各自定义 docs 标准会漂。
- **有何提升:** P1-B5 + P1-B6 关闭。
---

### v2.0.22 — 2026-09-17 · Item 7 Step 5: step_count 中央化

- **修改了什么:** `merge()` 集中 +1,10 处手动返回全删。
- **原理:** 10 处分散维护 silent budget failure 根因(预算漏算 / 多算)。
- **有何提升:** silent budget failure 根因消除。
---

### v2.0.22 (PR-B) — 2026-09-17 · Item 7 Step 6: 节点统一到 async-gen

- **修改了什么:** `_accepts_step` sniff 删,4 sync 节点改 `yield __delta__`,主循环直化。
- **原理:** sync/async 混用让主循环需要 sniff 区分,type confusion 风险。
- **有何提升:** 主循环简化。
---

### v2.0.22 (PR-C) — 2026-09-17 · Item 7 Step 7: NODES 改 NodeSpec

- **修改了什么:** 4 处 `if current == ...` 硬编码删除,wire event 节点自治。
- **原理:** 硬编码字符串比较 = P0-B1 typo 静默走错路径。
- **有何提升:** P0-B1 彻底关闭。
---

### v2.0.22 (PR-D) — 2026-09-17 · Item 7 Step 8: TOOL_METADATA 化

- **修改了什么:** react_agent 内 `tc_name == "web_search"` 硬编码删除,加新搜索类工具不再改 node。
- **原理:** P0-B3 工具名硬编码 = 加新工具必须改 node。
- **有何提升:** P0-B3 关闭。
---

### v2.0.22 (PR-G) — 2026-09-17 · Item 7 Step 11: 错误恢复中央化

- **修改了什么:** byte-identical LLM-failure apology text 集中 + 8 个 `logger.exception` site 统一 `{node_name} failed: {exc}` 格式 + react_agent silent bare-except 改成 loud 统一日志。
- **原理:** P2-B4 8 个 site 各自维护 apology 文本容易漂。
- **有何提升:** P2-B4 关闭。
---

### v2.0.22 (PR-F) — 2026-09-17 · Item 7 Step 10: route_decision 改 Literal

- **修改了什么:** 3 个模块独立声明 `str | None` / `Optional[str]` 收紧成单一 closed vocabulary。
- **原理:** P2-B2 typo 静默走 `direct` short-circuit,改 loud AssertionError。
- **有何提升:** P2-B2 关闭。
---

### v2.0.22 (PR-E) — 2026-09-17 · Item 7 Step 9: pending_tool_calls 移到 FSM

- **修改了什么:** runner 从工具 lifecycle 簿记彻底解耦,变纯事件 forwarder。
- **原理:** P2-B3 runner 兼做工具生命周期 = 关注点混乱。
- **有何提升:** P2-B3 关闭。
---

### v2.0.21 — 2026-09-17 · LangGraph 完全移除

- **修改了什么:** graph.py + history.py 删,FSM 自含,grep 0 imports。
- **原理:** LangGraph 是 Phase 3 之前的临时方案,FSM ship 后应该清理。
- **有何提升:** 依赖表面清干净。
### v2.0.20 — 2026-09-17 · Phase 3 Steps 4-6: 自有 checkpointer

- **修改了什么:** `add_messages` reducer 删 + `InjectedState` 替 `InjectedToolArg` + 自有 SQLite checkpointer(从 LangGraph checkpoints 表迁移 19 thread)。
- **原理:** LangGraph runtime 替换后,配套 reducer + 注入模式要换成自含版本。
- **有何提升:** 数据迁移成功,FSM 完整接管。
### v2.0.19 — 2026-09-17 · Phase 3 Step 3: LangGraph 换 async FSM

- **修改了什么:** runner + fsm.py live,graph.py 暂留待 Phase 3 Step 7 删。
- **原理:** LangGraph runtime 是过度抽象,FSM + 自有 checkpointer 更可控。
- **有何提升:** 控制流可视化,便于后续 step_count centralization。
---

### v2.0.18 — 2026-09-16 · Web search 链路简化

- **修改了什么:** 7 文件 → 1 包 + 1 seam + wire-format 锁定。
- **原理:** Phase 3 ready:web search 链路是 FSM 切换前的最大变量。
- **有何提升:** 接口边界清晰。
---

### v2.0.17 — 2026-09-16 · WS wire-event schema 验证

- **修改了什么:** Pydantic discriminated union + 发送边界 schema guard。
- **原理:** WS wire-event 多 producer 漂移会让 frontend 类型判断错。
- **有何提升:** wire 类型稳定。
---

### v2.0.16 — 2026-09-16 · ReAct 重构彻底完成

- **修改了什么:** legacy_helpers 收纳 + dead nodes 删 + 死测试清理。
- **原理:** ReAct 重构迭代 5 版,旧 helper 残留。
- **有何提升:** ReAct 路径全清。
---

### v2.0.15 — 2026-09-16 · 测试套件边界审计 + 前端 WS 单源化

- **修改了什么:** 测试边界审计 + 前端 WS wire 单源化 + dead wire type 清理。
- **原理:** 测试套件随重构漂移,边界不清;前端 WS 多处重复订阅。
- **有何提升:** 测试可靠 + 前端 WS 数据源唯一。
---

### v2.0.14 — 2026-09-15 · 项目大扫除

- **修改了什么:** 跨模块 dead code / dead config / dead test 清理。
- **原理:** Phase 3 之前的累积技术债。
- **有何提升:** 项目体积下降,认知负担减。
---

### v2.0.13 — 2026-09-15 · answer_complete whitespace 覆写流式内容

- **修改了什么:** `answer_complete` 收到 `"  "` 或 `"\n"` 也 canonical 覆写累积 bubble 内容。
- **原理:** 部分 LLM 在 answer_complete 返 whitespace-only,前端 display 空。
- **有何提升:** 流式末尾视觉一致。
---

### v2.0.12 — 2026-09-15 · Tool-using 回答开头不显示

- **修改了什么:** Tool-using query 流式期首字符被吞,F5 后才看到完整回答。
- **原理:** streaming deferred-prefix 事件次序问题,前端等累积完才挂载。
- **有何提升:** Tool-using 回答首字符即时显示。
---

### v2.0.11 — 2026-09-14 · GraphRecursionError + loop-breaker 缺 total 上限

- **修改了什么:** max_steps=10 + 假信号识别 + loop-breaker 加 total 上限。
- **原理:** ReAct 死循环让 GraphRecursionError 一直 raise,UX 挂死。
- **有何提升:** loop-breaker 早停 + 优雅 fallback。
---

### v2.0.10.4 — 2026-09-13 · 上传后前端仍显示「向量化中(N 块)」

- **修改了什么:** 上传成功后 sidebar 仍 stuck 在「向量化中(N 块)」,F5 才更新。
- **原理:** ingest 完成的 WS event 没正确 flip doc 状态,前端 polling 拿到 stale。
- **有何提升:** 经 v2.0.10 / .3 三轮迭代最终修复;**原则:ingest 完成事件 + sidebar invalidate + 缓存 invalidation 三处必须一起改**。
---

### v2.0.10.2 — 2026-09-13 · Wire-format 前缀 `]<]minimax>[` 泄漏

- **修改了什么:** WS event payload 开头偶现 `]<]minimax>[` 字符串污染。
- **原理:** Anthropic streaming token 边界与 wire 拼装顺序不一致。
- **有何提升:** 前缀清理,wire 干净。
---

### v2.0.10.1 — 2026-09-12 · Anthropic tool_use ↔ tool_result 配对 400

- **修改了什么:** 多轮 tool-using 时 Anthropic 返 400 invalid_request_error。
- **原理:** `ToolMessage` 在 sanitized history 中 parent AIMessage 不是 tool_calls 那条 → 配对错。
- **有何提升:** tool_use ↔ tool_result 配对保护,Anthropic 不再 400。
---

### v2.0.9 — 2026-09-11 · LLM 感知不到时间工具

- **修改了什么:** LLM 不调 `get_current_time`,回答靠训练数据时间。
- **原理:** 两阶段 4 真根因(sanitize 路径 / tool 描述 / hint wording / agent prompt)。
- **有何提升:** 4 真根因一次全修,LLM 看到 tool 后会主动调。
---

### v2.0.8 — 2026-09-11 · P1 i18n 长尾 + tokenizer 预热

- **修改了什么:** 局外专家评估 P1 项:剩余 i18n 字符串统一 + HybridChunker tokenizer 启动期预热。
- **原理:** i18n 长尾让 CJK 用户看到 EN fallback;tokenizer 首次 upload 5-15s 卡顿。
- **有何提升:** i18n 完备 + 首上传即时。
---

### v2.0.7 — 2026-09-11 · P0 真源 + 性能 + i18n 核心

- **修改了什么:** P0 真源治理(Source of Truth = `source_kinds` 单一来源)+ 性能基础(lancedb cache + chunk 缓存)+ i18n 核心 key(中英双轨)。
- **原理:** 局外专家评估结论:Source of Truth 漂移是首要 P0。
- **有何提升:** 后续 PR (.6 / .26.1) 持续收敛。
---

### v2.0.6 — 2026-09-11 · P2 前端打磨

- **修改了什么:** 前端视觉一致性 / 间距 / 字号 / icon 等 P2 细节。
- **原理:** 局外专家评估 P2 项。
- **有何提升:** 整体观感优化。
---

### v2.0.5 — 2026-09-10 · P1 UX 一致性

- **修改了什么:** UX 一致性相关 P1 项(loading 态 / 错误反馈 / 状态文案)。
- **原理:** 局外专家评估 P1 项。
- **有何提升:** UX 反馈更一致。
---

### v2.0.4 — 2026-09-10 · P0 事实安全

- **修改了什么:** LLM hallucination 时引用编号 / 时间戳 / 文档 ID 不能被编造。
- **原理:** 局外专家评估 P0 项:不能让 LLM 在 stream 期间编造引用。
- **有何提升:** 引用 ID 必须在 `[allowed_set]` 内,LLM 不能越界编造。
---

### v2.0.3 — 2026-09-07 · 前端渲染 + 蓝气泡时间戳 + 联网更严重

- **修改了什么:** 三件套:前端 message render fix + 蓝气泡显示时间戳 + 联网搜索结果更严重错误(LLM 编造引用)。
- **原理:** 时间戳无,user 看不到消息时间;LLM 联网结果 hallucinate 引用。
- **有何提升:** 蓝气泡带时间 + 引用编号必须存在才能 render。
---

### v2.0.2 — 2026-09-06 · 联网搜索工具触发 Anthropic 400/2013 + 1026

- **修改了什么:** 联网工具返 `list[Document]` 让 Anthropic 400 + tool error 2013 + 1026。
- **原理:** Anthropic tool result 不接受 Document 对象,必须 serialize 成 string。
- **有何提升:** 联网工具正常返 string,Anthropic 不再 400。
---

### v2.0.1 — 2026-09-06 · summary intent 走错 prompt + 死循环

- **修改了什么:** summary intent 走 direct prompt(没 web_search)+ ReAct web_search 死循环。
- **原理:** intent 分类错 + 没 max_step 保护。
- **有何提升:** summary 走对 prompt + ReAct 加上限。
---

### v2.0 — 2026-09-06 · Agentic RAG 重构(ReAct 整体重写)

- **修改了什么:** 从 RAG pipeline 改 Agentic RAG,ReAct 整体重写(intent / react_agent / react_generate FSM)。
- **原理:** 单轮检索 + 总结的 RAG 模式在多轮复杂 query 上效果差。
- **有何提升:** Agentic 多步推理 + tool 调度能力上线。
---

### v1.1.16 — 2026-09-06 · WS close code 门控 + dedup

- **修改了什么:** WS 偶发多余 close event 让前端 reconnect 风暴。
- **原理:** close code 不分类 → 临时错误触发重连,无限循环。
- **有何提升:** close code 门控 + dedup,reconnect 收口。
---

### v1.1.15 — 2026-09-06 · banner 按 source_kinds 真渲染

- **修改了什么:** 联网 banner 仍按旧 schema 显示(覆盖 v1.1.14 半成品)。
- **原理:** v1.1.14 没把 banner 跟 `source_kinds` 真正挂钩。
- **有何提升:** banner 跟 source_kinds 真挂钩,v1.1.14 半成品覆盖完成。
---

### v1.1.13 — 2026-09-06 · 上传文档后失去联网能力

- **修改了什么:** 上传任何文档后,联网搜索功能被禁用。
- **原理:** 联网 / 本地检索被设计成互斥而不是互补。
- **有何提升:** 联网 + 本地改为互补,可同时使用。
---

### v1.1.12b — 2026-09-05 · 摄入管线三层伪装根因

- **修改了什么:** v1.1.12 表层修复没根治,摄入管线还有三层伪装问题。
- **原理:** chunk / 编码 / metadata 三层各自有 stale path。
- **有何提升:** 三层同时修,摄入干净(v1.1.12 表层修复作废)。
---

### v1.1.11 — 2026-09-05 · 联网查询新闻回答质量差

- **修改了什么:** 联网结果经常错(LLM 编正文,搜索 snippet 当答案,域名偏好错)。
- **原理:** prompt 鼓励 LLM 编,搜索结果只取 snippet 不访问正文,域名偏好缺失。
- **有何提升:** 诚实 prompt + 真访问正文 + 域名偏好 + 防御层 strip 四件套,回答质量跟上。
---

### v1.1.10 — 2026-09-04 · 中国网络下 ddgs backend 全被掐

- **修改了什么:** ddgs 默认 backend 在中国网络下全被掐,联网搜不到东西。
- **原理:** ddgs 后端依赖的几个 endpoint 在中国网络受限。
- **有何提升:** 改 Bing 直连 + engine chain dispatcher 多引擎 fallback。
---

### v1.1.9 — 2026-09-03 · 历史对话的回答消失

- **修改了什么:** reload 历史对话看不到之前 assistant 的回答。
- **原理:** v1.1.7 持久化层回归漏洞 + 历史回放层 silent-drop。
- **有何提升:** 持久化补漏 + 回放层不再 silent-drop,assistant 回答可见。
---

### v1.1.8 — 2026-09-03 · events.web_search bool 类型塞 payload

- **修改了什么:** 潜伏多年的 typo:`events.web_search` 把 bool 直接塞 payload,序列化偶发失败。
- **原理:** schema 期待 dict 但收到 bool。
- **有何提升:** typo 修,wire 干净。
---

### v1.1.7 — 2026-09-03 · 思考只第一轮显示 + 工具不触发

- **修改了什么:** 思考过程仅首轮显示,后续轮空;有文档时 web_search 不触发。
- **原理:** reasoning event 在第 2 轮起不 emit;tool 选择时优先本地文档,跳过 web。
- **有何提升:** 两 bug 一并修,多轮 reasoning 全程可见 + 工具正确触发。
---

### v1.1.6 — 2026-09-02 · live 双倍 / history 单倍

- **修改了什么:** live streaming 显示两份内容,reload 后只剩一份。
- **原理:** runner 双层 dedup 漏一层。
- **有何提升:** runner 双层 dedup 修正,live 和 reload 一致。
---

### v1.1.5 — 2026-09-02 · 回答正文念两遍

- **修改了什么:** 每条回答正文被念两遍(连续重复一段)。
- **原理:** 持久化层真根因:synthesis 阶段把 synthesis output 又存了一份。
- **有何提升:** 持久化层根因修,回答正文正常念一次。
---

### v1.1.4 — 2026-09-01 · fetchOrThrow 自递归栈溢出

- **修改了什么:** `fetchOrThrow` 某些 error path 触发自递归栈溢出。
- **原理:** error handler 自身 fetch 又触发 error → 无限递归。
- **有何提升:** 递归改成 iterative,补强测试可观测性。
---

### v1.1.3 — 2026-09-01 · 「你是谁」两次回复

- **修改了什么:** 问「你是谁」收到两次回复,内容相同。
- **原理:** 联网 banner 误标(显示「联网中」)+ 引用 chip 泄露到 greeting bubble。
- **有何提升:** 两 bug 各自修,banner + chip 各自归位,回复只一次。
---

### v1.1.2 — 2026-09-01 · 上传 CJK 文件名回归

- **修改了什么:** 上传中文文件名失败,变乱码或报 not found。
- **原理:** `safe_filename` 走 ASCII 化丢失 CJK 字符。
- **有何提升:** `safe_filename` Unicode 化,中文文件名正常。
---

### v1.1.1 — 2026-09-01 · BGE-M3 启动期 [WinError 10060] 超时

- **修改了什么:** 启动期拉 HF 权重 Windows 下报 `[WinError 10060]` 连接超时。
- **原理:** 直连 HF Hub 在国内网络超时。
- **有何提升:** 加 HF 镜像 endpoint 兜底,启动稳。
---

### v1.1 — 2026-08-31 · 6 条 quick win

- **修改了什么:** 6 条 quick win 小修补(UI 文本 + minor bug + error toast + etc)。
- **原理:** 阶段性累积小修补集中发版。
- **有何提升:** 综合 UX 改善。
---

### v1.0 — 2026-08-31 · 流式期不冻屏 + 加固 + 全栈去耦

- **修改了什么:** 流式期间 UI 不再冻屏 + WS / 提示词两条加固线 + 全栈去耦(模块边界清晰)。
- **原理:** 单线程 sync fetch 让 UI 冻屏;模块耦合让修改一处牵动多处。
- **有何提升:** 流式响应顺畅 + 架构边界清晰,v1 第一个正式版。
---

### v0.9 — 2026-08-31 · 引用链路 / 上传安全 / /sessions N+1

- **修改了什么:** 三类 P0 一并修:引用 chip 链接不正确 + 上传文件类型校验缺 + `/sessions` 查询 N+1 性能。
- **原理:** 引用 chip 路径缺前缀;上传仅看 extension 不看 content_type;`/sessions` 每条 thread 一次子查询。
- **有何提升:** 引用可点 + 上传更安全 + `/sessions` 列表加载快。
---

### v0.8 — 2026-08-30 · 全栈提速与去重

- **修改了什么:** 19 个阶段全栈性能 + 去重优化(检索 cache + 重复查询合并 + 流式缓冲)。
- **原理:** 初版 0.7 性能基线不够,用户感知延迟高。
- **有何提升:** 整体 P50 延迟降一档。
---

### v0.7 — 2026-08-29 · 蓝白清新 + 上传可视化 + 文件生命周期

- **修改了什么:** UI 改蓝白清新风格 + 上传过程可视化(进度条 + chunk 计数)+ 文件生命周期(pending / indexed / failed)。
- **原理:** 用户反馈初版风格 + 上传看不到进度。
- **有何提升:** 视觉清爽 + 上传可观测 + 文件状态清晰。
---

### v0.6.2 — 2026-08-29 · 输入框自适应换行

- **修改了什么:** 输入框单行 hard-cap,长 query 看不到完整。
- **原理:** textarea 默认 height 固定,不自动 grow。
- **有何提升:** 输入框自适应换行,长 query 完整可见。
---

### v0.6.1 — 2026-08-29 · 色调焕新 + 「渊」→「源」

- **修改了什么:** 色调焕新,项目名从「渊 RAG」改为「源 RAG」。
- **原理:** 「渊」字 user 不熟,「源」更直白。
- **有何提升:** 项目名易记 + 色调现代。
---

### v0.6 — 2026-08-29 · UI/UX 全套重做

- **修改了什么:** 整套 UI/UX 重做(layout + sidebar + chat pane + 引用展示)。
- **原理:** 初版 0.5 视觉布局不稳定。
- **有何提升:** 整体观感立得住。
---

### v0.5 — 2026-08-30 · 性能扫一遍

- **修改了什么:** 性能基础扫一遍(检索 + 摄入 + 前端首屏)。
- **原理:** 初版 0.4 性能基线建立。
- **有何提升:** 性能可量化,后续 v0.8 深度优化基础。
---

### v0.4 — 2026-08-30 · 思考持久化 + 多线程基础设施

- **修改了什么:** 思考过程持久化(进 DB)+ 多线程基础设施(threading.Lock + 线程安全 cache)。
- **原理:** 用户希望 reload 看到思考过程;单线程后续卡并发。
- **有何提升:** 思考可见 + 并发基础设施就位。
---

### v0.3 — 2026-08-29 · 冷启动 UX 收尾

- **修改了什么:** 冷启动 loading + 模型下载进度 + warmup 提示。
- **原理:** 首次启动 BGE-M3 下载 10-30s,用户看不到进度会以为卡死。
- **有何提升:** 冷启动期间用户可观测,不焦虑。
---

### v0.2 — 2026-08-28 · 后台预热 + 内联加载提示

- **修改了什么:** 模型后台预热(用户首次 query 时已 ready)+ 内联 loading 提示(不阻挡输入)。
- **原理:** 首次 query 等模型加载 30s,用户等不住。
- **有何提升:** 首次 query 即时响应。
---

### v0.1 — 2026-08 · 初版

- **修改了什么:** 项目初始版本,核心 RAG pipeline 上线(检索 + 总结 + 引用 + 多轮)。
- **原理:** MVP 验证 RAG 体验可行性。
- **有何提升:** 项目从 0 → 1。
---

## 快速开始

> **没有 git?** 在 GitHub 页面点绿色的 **Code → Download ZIP**,解压,然后从第 2 步开始。

### Windows
```bat
git clone https://github.com/<your-username>/YuanRAG.git
cd YuanRAG
start.bat
```

### macOS / Linux
```bash
git clone https://github.com/<your-username>/YuanRAG.git
cd YuanRAG
chmod +x start.sh
./start.sh
```

启动器会:
1. 在 `.venv` 中创建 Python 虚拟环境(仅首次)
2. 安装依赖(首次约 5–10 分钟,Docling 拉模型最慢)
3. 构建 React 前端(仅首次)
4. 启动服务并在默认浏览器中打开 `http://127.0.0.1:8765`
5. 后台异步下载 BGE-M3 / BGE Reranker 模型(约 900 MB);浏览器一打开就能聊天,模型在后台预热无需等待

> **首次启动完成后**,检查 `.env` 是否需要填你的 API key —— 没有密钥聊天就只是 placeholder。继续往下看"首次配置"。

### 首次配置(在浏览器里)
1. 浏览器一打开就是干净的聊天界面 —— 模型在后台预热,无需等待。
2. 没配 API key 的话,设置对话框会主动弹出让你填;之后访问不再弹。填入 **OpenAI** 或 **Anthropic** 的 API key。密钥存在系统钥匙串里(Windows 凭据管理器 / macOS 钥匙串 / Linux Secret Service),不写到配置文件。
3. 选模型和供应商。
4. 上传文档,开始问问题。

### 想把数据放在项目里?

默认情况下用户数据落在 `~/.rag_assistant/`(系统用户目录)。如果你想所有文件都跟着项目走 —— 比如方便打包、备份,或者跑多份互不冲突 —— 在项目根目录的 `.env` 里加一行(`.env.example` 是模板,已有 `RAG_DATA_DIR=./data` 的注释示例):

```
RAG_DATA_DIR=./data
```

相对路径会被解析到项目根(`src/main.py` 所在目录)。这个目录已经被 `.gitignore` 忽略,放心用。

### 模型拉不下来 / 启动卡 `BGE-M3 load failed`?

启动期 `BGEM3FlagModel` 会先 `hf_api().list_repo_tree(...)` 探一下 `BAAI/bge-m3` 仓库元数据(用于缓存完整性校验 / 模板发现)。在 `huggingface.co` 不可达的网络环境下(典型:CN 出口封禁、企业网关屏蔽、IPv6-only 节点)这一步会卡在 `httpcore.ConnectTimeout: [WinError 10060]` 并把整个 warmup 拖死。

**.env 已经默认指向国内镜像** —— [`HF_ENDPOINT=https://hf-mirror.com`](.env),由 [`src/core/bootstrap.py`](src/core/bootstrap.py) 在 `.env` 没显式设置时自动填上。`hf-mirror.com` 代理所有 HuggingFace 仓库的 metadata + blob 请求,本地缓存(`<RAG_DATA_DIR>/hf_cache/`)命中后不再走远端,两边各管各的。

- 想强制走官方端点?在 `.env` 里覆盖成 `HF_ENDPOINT=`(空串)。
- 想换其他镜像(自建代理 / 公司内网)?改成 `HF_ENDPOINT=https://your-mirror.example.com`。
- 启动日志第一行就能看到生效的端点:搜 `BGE-M3 load failed` 或 `Loading BGE-M3 model`,如果还报 timeout,先 `curl -I $HF_ENDPOINT/api/models/BAAI/bge-m3` 验证端点本身可达。

### 环境要求

- **Python 3.11+**(3.13 已验证)
- **Node.js 18+**(前端构建用)
- **磁盘** ≈ 4 GB(`.venv` + 依赖 + 模型缓存)
- **首次运行**:`pip install` 拉取大约 900 MB 的 PyTorch + Docling 权重,慢机器可能 10 分钟以上

---

## 这些行为是有意的 —— 看不出来容易被当 bug

列在前面先说,免得第一次用就懵。

### 多线程并行,互不干扰

每个对话线程各自占一条 WebSocket(首次发消息时才打开,代理发完 `done` 后关闭)。在侧边栏切来切去,其他线程的连接不会被掐断 —— 它们继续在后台流式返回。你可以在线程 A 问个长问题,切到线程 B 接着问,两个答案同时跑。状态由 [`App.tsx`](src/frontend/src/App.tsx) 里的 `messagesByThread` / `streamingByThread` / `wsByThreadRef` 维护,ChatPane 只是个纯展示组件。

### 切走不丢答案

老版本里,WebSocket 中途断开(典型场景:你正等回答,顺手点了"+ 新对话"),`aclosing(stream_agent(...))` 会向代理抛 `GeneratorExit`,LangGraph 在写出最终 checkpoint 之前就被取消 —— 你切回去看到的是一份陈旧快照,答案就这么"丢了"。

现在 [`src/api/websocket.py`](src/api/websocket.py) 对每个 `stream_agent` 生成器持有强引用,一收到 `WebSocketDisconnect` 就把生成器交给 `_drain_generator` 在后台跑完。代理完整跑完,checkpointer 提交最终状态,答案在服务端稳稳地留着 —— 客户端在不在都无所谓。

### 冷启动 ≠ 页面刷新

[`App.tsx`](src/frontend/src/App.tsx) 用 `PerformanceNavigationTiming.type` 区分两种"打开":

| 操作 | nav type | 行为 |
|---|---|---|
| 输入 URL / 新标签页 / 重启浏览器 | `navigate` | 全新 UUID —— 每次冷启动从空白对话开始 |
| F5 / Ctrl+R / `location.reload()` | `reload` | 从 `localStorage["rag.active_thread"]` 恢复最近一个"真实"线程 |
| 浏览器前进 / 后退 | `back_forward` | 同 `reload` |

线程在第一次有意义的动作(上传、发消息、点侧边栏项)时被提交到 localStorage。"+ 新对话"和"删除当前会话"会主动清掉这个 commit,所以这两种操作之后再刷新,会从一个全新的 UUID 开始。

### 侧边栏什么都能看到

`GET /sessions` 把 LangGraph checkpoints 和 [`list_threads_with_documents()`](src/storage/lancedb_store.py) 合在一起,所以"只上传了文档、还没发消息"的线程也会出现,标题用第一个文档的文件名兜底。完整回退顺序:最后一条用户消息 → 第一个文件名 → `"新对话"`。

### 模型的思考过程可见,而且翻历史也能看到

Anthropic 的 extended-thinking 块会随答案一起流式返回,在助手气泡上方以可折叠抽屉呈现(默认展开)。聊天窗口顶部的 💭 按钮可以全局隐藏 / 显示;偏好存在 `localStorage["rag.show_thinking"]`。

**关键点:思考过程也会被持久化。** v0.4 起,`generate.py` 把推理文本写到 AIMessage 的 `additional_kwargs["reasoning"]`,跟 `sources` 走同一套路,`/sessions/{thread_id}/messages` 回放时一并取出 —— 重新打开三个月前的对话,💭 抽屉按原样展开。OpenAI gpt-4o 不发思考块就显示空抽屉,按钮仍然可见但表现成"无内容"。

### 空对话不留痕

"+ 新对话"和"删除当前会话"会主动清掉 localStorage 里的 commit。如果之后既没发消息也没上传,就不会再触发 `rag:session-updated` 去重新提交,下一次打开页面也就没有"复活"的依据,自然从一个全新 UUID 开始。

### 冷启动:模型后台预热,inline spinner,不弹框

[`src/app.py`](src/app.py) 的 FastAPI `lifespan` hook 在启动时 `asyncio.create_task(asyncio.to_thread(...))` 后台跑 BGE-M3 + BGE Reranker 的 RAM 加载,HTTP 服务立即就绪,前端可立即打开。前端 [`App.tsx`](src/frontend/src/App.tsx) 里**两条 effect 协作**:

- 第一条 mount 时单次 `getModelsStatus()` —— lifespan 已经跑完就直接 `setModelsReady(true)`,什么都不显示。
- 第二条**在 `modelsReady === false` 期间持续轮询 `/models/status`**(每 2 s),lifespan 一完成预热就触发清除。

聊天窗口头部的 `Loading models…` spinner(`<span class="chat-header-loading">`)是用户唯一看到的"还没好"信号。**不需要也不应该刷新页面** —— 刷新只会重新走一遍 nav type 判定、把 thread 改回 UUID、丢失刚写的草稿。

---

## 架构

- **后端**:FastAPI + Uvicorn(Python 3.11+)
- **前端**:React + Vite + TypeScript,构建后由 FastAPI 静态托管
- **代理**:LangGraph 状态机,六个节点 —— route → retrieve → grade → rewrite → generate → hallucination-check
- **检索**:BGE-M3(同时出 dense + sparse,经 FastEmbed)→ LanceDB(向量 + Tantivy BM25)→ RRF 融合(k=60)→ BGE Reranker v2-M3 精排
- **解析**:Docling 处理 PDF/DOCX/PPTX/HTML,openpyxl 处理 XLSX
- **历史**:LangGraph `SqliteSaver`,按 thread_id 分隔
- **前端状态**:`App.tsx` 里按 thread 维护 WebSocket 和消息缓冲,多线程同时流式不打架

```
浏览器(每线程一条 WS)
   │
   ├──► FastAPI /ws/chat ──► LangGraph 代理
   │                              │
   │                              ├──► LanceDB(BM25 + 向量)
   │                              ├──► BGE-M3(embeddings)
   │                              └──► BGE Reranker
   │
   └──► SqliteSaver(对话历史)
        + 后台 drain:WS 断了代理也会跑完
```

---

## 目录结构

```
config/             Pydantic Settings、常量
src/
  main.py           入口(uvicorn + 拉起浏览器)
  app.py            FastAPI 工厂
  core/             路径、日志、lifespan
  llm/              LLM 客户端(OpenAI、Anthropic)、prompts、schemas
  embeddings/       BGE-M3 封装
  reranker/         BGE Reranker v2-M3 封装
  ingestion/        格式检测 + 解析器 + chunker
  storage/          LanceDB + SqliteSaver
  retrieval/        混合检索(RRF)+ rerank
  agent/            LangGraph 状态机 + 节点 + runner
  api/              FastAPI 路由 + WebSocket + schemas
  frontend/         React + Vite 源码(构建到 dist/)
  security/         钥匙串封装
  utils/            工具函数
scripts/            模型下载、前端构建、dev runner
tests/              pytest 测试套件 + fixtures
```

用户数据落在 `~/.rag_assistant/` 下:
```
~/.rag_assistant/
├── config/settings.json
├── data/lancedb/         (向量 + FTS 索引)
├── data/history.db        (对话历史)
├── models/                (BGE-M3、BGE Reranker 缓存)
├── uploads/               (原始上传文件)
└── logs/app.log
```

---

## 开发

```bash
# 后端(带热重载)
python -m src.main
# 或者改代码自动重启
uvicorn src.app:app --reload --host 127.0.0.1 --port 8765

# 前端(带 HMR)
cd src/frontend
npm install
npm run dev    # → http://localhost:5173(自动把 /api 和 /ws 代理到 :8765)

# 测试
pytest

# Lint
ruff check .
```

---

## 配置

| 位置 | 内容 |
|---|---|
| `.env`(或 `RAG_*` 环境变量) | 启动期覆盖 —— host、port、log level |
| 浏览器 → 设置 | 供应商、模型、API key(存系统钥匙串) |
| `~/.rag_assistant/config/settings.json` | 持久化的用户偏好 |

---

## 技术栈(2026 年 1 月调研)

| 层 | 选择 | 理由 / 性能特征 |
|---|---|---|
| LLM | OpenAI / Anthropic(云 API) | 快、本地零占用;路由 / 改写 / 评估 / 幻觉判定走 `gpt-4o-mini` / `claude-3-5-haiku`,只有 `generate_answer` 走满血模型 —— 廉价模型节点 5–10× 便宜 |
| 代理 | LangGraph | 成熟的状态图,适合自决策 RAG;`max_retrieval_rounds=2` / `max_rewrites=1` / `max_iterations=6` / `top_k_pre_rerank=20` / `top_k_post_rerank=5` 给代理设了硬上限,杜绝死循环 |
| 向量库 | LanceDB | 真嵌入式,自带 Tantivy BM25;`where(thread_id='...')` 同时过滤 dense 与 BM25 两侧,免去 per-thread 索引维护 |
| Embedding | BGE-M3(int8 + fp16,FastEmbed) | 一次前向同时出 dense (1024-d) + sparse (BM25-like weights) —— 比"分开跑 embedder + BM25 indexer"少一半延迟 |
| Reranker | BGE Reranker v2-M3(int8 ONNX) | 只对 RRF 融合后的 top-20 重排到 top-5,reranker 是唯一的二阶段编码器;失败时回退 RRF 排序(fail-open) |
| 解析器 | Docling | 多格式结构保留最好;upload 硬封顶 100 MB 防止 Docling 解析 / LanceDB 写入失控 |
| 前端 | React + Vite | 现代、快 HMR;`wsByThreadRef` 用 `useRef` 而非 state,React 18 StrictMode 双 mount 不会把握手中途的 socket 给扔掉 |
| 后端 | FastAPI + WebSocket | 异步流式;每线程独立 socket 懒打开(首次发消息)、`done` 后关闭 —— 浏览型线程不占 idle 连接;WS 日志降级到 DEBUG,不再刷屏 |
| 存储 | SQLite(历史)+ LanceDB(向量) | 单文件,可移植;`list_sessions` 硬封顶 500 条保护 `saver.alist(None)` 的 OOM 风险;`list_chunks_by_thread` 封顶 100 块防止上下文爆窗 |
| 单例管理 | 手工 `Lock` + `_loaded` 标志 | 取代 `functools.lru_cache` —— 冷缓存并发调用不再各自跑一遍 10–30 s 的权重加载;FlagEmbedding 推理另起一把 `_infer_lock`,序列化并发(正确性 > 吞吐) |
| 生命周期 | FastAPI `lifespan` + `asyncio.to_thread` 后台预热 | 首次打开跳过 10–30 s 的 RAM 加载,HTTP / WebSocket / 轮询全程不阻塞;`WebSocketDisconnect` 后台 drain 不浪费已跑的代理步 |
| 联网搜索 | DDGS + 退避(`max_concurrency=2`, `min_interval_seconds=1.5`) | 单 IP 限速 ~30 req/min,信号量 + 全局时间窗让我们留在阈值之下 |
| 总结意图快路径 | 正则命中"总结 / 摘要 / 概述 / summarize"等 → 跳过 embedding,直接灌文档(封顶 100 块) | 绕开"meta 问题不进 embedding 空间"导致的零检索假阴性,节省约 200 ms 的向量化 + LanceDB 查询 |

---

## 许可证

待定