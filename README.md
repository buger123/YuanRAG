# Yuan RAG · 本地优先 RAG 智能问答

> **把资料丢给它,问什么都行 —— 回答自带出处,事实有保障。**

PDF / Word / Excel / PPT / HTML / TXT 全吃。每条回答都自带引用编号,点一下回到原文。  
不只是「搜资料」这么简单 —— 代理会自己判断问题该怎么查:BM25 关键词匹配、向量化语义检索、重排序精排,该用哪个就用哪个;检索结果不满意还能自动改写问题再试一次;本地资料实在不够,它会自己上网搜。

**最大的不同:反幻觉 9-Phase 链路** —— 从「明确拒绝」到「事实校验」到「过期淘汰」到「原文忠实」,把 LLM 自由发挥的空间一步步锁死。下面有表格。

数据完全留在本地,不出机器。

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green) ![Status active](https://img.shields.io/badge/status-active%20development-orange) ![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

[English tagline below · 文档与代码注释以中文为主]

**Local-first document Q&A with a 9-phase anti-hallucination chain.**  
Runs on your own machine: BGE-M3 embeddings, BGE reranker, Anthropic LLM. Nothing leaves your box.

---

## 目录

- [为什么是 YuanRAG?](#为什么是-yuanrag)
- [5 分钟快速开始](#5-分钟快速开始)
- [核心功能](#核心功能)
- [反幻觉 9-Phase 链路](#反幻觉-9-phase-链路)
- [架构概览](#架构概览)
- [配置](#配置)
- [开发与测试](#开发与测试)
- [路线图](#路线图)
- [贡献](#贡献)
- [License & 致谢](#license--致谢)

---

## 为什么是 YuanRAG?

大多数 RAG 系统都会在「具体数字」「某条规定」「引用是否忠实」这些点上翻车 —— 因为 LLM 即使引用了来源,数字 / 措辞也是它自己推的。YuanRAG 用 9 个 phase 把这一类问题系统性地压到最低:

| 根因组 | Phase | 治愈的幻觉类型 |
|---|---|---|
| **静默失败** | Day 0 / Phase 3 | LLM 拿到坏上下文也不报;同类 bug 一根接一根 |
| **无拒绝契约** | Phase 1 / Phase 6 | 「我也不知道」变成自信回答;「引用过」≠「数字对」 |
| **检索门松** | Phase 4 PR-2 / Phase 7 | 过期片段当现行;长文档只看到一半 |
| **引用完整性** | Phase 4 PR-1 | 引文对但内容是 LLM 编的 |
| **可观测盲区** | Phase 2 | 翻车没 log,无法定位 |
| **verbatim 失真** | Phase 8 | 法律 / 医疗 / 财务场景 LLM 自主改写措辞 |

**已 ship:** 9 个 Phase + Day 0,32 个 finding 全清,1600+ pytest 0 回归,Layer 5 hermetic verification 100% green。

---

## 5 分钟快速开始

### 前置依赖

- **Python 3.11+** + **Node.js 18+**
- 8GB 内存 / 5GB 磁盘(本地 BGE-M3 + reranker 模型约 900MB,首次启动下载)
- **LLM API key**:`MINIMAX_API_KEY`(MiniMax Anthropic-兼容 endpoint)或换 OpenAI / 任何 OpenAI-兼容 endpoint

### 安装

```bash
git clone https://github.com/buger123/YuanRAG.git
cd YuanRAG
cp .env.example .env
# 编辑 .env,把 MINIMAX_API_KEY 填上
```

### 启动

```bash
# 一行启动,自动装依赖、下载模型、build 前端、起服务
./start.sh        # macOS / Linux
start.bat         # Windows
```

第一次启动会从 HuggingFace 下载 BGE-M3 (≈600MB) + reranker (≈300MB),5-10 分钟。后续启动直接命中本地缓存。

### 第一个问题

浏览器打开 `http://127.0.0.1:8765/`,把 PDF / Word 拖进左侧「上传文档」,等「Indexed」绿灯亮,直接在右侧问问题:

> 这份合同里关于违约金的条款是怎么说的?

回答会带 `[1]` `[2]` `[3]` 引用编号,点编号跳到原文对应位置。

---

## 核心功能

- 🗂️ **多格式解析** —— PDF / Word / Excel / PPT / HTML / TXT,统一 chunk 化
- 🔍 **混合检索** —— BM25 + 向量 + BGE reranker,代理根据 query 自动选策略
- 🧭 **智能路由** —— 简单事实走 keyword,语义问题走向量,综合类走 hybrid + reranker
- 🔄 **查询改写** —— 第一轮检索失败,LLM 改写 query 重试,直到 confidence threshold 通过
- 🌐 **本地不够自动搜网** —— 多引擎(DDG / Bing / Tavily)并发 + fallback,`gather(return_exceptions=True)` 单引擎失败不阻塞
- 📎 **引用一体** —— 每条回答带 `[n]` 编号,点编号跳原文 + 高亮对应 chunk
- 🛡️ **反幻觉 9-Phase 链路**(见下表)
- 🧵 **多线程并发** —— A 线程跑长任务,B 线程问新问题,谁也不耽误
- 💬 **流式输出 + 思考过程** —— 实时看 LLM 思考,翻历史仍可见,跟随提问语言(中文提问中文思考)
- 🌗 **主题 / 语言切换** —— Light / Dark / System 三档,i18n zh ↔ en 镜像对称
- 📱 **响应式** —— 桌面双窗 / iPad 半窗 / 手机单窗,sidebar 抽屉化

---

## 反幻觉 9-Phase 链路

### Phase 0 · Day 0 — OOD 对抗性 fixture 集

7 fixture 覆盖「找不到 / 引文错位 / 内容是 LLM 编的 / 来源已过期 / 跨文档冲突」等典型翻车姿势。所有后续 phase 的回归测试都跑这 7 个 fixture + 11 个行为断言。

### Phase 1 · 拒绝契约

LLM 拿不到可信来源时,**显式拒绝**而不是 hedge。3 个拒绝模板(`not_found` / `insufficient_evidence` / `low_relevance`),通过 FSM `retrieval_status` signal 写入 state,前端透传展示。

### Phase 2 · 自建 metrics + loguru + `/debug/metrics`

9 个预注册 Counter 覆盖 retrieval / synthesis / verification 各关键路径,所有 hit site 都打点。env-flag gated endpoint,不污染生产环境。

### Phase 3 · 防御性 override registry + loop-break synthesis hint

同类 bug 一次性根除 —— 把所有「LLM 输出不对就改写」的兜底逻辑收进 3-registry,加新 tool = 1 行,call site 0 改动。Loop break post-script 防止 LLM 在同一坏上下文里反复 hallucinate。

### Phase 4 PR-1 · 引用 + 检索完整性(load-bearing)

6 个 load-bearing fix:rerank 失败要 stamp / range collapse WARNING log / dedupe key `page_content[:200]` / CitationChip 加 chunk_id+doc_id / `_SUMMARY_INTENT_RE` 锚定开头 / `min_top_relevance_score=0.3` gate。

### Phase 4 PR-2 · 检索侧硬化

- 文档片段加 `expires_at`,查询时 temporal filter
- 旧版本文档 `supersede()` 归档,新版本自然顶替
- `clean_chunks()` 4 规则清理脏数据(空 content / 重复 chunk / 已 supersede / 已过期)

### Phase 5 · 并发 + 清理 + UX

`_override_top_k` → keyword-only `top_k`(防 thread-safety race)+ `_hybrid_executor atexit.register(.shutdown)` + `_doc_id_for_url` 加 domain 段 + dispatcher `asyncio.gather(return_exceptions=True)` + URL dedup + `(-score, engine_priority)` 排序。

### Phase 6 · 后校验机制(rule engine + cheap-LLM)

双层 fail-closed:

1. **rule engine** —— regex 抓 date / currency / percentage / article_number 字段不一致,**零 LLM 调用,~ms 级**
2. **verifier LLM** —— 复用 `build_cheap_model(temperature=0.0)` second-pass 抓语义级 hallucination

LLM JSON parse fail / ainvoke exception → fail-closed `consistent=False`(防 verifier 挂掉时悄悄放过)。

### Phase 7 · 超长文档分片过滤

`MAX_CHUNKS_PER_DOC=50` sliding window(head 5 + middle 5 evenly-spaced + tail 5 = 15 chunks)per-doc overflow。选 head + tail 而非 top-K 因长文档关键信息多在 boundary。

### Phase 8 · 高精确模式(verbatim extraction,🔒 锁定)

法律 / 医疗 / 财务场景:**物理禁止 paraphrase**。User 选「开启 / On」→ FSM 路由到 `react_generate_extractive` 节点 → 严格 verbatim prompt(6 hard rules + 13 forbidden phrases + 「原文未提及」fallback)。回答顶部带 🔒 icon 视觉确认。

### 状态

**9 个 Phase + Day 0 全部 SHIPPED (2026-09-28)。** 详细 logs,见 [`CHANGELOG.md`](CHANGELOG.md)。

---

## 架构概览

### 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| Backend | FastAPI + Uvicorn | 异步 I/O 适合 LLM 流式 |
| Agent | LangChain + 自研 FSM | 节点状态机可观测,事件流清晰 |
| Embedding | BGE-M3 (FlagEmbedding) | 多语种 + 8192 token 长文 |
| Reranker | BGE-reranker-v2-m3 | 同源 reranker 准确度优于 cross-encoder |
| Vector DB | LanceDB | 单文件,zero infra,本地友好 |
| Search | BM25 + 向量 hybrid | 各擅胜场,互补 |
| Frontend | React + Vite + TypeScript | 现代开发体验,strict mode |
| State | LangGraph 自检 checkpoint (SQLite) | 多线程隔离,断电恢复 |
| LLM | Anthropic Claude (via MiniMax proxy) | 长 context + 工具调用稳 |

### 模块结构

```
src/
├── api/            # FastAPI routes (chat / documents / models / debug)
├── agent/          # LangChain FSM nodes + agent runner
│   ├── nodes/      # intent_analysis / retrieve / react_generate / ...
│   ├── tools/      # retrieve_docs / get_current_time
│   └── verification/  # Phase 6 rule_engine + verifier
├── retrieval/      # hybrid_search (BM25 + vector)
├── web_search/     # DDG / Bing / Tavily dispatchers
├── embeddings/     # BGE-M3 wrapper (lazy load + release)
├── reranker/       # BGE reranker wrapper
├── storage/        # LanceDB + checkpointer
├── llm/            # factory + structured output + retry
│   └── finetune/   # Phase 9 placeholder (MLOps deferred)
├── security/       # prompt safety + secret redaction
└── core/           # bootstrap / logging / trace_id

src/frontend/src/
├── components/     # ChatPane / Sidebar / SettingsDialog / ThinkingDrawer / ...
├── hooks/          # useModelsReady / useChatStream / ...
├── i18n/           # zh ↔ en symmetric
└── api/            # client (REST + WebSocket)

tests/
├── *_test.py       # pytest (1600+ tests)
└── e2e/            # Playwright Layer 5 hermetic verification
```

---

## 配置

### 必填:`MINIMAX_API_KEY`

`.env` 必填项。YuanRAG 默认指向 `https://api.minimaxi.com/anthropic` (MiniMax 代理),换 OpenAI / 任何 OpenAI-兼容 endpoint 改 `MINIMAX_BASE_URL` + `MINIMAX_MODEL` 即可。

### 常用可选

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_DATA_DIR` | `~/.rag_assistant` | 索引 / 历史 / 上传文件路径 |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8765` | server 绑定 |
| `RAG_LOG_LEVEL` | `INFO` | loguru level |
| `HF_HOME` | `<RAG_DATA_DIR>/hf_cache` | HuggingFace 模型缓存,改这里挪到 D 盘 |
| `RAG_LLM_PROVIDER` / `RAG_LLM_MODEL` | `anthropic` / `MiniMax-M3` | in-app default,Settings dialog 可覆盖 |
| `RAG_KEYRING_SERVICE` | `rag_assistant` | 备用 fallback keyring service name |

### Debug endpoint

```bash
# 默认 404,需环境变量开启
RAG_ENABLE_DEBUG=1 RAG_METRICS_ENABLED=1 \
  curl http://127.0.0.1:8765/debug/metrics
```

---

## 开发与测试

### 测试金字塔

```
              ╱ Layer 5 (Playwright + 真实 dist/ + 真实 WS + 真实 LLM) ╲
             ╱  任何 ship 必须全绿,真 Chromium + 录屏存档           ╲
            ──── pytest ────
            ╱─── 1600+ 单元 + 集成测试,LanceDB 真表,BM25 真 FTS  ╲
           ─── vitest ───
          ╱─── 140+ React hook / util / component 单测 ──────────╲
```

### 跑测试

```bash
# 后端
pytest tests/ -v --tb=short

# 前端
cd src/frontend
npm install
npm test

# Layer 5(必须:后端 + dist/ 都在跑)
node tests/e2e/verify-v2_0_31_1-layer5.js
```

### 调试叙事存档

每个 phase / 每个 fix 的完整 debugging narrative 在 `memory/<version>.md`。**该目录不进公开仓库**—— 是 per-developer / internal-only 的笔记,gitignore 已挡。如果想 clone 下来本地看,删 `memory/` 这行 ignore 即可。

---

## 路线图

- **Phase 9 · 模型微调** — MLOps effort,**deferred** 至 ≥30 天 Phase 1-8 telemetry 收齐(预计 2026-10-28+)。User 拍 infra 后启动。placeholder 已在 `src/llm/finetune/__init__.py`。
- **架构层 refactors** — 10 项 multi-week 队列,见 `memory/long-term-architecture.md`(本地)。
- 持续 UX polish(高精确模式 UI 美化 / 首次使用可发现性已 ship,见 CHANGELOG v2.0.31.x)

---

## 贡献

欢迎 PR / Issue。开发流程遵循 [`methodology`](/memory/development-methodology.md)(本地,clone 后可读):**任何 fix / refactor / feature ship 前必须 Layer 5 全绿** —— 真 Chromium + 真 dist/ + 真 WS + 真 LLM + 录屏存档。Mock 看不到 LangGraph / Anthropic streaming / React runtime 真行为,过去 45 版本一半是「修上一版回归」。

代码风格:Python 用 `ruff`(见 `pyproject.toml`),TypeScript 用 `prettier`(见 `src/frontend/.prettierrc`),**中文为主 commit message**(release notes 中文对齐);Chinese-first 因为项目用户 80% 是中文使用者。

---

## License & 致谢

[MIT](LICENSE)

**Built on the shoulders of:**

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) + [BGE Reranker v2-M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — BAAI
- [LanceDB](https://lancedb.github.io/lancedb/) — single-file vector DB
- [LangChain](https://www.langchain.com/) — agentic retrieval primitives
- [FastAPI](https://fastapi.tiangolo.com/) — async web framework
- [Anthropic Claude](https://www.anthropic.com/) — long-context LLM
- [MiniMax](https://api.minimaxi.com/anthropic) — Anthropic-兼容 proxy

---

<p align="center">
  <sub>Built with care by <a href="https://github.com/buger123">@buger123</a> and friends · 反幻觉,从这里开始</sub>
</p>