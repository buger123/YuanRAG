/**
 * v2.0.28.14 — tests for the history-replay helpers extracted
 * from App.tsx: ``loadHistory``. Two helpers under test:
 *
 *   * ``restoreMessages(records: MessageRecord[]): ChatMessage[]``
 *     — wire → ChatMessage mapping (was inline at App.tsx:679-770
 *     before this fix).
 *   * ``mergeAdjacentAssistantTurns(messages: ChatMessage[]): ChatMessage[]``
 *     — collapses the 2-AIMessage-per-ReAct-turn contract from
 *     v2.0.28.10 into 1 bubble on reload, fixing the double-bubble
 *     UX bug.
 *
 * No mocking required: both are pure transforms of plain objects.
 * ``crypto.randomUUID`` is provided by jsdom (test-setup.ts asserts
 * its presence).
 */

import { describe, expect, it } from "vitest";
import type { MessageRecord } from "../api/client";
import {
  mergeAdjacentAssistantTurns,
  restoreMessages,
  type ChatMessage,
  type ToolCall,
} from "../chat-utils";

// ---- helpers -----------------------------------------------------------

/** Build a minimal user MessageRecord. */
function userMsg(overrides: Partial<MessageRecord> = {}): MessageRecord {
  return {
    role: "user",
    content: "用户问题",
    ...overrides,
  };
}

/** Build a minimal assistant MessageRecord. */
function assistantMsg(
  overrides: Partial<MessageRecord> = {},
): MessageRecord {
  return {
    role: "assistant",
    content: "助手回答",
    ...overrides,
  };
}

/** Build a minimal ToolCall entry. */
function tc(overrides: Partial<ToolCall> = {}): ToolCall {
  return {
    id: "tc_1",
    name: "get_current_time",
    args: {},
    ok: true,
    ...overrides,
  };
}

// ---- TestRestoreMessages -----------------------------------------------

describe("v2.0.28.14 — restoreMessages", () => {
  it("returns empty array for empty records", () => {
    expect(restoreMessages([])).toEqual([]);
  });

  it("drops system and tool records (defense in depth — backend should already skip them)", () => {
    const records: MessageRecord[] = [
      userMsg(),
      { role: "system", content: "sys" } as MessageRecord,
      { role: "tool", content: "tool-result" } as unknown as MessageRecord,
      assistantMsg(),
    ];
    const out = restoreMessages(records);
    expect(out).toHaveLength(2);
    expect(out[0].role).toBe("user");
    expect(out[1].role).toBe("assistant");
  });

  it("assigns a fresh uuid to each message (no id collisions)", () => {
    const out = restoreMessages([userMsg(), assistantMsg()]);
    expect(out).toHaveLength(2);
    expect(out[0].id).not.toEqual(out[1].id);
    // UUID v4 format check (8-4-4-4-12 hex)
    expect(out[0].id).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("defaults source_kind to 'local' for legacy persisted rows (v2.0.7 SoT-4)", () => {
    const records: MessageRecord[] = [
      assistantMsg({
        // Cast through unknown: simulates a pre-v2.0.7 row that
        // lacks ``source_kind``. The defensive map in
        // ``restoreMessages`` defaults it to "local".
        sources: [
          {
            index: 1,
            filename: "doc.pdf",
            text: "t",
            score: 0.9,
          },
        ] as unknown as MessageRecord["sources"],
      }),
    ];
    const out = restoreMessages(records);
    expect(out[0].sources).toBeDefined();
    expect(out[0].sources![0].source_kind).toBe("local");
  });

  it("preserves source_kind when present", () => {
    const records: MessageRecord[] = [
      assistantMsg({
        sources: [
          {
            index: 1,
            filename: "x",
            text: "t",
            score: 0.5,
            source_kind: "web",
          },
        ],
      }),
    ];
    const out = restoreMessages(records);
    expect(out[0].sources![0].source_kind).toBe("web");
  });

  it("drops tool_calls entries that lack required id/name (defense)", () => {
    const records: MessageRecord[] = [
      assistantMsg({
        tool_calls: [
          { id: "tc_1", name: "get_current_time", ok: true } as ToolCall,
          { id: "no_name" } as unknown as ToolCall,
          { name: "no_id" } as unknown as ToolCall,
          null as unknown as ToolCall,
        ] as Array<ToolCall>,
      }),
    ];
    const out = restoreMessages(records);
    expect(out[0].tool_calls).toBeDefined();
    expect(out[0].tool_calls).toHaveLength(1);
    expect(out[0].tool_calls![0].id).toBe("tc_1");
  });

  it("preserves tool_calls timing fields verbatim", () => {
    const records: MessageRecord[] = [
      assistantMsg({
        tool_calls: [
          {
            id: "tc_1",
            name: "get_current_time",
            args: {},
            result: "2026-09-23",
            ok: true,
            step: 1,
            started_at: "2026-09-23T10:00:00Z",
            ended_at: "2026-09-23T10:00:01Z",
            elapsed_ms: 1000,
          },
        ] as Array<ToolCall>,
      }),
    ];
    const out = restoreMessages(records);
    const card = out[0].tool_calls![0];
    expect(card.started_at).toBe("2026-09-23T10:00:00Z");
    expect(card.ended_at).toBe("2026-09-23T10:00:01Z");
    expect(card.elapsed_ms).toBe(1000);
    expect(card.step).toBe(1);
    expect(card.result).toBe("2026-09-23");
  });

  it("preserves reasoning verbatim", () => {
    const records: MessageRecord[] = [
      assistantMsg({ reasoning: "Let me think about this." }),
    ];
    const out = restoreMessages(records);
    expect(out[0].reasoning).toBe("Let me think about this.");
  });
});

// ---- TestMergeAdjacentAssistantTurns -----------------------------------

describe("v2.0.28.14 — mergeAdjacentAssistantTurns", () => {
  it("passes through single assistant message (no merge needed)", () => {
    const messages: ChatMessage[] = [
      {
        id: "u1",
        role: "user",
        content: "你是谁?",
      },
      {
        id: "a1",
        role: "assistant",
        content: "我是 YuanRAG",
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2);
    expect(merged[1].content).toBe("我是 YuanRAG");
  });

  it("merges two adjacent assistants when both have tool_calls (the bug case)", () => {
    // Per v2.0.28.10 + v2.0.28.13 — both AIMessages' tool_calls
    // share the same tool_use.id. The merge should drop the
    // planning-step's tool_calls and keep only the synthesis's.
    const messages: ChatMessage[] = [
      {
        id: "u1",
        role: "user",
        content: "现在几点了?",
      },
      // Planning-step (from react_agent):
      {
        id: "a1",
        role: "assistant",
        content: "",
        reasoning: "I need to call get_current_time.",
        tool_calls: [
          // Minimal card from v2.0.28.13 synthesized fallback:
          // no timing, no result-step numbers
          { id: "tool_use_abc", name: "get_current_time", args: {}, ok: true, result: "t" },
        ],
      },
      // Synthesis (from react_generate):
      {
        id: "a2",
        role: "assistant",
        content: "现在是 2026年9月23日",
        tool_calls: [
          // Rich log from _build_tool_call_log: has timing
          {
            id: "tool_use_abc",
            name: "get_current_time",
            args: {},
            ok: true,
            result: "2026-09-23T18:43",
            step: 1,
            elapsed_ms: 0,
            started_at: "2026-09-23T18:43:00Z",
            ended_at: "2026-09-23T18:43:00Z",
          },
        ],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2); // [user, merged-assistant]
    expect(merged[1].content).toBe("现在是 2026年9月23日"); // synthesis wins
    expect(merged[1].reasoning).toBe("I need to call get_current_time."); // planning's preserved
    expect(merged[1].tool_calls).toHaveLength(1); // synthesis's kept, planning's dropped
    expect(merged[1].tool_calls![0].elapsed_ms).toBe(0); // synthesis's timing kept
  });

  it("does NOT merge when only the planning-step has tool_calls (synthesis is text-only)", () => {
    // Edge case: react_agent produced a tool-call AIMessage but
    // react_generate then produced a direct text-only synthesis
    // (rare path — synthesis might answer the question directly).
    // The two bubbles have DIFFERENT tool_calls states; merging
    // would be confusing.
    const messages: ChatMessage[] = [
      {
        id: "a1",
        role: "assistant",
        content: "",
        tool_calls: [tc({ id: "t1" })],
      },
      {
        id: "a2",
        role: "assistant",
        content: "Direct answer without showing the tool.",
        // no tool_calls
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2);
    expect(merged[0].tool_calls).toHaveLength(1);
    expect(merged[1].content).toBe("Direct answer without showing the tool.");
  });

  it("does NOT merge when only the synthesis has tool_calls", () => {
    // Another edge case: planning-step had no tool_calls (text-only
    // draft), synthesis added tool_calls post-hoc. Unusual; pass
    // through.
    const messages: ChatMessage[] = [
      { id: "a1", role: "assistant", content: "draft" },
      {
        id: "a2",
        role: "assistant",
        content: "answer",
        tool_calls: [tc({ id: "t1" })],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2);
  });

  it("does NOT merge when a user message separates the assistants (different turns)", () => {
    const messages: ChatMessage[] = [
      { id: "u1", role: "user", content: "Q1" },
      {
        id: "a1",
        role: "assistant",
        content: "A1",
        tool_calls: [tc({ id: "t1" })],
      },
      { id: "u2", role: "user", content: "Q2" },
      {
        id: "a2",
        role: "assistant",
        content: "A2",
        tool_calls: [tc({ id: "t2" })],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(4); // no merging
  });

  it("v2.0.28.15 — NEVER falls back to synthesis reasoning (was a bug)", () => {
    // Pre-v2.0.28.15 the rule was ``prev.reasoning ?? m.reasoning``
    // which fell back to synthesis reasoning when planning-step
    // lacked it. This was the v2.0.28.15 leak source: synthesis
    // AIMessage's "I'm about to write the answer" meta-commentary
    // surfaced in <ThinkingDrawer>. Post-v2.0.28.15 the rule is
    // ``reasoning: prev.reasoning`` (no fallback ever).
    // Synthesis reasoning must NEVER surface.
    const messages: ChatMessage[] = [
      { id: "u1", role: "user", content: "Q" },
      {
        id: "a1",
        role: "assistant",
        content: "",
        tool_calls: [tc()] /* no reasoning */,
      },
      {
        id: "a2",
        role: "assistant",
        content: "answer",
        reasoning: "synthesis's own thinking",
        tool_calls: [tc()],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    // [user, merged-assistant] → merged at index 1.
    expect(merged).toHaveLength(2);
    // v2.0.28.15: synthesis reasoning MUST NOT surface. Even when
    // planning-step lacks reasoning, the merged result has
    // `reasoning: undefined` (no fallback to synthesis).
    expect(merged[1].reasoning).toBeUndefined();
    expect(merged[1].reasoning).not.toBe("synthesis's own thinking");
  });

  it("carries synthesis's sources / sources_kinds / created_at verbatim", () => {
    const messages: ChatMessage[] = [
      { id: "u1", role: "user", content: "Q" },
      {
        id: "a1",
        role: "assistant",
        content: "",
        tool_calls: [tc()],
        sources: [],
        source_kinds: ["local"],
        created_at: "2026-09-23T18:00:00Z",
      },
      {
        id: "a2",
        role: "assistant",
        content: "answer with web sources",
        tool_calls: [tc()],
        sources: [
          {
            index: 1,
            filename: "x.pdf",
            text: "t",
            score: 0.9,
            source_kind: "web",
          },
        ],
        source_kinds: ["web"],
        created_at: "2026-09-23T18:43:00Z",
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2);
    expect(merged[1].sources).toHaveLength(1);
    expect(merged[1].sources![0].source_kind).toBe("web");
    expect(merged[1].source_kinds).toEqual(["web"]);
    expect(merged[1].created_at).toBe("2026-09-23T18:43:00Z");
  });

  it("handles 3 consecutive assistants (planning + mid-flight draft + synthesis) by only merging adjacent pairs", () => {
    // The merge only looks at ADJACENT pairs (no lookahead).
    // [a1(tools), a2(text), a3(tools)]: a1+a2 — a2 has no tools
    // so no merge; a2+a3 — a2 has no tools so no merge; result is
    // 3 messages, no collapsing. This is intentionally narrow —
    // see trap #4 (don't merge when prev has no tool_calls).
    const messages: ChatMessage[] = [
      {
        id: "a1",
        role: "assistant",
        content: "",
        tool_calls: [tc({ id: "t1" })],
      },
      {
        id: "a2",
        role: "assistant",
        content: "intermediate draft",
        // no tool_calls (text-only)
      },
      {
        id: "a3",
        role: "assistant",
        content: "final synthesis",
        tool_calls: [tc({ id: "t1" })],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    // No collapse happens because the merge is strictly
    // adjacent-only. The 3-message shape persists. If a future
    // FSM refactor produces this shape, a stronger pass would
    // be needed (defer).
    expect(merged).toHaveLength(3);
    expect(merged[2].content).toBe("final synthesis");
  });
});

// ---- TestRestoreMessagesThenMerge (end-to-end pipe) --------------------

describe("v2.0.28.14 — restoreMessages -> mergeAdjacentAssistantTurns (pipe)", () => {
  it("produces 1 ChatMessage per ReAct turn from 2 MessageRecords", () => {
    // The end-to-end scenario the user reported: reload of a
    // thread with a "现在几点了" turn. Wire returns 2 records;
    // the pipe collapses them to 1 ChatMessage.
    const records: MessageRecord[] = [
      userMsg({ content: "现在几点了?" }),
      // Planning-step AIMessage (v2.0.28.13 wire-enriched):
      assistantMsg({
        content: "",
        reasoning: "I should call get_current_time.",
        tool_calls: [
          {
            id: "tool_use_abc",
            name: "get_current_time",
            args: {},
            ok: true,
            result: "2026-09-23T18:43",
          },
        ] as Array<ToolCall>,
      }),
      // Synthesis AIMessage:
      assistantMsg({
        content: "当前时间是 **2026年9月23日 周三 18:43**(Asia/Shanghai 时区)。",
        tool_calls: [
          {
            id: "tool_use_abc",
            name: "get_current_time",
            args: {},
            ok: true,
            result: "2026-09-23T18:43",
            step: 1,
            elapsed_ms: 0,
            started_at: "2026-09-23T18:43:00Z",
            ended_at: "2026-09-23T18:43:00Z",
          },
        ] as Array<ToolCall>,
        created_at: "2026-09-23T18:43:00Z",
      }),
    ];

    const restored = restoreMessages(records);
    expect(restored).toHaveLength(3); // before merge

    const merged = mergeAdjacentAssistantTurns(restored);
    expect(merged).toHaveLength(2); // user + 1 merged assistant
    expect(merged[1].content).toBe(
      "当前时间是 **2026年9月23日 周三 18:43**(Asia/Shanghai 时区)。",
    );
    expect(merged[1].reasoning).toBe("I should call get_current_time.");
    expect(merged[1].tool_calls).toHaveLength(1);
    // Synthesis's timing preserved:
    expect(merged[1].tool_calls![0].elapsed_ms).toBe(0);
    expect(merged[1].tool_calls![0].started_at).toBe("2026-09-23T18:43:00Z");
  });
});

// ---- v2.0.28.15 — mergeAdjacentAssistantTurns no-fallback rule ----

describe("v2.0.28.15 — mergeAdjacentAssistantTurns never falls back to synthesis reasoning", () => {
  // v2.0.28.15 fix: pre-v2.0.28.15 the rule was
  // ``reasoning: prev.reasoning ?? m.reasoning`` which fell back
  // to synthesis reasoning when planning-step lacked it. Synthesis
  // reasoning is meta-commentary ("I'm about to write the answer")
  // that the LLM often hallucinated as prior-turn context (probe
  // Run 1: "I already provided the answer in my previous response
  // with the time 2026年9月23日 19:09" — false). Post-v2.0.28.15
  // the rule is ``reasoning: prev.reasoning`` (no fallback ever).
  // Synthesis reasoning must NEVER surface.

  it("merged reasoning === prev.reasoning (both have reasoning, prev wins)", () => {
    const messages: ChatMessage[] = [
      {
        id: "u1",
        role: "user",
        content: "Q",
      },
      {
        id: "a1",
        role: "assistant",
        content: "",
        reasoning: "plan trace: I need to call X tool",
        tool_calls: [tc()],
      },
      {
        id: "a2",
        role: "assistant",
        content: "answer",
        reasoning:
          "synth meta: I'm about to write the answer based on tool result",
        tool_calls: [tc()],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2);
    expect(merged[1].reasoning).toBe("plan trace: I need to call X tool");
  });

  it("merged reasoning === prev.reasoning when synthesis has no reasoning", () => {
    const messages: ChatMessage[] = [
      { id: "u1", role: "user", content: "Q" },
      {
        id: "a1",
        role: "assistant",
        content: "",
        reasoning: "plan trace",
        tool_calls: [tc()],
      },
      {
        id: "a2",
        role: "assistant",
        content: "answer",
        // no reasoning field
        tool_calls: [tc()],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2);
    expect(merged[1].reasoning).toBe("plan trace");
  });

  it("merged reasoning === undefined when prev has no reasoning (no fallback to synthesis)", () => {
    // This is the KEY test: even if synthesis reasoning exists,
    // it MUST NOT surface. The pre-fix `prev ?? m` rule would have
    // surfaced "synth meta" here.
    const messages: ChatMessage[] = [
      { id: "u1", role: "user", content: "Q" },
      {
        id: "a1",
        role: "assistant",
        content: "",
        // no reasoning
        tool_calls: [tc()],
      },
      {
        id: "a2",
        role: "assistant",
        content: "answer",
        reasoning: "synth meta: I'm about to write the answer",
        tool_calls: [tc()],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    expect(merged).toHaveLength(2);
    expect(merged[1].reasoning).toBeUndefined();
    // Belt-and-suspenders: assert NOT the synthesis text.
    expect(merged[1].reasoning).not.toBe(
      "synth meta: I'm about to write the answer",
    );
  });

  it("source-pin: mergeAdjacentAssistantTurns has no synthesis-reasoning fallback (functional coverage)", () => {
    // Source-pin via functional coverage: the previous test
    // ("merged reasoning === undefined when prev has no
    // reasoning (no fallback to synthesis)") proves the no-
    // fallback contract end-to-end. The chat-utils.tsx inline
    // comment block above ``mergeAdjacentAssistantTurns`` (lines
    // ~278-300 in source) carries the full design rationale and
    // is reviewed in PRs. Source-string scanning was attempted
    // here but the project tsconfig doesn't include @types/node
    // (would require adding a dev dep just for one test); the
    // functional assertion is sufficient.
    const messages: ChatMessage[] = [
      { id: "u1", role: "user", content: "Q" },
      {
        id: "a1",
        role: "assistant",
        content: "",
        // no reasoning
        tool_calls: [tc()],
      },
      {
        id: "a2",
        role: "assistant",
        content: "answer",
        reasoning: "synth meta: I'm about to write the answer",
        tool_calls: [tc()],
      },
    ];
    const merged = mergeAdjacentAssistantTurns(messages);
    // The fallback pattern `prev.reasoning ?? m.reasoning` would
    // surface "synth meta" here. The no-fallback rule returns
    // undefined — assertion below pins the contract.
    expect(merged[1].reasoning).toBeUndefined();
  });
});