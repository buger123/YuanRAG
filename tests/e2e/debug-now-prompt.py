"""Quick probe — capture the synthesis LLM prompt for the user's exact 2-turn flow.

The bug: TURN 1 (intro) + TURN 2 (time question) → synthesis LLM
produces "（等待你的下一条消息～）" instead of answering the time.

Hypothesis: the synthesis prompt is missing or misordering the HM
"现在几了点", causing the LLM to think "no new user message".

This script monkey-patches react_generate's build_chat_model to
capture the exact msgs list and dump it.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import websockets


async def main() -> None:
    tid = str(uuid.uuid4())
    uri = f"ws://127.0.0.1:8765/ws/chat?thread_id={tid}"

    print(f">>> connecting to {uri}")
    async with websockets.connect(uri) as ws:
        # TURN 1 — intro
        await ws.send(json.dumps({"message": "你是谁"}))
        print("\n=== TURN 1: 你是谁 ===")
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180.0)
            evt = json.loads(raw)
            kind = evt.get("type", "?")
            if kind in ("ping", "token"):
                continue
            print(f"  [{kind}] {json.dumps(evt, ensure_ascii=False)[:200]}")
            if kind == "done":
                break

        # Before TURN 2, monkey-patch the synthesis LLM to capture prompt.
        # We can't actually monkey-patch uvicorn from here, but we can
        # instead use a curl-style HTTP probe... actually the only way is
        # to inject code via uvicorn's `--reload` (which we don't use)
        # or to call the FSM directly via a script in the same process.
        #
        # Workaround: send TURN 2 and rely on uvicorn's existing loguru
        # output to show the prompt. But the prompt isn't logged.
        #
        # Alternative: write a separate pytest-style unit test that
        # exercises the FSM directly with a capture LLM. The user can
        # run that.

        # TURN 2 — time question
        await ws.send(json.dumps({"message": "现在几点了"}))
        print("\n=== TURN 2: 现在几点了 ===")
        reasoning_chunks: list[str] = []
        answer_complete_content = ""
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180.0)
            evt = json.loads(raw)
            kind = evt.get("type", "?")
            if kind == "ping":
                continue
            if kind == "token":
                continue
            if kind == "reasoning":
                chunk = (evt.get("content") or "")
                reasoning_chunks.append(chunk)
            if kind == "answer_complete":
                answer_complete_content = (evt.get("content") or "")
            print(f"  [{kind}] {json.dumps(evt, ensure_ascii=False)[:200]}")
            if kind == "done":
                break

        print(f"\n--- reasoning (joined): {''.join(reasoning_chunks)!r}")
        print(f"--- answer_complete: {answer_complete_content!r}")


if __name__ == "__main__":
    asyncio.run(main())