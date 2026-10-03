"""Reproduce the user's exact flow: intro turn + time question in same thread.

User reported: turn 1 = "你是谁" → intro; turn 2 = "现在几点了" → LLM
hallucinated "I don't have a time tool" instead of calling get_current_time.

Drive both turns via the same WS, capture events, dump saved DB messages.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import uuid

# Force UTF-8 stdout/stderr so we can print Chinese on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import websockets


async def send_and_capture(
    ws, content: str, label: str
) -> dict:
    """Send ``content`` via WS, drain events until done/timeout."""
    print(f"\n=== TURN: {label} — content={content!r} ===")
    msg = {"message": content}
    await ws.send(json.dumps(msg, ensure_ascii=False))

    seen = {
        "events": [],
        "tool_calls": [],
        "reasoning": [],
        "tokens": [],
        "answer_complete_content": None,
        "intent": None,
    }

    try:
        # Drain until we see "done" or timeout.
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180.0)
            try:
                evt = json.loads(raw)
            except Exception:
                continue
            kind = evt.get("type", "?")
            if kind == "ping":
                continue
            seen["events"].append(kind)
            if kind == "intent":
                seen["intent"] = evt.get("intent_value") or evt.get("intent")
                print(f"  [intent] {seen['intent']!r}")
            elif kind == "tool_call_start":
                # wire protocol field is 'name' (not 'tool_name')
                name = evt.get("name") or evt.get("tool_name")
                args = evt.get("args") or evt.get("tool_args") or {}
                seen["tool_calls"].append({"name": name, "args": args})
                print(f"  [tool_call_start] name={name!r} args={args!r}")
            elif kind == "tool_call_end":
                name = evt.get("name") or evt.get("tool_name")
                ok = evt.get("ok")
                print(f"  [tool_call_end] name={name!r} ok={ok}")
            elif kind == "reasoning":
                snippet = (evt.get("content") or "")[:120]
                seen["reasoning"].append(snippet)
                # print incrementally for visibility
                print(f"  [reasoning] {snippet!r}")
            elif kind == "token":
                content_token = (evt.get("content") or "")
                seen["tokens"].append(content_token)
            elif kind == "answer_complete":
                seen["answer_complete_content"] = (evt.get("content") or "")
                print(
                    f"  [answer_complete] content={seen['answer_complete_content']!r}"
                )
            elif kind == "done":
                print(f"  [done]")
                break
            elif kind == "error":
                print(f"  [error] {evt}")
                # Don't break — keep listening in case the agent run
                # continues despite the error.
    except asyncio.TimeoutError:
        print(f"!!! timeout (180s) waiting for events; partial events: {seen['events'][:30]}")
    return seen


async def main() -> None:
    tid = str(uuid.uuid4())
    uri = f"ws://127.0.0.1:8765/ws/chat?thread_id={tid}"

    print(f">>> connecting to {uri}")
    async with websockets.connect(uri) as ws:
        # Turn 1 — intro
        t1 = await send_and_capture(ws, "你是谁", "intro")
        print(f"\n--- TURN 1 SUMMARY ---")
        print(f"events: {t1['events']}")
        print(f"intent: {t1['intent']!r}")
        print(f"tool_calls: {t1['tool_calls']}")
        print(f"tokens joined: {''.join(t1['tokens'])[:200]!r}")
        print(f"answer_complete: {t1['answer_complete_content'][:200]!r}")

        # Turn 2 — time question (the user's bug)
        t2 = await send_and_capture(ws, "现在几点了", "time-question")
        print(f"\n--- TURN 2 SUMMARY ---")
        print(f"events: {t2['events']}")
        print(f"intent: {t2['intent']!r}")
        print(f"tool_calls: {t2['tool_calls']}")
        print(f"tokens joined: {''.join(t2['tokens'])[:200]!r}")
        print(f"answer_complete: {t2['answer_complete_content'][:300]!r}")

        # Also fetch DB state.
        import urllib.request
        url = f"http://127.0.0.1:8765/sessions/{tid}/messages"
        print(f"\n>>> GET {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"saved message count: {len(data)}")
                for i, m in enumerate(data):
                    role = m.get("role")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = " | ".join(str(c)[:80] for c in content)
                    tool_calls = m.get("tool_calls") or []
                    tc_names = [tc.get("name") for tc in tool_calls]
                    print(
                        f"  [{i}] role={role!r} tool_calls={tc_names} "
                        f"content={content[:200]!r}"
                    )
        except Exception as e:
            print(f"!!! HTTP fetch failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())