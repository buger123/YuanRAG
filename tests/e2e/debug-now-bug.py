"""Debug script — reproduce the user's '现在几点了' bug.

Sends "现在几点了" via the WebSocket API and captures every event the
backend emits, so we can see what intent / tools / messages the FSM
produced and pinpoint WHY the LLM didn't call get_current_time.
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid

import websockets


async def main() -> None:
    tid = str(uuid.uuid4())
    uri = f"ws://127.0.0.1:8765/ws/chat?thread_id={tid}"

    print(f">>> connecting to {uri}")
    async with websockets.connect(uri) as ws:
        # Per server contract: send {"message": "..."} (no type field).
        msg = {"message": "现在几点了"}
        print(f">>> send: {msg!r}")
        await ws.send(json.dumps(msg, ensure_ascii=False))

        tool_calls_seen: list[str] = []
        thinking_seen: list[str] = []
        assistant_texts: list[str] = []
        event_kinds: list[str] = []

        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=120.0)
                try:
                    evt = json.loads(raw)
                except Exception as e:
                    print(f"!!! failed to parse: {raw!r}: {e}")
                    continue
                kind = evt.get("type", "?")
                event_kinds.append(kind)
                # Print every event for full visibility.
                if kind == "ping":
                    print(".", end="", flush=True)
                    continue
                if kind == "tool_call_start":
                    tool_calls_seen.append(evt.get("tool_name", "?"))
                    print(f"\n[EVENT] tool_call_start: {evt.get('tool_name')}")
                elif kind == "tool_call_end":
                    print(f"[EVENT] tool_call_end: {evt.get('tool_name')}")
                elif kind == "thinking":
                    snippet = (evt.get("content") or "")[:80]
                    thinking_seen.append(snippet)
                    print(f"[EVENT] thinking: {snippet!r}")
                elif kind == "assistant_delta":
                    print(f"[EVENT] assistant_delta: {(evt.get('content') or '')[:60]!r}")
                elif kind == "answer_complete":
                    content = (evt.get("content") or "")
                    assistant_texts.append(content)
                    print(f"[EVENT] answer_complete: {content[:200]!r}")
                elif kind == "intent":
                    print(f"[EVENT] intent: {evt.get('intent_value')!r}")
                elif kind == "error":
                    print(f"[EVENT] error: {evt}")
                elif kind == "user_message_received":
                    print(f"[EVENT] user_message_received: {evt.get('content')!r}")
                else:
                    # Print first 100 chars of any other event.
                    print(f"[EVENT] {kind}: {json.dumps(evt, ensure_ascii=False)[:120]}")

                if kind == "answer_complete":
                    # Don't break — server may send follow-up. Wait for the
                    # final message envelope.
                    pass
                if kind == "thread_done":
                    print(">>> thread_done — exit")
                    break
        except asyncio.TimeoutError:
            print("!!! timeout waiting for events")

        print("\n=== SUMMARY ===")
        print(f"thread_id: {tid}")
        print(f"event_kinds: {event_kinds}")
        print(f"tool_calls_seen: {tool_calls_seen}")
        print(f"thinking_seen: {thinking_seen}")
        print(f"assistant_texts_count: {len(assistant_texts)}")
        for i, t in enumerate(assistant_texts):
            print(f"  [{i}]: {t[:300]!r}")

        # Fetch via HTTP wire cross-check.
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