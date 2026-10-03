import { useId, useState } from "react";
import type { ToolCall } from "../chat-utils";
import { useLocale } from "../i18n";

interface Props {
  call: ToolCall;
}

/** Pick a small icon per tool name so the user can scan the
 *  card at a glance. Unknown tools get a generic wrench. */
function iconForName(name: string): string {
  if (name === "retrieve_docs") return "📄";
  if (name === "web_search") return "🔎";
  if (name === "get_current_time") return "🕐";
  return "🔧";
}

/** ReAct tool-call card.
 *
 *  Renders a single tool invocation the agent made during its
 *  loop. Visual states:
 *
 *    running  — yellow border + spinner; ``call.ok === undefined``
 *               (the matching ``tool_call_end`` hasn't arrived yet,
 *               or the run was killed mid-flight).
 *    ok       — green border + checkmark; ``call.ok === true``.
 *    error    — red border + x; ``call.ok === false`` (tool raised
 *               or returned a Pydantic-validation error string).
 *
 *  Body is collapsible (default collapsed — the list of cards
 *  is verbose when there are 3-4 of them). Click the header to
 *  expand; ``args`` and a truncated ``result`` are shown when open.
 */
export function ToolCallCard({ call }: Props) {
  // v2.0.7 i18n-1 — translate the collapse / expand title.
  const { t } = useLocale();
  const [open, setOpen] = useState(false);
  // v2.0.26.2 (PR-5) — QW #13 carryover: ARIA plumbing. The body
  // gets a per-card stable id (``tool-call-card-{useId}``) that
  // the header button's ``aria-controls`` points at, so screen
  // readers can announce expanded/collapsed state.
  const bodyId = `tool-call-card-${useId()}`;
  const running = call.ok === undefined;
  const ok = call.ok === true;
  const stateClass = running ? "running" : ok ? "ok" : "error";
  const icon = running ? "⏳" : ok ? "✓" : "✗";
  const elapsed = call.elapsed_ms !== undefined ? `${call.elapsed_ms} ms` : "";
  return (
    <div className={`tool-call-card ${stateClass}`}>
      <button
        type="button"
        className="tool-call-card-header"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={bodyId}
        title={open ? t("toolCard.collapseDetails") : t("toolCard.expandDetails")}
      >
        <span className="tool-call-card-icon" aria-hidden="true">
          {icon}
        </span>
        <span className="tool-call-card-name">
          <span aria-hidden="true" className="tool-call-card-tool-icon">
            {iconForName(call.name)}
          </span>{" "}
          {call.name || t("toolCard.unknown")}
        </span>
        {call.step !== undefined && (
          // PR-4 (v2.0.26.1): was hardcoded English "step {n}". Now
          // localized via ``toolCard.step`` (matches the existing
          // ``chat.step`` placeholder shape).
          <span className="tool-call-card-step">
            {t("toolCard.step", { n: call.step })}
          </span>
        )}
        {elapsed && (
          <span className="tool-call-card-elapsed">{elapsed}</span>
        )}
        <span className="tool-call-card-chevron" aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div id={bodyId} className="tool-call-card-body" role="region">
          <div className="tool-call-card-args">
            <div className="tool-call-card-label">
              {t("toolCard.args")}
            </div>
            <pre>
              <code>
                {JSON.stringify(call.args ?? {}, null, 2)}
              </code>
            </pre>
          </div>
          {call.result !== undefined && call.result !== "" && (
            <div className="tool-call-card-result">
              <div className="tool-call-card-label">
                {t("toolCard.result")}
              </div>
              <pre>
                <code>
                  {call.result.length > 800
                    ? `${call.result.slice(0, 800)}…`
                    : call.result}
                </code>
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}