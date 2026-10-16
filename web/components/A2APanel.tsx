"use client";
import { useId, useState } from "react";
import type { A2AMessage, RunState } from "../lib/events";

/** A message longer than this many lines (or chars) starts collapsed. */
export const COLLAPSE_LINES = 6;
export const COLLAPSE_CHARS = 480;

const COORDINATOR = "coordinator";

/** Code points, as Python `len()` counts them (text_len is a Python length). */
const cpLen = (s: string): number => [...s].length;

/** The text pretty-printed when the whole of it is a JSON object; otherwise as given. */
export function displayText(text: string): { text: string; json: boolean } {
  const t = text.trim();
  if (t.startsWith("{")) {
    try {
      const v: unknown = JSON.parse(t);
      if (v !== null && typeof v === "object" && !Array.isArray(v)) return { text: JSON.stringify(v, null, 2), json: true };
    } catch {
      // not JSON (or cut mid-object): shown raw
    }
  }
  return { text, json: false };
}

/** The first COLLAPSE_LINES lines, at most COLLAPSE_CHARS chars, or null when it fits. */
function preview(text: string): string | null {
  const lines = text.split("\n");
  if (lines.length <= COLLAPSE_LINES && text.length <= COLLAPSE_CHARS) return null;
  let head = lines.slice(0, COLLAPSE_LINES).join("\n");
  if (head.length > COLLAPSE_CHARS) head = head.slice(0, COLLAPSE_CHARS);
  return `${head}…`;
}

/** Fixed class names: `kind` is untrusted, so it never becomes a class itself. */
const kindClass = (kind: string): string => (kind === "brief" ? "a2a-kind-brief" : kind === "result" ? "a2a-kind-result" : "a2a-kind-other");

/** Only the A2AMessage keys with the right types (a snapshot's state is untrusted). */
function clean(m: unknown): A2AMessage | null {
  if (m === null || typeof m !== "object") return null;
  const r = m as Record<string, unknown>;
  if (typeof r.from !== "string" || typeof r.to !== "string" || typeof r.kind !== "string") return null;
  return {
    from: r.from,
    to: r.to,
    kind: r.kind,
    ...(typeof r.text === "string" ? { text: r.text } : {}),
    ...(typeof r.ok === "boolean" ? { ok: r.ok } : {}),
    ...(typeof r.text_len === "number" && Number.isInteger(r.text_len) ? { text_len: r.text_len } : {}),
  };
}

export function a2aMessages(state: RunState): A2AMessage[] {
  const list: unknown = (state as { a2a?: unknown }).a2a;
  if (!Array.isArray(list)) return [];
  return list.map(clean).filter((m): m is A2AMessage => m !== null);
}

/** Download {run, messages} as a JSON file (client side; works in replay too). */
export function exportJson(run: string, messages: A2AMessage[]): boolean {
  if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function" || typeof document === "undefined") return false;
  const blob = new Blob([JSON.stringify({ run, messages }, null, 2)], { type: "application/json" });
  const href = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = href;
    a.download = `a2a-${run.replace(/[^A-Za-z0-9_.-]/g, "_") || "run"}.json`;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    setTimeout(() => URL.revokeObjectURL(href), 0);
  }
  return true;
}

function Message({ m }: { m: A2AMessage }) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const coord = m.from === COORDINATOR;
  const raw = m.text ?? "";
  const shown = displayText(raw);
  const short = preview(shown.text);
  const cut = m.text_len !== undefined && m.text_len > cpLen(raw);
  const kc = kindClass(m.kind);
  return (
    <li className={`a2a-msg ${coord ? "a2a-coord" : "a2a-sub"}`} data-kind={kc.slice("a2a-kind-".length)}>
      <div className="a2a-head">
        <span className={coord ? "a2a-from amber" : "a2a-from cyan"}>{m.from}</span>
        <span className="dim" aria-hidden="true"> → </span>
        <span className="sr-only">to </span>
        <span className={m.to === COORDINATOR ? "a2a-to amber" : "a2a-to cyan"}>{m.to}</span>{" "}
        <span className={`badge ${kc}`}>{m.kind.toUpperCase()}</span>
        {m.ok === true ? <span className="a2a-ok"> ✓ ok</span> : null}
        {m.ok === false ? <span className="a2a-fail"> ✗ fail</span> : null}
        {shown.json ? <span className="dim a2a-note"> json</span> : null}
      </div>
      {raw === "" ? (
        <p className="dim a2a-empty-text">(no text)</p>
      ) : (
        <pre className="a2a-text" id={id}>
          {short !== null && !open ? short : shown.text}
        </pre>
      )}
      <div className="a2a-foot">
        {short !== null ? (
          <button type="button" className="a2a-toggle" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
            {open ? "▾ collapse" : `▸ expand (${shown.text.split("\n").length} lines)`}
          </button>
        ) : null}
        {cut ? (
          <span className="dim a2a-cut">
            cut: showing {cpLen(raw)} of {m.text_len} chars
          </span>
        ) : null}
      </div>
    </li>
  );
}

/**
 * The coordinator <-> subagent conversation (state.a2a, the last 100). Rendered from the state it
 * is given, so a scrubbed Dashboard shows the conversation as folded up to that seq. Not a live
 * region: messages are never announced. Text is model output and is rendered as text only.
 */
export function A2APanel({ state, runId, simulated = false }: { state: RunState; runId?: string; simulated?: boolean }) {
  const messages = a2aMessages(state);
  return (
    <section className="panel a2a" aria-label="Agent-to-agent conversation">
      <h2 className="panel-title">
        A2A CONVERSATION <span className="dim a2a-count">{messages.length}</span>
        {simulated && messages.length > 0 ? (
          <span className="a2a-sim" title="bundled sample: these messages are simulated, not from a real run">
            SIMULATED
          </span>
        ) : null}
        <button
          type="button"
          className="a2a-export"
          disabled={messages.length === 0}
          onClick={() => exportJson(runId ?? "run", messages)}
          aria-label="Export conversation as JSON"
        >
          export JSON
        </button>
      </h2>
      {messages.length === 0 ? (
        <p className="dim">no agent-to-agent messages yet</p>
      ) : (
        // column-reverse (as in the event stream): pinned to the newest message while at the bottom.
        <div className="a2a-scroll" tabIndex={0} role="region" aria-label="Agent-to-agent messages (scrollable)">
          <ol className="a2a-list">
            {messages.map((m, i) => (
              <Message key={i} m={m} />
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
