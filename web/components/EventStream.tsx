import type { LogEntry, RunState } from "../lib/events";
import { clock } from "./format";

/** How many of the folded log entries (the reducer keeps 200) are rendered. */
export const STREAM_ROWS = 60;

function tone(e: LogEntry): string {
  if (/\bok=false\b/.test(e.summary) || /\bfallback=true\b/.test(e.summary)) return "ev-err";
  if (e.type.startsWith("approval") || e.type === "budget.update") return "ev-amber";
  if (e.type === "score" || e.type === "gate.verdict" || e.type === "checkpoint") return "ev-green";
  if (e.type.startsWith("agent") || e.type === "run.status" || e.type === "run.started") return "ev-cyan";
  return "";
}

export function EventStream({ state }: { state: RunState }) {
  const rows = state.log.slice(-STREAM_ROWS);
  return (
    <section className="panel stream" aria-label="Event stream">
      <h2 className="panel-title">EVENT STREAM</h2>
      {rows.length === 0 ? (
        <p className="dim">stream empty</p>
      ) : (
        <div className="stream-scroll">
          <ol className="stream-list" role="log" aria-live="polite" aria-relevant="additions">
            {rows.map((e) => (
              <li key={e.seq} className={`stream-row ${tone(e)}`}>
                <time className="stream-ts">{clock(e.ts)}</time>{" "}
                <span className="stream-type">{e.type}</span> <span className="stream-summary">{e.summary}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
