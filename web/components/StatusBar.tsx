import type { Conn, RunView } from "../lib/connection";
import type { RunState } from "../lib/events";
import { num } from "./format";

const CONN_LABEL: Record<Conn, string> = {
  CONNECTING: "CONNECTING",
  LIVE: "● LIVE",
  RECONNECTING: "RECONNECTING",
  OFFLINE: "OFFLINE",
  AUTH_FAILED: "AUTH FAILED",
};

/** The visible connection label. A server-side replay stream is never called LIVE. */
export function connLabel(view: Pick<RunView, "conn" | "mode">): string {
  if (view.conn === "LIVE" && view.mode === "REPLAY") return "● STREAMING";
  return CONN_LABEL[view.conn];
}

type Props = {
  view: RunView;
  /** The state on screen (historical while scrubbing). */
  state: RunState;
  runId?: string;
  /** Set while the timeline is scrubbed to a past seq. */
  scrubSeq?: number | null;
};

export function StatusBar({ view, state, runId, scrubSeq = null }: Props) {
  const budget = num(state.experiment?.experiment_budget);
  const sample = view.mode === "REPLAY" && view.source === "sample";
  return (
    <header className="statusbar" aria-label="Run status">
      <span className="statusbar-run">
        [ POSTTRAIN :: thread {runId ?? "?"} :: <b>{state.status ?? "NO STATUS"}</b> ]{" "}
        <span className="dim">
          exp {state.attempt}
          {budget !== null ? `/${budget}` : ""}
        </span>
        {state.backend ? <span className="dim"> · backend {state.backend}</span> : null}
      </span>
      <span className="statusbar-conn" role="status" aria-live="polite">
        {scrubSeq !== null ? <span className="label label-amber">HISTORICAL · seq {scrubSeq}</span> : null}
        {view.mode === "REPLAY" ? (
          <span className="label label-amber">{sample ? "REPLAY · SAMPLE DATA" : "REPLAY"}</span>
        ) : null}
        <span className={`label conn conn-${view.conn.toLowerCase()}`}>{connLabel(view)}</span>
        {view.detail ? <span className="dim">({view.detail})</span> : null}
        {view.conn === "RECONNECTING" && view.state.last_seq > 0 ? (
          <span className="dim">last received state shown</span>
        ) : null}
        <span className="dim">seq {view.state.last_seq}</span>
      </span>
    </header>
  );
}
