"use client";
import { useMemo, useState } from "react";
import type { History, RunView } from "../lib/connection";
import { foldEvent } from "../lib/reducer";
import { AgentBoard } from "./AgentBoard";
import { ApprovalBanner } from "./ApprovalBanner";
import { Budget } from "./Budget";
import { EventStream } from "./EventStream";
import { JobLog, JobPanel } from "./JobPanel";
import { ScoreGate } from "./ScoreGate";
import { StatusBar, connLabel } from "./StatusBar";
import { Telemetry } from "./Telemetry";
import { Timeline } from "./Timeline";

const seqOf = (ev: unknown): number =>
  ev !== null && typeof ev === "object" && typeof (ev as { seq?: unknown }).seq === "number" ? (ev as { seq: number }).seq : Infinity;

/** The history cut at `seq` (inclusive). */
export function historyUpTo(h: History, seq: number): History {
  return { base: h.base, events: h.events.filter((e) => seqOf(e) <= seq) };
}

/**
 * The whole mission-control screen for one RunView. Pure apart from the scrub position: while
 * scrubbed, every panel shows the history folded up to that seq (foldEvent over history.base).
 */
export function DashboardView({ view, runId }: { view: RunView; runId?: string }) {
  const [scrub, setScrub] = useState<number | null>(null);
  const liveSeq = view.state.last_seq;
  // A scrub position at or past the newest seq is the live view; one before the base is clamped.
  const target = scrub === null || scrub >= liveSeq ? null : Math.max(scrub, view.history.base.last_seq);

  const shownHistory = useMemo(
    () => (target === null ? view.history : historyUpTo(view.history, target)),
    [view.history, target],
  );
  const state = useMemo(
    () => (target === null ? view.state : shownHistory.events.reduce(foldEvent, shownHistory.base)),
    [view.state, shownHistory, target],
  );

  const empty = view.state.last_seq === 0;
  return (
    <div className="dashboard" data-conn={view.conn} data-mode={view.mode} data-historical={target !== null}>
      <StatusBar view={view} state={state} runId={runId} scrubSeq={target} />
      <ApprovalBanner state={state} historical={target !== null} />
      {empty ? (
        <p className="empty-notice" role="status">
          No events received yet · connection {connLabel(view)}
        </p>
      ) : null}
      <main className="grid-main">
        <AgentBoard state={state} />
        <EventStream state={state} />
        <aside className="panel rightcol" aria-label="Score, gate, budget and job">
          <ScoreGate state={state} />
          <Budget state={state} />
          <JobPanel state={state} />
        </aside>
      </main>
      <div className="grid-bottom">
        <JobLog state={state} />
        <Telemetry state={state} history={shownHistory} />
      </div>
      <Timeline
        history={view.history}
        liveSeq={liveSeq}
        value={target}
        onScrub={(seq) => setScrub(seq >= liveSeq ? null : seq)}
        onLive={() => setScrub(null)}
      />
    </div>
  );
}
