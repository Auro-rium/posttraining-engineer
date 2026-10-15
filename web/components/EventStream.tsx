"use client";
import { useEffect, useRef, useState } from "react";
import type { LogEntry, RunState } from "../lib/events";
import { clock } from "./format";

/** How many of the folded log entries (the reducer keeps 200) are rendered. */
export const STREAM_ROWS = 60;
/** At most one screen-reader announcement per this many ms; a burst is summarised in one. */
export const ANNOUNCE_MS = 1000;
const ANNOUNCE_MAX = 160;

function tone(e: LogEntry): string {
  if (/\bok=false\b/.test(e.summary) || /\bfallback=true\b/.test(e.summary)) return "ev-err";
  if (e.type.startsWith("approval") || e.type === "budget.update") return "ev-amber";
  if (e.type === "score" || e.type === "gate.verdict" || e.type === "checkpoint") return "ev-green";
  if (e.type.startsWith("agent") || e.type === "run.status" || e.type === "run.started") return "ev-cyan";
  return "";
}

/** True for an entry that renders as an error row (also used to pick the alert blip). */
export const isErrorEntry = (e: LogEntry): boolean => tone(e) === "ev-err";

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

export function announcementFor(batch: LogEntry[]): string {
  const latest = batch[batch.length - 1];
  const line = `${latest.type} ${latest.summary}`.trim();
  return clip(batch.length === 1 ? line : `${batch.length} new events, latest: ${line}`, ANNOUNCE_MAX);
}

/**
 * Throttled live-region text: new log entries (by seq) are batched and announced at most once per
 * ANNOUNCE_MS. The backlog present on mount is never read out, nor is anything while `live` is
 * false (scrubbing); a seq going backwards just resets the baseline.
 */
function useThrottledAnnouncement(log: LogEntry[], live: boolean): string {
  const [text, setText] = useState("");
  const seen = useRef<number>(log.length ? log[log.length - 1].seq : 0);
  const pending = useRef<LogEntry[]>([]);
  const lastAt = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const newest = log.length ? log[log.length - 1].seq : 0;
    if (!live || newest < seen.current) {
      seen.current = newest;
      pending.current = [];
      if (timer.current) clearTimeout(timer.current);
      timer.current = null;
      return;
    }
    const fresh = log.filter((e) => e.seq > seen.current);
    if (fresh.length === 0) return;
    seen.current = newest;
    pending.current.push(...fresh);
    const flush = () => {
      timer.current = null;
      if (pending.current.length === 0) return;
      lastAt.current = Date.now();
      setText(announcementFor(pending.current));
      pending.current = [];
    };
    if (timer.current) return; // already scheduled: this burst joins it
    const wait = lastAt.current + ANNOUNCE_MS - Date.now();
    if (wait <= 0) flush();
    else timer.current = setTimeout(flush, wait);
  }, [log, live]);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  return text;
}

export function EventStream({ state, live = true }: { state: RunState; live?: boolean }) {
  const rows = state.log.slice(-STREAM_ROWS);
  const announcement = useThrottledAnnouncement(state.log, live);
  return (
    <section className="panel stream" aria-label="Event stream">
      <h2 className="panel-title">EVENT STREAM</h2>
      {/* The visible list is not a live region (every row would be read); this throttled summary is. */}
      <p className="sr-only stream-announcer" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>
      {rows.length === 0 ? (
        <p className="dim">stream empty</p>
      ) : (
        // column-reverse: the first child sits at the bottom, so the cursor follows the newest row.
        <div className="stream-scroll" tabIndex={0} role="region" aria-label="Event rows (scrollable)">
          <span className="stream-cursor" aria-hidden="true">
            ▌
          </span>
          <ol className="stream-list">
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
