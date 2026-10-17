"use client";
import { useState } from "react";
import type { RunState } from "../lib/events";
import { useReducedMotion } from "./motion";

/** How many of the newest log entries scroll through the ticker. */
export const TICKER_ITEMS = 14;

/**
 * Mini event ticker for the top line. Decorative duplicate of the event stream (aria-hidden: the
 * stream and its single throttled announcer are the accessible version). The list is rendered
 * twice so a CSS translateX loop is seamless; under reduced motion it is static.
 */
export function Ticker({ state }: { state: RunState }) {
  const items = state.log.slice(-TICKER_ITEMS).reverse();
  if (items.length === 0) return <div className="ticker ticker-empty" aria-hidden="true" />;
  const row = (dup: boolean) => (
    <span className="ticker-row" key={dup ? "b" : "a"}>
      {items.map((e) => (
        <span key={e.seq} className="ticker-item">
          <span className="ticker-seq">#{e.seq}</span> <span className="ticker-type">{e.type}</span>
          {e.summary ? <span className="ticker-sum"> {e.summary}</span> : null}
        </span>
      ))}
    </span>
  );
  return (
    <div className="ticker" aria-hidden="true">
      <div className="ticker-track" style={{ ["--n" as string]: items.length }}>
        {row(false)}
        {row(true)}
      </div>
    </div>
  );
}

const BOOT_LINES = ["telemetry bus", "event journal", "plush crew", "gauges", "gate reactor"];

/**
 * One-time boot sequence over the panels on first mount (decorative, aria-hidden, no pointer
 * events). It sits under the status bar and the approval banner, so the honesty labels are never
 * covered. It removes itself when its fade-out ends; under reduced motion it is not rendered (and
 * the CSS reduced-motion block hides it before hydration).
 */
export function BootSequence() {
  const reduced = useReducedMotion(false);
  const [done, setDone] = useState(false);
  if (reduced || done) return null;
  return (
    <div
      className="bootseq"
      aria-hidden="true"
      onAnimationEnd={(e) => {
        if (e.target === e.currentTarget) setDone(true);
      }}
    >
      <div className="bootseq-box">
        <div className="bootseq-title">POSTTRAIN · TELEMETRY CORE</div>
        <ol className="bootseq-lines">
          {BOOT_LINES.map((l, i) => (
            <li key={l} style={{ ["--i" as string]: i }}>
              <span className="bootseq-arrow">&gt;</span> linking {l}
              <span className="bootseq-dots"> ···</span>
            </li>
          ))}
        </ol>
        <div className="bootseq-bar">
          <span className="bootseq-fill" />
        </div>
      </div>
    </div>
  );
}
