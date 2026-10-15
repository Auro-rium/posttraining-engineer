import type { History } from "../lib/connection";

type Marker = { seq: number; kind: "checkpoint" | "gate" | "approval"; label: string };

const isObj = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === "object" && !Array.isArray(v);

/** Checkpoints, gate verdicts and approval requests in the history, with their seq. */
export function timelineMarkers(history: History): Marker[] {
  const out: Marker[] = [];
  for (const ev of history.events) {
    if (!isObj(ev) || typeof ev.seq !== "number" || !isObj(ev.data)) continue;
    const d = ev.data;
    if (ev.type === "checkpoint") {
      out.push({ seq: ev.seq, kind: "checkpoint", label: `checkpoint ${typeof d.checkpoint_id === "string" ? d.checkpoint_id : "?"}` });
    } else if (ev.type === "gate.verdict") {
      out.push({ seq: ev.seq, kind: "gate", label: `gate ${typeof d.decision === "string" ? d.decision : "?"}` });
    } else if (ev.type === "approval.pending") {
      out.push({ seq: ev.seq, kind: "approval", label: "approval requested" });
    }
  }
  return out;
}

type Props = {
  history: History;
  /** The newest seq (the live end of the scrubber). */
  liveSeq: number;
  /** The scrubbed seq, or null when following live. */
  value: number | null;
  onScrub(seq: number): void;
  onLive(): void;
};

/**
 * Scrubber over the history. Its range starts at the history base (the snapshot's last_seq, or 0
 * for the sample): earlier events were never received, so they cannot be shown.
 */
export function Timeline({ history, liveSeq, value, onScrub, onLive }: Props) {
  const min = history.base.last_seq;
  const max = Math.max(liveSeq, min);
  const span = max - min;
  const markers = timelineMarkers(history);
  const pos = (seq: number) => (span > 0 ? ((seq - min) / span) * 100 : 100);
  return (
    <section className="panel timeline" aria-label="Timeline">
      <h2 className="panel-title">TIMELINE</h2>
      <div className="timeline-track">
        <input
          type="range"
          className="timeline-range"
          aria-label="Timeline scrubber (event seq)"
          min={min}
          max={max}
          step={1}
          value={value ?? max}
          disabled={span === 0}
          onChange={(e) => onScrub(Number(e.target.value))}
        />
        <div className="timeline-markers">
          {markers.map((m) =>
            m.kind === "checkpoint" ? (
              <button
                key={m.seq}
                type="button"
                className="marker marker-checkpoint"
                style={{ left: `${pos(m.seq)}%` }}
                aria-label={`Jump to ${m.label} (seq ${m.seq})`}
                title={`${m.label} · seq ${m.seq}`}
                onClick={() => onScrub(m.seq)}
              >
                ●
              </button>
            ) : (
              <span key={m.seq} className={`marker marker-${m.kind}`} style={{ left: `${pos(m.seq)}%` }} title={`${m.label} · seq ${m.seq}`} aria-hidden="true">
                ◆
              </span>
            ),
          )}
        </div>
      </div>
      <span className="dim timeline-pos">
        seq {value ?? max} / {max}
      </span>
      <button type="button" className="live-btn" aria-label="Return to live (latest state)" disabled={value === null} onClick={onLive}>
        LIVE ▸
      </button>
    </section>
  );
}
