import type { Json, RunState } from "../lib/events";
import { EvidenceBadge, num, str } from "./format";
import { ProgressRing } from "./Gauges";

/** Training job progress (right column). */
export function JobPanel({ state }: { state: RunState }) {
  const j = state.job;
  const id = str(j.job_id);
  const status = str(j.status) ?? "unknown";
  const done = num(j.trained_steps);
  const total = num(j.total_steps);
  const ratio = done !== null && total !== null && total > 0 ? Math.min(1, Math.max(0, done / total)) : null;
  return (
    <section className="panel-sub job" aria-label={id ? `Job ${id}` : "Job"}>
      <h2 className="panel-title cyan">JOB {id ?? ""}</h2>
      {!id && Object.keys(j).length === 0 ? (
        <p className="dim">no training job</p>
      ) : (
        <div className="job-row">
          <ProgressRing done={ratio !== null ? done : null} total={total} label="Training steps" />
          <div className="job-text">
            <div>
              {status} · {done !== null && total !== null ? `${done}/${total} steps` : "steps unknown"}
            </div>
            {"evidence" in j ? (
              <div>
                <EvidenceBadge evidence={j.evidence as Json} />
              </div>
            ) : null}
          </div>
        </div>
      )}
    </section>
  );
}

export const JOB_LOG_ROWS = 8;

/** Tail of the provider's job log (bottom row). */
export function JobLog({ state }: { state: RunState }) {
  const rows = state.job_log.slice(-JOB_LOG_ROWS);
  return (
    <section className="panel bottom joblog" aria-label="Job log">
      <h2 className="panel-title">JOB LOG</h2>
      {rows.length === 0 ? (
        <p className="dim">log empty</p>
      ) : (
        <ol className="joblog-list">
          {rows.map((l, i) => (
            <li key={i} className={str(l.level) === "error" ? "ev-err" : "dim"}>
              {str(l.level) ? <span className="joblog-level">{str(l.level)}</span> : null} {str(l.message) ?? ""}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
