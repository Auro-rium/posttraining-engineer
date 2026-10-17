import type { Json, RunState, Score } from "../lib/events";
import { EvidenceBadge, fixed, num, pct, str } from "./format";
import { BarGauge, RadialGauge, ratioOf } from "./Gauges";
import { CountUp } from "./motion";

/** How many earlier scores are listed under the headline one. */
export const SCORE_ROWS = 5;

function scoreLabel(s: Score): string {
  return [str(s.checkpoint as Json), str(s.split as Json)].filter(Boolean).join(" ") || "score";
}

/** The newest score of checkpoint "base" on the same split as `s` (the bar's base marker). */
export function baseFor(scores: Score[], s: Score): Score | null {
  for (let i = scores.length - 1; i >= 0; i--) {
    const b = scores[i];
    if (b !== s && b.checkpoint === "base" && b.split === s.split && num(b.value as Json) !== null) return b;
  }
  return null;
}

function Scores({ scores, live }: { scores: Score[]; live: boolean }) {
  const latest = scores.at(-1);
  const earlier = scores.slice(0, -1).slice(-SCORE_ROWS).reverse();
  const value = latest ? num(latest.value as Json) : null;
  const base = latest ? baseFor(scores, latest) : null;
  const baseValue = base ? num(base.value as Json) : null;
  return (
    <section className="panel-sub scores" aria-label="Scores">
      <h2 className="panel-title">SCORE</h2>
      {latest ? (
        <>
          <div className="score-big">
            <CountUp value={num(latest.value as Json)} animate={live} className="score-value" />{" "}
            <EvidenceBadge evidence={latest.evidence as Json} />
          </div>
          <BarGauge
            ratio={ratioOf(value)}
            tone="cyan"
            label={`Latest score ${scoreLabel(latest)}`}
            min={0}
            max={1}
            now={value}
            marks={baseValue !== null ? [{ at: baseValue, label: `base ${baseValue.toFixed(2)}` }] : []}
          />
          <div className="dim">
            {scoreLabel(latest)}
            {num(latest.n as Json) !== null ? ` · n=${latest.n}` : ""}
            {baseValue !== null ? ` · base ${baseValue.toFixed(2)}` : ""}
          </div>
          {earlier.length > 0 ? (
            <ul className="score-list">
              {earlier.map((s, i) => (
                <li key={i}>
                  {scoreLabel(s)} {fixed(s.value as Json)} <EvidenceBadge evidence={s.evidence as Json} />
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : (
        <p className="dim">no scores yet</p>
      )}
    </section>
  );
}

/** Evidence values that count as measured (spec 4.3); anything else is simulated or unknown. */
export const MEASURED_EVIDENCE: ReadonlySet<string> = new Set(["LIVE", "PRIOR_VERIFIED_RUN"]);

/**
 * gate.verdict carries no evidence, so the gate inherits the run's: it is measured only when the
 * run has scores and every one of them has measured evidence.
 */
export function gateIsMeasured(scores: Score[]): boolean {
  return scores.length > 0 && scores.every((s) => typeof s.evidence === "string" && MEASURED_EVIDENCE.has(s.evidence));
}

/** Longest gate reason rendered; the full text stays in the title tooltip. */
export const REASON_MAX = 200;
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function Gate({ state }: { state: RunState }) {
  const g = state.gate;
  const measured = gateIsMeasured(state.scores);
  const decision = g ? str(g.decision as Json) : null;
  const reasons = g && Array.isArray(g.reasons) ? g.reasons.filter((r): r is string => typeof r === "string") : [];
  const rel = g ? pct(g.relative_improvement as Json) : null;
  const champ = g ? num(g.champion_score as Json) : null;
  const cand = g ? num(g.candidate_score as Json) : null;
  // The colour follows the verdict, never the score difference.
  const tone = decision === "PROMOTE" ? "lime" : decision === "REJECT" ? "red" : "cyan";
  return (
    <section className="panel-sub gate" aria-label="Promotion gate">
      <h2 className="panel-title">GATE</h2>
      {g ? (
        <>
          <div className="reactor-row">
            <RadialGauge
              className="gate-gauge"
              rings={[
                { ratio: ratioOf(cand), tone, label: "candidate" },
                { ratio: ratioOf(champ), tone: "steel", label: "champion" },
              ]}
              center={cand !== null ? cand.toFixed(2) : "no data"}
              caption={rel ? `rel ${rel}` : "gate scores"}
              label={`Gate scores: candidate ${cand !== null ? cand.toFixed(2) : "no data"} (outer ring), champion ${champ !== null ? champ.toFixed(2) : "no data"} (inner ring)`}
            />
            {/* keyed by decision: the PROMOTE glow / REJECT shake replays only when the verdict changes */}
            <div key={decision ?? "none"} className={`reactor reactor-${tone}`}>
              <span className="reactor-core" aria-hidden="true" />
              <div className={`gate-decision gate-${(decision ?? "none").toLowerCase()}`}>{decision ?? "NO DECISION"}</div>
            </div>
          </div>
          {measured ? null : (
            <div>
              <span className="badge badge-sim" title="gate inherits the run's score evidence: simulated or undetermined">
                SIMULATED
              </span>
            </div>
          )}
          <div className="dim">
            champion {fixed(g.champion_score as Json)} · candidate {fixed(g.candidate_score as Json)}
            {rel ? ` · rel ${rel}` : ""}
          </div>
          {reasons.length > 0 ? (
            <ul className="gate-reasons" aria-label="Gate reasons">
              {reasons.map((r, i) => (
                <li key={i} title={r.length > REASON_MAX ? r : undefined}>
                  {clip(r, REASON_MAX)}
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : (
        <p className="dim">no verdict yet</p>
      )}
    </section>
  );
}

/** `live` is false while the timeline is scrubbed: the headline score jumps instead of counting. */
export function ScoreGate({ state, live = true }: { state: RunState; live?: boolean }) {
  return (
    <div className="scoregate">
      <Scores scores={state.scores} live={live} />
      <Gate state={state} />
    </div>
  );
}
