import type { RunState } from "../lib/events";
import { num, usd } from "./format";
import { BarGauge } from "./Gauges";

export function Budget({ state }: { state: RunState }) {
  const b = state.budget;
  const spent = num(b.spent_usd);
  const cap = num(b.cap_usd);
  const est = usd(b.estimate_usd);
  const unknown = b.price_known === false || spent === null;
  const hasData = Object.keys(b).length > 0;
  const ratio = !unknown && cap !== null && cap > 0 ? Math.min(1, Math.max(0, spent! / cap)) : null;
  // Spend over 80% of the cap is drawn as a warning (amber stays the budget colour otherwise).
  const tone = ratio !== null && ratio >= 0.8 ? "red" : "amber";
  return (
    <section className="panel-sub budget" aria-label="Budget">
      <h2 className="panel-title amber">BUDGET</h2>
      {!hasData ? (
        <p className="dim">no budget data</p>
      ) : (
        <>
          {ratio !== null ? (
            <BarGauge ratio={ratio} tone={tone} label="Budget spent" min={0} max={cap!} now={spent!} />
          ) : null}
          <div className="budget-line">
            <span className="amber budget-amount">
              {unknown ? "price unknown" : usd(spent)} / {usd(cap) ?? "no cap"}
            </span>
            {ratio !== null ? <span className="dim budget-pct">{(ratio * 100).toFixed(ratio < 0.01 ? 3 : 1)}% of cap</span> : null}
          </div>
          {est ? <div className="dim">estimate {est}</div> : null}
        </>
      )}
    </section>
  );
}
