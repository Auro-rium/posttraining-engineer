import type { RunState } from "../lib/events";
import { num, usd } from "./format";

export function Budget({ state }: { state: RunState }) {
  const b = state.budget;
  const spent = num(b.spent_usd);
  const cap = num(b.cap_usd);
  const est = usd(b.estimate_usd);
  const unknown = b.price_known === false || spent === null;
  const hasData = Object.keys(b).length > 0;
  const ratio = !unknown && cap !== null && cap > 0 ? Math.min(1, Math.max(0, spent! / cap)) : null;
  return (
    <section className="panel-sub budget" aria-label="Budget">
      <h2 className="panel-title amber">BUDGET</h2>
      {!hasData ? (
        <p className="dim">no budget data</p>
      ) : (
        <>
          {ratio !== null ? (
            <div
              className="bar"
              role="meter"
              aria-label="Budget spent"
              aria-valuemin={0}
              aria-valuemax={cap!}
              aria-valuenow={spent!}
            >
              <div className="bar-fill bar-amber" style={{ width: `${(ratio * 100).toFixed(1)}%` }} />
            </div>
          ) : null}
          <div className="amber">
            {unknown ? "price unknown" : usd(spent)} / {usd(cap) ?? "no cap"}
          </div>
          {est ? <div className="dim">estimate {est}</div> : null}
        </>
      )}
    </section>
  );
}
