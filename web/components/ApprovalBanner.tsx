import type { RunState } from "../lib/events";
import { str, usd } from "./format";

/**
 * Display-only: shown while an `approval.pending` is unresolved. Approval happens in the CLI, so
 * this renders the command (`cli_hint`) as plain text and has no buttons.
 */
export function ApprovalBanner({ state, historical = false }: { state: RunState; historical?: boolean }) {
  const a = state.approval;
  if (!a) return null;
  const action = str(a.action) ?? "unknown action";
  const model = str(a.base_model);
  const dataset = str(a.dataset_ref);
  const est = usd(a.estimate_usd);
  const cap = usd(state.budget.cap_usd);
  const hint = str(a.cli_hint);
  return (
    <section className="approval" role="alert" aria-label={historical ? "Approval was required (historical)" : "Approval required"}>
      <b>⚠ APPROVAL REQUIRED</b>
      {historical ? <span> (at this point in the run)</span> : null} {action}
      {model ? <span> · {model}</span> : null}
      {dataset ? <span> · data {dataset}</span> : null}
      {est ? (
        <span>
          {" "}
          · est {est}
          {cap ? ` of ${cap}` : ""}
        </span>
      ) : null}
      <span> · approve via CLI: </span>
      {hint ? <code>{hint}</code> : <span>(no CLI hint provided)</span>}
    </section>
  );
}
