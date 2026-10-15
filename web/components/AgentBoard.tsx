import type { AgentState, RunState } from "../lib/events";

/** The sub-agent roles of src/posttrain_agent/subagents.py, in pipeline order. */
export const ROLES = [
  "benchmark",
  "failure_analyst",
  "researcher",
  "data_curator",
  "training_designer",
  "training_executor",
  "evaluator",
  "champion_manager",
] as const;

export const MAX_SLOTS = 2;

function mark(a: AgentState | undefined): string {
  if (!a) return "·";
  if (a.active) return "▶";
  if (a.ok === true) return "✓";
  if (a.ok === false) return "✗";
  return "·";
}

function stateText(a: AgentState | undefined): string {
  if (!a) return "idle";
  if (a.active) return "active";
  if (a.ok === true) return "done";
  if (a.ok === false) return "failed";
  return "ended";
}

export function AgentBoard({ state }: { state: RunState }) {
  const known = new Set<string>(ROLES);
  const extra = Object.keys(state.agents).filter((r) => !known.has(r)).sort();
  const roles = [...ROLES, ...extra];
  return (
    <section className="panel agents" aria-label="Agents">
      <h2 className="panel-title">
        AGENTS{" "}
        <span className="amber" aria-label={`${state.active_slots.length} of ${MAX_SLOTS} slots busy`}>
          {state.active_slots.length}/{MAX_SLOTS}
        </span>
      </h2>
      <ul className="agent-list">
        <li className="agent agent-coordinator">
          ◆ coordinator <span className="dim agent-slot">{state.status ? state.status.toLowerCase() : "—"}</span>
        </li>
        {roles.map((role) => {
          const a = Object.prototype.hasOwnProperty.call(state.agents, role) ? state.agents[role] : undefined;
          const active = !!a?.active;
          return (
            <li
              key={role}
              className={`agent agent-${stateText(a)}`}
              data-active={active ? "true" : "false"}
              aria-label={`${role}: ${stateText(a)}${active ? `, slot ${a!.slot}` : ""}`}
            >
              <span aria-hidden="true">{mark(a)}</span> {role}
              {active ? <span className="agent-slot">slot {a!.slot}</span> : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
