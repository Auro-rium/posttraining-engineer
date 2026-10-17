"use client";
import { useEffect, useRef, useState } from "react";
import type { A2AMessage, AgentState, RunState } from "../lib/events";
import { a2aMessages } from "./A2APanel";
import { useReducedMotion } from "./motion";
import { COORDINATOR, lookOf, Plush, PlushDefs, type Pose, ROLES } from "./Plush";

export { ROLES } from "./Plush";

export const MAX_SLOTS = 2;
/** Speech bubbles show this many code points; the full text is one click (or hover) away. */
export const BUBBLE_MAX = 90;

const own = (o: object, k: string): boolean => Object.prototype.hasOwnProperty.call(o, k);

/**
 * The pose comes from agent events only. A2A messages never make a plush cheer or slump (a reply
 * with ok:true is not a win: the sample's champion_manager answers ok after a REJECT); they only
 * wake a never-started role into standby and feed its speech bubble.
 */
export function poseOf(a: AgentState | undefined, spoke: boolean): Pose {
  if (!a) return spoke ? "standby" : "asleep";
  if (a.active) return "working";
  if (a.ok === true) return "done";
  if (a.ok === false) return "failed";
  return "idle";
}

/** The coordinator has no agent events: it works while the run status is RUNNING. */
export function coordinatorPose(status: string | null, spoke: boolean): Pose {
  if (status === null) return spoke ? "standby" : "asleep";
  return status === "RUNNING" ? "working" : "idle";
}

const STATE_TEXT: Record<Pose, string> = {
  asleep: "asleep",
  standby: "standby",
  idle: "ended",
  working: "working",
  done: "done ✓",
  failed: "failed ✗",
};
/** Screen-reader wording: says where the pose comes from. */
const STATE_SR: Record<Pose, string> = {
  asleep: "not seen yet",
  standby: "standby, seen only in the agent-to-agent conversation",
  idle: "ended, no ok flag",
  working: "working",
  done: "finished ok",
  failed: "finished not ok",
};

/** Code-point safe cut (model text may hold astral characters). */
export function bubbleText(text: string, max = BUBBLE_MAX): { short: string; cut: boolean } {
  const flat = text.replace(/\s+/g, " ").trim();
  const cps = [...flat];
  return cps.length > max ? { short: `${cps.slice(0, max - 1).join("")}…`, cut: true } : { short: flat, cut: false };
}

/** The newest message a role sent or received (for the coordinator: the newest one it sent). */
export function latestMessageFor(role: string, msgs: A2AMessage[]): A2AMessage | null {
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (role === COORDINATOR ? m.from === COORDINATOR : m.from === role || m.to === role) return m;
  }
  return null;
}

function bubbleHead(role: string, m: A2AMessage): string {
  if (m.from === role) {
    const mark = m.ok === true ? " ✓" : m.ok === false ? " ✗" : "";
    return role === COORDINATOR ? `to ${m.to}` : `reply${mark}`;
  }
  return "brief in";
}

function Bubble({ role, m }: { role: string; m: A2AMessage | null }) {
  const [open, setOpen] = useState(false);
  if (!m) return <div className="crew-bubble crew-bubble-empty" aria-hidden="true" />;
  const raw = m.text ?? "";
  const { short, cut } = bubbleText(raw === "" ? "(no text)" : raw);
  const head = bubbleHead(role, m);
  const body = (
    <>
      <span className="crew-bubble-head">{head}</span> <span className="crew-bubble-text">{open ? raw : short}</span>
    </>
  );
  // Model output: rendered as text only. Not a live region.
  return cut ? (
    <button
      type="button"
      className="crew-bubble"
      data-open={open ? "true" : "false"}
      aria-expanded={open}
      aria-label={`${role} latest message (${head}), ${open ? "collapse" : "show full text"}: ${short}`}
      title={raw}
      onClick={() => setOpen(!open)}
    >
      {body}
    </button>
  ) : (
    <div className="crew-bubble" title={raw}>
      {body}
    </div>
  );
}

type Packet = { seq: number; role: string; dir: "down" | "up" };

/**
 * A message packet for a newly arrived a2a.message: only while following the latest state, only
 * for events that arrive after mount (never the backlog), never under reduced motion.
 */
function usePacket(state: RunState, live: boolean, reduced: boolean): [Packet | null, () => void] {
  const [packet, setPacket] = useState<Packet | null>(null);
  const seen = useRef(state.last_seq);
  useEffect(() => {
    const seq = state.last_seq;
    const fresh = seq > seen.current;
    seen.current = Math.max(seen.current, seq);
    if (!fresh || !live || reduced) return;
    const last = state.log.at(-1);
    const m = state.a2a.at(-1);
    if (!last || last.seq !== seq || last.type !== "a2a.message" || !m) return;
    if (m.from === COORDINATOR) setPacket({ seq, role: m.to, dir: "down" });
    else if (m.to === COORDINATOR) setPacket({ seq, role: m.from, dir: "up" });
  }, [state, live, reduced]);
  return [packet, () => setPacket(null)];
}

type CardProps = {
  role: string;
  pose: Pose;
  slot: number | null;
  msg: A2AMessage | null;
  packet: Packet | null;
  onPacketDone(): void;
  trophyLit: boolean;
};

function CrewCard({ role, pose, slot, msg, packet, onPacketDone, trophyLit }: CardProps) {
  const look = lookOf(role);
  const active = pose === "working";
  return (
    <li
      className={`crew-card crew-${pose}`}
      data-active={active ? "true" : "false"}
      data-pose={pose}
      style={{ ["--acc" as string]: look.accent }}
    >
      <span className="crew-wire" aria-hidden="true">
        {packet && packet.role === role ? (
          <span key={packet.seq} className={`crew-packet crew-packet-${packet.dir}`} onAnimationEnd={onPacketDone} />
        ) : null}
      </span>
      <Bubble role={role} m={msg} />
      <Plush role={role} pose={pose} trophyLit={trophyLit} />
      <div className="crew-name">{role}</div>
      <div className="crew-state">
        <span className="crew-chip">{STATE_TEXT[pose]}</span>
        {active && slot !== null ? <span className="crew-slot">slot {slot}</span> : null}
      </div>
      <span className="sr-only">
        {role} ({look.creature}): {STATE_SR[pose]}
        {msg ? `. Latest message, ${bubbleHead(role, msg)}: ${bubbleText(msg.text ?? "(no text)").short}` : ""}
      </span>
    </li>
  );
}

type Props = { state: RunState; live?: boolean; simulated?: boolean };

/**
 * The plush crew: the coordinator centre stage, the eight specialists below it in pipeline order.
 * Every pose is read from the state on screen (scrubbing shows the crew as it was).
 */
export function AgentBoard({ state, live = true, simulated = false }: Props) {
  const reduced = useReducedMotion();
  const [packet, clearPacket] = usePacket(state, live, reduced);
  const msgs = a2aMessages(state);
  const known = new Set<string>(ROLES);
  const extra = Object.keys(state.agents).filter((r) => !known.has(r) && r !== COORDINATOR).sort();
  const roles = [...ROLES, ...extra];
  const trophyLit = state.gate !== null && state.gate.decision === "PROMOTE";

  const coordMsg = latestMessageFor(COORDINATOR, msgs);
  const coordPose = coordinatorPose(state.status, coordMsg !== null);
  const gesture = packet && packet.dir === "down" ? packet.seq : null;
  const busy = state.active_slots.length;

  return (
    <section className="panel agents crew" aria-label="Agents">
      <PlushDefs />
      <h2 className="panel-title">
        PLUSH CREW · AGENTS{" "}
        <span className="amber crew-slots" aria-label={`${busy} of ${MAX_SLOTS} slots busy`}>
          {busy}/{MAX_SLOTS}
        </span>
        {simulated ? (
          <span className="crew-sim" title="bundled sample: the crew replays simulated events, not a real run">
            SIMULATED
          </span>
        ) : null}
      </h2>
      <ul className="crew-stage">
        <li className="crew-card crew-coordinator" data-active={coordPose === "working" ? "true" : "false"} data-pose={coordPose}>
          <Bubble role={COORDINATOR} m={coordMsg} />
          <div className="crew-coord-core">
            {gesture !== null ? <span key={gesture} className="crew-pulse" aria-hidden="true" /> : null}
            <Plush role={COORDINATOR} pose={coordPose} gesture={gesture} />
          </div>
          <div className="crew-name">coordinator</div>
          <div className="crew-state">
            <span className="crew-chip">{state.status ? state.status.toLowerCase() : "—"}</span>
          </div>
          <span className="sr-only">
            coordinator (star core): run status {state.status ?? "not reported yet"}
            {coordMsg ? `. Latest brief, to ${coordMsg.to}: ${bubbleText(coordMsg.text ?? "(no text)").short}` : ""}
          </span>
        </li>
        {roles.map((role) => {
          const a = own(state.agents, role) ? state.agents[role] : undefined;
          const msg = latestMessageFor(role, msgs);
          return (
            <CrewCard
              key={role}
              role={role}
              pose={poseOf(a, msg !== null)}
              slot={a?.active ? a.slot : null}
              msg={msg}
              packet={packet}
              onPacketDone={clearPacket}
              trophyLit={role === "champion_manager" && trophyLit}
            />
          );
        })}
      </ul>
    </section>
  );
}
