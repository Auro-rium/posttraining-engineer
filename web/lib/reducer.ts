/**
 * Pure snapshot reducer, a byte-for-byte port of src/posttrain_agent/fold.py (read its module
 * docstring for the full contract). Shared golden fixtures live in contract/golden.json.
 *
 * - foldEvent never mutates its input and never throws. An ignored event (unknown type,
 *   malformed, or seq <= last_seq) returns the input state object itself, untouched.
 * - Integer-valued numbers (2.0) count as ints; booleans never do.
 * - Keys come from untrusted data, so every write uses own-property definition (object spread or
 *   defineProperty), never `obj[key] = v`, to keep "__proto__" from touching prototypes.
 */
import { type A2AMessage, EVENT_TYPES, type EventType, type Json, type JsonObject, type RunState } from "./events";

export const LOG_MAX = 200;
export const JOB_LOG_MAX = 50;
export const A2A_MAX = 100;

const TYPES: ReadonlySet<string> = new Set(EVENT_TYPES);

const SUMMARY_FIELDS: Record<EventType, readonly string[]> = {
  "run.started": ["backend"],
  "run.status": ["status", "attempt"],
  "agent.start": ["role", "slot"],
  "agent.end": ["role", "slot", "ok"],
  "model.call": ["name", "ok", "fallback"],
  "tool.call": ["name", "ok"],
  score: ["split", "checkpoint", "evidence"],
  "gate.verdict": ["decision"],
  "budget.update": ["price_known"],
  "job.update": ["job_id", "status"],
  "job.log": ["level", "message"],
  "approval.pending": ["action"],
  "approval.resolved": ["decision"],
  checkpoint: ["checkpoint_id", "step"],
  "a2a.message": ["from", "to", "kind"],
  "telemetry.snapshot": [],
};
const A2A_KEYS = ["from", "to", "kind", "text", "ok", "text_len"] as const;
const SCORE_KEYS = ["split", "checkpoint", "value", "n", "evidence"] as const;
const GATE_KEYS = [
  "decision",
  "reasons",
  "champion_score",
  "candidate_score",
  "relative_improvement",
] as const;

export function emptyState(): RunState {
  return {
    status: null,
    attempt: 0,
    backend: null,
    experiment: null,
    agents: {},
    active_slots: [],
    log: [],
    scores: [],
    gate: null,
    budget: {},
    job: {},
    job_log: [],
    approval: null,
    telemetry: {},
    checkpoints: [],
    a2a: [],
    last_seq: 0,
  };
}

const has = (o: object, k: string): boolean => Object.prototype.hasOwnProperty.call(o, k);

function isPlainObject(v: unknown): v is Record<string, unknown> {
  if (v === null || typeof v !== "object" || Array.isArray(v)) return false;
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
}

/** Python `_is_int`: finite integer-valued number, never a bool. */
function isInt(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v);
}

/** Normalize an int for storage: -0 becomes 0 (Python int(-0.0) == 0). */
const toInt = (v: number): number => v + 0;

/** Python `_is_json`: JSON scalars, finite numbers, lists and str-keyed dicts, depth <= 8. */
function isJson(v: unknown, depth = 0): boolean {
  if (depth > 8) return false;
  if (v === null || typeof v === "boolean" || typeof v === "string") return true;
  if (typeof v === "number") return Number.isFinite(v);
  if (Array.isArray(v)) {
    for (let i = 0; i < v.length; i++) {
      if (!(i in v) || !isJson(v[i], depth + 1)) return false;
    }
    return true;
  }
  if (isPlainObject(v)) {
    for (const k of Object.keys(v)) if (!isJson(v[k], depth + 1)) return false;
    return true;
  }
  return false;
}

/** Deep copy of an already-validated JSON value with own-property-safe objects. */
function cloneJson(v: unknown): Json {
  if (Array.isArray(v)) return v.map(cloneJson);
  if (v !== null && typeof v === "object") {
    const out: JsonObject = {};
    for (const k of Object.keys(v)) setOwn(out, k, cloneJson((v as Record<string, unknown>)[k]));
    return out;
  }
  return v as Json;
}

function setOwn<T>(o: Record<string, T>, k: string, v: T): void {
  Object.defineProperty(o, k, { value: v, enumerable: true, writable: true, configurable: true });
}

/** Python `_pick`: copy the present keys, in the given order. */
function pick(data: Record<string, unknown>, keys: readonly string[]): JsonObject {
  const out: JsonObject = {};
  for (const k of keys) if (has(data, k)) setOwn(out, k, cloneJson(data[k]));
  return out;
}

/** Python `dict.update` returning a new object. */
function merge(base: JsonObject, data: Record<string, unknown>): JsonObject {
  const out: JsonObject = {};
  for (const k of Object.keys(base)) setOwn(out, k, base[k]);
  for (const k of Object.keys(data)) setOwn(out, k, cloneJson(data[k]));
  return out;
}

type Parsed = { seq: number; ts: number; type: EventType; data: Record<string, unknown> };

function parse(ev: unknown): Parsed | null {
  if (!isPlainObject(ev)) return null;
  const get = (k: string): unknown => (has(ev, k) ? ev[k] : undefined);
  const seq = get("seq");
  const ts = get("ts");
  const type = get("type");
  const data = get("data");
  if (!isInt(seq) || typeof type !== "string" || !TYPES.has(type)) return null;
  if (typeof ts !== "number" || !Number.isFinite(ts)) return null;
  if (!isPlainObject(data) || !isJson(data)) return null;
  return { seq: toInt(seq), ts, type: type as EventType, data };
}

function summary(type: EventType, data: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const key of SUMMARY_FIELDS[type]) {
    const v = has(data, key) ? data[key] : undefined;
    if (typeof v === "boolean") parts.push(`${key}=${v ? "true" : "false"}`);
    // BigInt prints every digit like Python int(); String(1e21) would give "1e+21".
    else if (isInt(v)) parts.push(`${key}=${BigInt(v).toString()}`);
    else if (typeof v === "string") parts.push(`${key}=${v}`);
  }
  return parts.join(" ");
}

/** The per-type state changes; null means malformed: discard the event. */
function apply(s: RunState, type: EventType, data: Record<string, unknown>): Partial<RunState> | null {
  const d = (k: string): unknown => (has(data, k) ? data[k] : undefined);
  switch (type) {
    case "run.started": {
      const backend = d("backend");
      return {
        ...(typeof backend === "string" ? { backend } : {}),
        experiment: pick(data, ["goal_len", "experiment_budget"]),
      };
    }
    case "run.status": {
      const status = d("status");
      const attempt = d("attempt");
      return {
        ...(typeof status === "string" ? { status } : {}),
        ...(isInt(attempt) ? { attempt: toInt(attempt) } : {}),
      };
    }
    case "agent.start":
    case "agent.end": {
      const role = d("role");
      const slot = d("slot");
      if (typeof role !== "string" || !isInt(slot)) return null;
      const ok = d("ok");
      const agent =
        type === "agent.start"
          ? { active: true, slot: toInt(slot), ok: null }
          : { active: false, slot: toInt(slot), ok: typeof ok === "boolean" ? ok : null };
      const agents: RunState["agents"] = {};
      for (const k of Object.keys(s.agents)) setOwn(agents, k, s.agents[k]);
      setOwn(agents, role, agent);
      const active_slots = Object.values(agents)
        .filter((a) => a.active)
        .map((a) => a.slot)
        .sort((a, b) => a - b);
      return { agents, active_slots };
    }
    case "score":
      return { scores: [...s.scores, pick(data, SCORE_KEYS)] };
    case "gate.verdict":
      return { gate: pick(data, GATE_KEYS) };
    case "budget.update":
      return { budget: merge(s.budget, data) };
    case "job.update":
      return { job: merge(s.job, data) };
    case "job.log":
      return { job_log: [...s.job_log, pick(data, ["level", "message"])].slice(-JOB_LOG_MAX) };
    case "approval.pending":
      return { approval: merge({}, data) };
    case "approval.resolved":
      return { approval: null };
    case "a2a.message": {
      // Presence is `has`, as in Python `"text" in data`: an explicit null is malformed too.
      if (!["from", "to", "kind"].every((k) => typeof d(k) === "string")) return null;
      if (has(data, "text") && typeof data.text !== "string") return null;
      if (has(data, "ok") && typeof data.ok !== "boolean") return null;
      if (has(data, "text_len") && !isInt(data.text_len)) return null;
      const msg = pick(data, A2A_KEYS);
      if (has(msg, "text_len")) setOwn(msg, "text_len", toInt(msg.text_len as number));
      return { a2a: [...s.a2a, msg as A2AMessage].slice(-A2A_MAX) };
    }
    case "checkpoint":
      return { checkpoints: [...s.checkpoints, pick(data, ["checkpoint_id", "step"])] };
    case "telemetry.snapshot":
      return { telemetry: merge(s.telemetry, data) };
    case "model.call":
    case "tool.call":
      return {};
  }
}

/** Return a new state with `ev` applied; the input state is never mutated. */
export function foldEvent(state: RunState, ev: unknown): RunState {
  try {
    const parsed = parse(ev);
    if (parsed === null) return state;
    const { seq, ts, type, data } = parsed;
    const lastSeq = has(state, "last_seq") ? state.last_seq : 0;
    if (seq <= lastSeq) return state;
    const changes = apply(state, type, data);
    if (changes === null) return state;
    const entry = { seq, ts, type, summary: summary(type, data) };
    return {
      ...state,
      ...changes,
      log: [...state.log, entry].slice(-LOG_MAX),
      last_seq: seq,
    };
  } catch {
    return state; // the reducer must never throw
  }
}
