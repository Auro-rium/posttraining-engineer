/** Event wire types shared with the Python journal (src/posttrain_agent/events.py). */

export const EVENT_TYPES = [
  "run.started",
  "run.status",
  "agent.start",
  "agent.end",
  "model.call",
  "tool.call",
  "score",
  "gate.verdict",
  "budget.update",
  "job.update",
  "job.log",
  "approval.pending",
  "approval.resolved",
  "checkpoint",
  "telemetry.snapshot",
] as const;

export type EventType = (typeof EVENT_TYPES)[number];

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: Json };

export type Ev = {
  seq: number;
  ts: number;
  run: string;
  type: EventType | string;
  data: Record<string, unknown>;
};

export type AgentState = { active: boolean; slot: number; ok: boolean | null };

export type LogEntry = { seq: number; ts: number; type: EventType; summary: string };

/** Keys present only when the event carried them (mirrors fold.py `_pick`). */
export type Score = Partial<{
  split: Json;
  checkpoint: Json;
  value: Json;
  n: Json;
  evidence: Json;
}>;

export type GateVerdict = Partial<{
  decision: Json;
  reasons: Json;
  champion_score: Json;
  candidate_score: Json;
  relative_improvement: Json;
}>;

export type JobLogLine = Partial<{ level: Json; message: Json }>;
export type Checkpoint = Partial<{ checkpoint_id: Json; step: Json }>;
export type Experiment = Partial<{ goal_len: Json; experiment_budget: Json }>;

/** Exactly the keys of fold.py `empty_state()`. */
export type RunState = {
  status: string | null;
  attempt: number;
  backend: string | null;
  experiment: Experiment | null;
  agents: Record<string, AgentState>;
  active_slots: number[];
  log: LogEntry[];
  scores: Score[];
  gate: GateVerdict | null;
  budget: JsonObject;
  job: JsonObject;
  job_log: JobLogLine[];
  approval: JsonObject | null;
  telemetry: JsonObject;
  checkpoints: Checkpoint[];
  last_seq: number;
};
