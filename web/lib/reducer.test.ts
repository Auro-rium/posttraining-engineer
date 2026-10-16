import { describe, expect, it } from "vitest";
import goldenJson from "../../contract/golden.json";
import { EVENT_TYPES, type RunState } from "./events";
import { A2A_MAX, emptyState, foldEvent } from "./reducer";

type Case = {
  name: string;
  events: unknown[];
  expected: Record<string, unknown>;
  expected_len?: Record<string, number>;
  expected_first?: Record<string, unknown>;
};

const CASE_KEYS = new Set(["name", "events", "expected", "expected_len", "expected_first"]);
const golden = goldenJson as unknown as Case[];
const STATE_KEYS = Object.keys(emptyState());

function foldAll(events: unknown[], start: RunState = emptyState()): RunState {
  return events.reduce<RunState>((s, ev) => foldEvent(s, ev), start);
}

function field(state: RunState, key: string): unknown {
  if (!STATE_KEYS.includes(key)) throw new Error(`unknown state key in fixture: ${key}`);
  expect(Object.hasOwn(state, key), `state has key ${key}`).toBe(true);
  return (state as unknown as Record<string, unknown>)[key];
}

function deepFreeze<T>(v: T): T {
  if (v !== null && typeof v === "object") {
    for (const x of Object.values(v)) deepFreeze(x);
    Object.freeze(v);
  }
  return v;
}

describe("golden fixtures", () => {
  it("loaded a non-empty fixture list", () => {
    expect(Array.isArray(golden)).toBe(true);
    expect(golden.length).toBeGreaterThanOrEqual(20);
  });

  it.each(golden.map((c) => [c.name, c] as const))(
    "golden fixtures match the Python expectations: %s",
    (_name, c) => {
      for (const k of Object.keys(c)) {
        if (!CASE_KEYS.has(k)) throw new Error(`unknown fixture key: ${k}`);
      }
      const state = foldAll(c.events);
      for (const [k, v] of Object.entries(c.expected)) {
        expect(field(state, k), `expected.${k}`).toStrictEqual(v);
      }
      for (const [k, n] of Object.entries(c.expected_len ?? {})) {
        expect(field(state, k) as unknown[], `expected_len.${k}`).toHaveLength(n);
      }
      for (const [k, v] of Object.entries(c.expected_first ?? {})) {
        expect((field(state, k) as unknown[])[0], `expected_first.${k}`).toStrictEqual(v);
      }
    },
  );
});

describe("reducer", () => {
  it("exposes the 16 event types and exactly the fold.py state keys", () => {
    expect(EVENT_TYPES).toHaveLength(16);
    expect(EVENT_TYPES).toContain("a2a.message");
    expect(STATE_KEYS.sort()).toStrictEqual(
      [
        "status", "attempt", "backend", "experiment", "agents", "active_slots", "log", "scores",
        "gate", "budget", "job", "job_log", "approval", "telemetry", "checkpoints", "a2a", "last_seq",
      ].sort(),
    );
  });

  it("unknown event types are ignored", () => {
    const s1 = foldEvent(emptyState(), {
      seq: 1, ts: 1, run: "r", type: "run.status", data: { status: "RUNNING", attempt: 1 },
    });
    expect(s1.last_seq).toBe(1);
    for (const type of ["future.thing", "", "constructor", "__proto__", "toString"]) {
      const s2 = foldEvent(s1, { seq: 2, ts: 2, run: "r", type, data: { x: 1 } });
      expect(s2).toStrictEqual(s1);
    }
  });

  it("reducer never throws on malformed events", () => {
    const base = foldEvent(emptyState(), {
      seq: 1, ts: 1, run: "r", type: "agent.start", data: { role: "a", slot: 1 },
    });
    const cyclic: Record<string, unknown> = { a: 1 };
    cyclic.self = cyclic;
    let deep: unknown = 1;
    for (let i = 0; i < 20; i++) deep = { d: deep };
    const bad: unknown[] = [
      undefined, null, 0, "x", true, [], [1, 2], Symbol("s"), () => 1, 10n, new Date(),
      { seq: 2 },
      { seq: 2, ts: 2, type: "run.status" },
      { seq: 2, ts: 2, type: "run.status", data: null },
      { seq: 2, ts: 2, type: "run.status", data: [] },
      { seq: 2, ts: 2, type: "run.status", data: cyclic },
      { seq: 2, ts: 2, type: "run.status", data: { status: deep } },
      { seq: 2, ts: 2, type: "run.status", data: { status: "X", n: Number.NaN } },
      { seq: 2, ts: 2, type: "run.status", data: { status: "X", n: undefined } },
      { seq: 2, ts: 2, type: "run.status", data: { status: "X", n: 1n } },
      { seq: 2, ts: Number.POSITIVE_INFINITY, type: "run.status", data: {} },
      { seq: Number.NaN, ts: 2, type: "run.status", data: {} },
      { seq: 2.5, ts: 2, type: "run.status", data: {} },
      { seq: true, ts: 2, type: "run.status", data: {} },
      { seq: 2, ts: 2, type: "agent.end", data: { role: "a", slot: Number.POSITIVE_INFINITY } },
      { seq: 2, ts: 2, type: "agent.start", data: { role: "a", slot: "1" } },
    ];
    for (const ev of bad) {
      let out: RunState | undefined;
      expect(() => {
        out = foldEvent(base, ev);
      }).not.toThrow();
      expect(out).toStrictEqual(base);
    }
    // A malformed state never makes it throw either.
    expect(() => foldEvent({} as RunState, { seq: 1, ts: 1, type: "job.log", data: {} })).not.toThrow();
    expect(() => foldEvent(null as unknown as RunState, { seq: 1, ts: 1, type: "score", data: {} })).not.toThrow();
  });

  it("does not mutate its input", () => {
    for (const c of golden) {
      let state = deepFreeze(emptyState());
      for (const ev of c.events) {
        const before = structuredClone(state);
        const evCopy = structuredClone(ev);
        const next = foldEvent(state, deepFreeze(structuredClone(ev)));
        expect(state).toStrictEqual(before);
        // The frozen input must not make it bail out: same result as an unfrozen fold.
        expect(next).toStrictEqual(foldEvent(structuredClone(before), evCopy));
        state = deepFreeze(next);
      }
      // Every golden case still reaches its expected state under freezing.
      for (const [k, v] of Object.entries(c.expected)) {
        expect(field(state, k)).toStrictEqual(v);
      }
    }
  });

  it("does not let event data alias into state or pollute prototypes", () => {
    const data = { action: "launch", hyperparameters: { lr: 1 } };
    const s = foldEvent(emptyState(), { seq: 1, ts: 1, type: "approval.pending", data });
    data.hyperparameters.lr = 2;
    expect(s.approval).toStrictEqual({ action: "launch", hyperparameters: { lr: 1 } });

    const evil = JSON.parse(
      '{"seq":2,"ts":2,"type":"job.update","data":{"__proto__":{"polluted":1},"job_id":"j"}}',
    );
    const s2 = foldEvent(s, evil);
    expect(Object.getPrototypeOf(s2.job)).toBe(Object.prototype);
    expect(Object.hasOwn(s2.job, "__proto__")).toBe(true);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();

    const s3 = foldEvent(s2, { seq: 3, ts: 3, type: "agent.start", data: { role: "__proto__", slot: 1 } });
    expect(Object.getPrototypeOf(s3.agents)).toBe(Object.prototype);
    expect(s3.active_slots).toStrictEqual([1]);
  });

  it("matches Python on summary edge cases", () => {
    const s = foldAll([
      { seq: 1, ts: 1, type: "checkpoint", data: { checkpoint_id: "c", step: 1e21 } },
      { seq: 2, ts: 2, type: "run.status", data: { status: "S", attempt: -0 } },
      { seq: 3, ts: 3, type: "score", data: { split: "h", checkpoint: 1.5, evidence: ["x"] } },
      { seq: 4, ts: 4, type: "agent.start", data: { role: "b", slot: 10 } },
      { seq: 5, ts: 5, type: "agent.start", data: { role: "a", slot: 9 } },
    ]);
    expect(s.log.map((e) => e.summary)).toStrictEqual([
      "checkpoint_id=c step=1000000000000000000000",
      "status=S attempt=0",
      "split=h",
      "role=b slot=10",
      "role=a slot=9",
    ]);
    expect(Object.is(s.attempt, 0)).toBe(true);
    expect(s.active_slots).toStrictEqual([9, 10]);
  });
});

describe("a2a.message", () => {
  const msg = (seq: number, data: Record<string, unknown>) => ({ seq, ts: 100 + seq, run: "r", type: "a2a.message", data });
  const brief = { from: "coordinator", to: "evaluator", kind: "brief", text: "go", text_len: 2 };

  it("appends present keys only, in order, and logs from/to/kind", () => {
    const s = foldAll([
      msg(1, brief),
      msg(2, { from: "evaluator", to: "coordinator", kind: "result", ok: false }),
      msg(3, { from: "x", to: "y", kind: "note", extra: [1, 2], text: "" }),
    ]);
    expect(s.a2a).toStrictEqual([
      brief,
      { from: "evaluator", to: "coordinator", kind: "result", ok: false },
      { from: "x", to: "y", kind: "note", text: "" },
    ]);
    expect(Object.keys(s.a2a[0])).toStrictEqual(["from", "to", "kind", "text", "text_len"]);
    expect(s.log.map((e) => e.summary)).toStrictEqual([
      "from=coordinator to=evaluator kind=brief",
      "from=evaluator to=coordinator kind=result",
      "from=x to=y kind=note",
    ]);
    expect(s.last_seq).toBe(3);
  });

  it("ignores malformed messages entirely (no state, log or last_seq change)", () => {
    const base = foldEvent(emptyState(), msg(1, brief));
    const bad: Record<string, unknown>[] = [
      {},
      { to: "b", kind: "brief" },
      { from: "a", kind: "brief" },
      { from: "a", to: "b" },
      { from: 1, to: "b", kind: "brief" },
      { from: "a", to: null, kind: "brief" },
      { from: "a", to: "b", kind: true },
      { from: "a", to: "b", kind: "brief", text: null },
      { from: "a", to: "b", kind: "brief", text: 5 },
      { from: "a", to: "b", kind: "brief", text: ["x"] },
      { from: "a", to: "b", kind: "result", ok: "yes" },
      { from: "a", to: "b", kind: "result", ok: null },
      { from: "a", to: "b", kind: "result", ok: 1 },
      { from: "a", to: "b", kind: "brief", text_len: 1.5 },
      { from: "a", to: "b", kind: "brief", text_len: "3" },
      { from: "a", to: "b", kind: "brief", text_len: true },
      { from: "a", to: "b", kind: "brief", text_len: null },
    ];
    for (const data of bad) {
      expect(foldEvent(base, msg(2, data)), JSON.stringify(data)).toBe(base);
    }
  });

  it("int rule: an integer-valued text_len is stored as an int, -0 as 0", () => {
    const s = foldAll([
      msg(1, { ...brief, text_len: 2.0 }),
      msg(2, { ...brief, text_len: -0 }),
      msg(3.0, { ...brief, text_len: 1e3 }),
    ]);
    expect(s.a2a.map((m) => m.text_len)).toStrictEqual([2, 0, 1000]);
    expect(Object.is(s.a2a[1].text_len, 0)).toBe(true);
    expect(s.last_seq).toBe(3);
  });

  it(`keeps the last ${A2A_MAX} messages`, () => {
    const events = Array.from({ length: A2A_MAX + 1 }, (_, i) => msg(i + 1, { ...brief, text: `m${i + 1}` }));
    const s = foldAll(events);
    expect(s.a2a).toHaveLength(A2A_MAX);
    expect(s.a2a[0].text).toBe("m2");
    expect(s.a2a.at(-1)!.text).toBe(`m${A2A_MAX + 1}`);
    expect(s.last_seq).toBe(A2A_MAX + 1);
  });

  it("a duplicate or stale seq is ignored", () => {
    const s = foldAll([msg(1, brief), msg(2, { ...brief, text: "second" })]);
    expect(foldEvent(s, msg(2, { ...brief, text: "dup" }))).toBe(s);
    expect(foldEvent(s, msg(1, { ...brief, text: "stale" }))).toBe(s);
    expect(s.a2a.map((m) => m.text)).toStrictEqual(["go", "second"]);
  });

  it("does not alias event data or pollute prototypes", () => {
    const data = { from: "__proto__", to: "constructor", kind: "brief", text: "t" };
    const s = foldEvent(emptyState(), msg(1, data));
    data.text = "changed";
    expect(s.a2a[0]).toStrictEqual({ from: "__proto__", to: "constructor", kind: "brief", text: "t" });
    expect(Object.getPrototypeOf(s.a2a[0])).toBe(Object.prototype);
  });
});
