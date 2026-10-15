import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  appendHistory,
  HISTORY_MAX,
  type History,
  backoffDelay,
  createConnection,
  type Connection,
  type ConnectionOptions,
  type EventSourceLike,
  type FetchLike,
  type FetchResponseLike,
  type RunView,
} from "./connection";
import { emptyState, foldEvent } from "./reducer";
import { createRunStore } from "./store";
import type { RunState } from "./events";

const API = "http://api.test/";
const RUN = "thread-1";
const TOKEN = "tok";
const SAMPLE = "/sample-run.jsonl";

// ---------------------------------------------------------------- fakes

type Listener = (ev: { data: string }) => void;

class FakeES implements EventSourceLike {
  static instances: FakeES[] = [];
  readonly url: string;
  closed = false;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private listeners = new Map<string, Listener[]>();

  constructor(url: string) {
    this.url = url;
    FakeES.instances.push(this);
  }
  addEventListener(type: string, listener: Listener): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }
  close(): void {
    this.closed = true;
  }
  // test drivers
  open(): void {
    this.onopen?.();
  }
  emit(ev: { seq: number; type: string; data?: Record<string, unknown> }): void {
    const envelope = { seq: ev.seq, ts: 1000 + ev.seq, run: RUN, type: ev.type, data: ev.data ?? {} };
    for (const l of this.listeners.get(ev.type) ?? []) l({ data: JSON.stringify(envelope) });
  }
  fail(): void {
    this.onerror?.();
  }
  get since(): string | null {
    return new URL(this.url).searchParams.get("since");
  }
}

const last = (): FakeES => {
  const es = FakeES.instances.at(-1);
  if (!es) throw new Error("no EventSource created");
  return es;
};
const open = (): FakeES[] => FakeES.instances.filter((e) => !e.closed);

function res(status: number, body: unknown = {}): FetchResponseLike {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  };
}

type Route = (url: URL) => FetchResponseLike | "network-error";

/** A fetch fake routed by path; each route can be swapped mid-test. */
function fakeFetch(routes: Record<string, Route>) {
  const calls: string[] = [];
  const fn = vi.fn(async (input: string) => {
    calls.push(input);
    const url = new URL(input, "http://web.test");
    const key = url.pathname.startsWith("/v1/runs/") ? "snapshot" : url.pathname;
    const route = routes[key];
    const out = route ? route(url) : res(404);
    if (out === "network-error") throw new TypeError("Failed to fetch");
    return out;
  });
  return { fn: fn as unknown as FetchLike, calls, routes, mock: fn };
}

function snapshotOf(n: number): { state: RunState; last_seq: number } {
  let s = emptyState();
  for (let seq = 1; seq <= n; seq++) {
    s = foldEvent(s, { seq, ts: 1000 + seq, run: RUN, type: "run.status", data: { status: "RUNNING", attempt: seq } });
  }
  return { state: s, last_seq: s.last_seq };
}

const SAMPLE_TEXT = [
  { seq: 1, ts: 10, run: "sample", type: "run.started", data: { backend: "stub" } },
  { seq: 2, ts: 11, run: "sample", type: "run.status", data: { status: "RUNNING", attempt: 1 } },
  { seq: 3, ts: 12, run: "sample", type: "run.status", data: { status: "COMPLETED", attempt: 1 } },
]
  .map((e) => JSON.stringify(e))
  .join("\n");

function liveRoutes(snapshot = snapshotOf(5)): Record<string, Route> {
  return {
    "/v1/health": () => res(200, { ok: true, mode: "live", journal: true }),
    snapshot: () => res(200, snapshot),
    "/v1/stream": () => res(200, ""),
    [SAMPLE]: () => res(200, SAMPLE_TEXT),
  };
}

function make(routes: Record<string, Route>, over: Partial<ConnectionOptions> = {}) {
  const f = fakeFetch(routes);
  const conn = createConnection({
    runId: RUN,
    apiUrl: API,
    token: TOKEN,
    sampleUrl: SAMPLE,
    ...over,
    deps: { EventSource: FakeES, fetch: f.fn, ...(over.deps ?? {}) },
  });
  return { conn, f };
}

const flush = () => vi.advanceTimersByTimeAsync(0);
const view = (c: Connection): RunView => c.getSnapshot();

beforeEach(() => {
  vi.useFakeTimers();
  FakeES.instances = [];
});
afterEach(() => {
  vi.useRealTimers();
});

// ---------------------------------------------------------------- tests

describe("connection", () => {
  it("loads the snapshot then streams from its last_seq", async () => {
    const { conn, f } = make(liveRoutes());
    conn.start();
    expect(view(conn).conn).toBe("CONNECTING");
    await flush();
    expect(f.calls[0]).toBe("http://api.test/v1/health");
    expect(f.calls[1]).toBe(`http://api.test/v1/runs/${RUN}/snapshot?token=${TOKEN}`);
    expect(view(conn).state.last_seq).toBe(5);
    const es = last();
    const u = new URL(es.url);
    expect(u.pathname).toBe("/v1/stream");
    expect(u.searchParams.get("run")).toBe(RUN);
    expect(u.searchParams.get("since")).toBe("5");
    expect(u.searchParams.get("token")).toBe(TOKEN);
    es.open();
    expect(view(conn)).toMatchObject({ mode: "LIVE", conn: "LIVE", source: "api" });
    es.emit({ seq: 6, type: "run.status", data: { status: "COMPLETED" } });
    expect(view(conn).state.status).toBe("COMPLETED");
    conn.stop();
  });

  it("resumes_from_last_seq_after_drop", async () => {
    const { conn } = make(liveRoutes());
    conn.start();
    await flush();
    const first = last();
    first.open();
    first.emit({ seq: 6, type: "run.status", data: { status: "RUNNING" } });
    first.emit({ seq: 7, type: "run.status", data: { status: "RUNNING" } });
    first.fail();
    expect(first.closed).toBe(true); // native auto-reconnect must not run alongside ours
    expect(FakeES.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(999);
    expect(FakeES.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    const second = last();
    expect(second).not.toBe(first);
    expect(second.since).toBe("7");
    second.open();
    second.emit({ seq: 8, type: "run.status", data: { status: "COMPLETED" } });
    expect(view(conn).state.last_seq).toBe(8);
    expect(view(conn).conn).toBe("LIVE");
    expect(open()).toHaveLength(1);
    conn.stop();
  });

  it("keeps_live_state_and_shows_reconnecting_on_drop", async () => {
    const routes = liveRoutes();
    const { conn, f } = make(routes);
    conn.start();
    await flush();
    last().open();
    last().emit({ seq: 6, type: "run.status", data: { status: "COMPLETED" } });
    const before = view(conn).state;

    // The whole backend disappears.
    routes["/v1/health"] = () => "network-error";
    routes.snapshot = () => "network-error";
    routes["/v1/stream"] = () => "network-error";
    last().fail();
    expect(view(conn)).toMatchObject({ mode: "LIVE", conn: "RECONNECTING", source: "api" });
    expect(view(conn).state).toBe(before);

    // Many failed retries later the live state is still there, still not sample data.
    for (let i = 0; i < 8; i++) {
      await vi.advanceTimersByTimeAsync(15_000);
      last().fail();
      await flush();
    }
    expect(view(conn)).toMatchObject({ mode: "LIVE", conn: "RECONNECTING", source: "api" });
    expect(view(conn).state).toBe(before);
    expect(view(conn).state.status).toBe("COMPLETED");
    expect(f.calls.some((c) => c.includes(SAMPLE))).toBe(false);
    conn.stop();
  });

  it("falls_back_to_sample_only_when_never_connected", async () => {
    // (a) No API URL at all: sample replay, no network to the API, no EventSource.
    {
      const { conn, f } = make(liveRoutes(), { apiUrl: "" });
      conn.start();
      await flush();
      expect(view(conn)).toMatchObject({ mode: "REPLAY", source: "sample", conn: "OFFLINE" });
      expect(f.calls).toEqual([SAMPLE]);
      expect(FakeES.instances).toHaveLength(0);
      await vi.advanceTimersByTimeAsync(10_000);
      expect(view(conn).state.status).toBe("COMPLETED");
      expect(view(conn).state.last_seq).toBe(3);
      conn.stop();
    }
    // (b) API URL set but the backend never answers: sample replay, labelled OFFLINE.
    {
      const routes = liveRoutes();
      routes["/v1/health"] = () => "network-error";
      const { conn, f } = make(routes);
      conn.start();
      await flush();
      expect(view(conn)).toMatchObject({ mode: "REPLAY", source: "sample", conn: "OFFLINE" });
      expect(f.calls).toEqual(["http://api.test/v1/health", SAMPLE]);
      await vi.advanceTimersByTimeAsync(10_000);
      expect(view(conn).state.last_seq).toBe(3);
      conn.stop();
    }
    // (c) Backend reachable (journal down): no sample, honest RECONNECTING with an empty state.
    {
      const routes = liveRoutes();
      routes["/v1/health"] = () => res(503, { ok: false, mode: "live", journal: false });
      const { conn, f } = make(routes);
      conn.start();
      await flush();
      expect(view(conn)).toMatchObject({ mode: "LIVE", source: "api", conn: "RECONNECTING" });
      expect(view(conn).detail).toMatch(/503/);
      expect(f.calls.some((c) => c.includes(SAMPLE))).toBe(false);
      conn.stop();
    }
  });

  it("labels server-side replay (REPLAY_FILE) as REPLAY, not LIVE", async () => {
    const routes = liveRoutes();
    routes["/v1/health"] = () => res(200, { ok: true, mode: "replay", journal: true });
    const { conn } = make(routes);
    conn.start();
    await flush();
    last().open();
    expect(view(conn)).toMatchObject({ mode: "REPLAY", source: "api", conn: "LIVE" });
    conn.stop();
  });

  it("401_sets_auth_failed", async () => {
    // On the snapshot.
    {
      const routes = liveRoutes();
      routes.snapshot = () => res(401, { detail: "invalid token" });
      const { conn, f } = make(routes);
      conn.start();
      await flush();
      expect(view(conn).conn).toBe("AUTH_FAILED");
      expect(view(conn).mode).toBe("LIVE");
      expect(FakeES.instances).toHaveLength(0);
      expect(vi.getTimerCount()).toBe(0);
      const n = f.calls.length;
      await vi.advanceTimersByTimeAsync(120_000);
      expect(f.calls.length).toBe(n); // no retry loop
      expect(f.calls.some((c) => c.includes(SAMPLE))).toBe(false);
      conn.stop();
    }
    // On the stream (token rotated mid-run): EventSource hides the status, the probe reveals it.
    {
      const routes = liveRoutes();
      const { conn, f } = make(routes);
      conn.start();
      await flush();
      last().open();
      last().emit({ seq: 6, type: "run.status", data: { status: "RUNNING" } });
      routes["/v1/stream"] = () => res(401, { detail: "invalid token" });
      last().fail(); // dropped after open: reconnect
      await vi.advanceTimersByTimeAsync(1000);
      last().fail(); // fails before open: probe
      await flush();
      expect(view(conn).conn).toBe("AUTH_FAILED");
      expect(view(conn).state.last_seq).toBe(6); // last live state kept
      expect(open()).toHaveLength(0);
      expect(vi.getTimerCount()).toBe(0);
      const n = f.calls.length;
      const esCount = FakeES.instances.length;
      await vi.advanceTimersByTimeAsync(120_000);
      expect(f.calls.length).toBe(n);
      expect(FakeES.instances.length).toBe(esCount);
      conn.stop();
    }
  });

  it("429_and_503_on_the_stream_show_reconnecting_and_back_off", async () => {
    const routes = liveRoutes();
    routes["/v1/stream"] = () => res(429, { detail: "too many streams" });
    const { conn } = make(routes);
    conn.start();
    await flush();
    expect(FakeES.instances).toHaveLength(1);
    last().fail();
    await flush();
    expect(view(conn).conn).toBe("RECONNECTING");
    expect(view(conn).detail).toMatch(/429/);
    expect(view(conn).state.last_seq).toBe(5);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeES.instances).toHaveLength(2);
    expect(last().since).toBe("5");

    routes["/v1/stream"] = () => res(503, { detail: "journal unavailable" });
    last().fail();
    await flush();
    expect(view(conn).conn).toBe("RECONNECTING");
    expect(view(conn).detail).toMatch(/503/);
    await vi.advanceTimersByTimeAsync(1999);
    expect(FakeES.instances).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeES.instances).toHaveLength(3);

    last().open();
    expect(view(conn)).toMatchObject({ conn: "LIVE", detail: null });
    conn.stop();
  });

  it("503_on_the_snapshot_retries_with_backoff", async () => {
    const routes = liveRoutes();
    routes.snapshot = () => res(503, { detail: "journal unavailable" });
    const { conn, f } = make(routes);
    conn.start();
    await flush();
    expect(view(conn).conn).toBe("RECONNECTING");
    expect(view(conn).detail).toMatch(/503/);
    expect(FakeES.instances).toHaveLength(0);
    routes.snapshot = () => res(200, snapshotOf(3));
    await vi.advanceTimersByTimeAsync(1000);
    expect(f.calls.filter((c) => c.includes("/snapshot"))).toHaveLength(2);
    expect(last().since).toBe("3");
    conn.stop();
  });

  it("backoff grows 1s,2s,4s,8s,15s,15s and resets only on open", async () => {
    expect([0, 1, 2, 3, 4, 5, 10].map(backoffDelay)).toEqual([1000, 2000, 4000, 8000, 15000, 15000, 15000]);
    const routes = liveRoutes();
    routes["/v1/stream"] = () => "network-error";
    const { conn } = make(routes);
    conn.start();
    await flush();
    for (const delay of [1000, 2000, 4000, 8000, 15000, 15000]) {
      const n = FakeES.instances.length;
      last().fail();
      await flush();
      await vi.advanceTimersByTimeAsync(delay - 1);
      expect(FakeES.instances.length).toBe(n);
      await vi.advanceTimersByTimeAsync(1);
      expect(FakeES.instances.length).toBe(n + 1);
    }
    last().open();
    last().fail(); // after open: back to 1 s
    await vi.advanceTimersByTimeAsync(1000);
    expect(open()).toHaveLength(1);
    conn.stop();
  });

  it("events_are_deduped_by_seq", async () => {
    const { conn } = make(liveRoutes());
    const seen: RunView[] = [];
    conn.subscribe(() => seen.push(conn.getSnapshot()));
    conn.start();
    await flush();
    const es = last();
    es.open();
    const atOpen = seen.length;
    es.emit({ seq: 3, type: "run.status", data: { status: "STALE" } }); // already in snapshot
    es.emit({ seq: 5, type: "run.status", data: { status: "STALE" } });
    expect(seen.length).toBe(atOpen); // no notification, no new snapshot object
    es.emit({ seq: 6, type: "run.status", data: { status: "RUNNING" } });
    const after6 = conn.getSnapshot();
    es.emit({ seq: 6, type: "run.status", data: { status: "DUP" } });
    expect(conn.getSnapshot()).toBe(after6);
    es.emit({ seq: 7, type: "run.status", data: { status: "COMPLETED" } });
    const s = conn.getSnapshot().state;
    expect(s.log.map((e) => e.seq)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(s.status).toBe("COMPLETED");
    expect(seen.length).toBe(atOpen + 2);
    conn.stop();
  });

  it("ignores malformed frames", async () => {
    const { conn } = make(liveRoutes());
    conn.start();
    await flush();
    const es = last();
    es.open();
    const before = conn.getSnapshot();
    // reach into the listener with a garbage payload
    (es as unknown as { listeners: Map<string, Listener[]> }).listeners
      .get("run.status")!
      .forEach((l) => l({ data: "{not json" }));
    expect(conn.getSnapshot()).toBe(before);
    conn.stop();
  });

  it("getSnapshot is referentially stable between changes", async () => {
    const { conn } = make(liveRoutes());
    expect(conn.getSnapshot()).toBe(conn.getSnapshot());
    expect(conn.getServerSnapshot()).toBe(conn.getServerSnapshot());
    conn.start();
    await flush();
    expect(conn.getSnapshot()).toBe(conn.getSnapshot());
    conn.stop();
  });

  it("does nothing without an EventSource implementation (server render)", async () => {
    const f = fakeFetch(liveRoutes());
    const conn = createConnection({ runId: RUN, apiUrl: API, token: TOKEN, deps: { fetch: f.fn, EventSource: undefined } });
    conn.start();
    await flush();
    expect(f.calls).toHaveLength(0);
    expect(conn.getSnapshot()).toBe(conn.getServerSnapshot());
    conn.stop();
  });

  it("plays the bundled sample-run.jsonl as REPLAY", async () => {
    const text = readFileSync(join(__dirname, "..", "public", "sample-run.jsonl"), "utf8");
    const routes = liveRoutes();
    routes[SAMPLE] = () => res(200, text);
    const { conn } = make(routes, { apiUrl: null });
    conn.start();
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    const lines = text.split("\n").filter((l) => l.trim());
    const v = view(conn);
    expect(v.mode).toBe("REPLAY");
    expect(v.source).toBe("sample");
    expect(v.state.last_seq).toBe(JSON.parse(lines.at(-1)!).seq);
    expect(vi.getTimerCount()).toBe(0);
    conn.stop();
  });

  it("history_refolds_to_the_current_state (sample and live)", async () => {
    const text = readFileSync(join(__dirname, "..", "public", "sample-run.jsonl"), "utf8");
    const routes = liveRoutes();
    routes[SAMPLE] = () => res(200, text);
    const sample = make(routes, { apiUrl: null }).conn;
    sample.start();
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    const sv = view(sample);
    expect(sv.history.base).toEqual(emptyState());
    expect(sv.history.events.reduce(foldEvent, sv.history.base)).toEqual(sv.state);
    sample.stop();

    const { conn } = make(liveRoutes());
    conn.start();
    await flush();
    const es = last();
    es.open();
    es.emit({ seq: 6, type: "run.status", data: { status: "COMPLETED" } });
    es.emit({ seq: 6, type: "run.status", data: { status: "DUP" } }); // dropped: not in history
    const v = view(conn);
    expect(v.history.base.last_seq).toBe(5); // cannot scrub before the snapshot
    expect(v.history.events).toHaveLength(1);
    expect(v.history.events.reduce(foldEvent, v.history.base)).toEqual(v.state);
    conn.stop();
  });

  it("history is capped and folds the overflow into base", () => {
    let h: History = { base: emptyState(), events: [] };
    for (let seq = 1; seq <= HISTORY_MAX + 5; seq++) {
      h = appendHistory(h, { seq, ts: seq, run: RUN, type: "run.status", data: { attempt: seq } });
    }
    expect(h.events).toHaveLength(HISTORY_MAX);
    expect(h.base.last_seq).toBe(5);
    expect(h.events.reduce(foldEvent, h.base).last_seq).toBe(HISTORY_MAX + 5);
  });
});

describe("unmount cleanup", () => {
  it("closes the stream, clears timers and makes no requests after the last unsubscribe", async () => {
    const { conn, f } = make(liveRoutes());
    const store = createRunStore(conn);
    const unsub = store.subscribe(() => {});
    await flush();
    const es = last();
    es.open();
    es.fail(); // a retry timer is now pending
    expect(vi.getTimerCount()).toBe(1);
    unsub();
    expect(es.closed).toBe(true);
    expect(open()).toHaveLength(0);
    expect(vi.getTimerCount()).toBe(0);
    const n = f.calls.length;
    await vi.advanceTimersByTimeAsync(120_000);
    expect(f.calls.length).toBe(n);
    expect(FakeES.instances).toHaveLength(1);
  });

  it("ignores a fetch that resolves after unmount", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const routes = liveRoutes();
    const { conn } = make(routes);
    const slow = vi.fn(async (input: string) => {
      await gate;
      return res(200, input.includes("snapshot") ? snapshotOf(2) : { ok: true, mode: "live" });
    });
    const c2 = createConnection({ runId: RUN, apiUrl: API, token: TOKEN, deps: { EventSource: FakeES, fetch: slow as unknown as FetchLike } });
    const store = createRunStore(c2);
    const unsub = store.subscribe(() => {});
    await flush();
    unsub();
    release();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(FakeES.instances).toHaveLength(0);
    expect(c2.getSnapshot().state.last_seq).toBe(0);
    conn.stop();
  });

  it("stops the sample replay timers on unmount", async () => {
    const { conn } = make(liveRoutes(), { apiUrl: "" });
    const store = createRunStore(conn);
    const unsub = store.subscribe(() => {});
    await flush();
    expect(vi.getTimerCount()).toBe(1);
    unsub();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("subscribe, unsubscribe, subscribe (strict mode) ends with exactly one open stream", async () => {
    const { conn } = make(liveRoutes());
    const store = createRunStore(conn);
    store.subscribe(() => {})();
    const unsub = store.subscribe(() => {});
    await flush();
    expect(open()).toHaveLength(1);
    last().open();
    expect(view(conn).conn).toBe("LIVE");
    // A second component on the same run shares the connection.
    const unsub2 = store.subscribe(() => {});
    expect(open()).toHaveLength(1);
    unsub();
    expect(open()).toHaveLength(1);
    unsub2();
    expect(open()).toHaveLength(0);
    // Remount resumes from the last seq without refetching the snapshot.
    const unsub3 = store.subscribe(() => {});
    expect(open()).toHaveLength(1);
    expect(last().since).toBe("5");
    expect(view(conn).conn).toBe("RECONNECTING");
    unsub3();
  });
});
