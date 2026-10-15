/**
 * Framework-free connection to the read-only run API, with the honesty rules of the spec (6.5).
 *
 * Flow (API configured): GET /v1/health (unauthenticated; tells live from server-side replay and
 * reachable from unreachable) -> GET /v1/runs/{id}/snapshot -> EventSource /v1/stream?since=last_seq.
 *
 * Honesty rules:
 * - mode "REPLAY" whenever the data is recorded: the bundled sample, or a backend serving
 *   REPLAY_FILE (health.mode === "replay"). `source` tells which.
 * - The bundled sample is used only when there is no API URL or the backend never answered at all.
 *   Once any HTTP response came back, the state is never replaced by sample data; a drop keeps the
 *   last live state and shows conn "RECONNECTING".
 * - 401 anywhere -> conn "AUTH_FAILED", everything stops (no retry loop).
 * - 429 / 503 / network errors -> conn "RECONNECTING" with `detail`, retried with backoff 1 s..15 s
 *   (reset only when a stream opens), resuming from the last folded seq.
 *
 * EventSource never exposes the HTTP status, so when a stream fails before it opened, the same URL
 * is probed once with fetch (aborted as soon as headers arrive) to tell 401 / 429 / 503 apart.
 *
 * SSR-safe: nothing touches window, EventSource or fetch at import or construction. Dependencies are
 * resolved in start(); with no EventSource implementation (server render) start() is a no-op.
 */
import { EVENT_TYPES, type RunState } from "./events";
import { emptyState, foldEvent } from "./reducer";

export type Mode = "LIVE" | "REPLAY";
export type Conn = "CONNECTING" | "LIVE" | "RECONNECTING" | "OFFLINE" | "AUTH_FAILED";
export type Source = "api" | "sample";

/**
 * What the timeline scrubber needs: `state` equals `events` folded onto `base` with foldEvent.
 * `base` is the empty state (sample) or the loaded snapshot (API), so history cannot reach before
 * the snapshot's last_seq. Only events that changed the state are kept, at most HISTORY_MAX; older
 * ones are folded into `base`.
 */
export type History = { base: RunState; events: readonly unknown[] };

export type RunView = {
  state: RunState;
  mode: Mode;
  conn: Conn;
  /** Where the data comes from: the API, or the bundled sample (label it SAMPLE DATA). */
  source: Source;
  /** Human-readable reason for a non-LIVE conn (e.g. "too many streams (429)"), else null. */
  detail: string | null;
  history: History;
};

export const HISTORY_MAX = 2_000;

/** Append an accepted event to the history, folding the oldest into `base` past HISTORY_MAX. */
export function appendHistory(h: History, ev: unknown): History {
  let base = h.base;
  let events = [...h.events, ev];
  while (events.length > HISTORY_MAX) {
    base = foldEvent(base, events[0]);
    events = events.slice(1);
  }
  return { base, events };
}

const freshHistory = (base: RunState): History => ({ base, events: [] });

/** The subset of the DOM EventSource this module uses (so fakes type-check). */
export interface EventSourceLike {
  addEventListener(type: string, listener: (ev: { data: string }) => void): void;
  onopen: ((ev?: unknown) => void) | null;
  onerror: ((ev?: unknown) => void) | null;
  close(): void;
}
export type EventSourceCtor = new (url: string) => EventSourceLike;

export interface FetchResponseLike {
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
  text(): Promise<string>;
}
export type FetchLike = (url: string, init?: { signal?: AbortSignal }) => Promise<FetchResponseLike>;

export type Deps = {
  EventSource?: EventSourceCtor;
  fetch?: FetchLike;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (handle: unknown) => void;
};

export type ConnectionOptions = {
  runId: string;
  /** Base URL of the API; empty/null means "no backend": play the bundled sample. */
  apiUrl?: string | null;
  token?: string | null;
  /** URL of the bundled sample journal (JSON lines). */
  sampleUrl?: string;
  /** Sample playback speed multiplier on the recorded timestamps. */
  replaySpeed?: number;
  /** Injected for tests; missing entries are taken from globalThis inside start(). */
  deps?: Deps;
};

export interface Connection {
  start(): void;
  stop(): void;
  subscribe(listener: () => void): () => void;
  getSnapshot(): RunView;
  getServerSnapshot(): RunView;
}

export const BACKOFF_MIN_MS = 1_000;
export const BACKOFF_MAX_MS = 15_000;
/** Longest pause between two sample events, whatever the recorded gap. */
export const REPLAY_MAX_GAP_MS = 2_000;
export const DEFAULT_SAMPLE_URL = "/sample-run.jsonl";

/** Delay before retry number `attempt` (0-based): 1 s, 2 s, 4 s, 8 s, then 15 s. */
export function backoffDelay(attempt: number): number {
  return Math.min(BACKOFF_MIN_MS * 2 ** Math.max(0, attempt), BACKOFF_MAX_MS);
}

const isObject = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

/** True when `base` (already trimmed, no trailing slash) is an absolute http(s) URL with a host. */
function isValidApiBase(base: string): boolean {
  try {
    const u = new URL(base);
    return (u.protocol === "http:" || u.protocol === "https:") && u.host !== "";
  } catch {
    return false;
  }
}

function statusDetail(status: number): string {
  if (status === 429) return "too many streams (429)";
  if (status === 503) return "journal unavailable (503)";
  return `HTTP ${status}`;
}

export function createConnection(opts: ConnectionOptions): Connection {
  const base = (opts.apiUrl ?? "").trim().replace(/\/+$/, "");
  const token = opts.token ?? "";
  const sampleUrl = opts.sampleUrl ?? DEFAULT_SAMPLE_URL;
  const speed = opts.replaySpeed && opts.replaySpeed > 0 ? opts.replaySpeed : 1;

  const initialState = emptyState();
  const initial: RunView = {
    state: initialState,
    mode: base ? "LIVE" : "REPLAY",
    conn: "CONNECTING",
    source: base ? "api" : "sample",
    detail: null,
    history: freshHistory(initialState),
  };
  let view = initial;
  const listeners = new Set<() => void>();

  // Lifecycle
  let epoch = 0; // bumped by stop(); async callbacks from an older epoch are dropped
  let running = false;
  let deps: Required<Pick<Deps, "fetch" | "setTimeout" | "clearTimeout">> & { EventSource: EventSourceCtor } | null = null;
  let timer: unknown = null;
  let inflight: AbortController | null = null;
  let es: EventSourceLike | null = null;

  // Progress
  let reached = false; // any HTTP response from the backend: never fall back to sample after this
  let snapshotLoaded = false;
  let authFailed = false;
  let attempt = 0;
  let sample: unknown[] | null = null;
  let sampleIndex = 0;

  function set(patch: Partial<RunView>): void {
    let changed = false;
    for (const k of Object.keys(patch) as (keyof RunView)[]) {
      if (view[k] !== patch[k]) changed = true;
    }
    if (!changed) return;
    view = { ...view, ...patch };
    for (const l of [...listeners]) l();
  }

  function url(path: string, params: Record<string, string>): string {
    const u = new URL(base + path);
    for (const [k, v] of Object.entries(params)) u.searchParams.set(k, v);
    return u.toString();
  }

  function clearTimer(): void {
    if (timer !== null && deps) deps.clearTimeout(timer);
    timer = null;
  }

  function later(ms: number, fn: () => void): void {
    clearTimer();
    const my = epoch;
    timer = deps!.setTimeout(() => {
      timer = null;
      if (my === epoch) fn();
    }, ms);
  }

  function retry(step: () => void): void {
    later(backoffDelay(attempt++), step);
  }

  /** fetch with an abort handle owned by this connection; resolves null when stopped meanwhile. */
  async function request(u: string): Promise<FetchResponseLike | "network-error" | null> {
    const my = epoch;
    const ctl = new AbortController();
    inflight = ctl;
    try {
      const r = await deps!.fetch(u, { signal: ctl.signal });
      return my === epoch ? r : null;
    } catch {
      return my === epoch ? "network-error" : null;
    } finally {
      if (inflight === ctl) inflight = null;
    }
  }

  function closeStream(): void {
    if (es) {
      es.onopen = null;
      es.onerror = null;
      es.close();
      es = null;
    }
  }

  function failAuth(): void {
    authFailed = true;
    closeStream();
    clearTimer();
    inflight?.abort();
    inflight = null;
    set({ conn: "AUTH_FAILED", detail: "invalid read token (401)" });
  }

  function reconnecting(detail: string): void {
    set({ conn: "RECONNECTING", detail });
  }

  // ---- API path -------------------------------------------------------------------------------

  async function checkHealth(): Promise<void> {
    const my = epoch;
    const r = await request(url("/v1/health", {}));
    if (r === null) return;
    if (r === "network-error") {
      if (!reached) return startSample("backend unreachable");
      reconnecting("backend unreachable");
      return retry(checkHealth);
    }
    reached = true;
    if (view.source === "sample") {
      // Only reachable when a sample load was interrupted before any sample data was shown.
      const empty = emptyState();
      set({ source: "api", mode: "LIVE", state: empty, history: freshHistory(empty), conn: "CONNECTING", detail: null });
    }
    let body: unknown = null;
    try {
      body = await r.json();
    } catch {
      /* treated as a bad status below */
    }
    if (my !== epoch) return;
    if (isObject(body) && (body.mode === "replay" || body.mode === "live")) {
      set({ mode: body.mode === "replay" ? "REPLAY" : "LIVE" });
    }
    if (!r.ok) {
      reconnecting(statusDetail(r.status));
      return retry(checkHealth);
    }
    void loadSnapshot();
  }

  async function loadSnapshot(): Promise<void> {
    const my = epoch;
    const r = await request(url(`/v1/runs/${encodeURIComponent(opts.runId)}/snapshot`, { token }));
    if (r === null) return;
    if (r === "network-error") {
      reconnecting("backend unreachable");
      return retry(loadSnapshot);
    }
    reached = true;
    if (r.status === 401) return failAuth();
    if (!r.ok) {
      reconnecting(statusDetail(r.status));
      return retry(loadSnapshot);
    }
    let body: unknown;
    try {
      body = await r.json();
    } catch {
      body = null;
    }
    if (my !== epoch) return;
    if (!isObject(body) || !isObject(body.state)) {
      reconnecting("malformed snapshot");
      return retry(loadSnapshot);
    }
    const state = { ...emptyState(), ...(body.state as Partial<RunState>) } as RunState;
    if (typeof state.last_seq !== "number" || !Number.isInteger(state.last_seq)) state.last_seq = 0;
    snapshotLoaded = true;
    if (state.last_seq >= view.state.last_seq) set({ state, history: freshHistory(state) });
    openStream();
  }

  function onFrame(data: string): void {
    let ev: unknown;
    try {
      ev = JSON.parse(data);
    } catch {
      return;
    }
    // Dedupe by seq before folding (foldEvent also drops seq <= last_seq).
    if (isObject(ev) && typeof ev.seq === "number" && ev.seq <= view.state.last_seq) return;
    const next = foldEvent(view.state, ev);
    if (next !== view.state) set({ state: next, history: appendHistory(view.history, ev) });
  }

  function openStream(): void {
    closeStream();
    const params = { run: opts.runId, since: String(view.state.last_seq), token };
    const streamUrl = url("/v1/stream", params);
    const source = new deps!.EventSource(streamUrl);
    es = source;
    let opened = false;
    source.onopen = () => {
      if (es !== source) return;
      opened = true;
      attempt = 0;
      set({ conn: "LIVE", detail: null });
    };
    for (const type of EVENT_TYPES) {
      source.addEventListener(type, (m) => {
        if (es === source) onFrame(m.data);
      });
    }
    source.onerror = () => {
      if (es !== source) return;
      closeStream(); // stop the browser's own reconnect; ours resumes from last_seq
      if (opened) {
        reconnecting("connection lost");
        return retry(openStream);
      }
      reconnecting("connecting");
      void probe(streamUrl);
    };
  }

  /** The stream failed before opening: learn why (EventSource hides HTTP status). */
  async function probe(streamUrl: string): Promise<void> {
    const my = epoch;
    const ctl = new AbortController();
    inflight = ctl;
    let status: number | null = null;
    try {
      const r = await deps!.fetch(streamUrl, { signal: ctl.signal });
      status = r.status;
    } catch {
      status = null;
    } finally {
      ctl.abort(); // never keep a probe stream (and its server slot) open
      if (inflight === ctl) inflight = null;
    }
    if (my !== epoch) return;
    if (status === 401) return failAuth();
    if (status === null) reconnecting("backend unreachable");
    else if (status >= 200 && status < 300) reconnecting("stream error");
    else reconnecting(statusDetail(status));
    retry(openStream);
  }

  // ---- sample path ----------------------------------------------------------------------------

  async function startSample(reason: string): Promise<void> {
    const my = epoch;
    const empty = emptyState();
    set({ mode: "REPLAY", source: "sample", conn: "OFFLINE", detail: reason, state: empty, history: freshHistory(empty) });
    sampleIndex = 0;
    if (sample === null) {
      const r = await request(sampleUrl);
      if (r === null) return;
      if (r === "network-error" || !r.ok) return set({ detail: "sample data unavailable" });
      let text = "";
      try {
        text = await r.text();
      } catch {
        return set({ detail: "sample data unavailable" });
      }
      if (my !== epoch) return;
      sample = parseLines(text);
    }
    playSample();
  }

  function playSample(): void {
    const events = sample ?? [];
    if (sampleIndex >= events.length) return;
    const ev = events[sampleIndex++];
    const next = foldEvent(view.state, ev);
    if (next !== view.state) set({ state: next, history: appendHistory(view.history, ev) });
    if (sampleIndex >= events.length) return;
    const gap = tsOf(events[sampleIndex]) - tsOf(ev);
    const ms = Number.isFinite(gap) ? Math.min(Math.max(gap, 0) * 1000 / speed, REPLAY_MAX_GAP_MS) : 0;
    later(ms, playSample);
  }

  // ---- lifecycle ------------------------------------------------------------------------------

  function start(): void {
    if (running || authFailed) return;
    if (base && !isValidApiBase(base)) {
      // A misconfigured NEXT_PUBLIC_API_URL: say so, request nothing, and never show sample data.
      set({ conn: "OFFLINE", detail: "invalid API URL (NEXT_PUBLIC_API_URL needs http:// or https://)" });
      return;
    }
    const g = globalThis as unknown as Record<string, unknown>;
    const injected = opts.deps ?? {};
    const ES = "EventSource" in injected ? injected.EventSource : (g.EventSource as EventSourceCtor | undefined);
    const f = injected.fetch ?? (typeof g.fetch === "function" ? (g.fetch as FetchLike).bind(globalThis) : undefined);
    if (!ES || !f) return; // server render or an environment without streaming: stay idle
    deps = {
      EventSource: ES,
      fetch: f,
      setTimeout: injected.setTimeout ?? ((fn, ms) => globalThis.setTimeout(fn, ms)),
      clearTimeout: injected.clearTimeout ?? ((h) => globalThis.clearTimeout(h as ReturnType<typeof setTimeout>)),
    };
    running = true;
    if (view.source === "sample" && (sample !== null || !base)) {
      // Resume (or begin) the sample replay; never switch a sample view to live mid-session.
      if (sample === null) void startSample(base ? "backend unreachable" : "no API configured");
      else playSample();
      return;
    }
    if (snapshotLoaded) {
      reconnecting("resuming");
      openStream();
    } else {
      void checkHealth();
    }
  }

  function stop(): void {
    epoch++;
    running = false;
    closeStream();
    clearTimer();
    inflight?.abort();
    inflight = null;
  }

  return {
    start,
    stop,
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getSnapshot: () => view,
    getServerSnapshot: () => initial,
  };
}

function tsOf(ev: unknown): number {
  return isObject(ev) && typeof ev.ts === "number" ? ev.ts : NaN;
}

function parseLines(text: string): unknown[] {
  const out: unknown[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      out.push(JSON.parse(line));
    } catch {
      /* skip a bad line; the reducer would ignore it anyway */
    }
  }
  return out;
}
