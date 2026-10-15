"use client";
import Link from "next/link";
import { useEffect, useState } from "react";

/** The run id inside the bundled public/sample-run.jsonl. */
export const SAMPLE_RUN_ID = "sample-simulated-run";

type Run = { run: string; started: number | null; last_seq: number | null; status: string | null };
type RunsMode = "LIVE" | "REPLAY" | "UNKNOWN";
const MODE_LABEL: Record<RunsMode, { text: string; cls: string }> = {
  LIVE: { text: "● LIVE", cls: "label conn-live" },
  REPLAY: { text: "REPLAY", cls: "label label-amber" },
  UNKNOWN: { text: "unknown mode", cls: "label dim" },
};
type ListState =
  | { kind: "loading" }
  | { kind: "ok"; mode: RunsMode; runs: Run[] }
  | { kind: "error"; label: "AUTH FAILED" | "OFFLINE"; detail: string };

const API = (process.env.NEXT_PUBLIC_API_URL ?? "").trim().replace(/\/+$/, "");
const TOKEN = process.env.NEXT_PUBLIC_READ_TOKEN ?? "";

function SampleLink() {
  return (
    <p>
      <span className="label label-amber">REPLAY · SAMPLE DATA</span>{" "}
      <Link href={`/run/${SAMPLE_RUN_ID}`}>{SAMPLE_RUN_ID}</Link> <span className="dim">(simulated run, stub backend)</span>
    </p>
  );
}

/** Home page run list: the API's /v1/runs, or the bundled sample when no API is configured. */
export function RunList() {
  const [list, setList] = useState<ListState>({ kind: "loading" });

  useEffect(() => {
    if (!API) return;
    const ctl = new AbortController();
    (async () => {
      try {
        const h = await fetch(`${API}/v1/health`, { signal: ctl.signal });
        const health: unknown = await h.json().catch(() => null);
        // Never assume LIVE: a failed or unparseable health answer leaves the mode unknown.
        const reported = h.ok ? (health as { mode?: unknown } | null)?.mode : undefined;
        const mode: RunsMode = reported === "replay" ? "REPLAY" : reported === "live" ? "LIVE" : "UNKNOWN";
        const r = await fetch(`${API}/v1/runs?token=${encodeURIComponent(TOKEN)}`, { signal: ctl.signal });
        if (r.status === 401) return setList({ kind: "error", label: "AUTH FAILED", detail: "invalid read token (401)" });
        if (!r.ok) return setList({ kind: "error", label: "OFFLINE", detail: `HTTP ${r.status}` });
        const body: unknown = await r.json();
        const runs = Array.isArray((body as { runs?: unknown })?.runs) ? ((body as { runs: Run[] }).runs) : [];
        setList({ kind: "ok", mode, runs: runs.filter((x) => x && typeof x.run === "string") });
      } catch {
        if (!ctl.signal.aborted) setList({ kind: "error", label: "OFFLINE", detail: "backend unreachable" });
      }
    })();
    return () => ctl.abort();
  }, []);

  if (!API) {
    return (
      <section aria-label="Runs">
        <p className="dim">No API configured (NEXT_PUBLIC_API_URL unset).</p>
        <SampleLink />
      </section>
    );
  }
  return (
    <section aria-label="Runs">
      {list.kind === "loading" ? <p className="conn-connecting">CONNECTING</p> : null}
      {list.kind === "error" ? (
        <>
          <p>
            <span className={`label conn-${list.label === "AUTH FAILED" ? "auth_failed" : "offline"}`}>{list.label}</span>{" "}
            <span className="dim">({list.detail})</span>
          </p>
          {list.label === "OFFLINE" ? <SampleLink /> : null}
        </>
      ) : null}
      {list.kind === "ok" ? (
        <>
          <p>
            <span className={MODE_LABEL[list.mode].cls}>{MODE_LABEL[list.mode].text}</span>
          </p>
          {list.runs.length === 0 ? (
            <p className="dim">no runs in the journal yet</p>
          ) : (
            <ul>
              {[...list.runs].reverse().map((r) => (
                <li key={r.run}>
                  <Link href={`/run/${encodeURIComponent(r.run)}`}>{r.run}</Link>{" "}
                  <span className="dim">
                    {r.status ?? "no status"} · seq {r.last_seq ?? "?"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </>
      ) : null}
    </section>
  );
}
