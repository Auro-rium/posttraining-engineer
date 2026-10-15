"use client";
import { useEffect, useRef } from "react";
import type { History } from "../lib/connection";
import type { Json, JsonObject, RunState } from "../lib/events";
import { ms, num } from "./format";

type Series = { p50: number[]; p95: number[]; err: number[] };

const isObj = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === "object" && !Array.isArray(v);

/**
 * The reducer keeps only the latest telemetry, so the series is rebuilt from the history's
 * `telemetry.snapshot` events (plus the snapshot base, if it had telemetry).
 */
export function telemetrySeries(history: History): Series {
  const out: Series = { p50: [], p95: [], err: [] };
  const push = (d: JsonObject | Record<string, unknown>) => {
    const p50 = num(d.p50_ms as Json);
    const p95 = num(d.p95_ms as Json);
    const err = num(d.error_rate as Json);
    if (p50 !== null) out.p50.push(p50);
    if (p95 !== null) out.p95.push(p95);
    if (err !== null) out.err.push(err);
  };
  if (Object.keys(history.base.telemetry).length > 0) push(history.base.telemetry);
  for (const ev of history.events) {
    if (isObj(ev) && ev.type === "telemetry.snapshot" && isObj(ev.data)) push(ev.data);
  }
  return out;
}

/**
 * Canvas sparkline. Draws only in an effect, so server render emits an empty canvas, and it does
 * nothing where 2D canvas is unavailable (jsdom). The numbers are always also rendered as text.
 */
export function Sparkline({ values, label, color = "#7dff9b" }: { values: number[]; label: string; color?: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || values.length === 0) return;
    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = typeof canvas.getContext === "function" ? canvas.getContext("2d") : null;
    } catch {
      ctx = null;
    }
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const span = hi - lo || 1;
    const x = (i: number) => (values.length === 1 ? w / 2 : (i / (values.length - 1)) * (w - 2) + 1);
    const y = (v: number) => h - 2 - ((v - lo) / span) * (h - 4);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    if (values.length === 1) {
      ctx.fillRect(x(0) - 1.5, h / 2 - 1.5, 3, 3);
      return;
    }
    ctx.beginPath();
    values.forEach((v, i) => (i === 0 ? ctx!.moveTo(x(i), y(v)) : ctx!.lineTo(x(i), y(v))));
    ctx.stroke();
  }, [values, color]);
  return <canvas ref={ref} className="sparkline" width={120} height={24} role="img" aria-label={`${label} sparkline, ${values.length} points`} />;
}

export function Telemetry({ state, history }: { state: RunState; history: History }) {
  const t = state.telemetry;
  const series = telemetrySeries(history);
  const err = num(t.error_rate);
  const tin = num(t.input_tokens);
  const tout = num(t.output_tokens);
  const has = Object.keys(t).length > 0;
  return (
    <section className="panel bottom telemetry" aria-label="Telemetry">
      <h2 className="panel-title">TELEMETRY</h2>
      {!has ? (
        <p className="dim">no telemetry yet</p>
      ) : (
        <>
          <div>
            p50 {ms(t.p50_ms)} · p95 {ms(t.p95_ms)} · err {err !== null ? `${Math.round(err * 100)}%` : "—"}
            {tin !== null || tout !== null ? ` · tokens ${tin ?? "—"} in / ${tout ?? "—"} out` : ""}
          </div>
          <div className="sparklines">
            {series.p50.length > 0 ? <Sparkline values={series.p50} label="p50 latency" /> : null}
            {series.p95.length > 0 ? <Sparkline values={series.p95} label="p95 latency" color="#ffb000" /> : null}
            {series.err.length > 0 ? <Sparkline values={series.err} label="error rate" color="#ff5a5a" /> : null}
          </div>
        </>
      )}
    </section>
  );
}
