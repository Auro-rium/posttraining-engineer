"use client";
import { useEffect, useMemo, useRef } from "react";
import type { History } from "../lib/connection";
import type { Json, JsonObject, RunState } from "../lib/events";
import { ms, num } from "./format";
import { Dial } from "./Gauges";
import { CountUp } from "./motion";

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
export function Sparkline({ values, label, color = "#3ef0ff" }: { values: number[]; label: string; color?: string }) {
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
    ctx.lineWidth = 2;
    if (values.length === 1) {
      ctx.fillRect(x(0) - 2, h / 2 - 2, 4, 4);
      return;
    }
    ctx.beginPath();
    values.forEach((v, i) => (i === 0 ? ctx!.moveTo(x(i), y(v)) : ctx!.lineTo(x(i), y(v))));
    ctx.stroke();
  }, [values, color]);
  return <canvas ref={ref} className="sparkline" width={180} height={40} role="img" aria-label={`${label} sparkline, ${values.length} points`} />;
}

function Spark({ title, values, color, now }: { title: string; values: number[]; color: string; now: string }) {
  return (
    <div className="spark-card">
      <div className="spark-head">
        <span className="spark-title">{title}</span> <span className="spark-now">{now}</span>
      </div>
      {values.length > 0 ? <Sparkline values={values} label={title} color={color} /> : <div className="spark-empty dim">no data</div>}
    </div>
  );
}

export function Telemetry({ state, history, live = true }: { state: RunState; history: History; live?: boolean }) {
  const t = state.telemetry;
  // Memoized so the arrays keep their identity and the sparkline effects only redraw on change.
  const series = useMemo(() => telemetrySeries(history), [history]);
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
          <div className="tele-grid">
            <Dial value={err} label="Error rate" tone={err !== null && err >= 0.05 ? "red" : "lime"} text={err !== null ? `${(err * 100).toFixed(1)}%` : ""} />
            <div className="sparklines">
              <Spark title="p50 latency" values={series.p50} color="#3ef0ff" now={ms(t.p50_ms)} />
              <Spark title="p95 latency" values={series.p95} color="#ff5cf0" now={ms(t.p95_ms)} />
              <Spark title="error rate" values={series.err} color="#ff5a6e" now={err !== null ? `${(err * 100).toFixed(1)}%` : "—"} />
            </div>
            <dl className="tokens">
              <div>
                <dt>TOKENS IN</dt>
                <dd>{tin !== null ? <CountUp value={tin} digits={0} animate={live} /> : <span className="dim">no data</span>}</dd>
              </div>
              <div>
                <dt>TOKENS OUT</dt>
                <dd>{tout !== null ? <CountUp value={tout} digits={0} animate={live} /> : <span className="dim">no data</span>}</dd>
              </div>
            </dl>
          </div>
        </>
      )}
    </section>
  );
}
