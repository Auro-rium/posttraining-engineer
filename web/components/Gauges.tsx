/**
 * Telemetry gauges (inline SVG). Values come from untrusted JSON: a missing or non-finite value is
 * drawn as "no data", never as zero; a value outside the range is clamped for the drawing only and
 * the real number is printed as text. The arcs are static strokes set to the final value; motion
 * is a CSS transition on the needle / fill transform only (off under reduced motion).
 */
import type { CSSProperties, ReactNode } from "react";

export const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);

/** `value` mapped to 0..1 of [min, max], clamped; null when it cannot be drawn. */
export function ratioOf(value: number | null | undefined, min = 0, max = 1): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || !(max > min)) return null;
  return clamp01((value - min) / (max - min));
}

export const NO_DATA = "no data";

/** The 270° arc, from 135° (bottom left) clockwise. pathLength=100 keeps the dash maths simple. */
const SWEEP = 75; // percent of the circle's pathLength covered by the 270° track

function Arc({ r, ratio, className }: { r: number; ratio: number; className: string }) {
  return (
    <circle
      className={className}
      cx={60}
      cy={60}
      r={r}
      pathLength={100}
      fill="none"
      strokeDasharray={`${(ratio * SWEEP).toFixed(2)} 100`}
      transform="rotate(135 60 60)"
    />
  );
}

export type Ring = { ratio: number | null; tone: string; label: string };

type RadialProps = {
  /** Outermost first. A ring with ratio null draws its track only. */
  rings: Ring[];
  /** Big centre text (already formatted) and a caption under it. */
  center: ReactNode;
  caption?: ReactNode;
  label: string;
  className?: string;
};

/** Radial gauge with up to three concentric rings and a needle on the outer ring. */
export function RadialGauge({ rings, center, caption, label, className }: RadialProps) {
  const outer = rings[0]?.ratio ?? null;
  return (
    <figure className={className ? `gauge gauge-radial ${className}` : "gauge gauge-radial"} role="img" aria-label={label}>
      <svg viewBox="0 0 120 120" aria-hidden="true" focusable="false">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <line
            key={t}
            className="gauge-tick"
            x1={60}
            y1={6}
            x2={60}
            y2={11}
            transform={`rotate(${-135 + t * 270} 60 60)`}
          />
        ))}
        {rings.map((ring, i) => {
          const r = 48 - i * 11;
          return (
            <g key={i} className={`gauge-ring tone-${ring.tone}`}>
              <Arc r={r} ratio={1} className="gauge-track" />
              {ring.ratio !== null && ring.ratio > 0 ? <Arc r={r} ratio={ring.ratio} className="gauge-value" /> : null}
            </g>
          );
        })}
        {outer !== null ? (
          <g className={`gauge-needle tone-${rings[0].tone}`} style={{ transform: `rotate(${-135 + outer * 270}deg)` } as CSSProperties}>
            <circle cx={60} cy={12} r={3.2} />
          </g>
        ) : null}
      </svg>
      <figcaption className="gauge-center">
        <span className="gauge-big">{center}</span>
        {caption ? <span className="gauge-cap">{caption}</span> : null}
      </figcaption>
    </figure>
  );
}

type BarProps = {
  ratio: number | null;
  tone: string;
  label: string;
  /** role=meter attributes in the caller's units. */
  min: number;
  max: number;
  now: number | null;
  /** Small markers (0..1) drawn over the track, e.g. a base score. */
  marks?: { at: number; label: string }[];
};

/** Horizontal bar gauge; the fill is scaled with transform (never width). */
export function BarGauge({ ratio, tone, label, min, max, now, marks = [] }: BarProps) {
  if (ratio === null || now === null) return <div className="bar bar-empty dim">{NO_DATA}</div>;
  return (
    <div className={`bar bar-gauge tone-${tone}`} role="meter" aria-label={label} aria-valuemin={min} aria-valuemax={max} aria-valuenow={now}>
      <div className="bar-fill" style={{ transform: `scaleX(${ratio.toFixed(4)})` }} />
      {[0.25, 0.5, 0.75].map((t) => (
        <span key={t} className="bar-tick" style={{ left: `${t * 100}%` }} aria-hidden="true" />
      ))}
      {marks.map((m) => (
        <span key={m.label} className="bar-mark" style={{ left: `${clamp01(m.at) * 100}%` }} title={m.label} aria-hidden="true" />
      ))}
    </div>
  );
}

type DialProps = { value: number | null; label: string; tone: string; text: string };

/** A half-circle dial (0..1) with a rotating needle; "no data" when value is null. */
export function Dial({ value, label, tone, text }: DialProps) {
  const r = ratioOf(value);
  return (
    <div
      className={`gauge dial tone-${tone}`}
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={1}
      {...(value !== null && r !== null ? { "aria-valuenow": value, "aria-valuetext": text } : { "aria-valuetext": NO_DATA })}
    >
      <svg viewBox="0 0 100 58" aria-hidden="true" focusable="false">
        <path className="dial-track" d="M8 52 A42 42 0 0 1 92 52" pathLength={100} />
        {r !== null && r > 0 ? <path className="dial-value" d="M8 52 A42 42 0 0 1 92 52" pathLength={100} strokeDasharray={`${(r * 100).toFixed(2)} 100`} /> : null}
        {r !== null ? (
          <g className="dial-needle" style={{ transform: `rotate(${-90 + r * 180}deg)` }}>
            <line x1={50} y1={52} x2={50} y2={18} />
            <circle cx={50} cy={52} r={3.5} />
          </g>
        ) : null}
      </svg>
      <span className="dial-text">{r !== null ? text : NO_DATA}</span>
    </div>
  );
}

type ProgressRingProps = { done: number | null; total: number | null; label: string };

/** Job progress ring (trained_steps / total_steps). Missing or zero total: "no data". */
export function ProgressRing({ done, total, label }: ProgressRingProps) {
  const r = done !== null && total !== null && total > 0 ? clamp01(done / total) : null;
  const inner = (
    <svg viewBox="0 0 120 120" aria-hidden="true" focusable="false">
      <circle className="ring-track" cx={60} cy={60} r={46} pathLength={100} />
      {r !== null && r > 0 ? (
        <circle className="ring-value" cx={60} cy={60} r={46} pathLength={100} strokeDasharray={`${(r * 100).toFixed(2)} 100`} transform="rotate(-90 60 60)" />
      ) : null}
      {r !== null ? (
        <g className="ring-head" style={{ transform: `rotate(${r * 360}deg)` }}>
          <circle cx={60} cy={14} r={4} />
        </g>
      ) : null}
    </svg>
  );
  return r !== null ? (
    <div className="gauge ring" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={total!} aria-valuenow={done!}>
      {inner}
      <span className="ring-text">{Math.round(r * 100)}%</span>
    </div>
  ) : (
    <div className="gauge ring ring-empty">
      {inner}
      <span className="ring-text dim">{NO_DATA}</span>
    </div>
  );
}
