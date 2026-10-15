/** Small display helpers shared by the panels. Values come from untrusted JSON: never assume types. */
import type { Json } from "../lib/events";

export const num = (v: Json | undefined): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export const str = (v: Json | undefined): string | null => (typeof v === "string" ? v : null);

export function fixed(v: Json | undefined, digits = 2): string {
  const n = num(v);
  return n === null ? "—" : n.toFixed(digits);
}

export function usd(v: Json | undefined): string | null {
  const n = num(v);
  if (n === null) return null;
  return `$${n < 0.01 && n > 0 ? n.toFixed(4) : n.toFixed(2)}`;
}

export function pct(v: Json | undefined): string | null {
  const n = num(v);
  if (n === null) return null;
  const p = Math.round(n * 100);
  return `${p >= 0 ? "+" : ""}${p}%`;
}

export function ms(v: Json | undefined): string {
  const n = num(v);
  if (n === null) return "—";
  return n >= 1000 ? `${(n / 1000).toFixed(1)}s` : `${Math.round(n)}ms`;
}

/** HH:MM:SS in the viewer's local time. */
export function clock(ts: number): string {
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "--:--:--";
  const p = (x: number) => String(x).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/**
 * Evidence label for a score or job. EXPLANATION means simulated (stub backend or a narrative
 * estimate): badged SIMULATED in amber. Anything else is shown as given; missing is "NO EVIDENCE".
 */
export function EvidenceBadge({ evidence }: { evidence: Json | undefined }) {
  if (evidence === "EXPLANATION") {
    return (
      <span className="badge badge-sim" title="evidence: EXPLANATION (simulated, not measured)">
        SIMULATED
      </span>
    );
  }
  const label = typeof evidence === "string" && evidence ? evidence : "NO EVIDENCE";
  return (
    <span className={label === "LIVE" ? "badge badge-live" : "badge"} title={`evidence: ${label}`}>
      {label}
    </span>
  );
}
