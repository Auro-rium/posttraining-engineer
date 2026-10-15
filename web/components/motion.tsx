"use client";
/**
 * Motion helpers. Every JS-driven animation reads `useReducedMotion()` and jumps straight to its
 * final value when the viewer asked for reduced motion (CSS animations are switched off by the
 * `prefers-reduced-motion` block in styles/terminal.css). Final values are always in the DOM, so no
 * text depends on an animation finishing to be readable.
 */
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

export const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

function mediaQuery(): MediaQueryList | null {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return null;
  try {
    return window.matchMedia(REDUCED_MOTION_QUERY);
  } catch {
    return null;
  }
}

/** True when the viewer prefers reduced motion, or when that cannot be determined. */
export function prefersReducedMotion(): boolean {
  const m = mediaQuery();
  return m ? m.matches : true;
}

function subscribe(onChange: () => void): () => void {
  const m = mediaQuery();
  if (!m) return () => {};
  if (typeof m.addEventListener === "function") {
    m.addEventListener("change", onChange);
    return () => m.removeEventListener("change", onChange);
  }
  m.addListener?.(onChange);
  return () => m.removeListener?.(onChange);
}

/**
 * Live `prefers-reduced-motion`. `serverDefault` is what server render and hydration assume
 * (default: reduced, so the first paint shows final values).
 */
export function useReducedMotion(serverDefault = true): boolean {
  return useSyncExternalStore(subscribe, prefersReducedMotion, () => serverDefault);
}

export const COUNT_UP_MS = 650;

type CountUpProps = {
  value: number | null;
  digits?: number;
  /** False while scrubbing: jump instead of counting. */
  animate?: boolean;
  className?: string;
};

/**
 * A number that counts from its previous value to the new one. The first render (and server
 * render) shows the final value; it never counts up from zero. Screen readers and text queries get
 * the final value from visually-hidden text; the moving digits are aria-hidden.
 */
export function CountUp({ value, digits = 2, animate = true, className }: CountUpProps) {
  const reduced = useReducedMotion();
  const final = value === null ? "—" : value.toFixed(digits);
  const [shown, setShown] = useState<number | null>(value);
  const last = useRef<number | null>(value);

  useEffect(() => {
    const from = last.current;
    const canAnimate =
      animate && !reduced && value !== null && from !== null && from !== value && typeof requestAnimationFrame === "function";
    if (!canAnimate) {
      last.current = value;
      setShown(value);
      return;
    }
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    let raf = 0;
    const step = (now: number) => {
      const p = Math.min(1, Math.max(0, (now - t0) / COUNT_UP_MS));
      const eased = 1 - (1 - p) ** 3;
      const v = p >= 1 ? value : from + (value - from) * eased;
      last.current = v;
      setShown(v);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    // A count interrupted by a newer value continues from where the digits are now.
    return () => cancelAnimationFrame(raf);
  }, [value, reduced, animate]);

  const display = reduced || shown === null || value === null ? final : shown.toFixed(digits);
  return (
    <span className={className ? `countup ${className}` : "countup"} data-final={final}>
      <span aria-hidden="true">{display}</span>
      <span className="sr-only">{final}</span>
    </span>
  );
}
