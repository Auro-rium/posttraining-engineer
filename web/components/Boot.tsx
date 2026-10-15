"use client";
import { useReducedMotion } from "./motion";

/**
 * Boot line: a typewriter header that runs once per mount. The full text is always in the DOM
 * (readable without the animation and to screen readers); an aria-hidden cover in the background
 * colour slides off it one character at a time (transform + steps(), no layout work), carrying the
 * block cursor at its leading edge. Under reduced motion the cover is not rendered at all, and the
 * CSS reduced-motion block disables it too before hydration.
 */
export function Boot({ text, className }: { text: string; className?: string }) {
  // Server render assumes motion is allowed so the cover is there on first paint (no flash of the
  // full line before it types); reduced-motion viewers are covered by CSS until hydration.
  const reduced = useReducedMotion(false);
  return (
    <div className={className ? `boot ${className}` : "boot"} data-typing={reduced ? "false" : "true"}>
      <span className="boot-text" style={{ ["--chars" as string]: Math.max(1, text.length) }}>
        <span className="boot-line">{text}</span>
        {reduced ? null : <span className="boot-cover" aria-hidden="true" />}
      </span>
    </div>
  );
}
