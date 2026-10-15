// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import type { RunView } from "../lib/connection";
import type { RunState } from "../lib/events";
import { emptyState, foldEvent } from "../lib/reducer";
import { Boot } from "./Boot";
import { DashboardView } from "./Dashboard";
import { ANNOUNCE_MS, EventStream } from "./EventStream";
import { CountUp, REDUCED_MOTION_QUERY } from "./motion";

const CSS = readFileSync(join(__dirname, "..", "styles", "terminal.css"), "utf8");
const SAMPLE: unknown[] = readFileSync(join(__dirname, "..", "public", "sample-run.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));

const foldAll = (events: readonly unknown[], base: RunState = emptyState()): RunState => events.reduce<RunState>(foldEvent, base);
let n = 5000;
const ev = (type: string, data: Record<string, unknown>) => ({ seq: ++n, ts: 1_760_000_000 + n, run: "t", type, data });

function viewOf(events: readonly unknown[] = SAMPLE, over: Partial<RunView> = {}): RunView {
  const base = emptyState();
  return { state: foldAll(events, base), mode: "REPLAY", conn: "OFFLINE", source: "sample", detail: "no API configured", history: { base, events }, ...over };
}

/** Installs window.matchMedia answering `prefers-reduced-motion` with `reduce`. */
function setReducedMotion(reduce: boolean) {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn((q: string) => ({
      matches: q === REDUCED_MOTION_QUERY ? reduce : false,
      media: q,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => true,
    })),
  });
}

/** Plain CSS rules (selector, body), including those nested in @media blocks. */
function cssRules(css: string): { sel: string; body: string }[] {
  const clean = css.replace(/\/\*[\s\S]*?\*\//g, "");
  return [...clean.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ sel: m[1].trim(), body: m[2] }));
}

function mediaBlock(css: string, query: string): string {
  const clean = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const at = clean.indexOf(`@media ${query}`);
  expect(at).toBeGreaterThanOrEqual(0);
  let depth = 0;
  const open = clean.indexOf("{", at);
  for (let i = open; i < clean.length; i++) {
    if (clean[i] === "{") depth++;
    if (clean[i] === "}" && --depth === 0) return clean.slice(open + 1, i);
  }
  throw new Error("unterminated @media block");
}

let getContext: MockInstance;
beforeEach(() => {
  getContext = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
});
afterEach(() => {
  cleanup();
  getContext.mockRestore();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  delete (window as { matchMedia?: unknown }).matchMedia;
});

describe("motion and accessibility", () => {
  it("all_animations_disabled_under_prefers_reduced_motion", () => {
    // CSS: one block switches off every animation and transition, pseudo-elements included.
    const block = mediaBlock(CSS, "(prefers-reduced-motion: reduce)");
    const universal = cssRules(block).find((r) => r.sel.replace(/\s+/g, "") === "*,*::before,*::after");
    expect(universal).toBeDefined();
    expect(universal!.body).toMatch(/animation:\s*none\s*!important/);
    expect(universal!.body).toMatch(/transition:\s*none\s*!important/);
    // Animated elements rest in their final, readable state: the boot cover is parked off the text.
    const cover = cssRules(CSS).find((r) => r.sel === ".boot-cover")!;
    expect(cover.body).toMatch(/transform:\s*translateX\(100%\)/);
    // Honesty labels and badges are never animated.
    for (const r of cssRules(CSS)) {
      if (/\.(label|badge|conn)\b/.test(r.sel)) expect(r.body, r.sel).not.toMatch(/animation/);
    }
    // Only compositor-friendly properties are animated.
    const clean = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
    for (const kf of clean.matchAll(/@keyframes\s+([\w-]+)\s*\{([\s\S]*?\}\s*)\}/g)) {
      const props = [...kf[2].matchAll(/([\w-]+)\s*:/g)].map((m) => m[1]);
      for (const p of props) expect(["transform", "opacity"], `@keyframes ${kf[1]}`).toContain(p);
    }

    // JS: with reduced motion the typewriter and count-up render their final values at once.
    setReducedMotion(true);
    const before = SAMPLE;
    const after = [...SAMPLE, ev("score", { split: "heldout", checkpoint: "ft:z", value: 0.91, n: 50, evidence: "LIVE" })];
    const { container, rerender } = render(<DashboardView view={viewOf(before)} runId="r1" />);
    const boot = container.querySelector(".boot")!;
    expect(boot.getAttribute("data-typing")).toBe("false");
    expect(container.querySelector(".boot-cover")).toBeNull();
    rerender(<DashboardView view={viewOf(after)} runId="r1" />);
    const value = container.querySelector(".score-value")!;
    expect(value.querySelector('[aria-hidden="true"]')!.textContent).toBe("0.91");
    expect(value.getAttribute("data-final")).toBe("0.91");
  });

  it("count_up_skips_to_final_value_under_reduced_motion", async () => {
    setReducedMotion(true);
    const { container, rerender } = render(<CountUp value={0.5} />);
    const shown = () => container.querySelector('[aria-hidden="true"]')!.textContent;
    expect(shown()).toBe("0.50");
    rerender(<CountUp value={0.9} />);
    expect(shown()).toBe("0.90");
    cleanup();

    // With motion: first render is already final (never counts up from zero); a change counts,
    // but the final value is in the visually-hidden text immediately.
    setReducedMotion(false);
    const r = render(<CountUp value={0.5} />);
    const shown2 = () => r.container.querySelector('[aria-hidden="true"]')!.textContent;
    expect(shown2()).toBe("0.50");
    r.rerender(<CountUp value={0.9} />);
    expect(r.container.querySelector(".sr-only")!.textContent).toBe("0.90");
    await waitFor(() => expect(shown2()).toBe("0.90"), { timeout: 2000 });

    // Not animated while scrubbing.
    r.rerender(<CountUp value={0.3} animate={false} />);
    expect(shown2()).toBe("0.30");
  });

  it("boot_typewriter_exposes_full_text_and_types_only_with_motion", () => {
    const text = "> mission control :: thread r1 :: panels online";
    setReducedMotion(false);
    const { container } = render(<Boot text={text} />);
    // The whole line is in the DOM from the first frame; the moving cover is decoration.
    expect(container.querySelector(".boot-line")!.textContent).toBe(text);
    screen.getByText(text);
    const cover = container.querySelector(".boot-cover")!;
    expect(cover.getAttribute("aria-hidden")).toBe("true");
    expect(cover.textContent).toBe("");
    expect(container.querySelector(".boot")!.getAttribute("data-typing")).toBe("true");
    expect((container.querySelector(".boot-text") as HTMLElement).style.getPropertyValue("--chars")).toBe(String(text.length));
    cleanup();

    setReducedMotion(true);
    const r = render(<Boot text={text} />);
    screen.getByText(text);
    expect(r.container.querySelector(".boot-cover")).toBeNull();
    expect(r.container.querySelector(".boot")!.getAttribute("data-typing")).toBe("false");
  });

  it("event_log_uses_aria_live_polite_and_is_throttled", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    let s = foldAll([ev("run.started", { thread_id: "t" }), ev("score", { split: "heldout", checkpoint: "base", value: 0.5, n: 5, evidence: "LIVE" })]);
    const { container, rerender } = render(<EventStream state={s} />);
    const region = screen.getByRole("region", { name: "Event stream" });
    const live = region.querySelector('[aria-live="polite"]')!;
    expect(live).not.toBeNull();
    expect(live.getAttribute("aria-atomic")).toBe("true");
    // The visible rows are not a live region (that would read every event).
    const list = container.querySelector(".stream-list")!;
    expect(list.getAttribute("role")).toBeNull();
    expect(list.closest("[aria-live]")).toBeNull();
    expect(region.querySelectorAll("[aria-live]")).toHaveLength(1);
    // The backlog present on mount is not read out.
    expect(live.textContent).toBe("");

    // First new event: announced at once.
    s = foldEvent(s, ev("checkpoint", { checkpoint_id: "ck-1" }));
    rerender(<EventStream state={s} />);
    expect(live.textContent).toMatch(/^checkpoint/);
    const first = live.textContent;

    // A burst inside the window: nothing more until the window ends, then one summary.
    const texts = new Set<string>();
    for (let i = 0; i < 5; i++) {
      s = foldEvent(s, ev("checkpoint", { checkpoint_id: `ck-burst-${i}` }));
      rerender(<EventStream state={s} />);
      texts.add(live.textContent ?? "");
      act(() => vi.advanceTimersByTime(100));
    }
    expect([...texts]).toEqual([first]);
    act(() => vi.advanceTimersByTime(ANNOUNCE_MS));
    expect(live.textContent).toBe("5 new events, latest: checkpoint checkpoint_id=ck-burst-4");

    // While scrubbing (not live) nothing is announced.
    act(() => vi.advanceTimersByTime(ANNOUNCE_MS * 2));
    const settled = live.textContent;
    s = foldEvent(s, ev("checkpoint", { checkpoint_id: "ck-hist" }));
    rerender(<EventStream state={s} live={false} />);
    act(() => vi.advanceTimersByTime(ANNOUNCE_MS * 2));
    expect(live.textContent).toBe(settled);
  });

  it("blips_off_by_default", () => {
    const made: unknown[] = [];
    const osc = vi.fn();
    class FakeAudioContext {
      currentTime = 0;
      destination = {};
      constructor() {
        made.push(this);
      }
      resume = vi.fn(() => Promise.resolve());
      suspend = vi.fn(() => Promise.resolve());
      close = vi.fn(() => Promise.resolve());
      createGain() {
        return { gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() }, connect: (d: unknown) => d };
      }
      createOscillator() {
        osc();
        const gainNode = this.createGain();
        return { type: "sine", frequency: { value: 0 }, connect: () => gainNode, start: vi.fn(), stop: vi.fn() };
      }
    }
    vi.stubGlobal("AudioContext", FakeAudioContext);

    const events = [...SAMPLE];
    const { rerender } = render(<DashboardView view={viewOf(events)} runId="r1" />);
    const toggle = screen.getByRole("button", { name: /terminal sound blips/i });
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    expect(toggle.textContent).toMatch(/blips off/);

    // New events while muted: no AudioContext is ever created (no autoplay).
    events.push(ev("checkpoint", { checkpoint_id: "ck-a" }));
    rerender(<DashboardView view={viewOf([...events])} runId="r1" />);
    expect(made).toHaveLength(0);
    expect(osc).not.toHaveBeenCalled();

    // The user's click creates it; then a new event blips.
    fireEvent.click(toggle);
    expect(made).toHaveLength(1);
    expect(toggle.getAttribute("aria-pressed")).toBe("true");
    within(toggle).getByText(/blips on/);
    events.push(ev("checkpoint", { checkpoint_id: "ck-b" }));
    rerender(<DashboardView view={viewOf([...events])} runId="r1" />);
    expect(osc).toHaveBeenCalledTimes(1);

    // Off again: silent, and no second context.
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    events.push(ev("checkpoint", { checkpoint_id: "ck-c" }));
    rerender(<DashboardView view={viewOf([...events])} runId="r1" />);
    expect(osc).toHaveBeenCalledTimes(1);
    expect(made).toHaveLength(1);
  });

  it("body_text_contrast_is_at_least_4_5_to_1", () => {
    const token = (name: string): string => {
      const m = CSS.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
      if (!m) throw new Error(`token --${name} missing`);
      return m[1];
    };
    const lum = (hex: string) => {
      const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const ratio = (a: string, b: string) => {
      const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
      return (x + 0.05) / (y + 0.05);
    };
    const pairs: [string, string][] = [];
    for (const fg of ["green", "green-dim", "idle", "amber", "cyan", "red", "focus"]) {
      pairs.push([fg, "bg"], [fg, "bg-glow"]);
    }
    pairs.push(["amber", "amber-bg"], ["bg", "red"]); // approval banner / RECONNECTING; AUTH FAILED
    for (const [fg, bg] of pairs) {
      expect(ratio(token(fg), token(bg)), `--${fg} on --${bg}`).toBeGreaterThanOrEqual(4.5);
    }
  });
});
