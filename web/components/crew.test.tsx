// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import type { RunState } from "../lib/events";
import { emptyState, foldEvent } from "../lib/reducer";
import { AgentBoard, BUBBLE_MAX, bubbleText, poseOf } from "./AgentBoard";
import { Budget } from "./Budget";
import { DashboardView } from "./Dashboard";
import { BarGauge, Dial, NO_DATA, ProgressRing, RadialGauge, ratioOf } from "./Gauges";
import { JobPanel } from "./JobPanel";
import { REDUCED_MOTION_QUERY } from "./motion";
import { ROLES } from "./Plush";
import { ScoreGate } from "./ScoreGate";
import { Telemetry } from "./Telemetry";

const CSS = readFileSync(join(__dirname, "..", "styles", "terminal.css"), "utf8");
const SAMPLE: unknown[] = readFileSync(join(__dirname, "..", "public", "sample-run.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));

let n = 9000;
const ev = (type: string, data: Record<string, unknown>) => ({ seq: ++n, ts: 1_760_000_000 + n, run: "t", type, data });
const foldAll = (events: readonly unknown[], base: RunState = emptyState()): RunState => events.reduce<RunState>(foldEvent, base);

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

const card = (role: string): HTMLElement => {
  const board = screen.getByRole("region", { name: /agents/i });
  const li = within(board)
    .getAllByRole("listitem")
    .find((el) => el.querySelector(".crew-name")?.textContent === role);
  if (!li) throw new Error(`no card for ${role}`);
  return li;
};
const poseIn = (role: string) => card(role).querySelector("svg.plush")!.getAttribute("data-pose");

let getContext: MockInstance;
beforeEach(() => {
  getContext = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
});
afterEach(() => {
  cleanup();
  getContext.mockRestore();
  delete (window as { matchMedia?: unknown }).matchMedia;
});

describe("plush crew", () => {
  it("poses follow agent events: asleep, working, done, failed, ended", () => {
    const s = foldAll([
      ev("agent.start", { role: "evaluator", slot: 1 }),
      ev("agent.start", { role: "benchmark", slot: 2 }),
      ev("agent.start", { role: "researcher", slot: 1 }),
      ev("agent.end", { role: "researcher", slot: 1, ok: true }),
      ev("agent.start", { role: "data_curator", slot: 1 }),
      ev("agent.end", { role: "data_curator", slot: 1, ok: false }),
      ev("agent.start", { role: "training_designer", slot: 1 }),
      ev("agent.end", { role: "training_designer", slot: 1 }),
    ]);
    render(<AgentBoard state={s} />);
    expect(poseIn("evaluator")).toBe("working");
    expect(poseIn("benchmark")).toBe("working");
    expect(poseIn("researcher")).toBe("done");
    expect(poseIn("data_curator")).toBe("failed");
    expect(poseIn("training_designer")).toBe("idle");
    expect(poseIn("champion_manager")).toBe("asleep");
    // Two concurrent agents both show as working, each with its slot.
    within(card("evaluator")).getByText("slot 1");
    within(card("benchmark")).getByText("slot 2");
    // Every plush is decorative; each card has a text equivalent of its state.
    for (const svg of document.querySelectorAll("svg.plush")) expect(svg.getAttribute("aria-hidden")).toBe("true");
    expect(card("data_curator").querySelector(".sr-only")!.textContent).toMatch(/data_curator \(fox\): finished not ok/);
    expect(card("champion_manager").querySelector(".sr-only")!.textContent).toMatch(/not seen yet/);
    // Nine distinct creatures, one per role, coordinator first.
    const roles = [...document.querySelectorAll("svg.plush")].map((e) => e.getAttribute("data-role"));
    expect(roles).toEqual(["coordinator", ...ROLES]);
  });

  it("a2a messages fill bubbles and wake a role to standby, but never make it cheer or slump", () => {
    expect(poseOf(undefined, true)).toBe("standby");
    expect(poseOf(undefined, false)).toBe("asleep");
    const s = foldAll([
      ev("a2a.message", { from: "coordinator", to: "evaluator", kind: "brief", text: "check it" }),
      ev("a2a.message", { from: "evaluator", to: "coordinator", kind: "result", ok: false, text: "refused" }),
      ev("a2a.message", { from: "champion_manager", to: "coordinator", kind: "result", ok: true, text: "unchanged" }),
    ]);
    render(<AgentBoard state={s} />);
    expect(poseIn("evaluator")).toBe("standby");
    expect(poseIn("champion_manager")).toBe("standby");
    within(card("evaluator")).getByText("refused");
    within(card("evaluator")).getByText("reply ✗");
    // The coordinator's bubble is its newest brief.
    const coord = card("coordinator");
    within(coord).getByText("check it");
    within(coord).getByText("to evaluator");
  });

  it("long bubble text is cut by code points and expands on click; full text in the title", () => {
    const long = `${"😀".repeat(50)} ${"word ".repeat(40)}`;
    const cut = bubbleText(long);
    expect([...cut.short].length).toBe(BUBBLE_MAX);
    expect(cut.short.endsWith("…")).toBe(true);
    render(<AgentBoard state={foldAll([ev("a2a.message", { from: "researcher", to: "coordinator", kind: "result", ok: true, text: long })])} />);
    const btn = within(card("researcher")).getByRole("button");
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    expect(btn.getAttribute("title")).toBe(long);
    expect(btn.querySelector(".crew-bubble-text")!.textContent).toBe(cut.short);
    fireEvent.click(btn);
    expect(btn.getAttribute("aria-expanded")).toBe("true");
    expect(btn.querySelector(".crew-bubble-text")!.textContent).toBe(long);
    // Bubbles are never live regions.
    expect(card("researcher").querySelector("[aria-live]")).toBeNull();
  });

  it("model text with markup stays text", () => {
    const evil = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
    render(<AgentBoard state={foldAll([ev("a2a.message", { from: "benchmark", to: "coordinator", kind: "result", text: evil })])} />);
    const c = card("benchmark");
    expect(c.querySelector("img")).toBeNull();
    expect(c.querySelector("script")).toBeNull();
    expect(c.querySelector(".crew-bubble-text")!.textContent).toBe(evil);
  });

  it("a packet flies only for a new a2a message while live, never for the backlog or under reduced motion", () => {
    setReducedMotion(false);
    const brief = ev("a2a.message", { from: "coordinator", to: "evaluator", kind: "brief", text: "go" });
    let s = foldAll([brief]);
    const { container, rerender } = render(<AgentBoard state={s} />);
    expect(container.querySelector(".crew-packet")).toBeNull(); // backlog on mount
    s = foldEvent(s, ev("a2a.message", { from: "evaluator", to: "coordinator", kind: "result", ok: true, text: "done" }));
    rerender(<AgentBoard state={s} />);
    const up = container.querySelector(".crew-packet")!;
    expect(up.className).toMatch(/crew-packet-up/);
    expect(up.closest("li")!.querySelector(".crew-name")!.textContent).toBe("evaluator");
    s = foldEvent(s, ev("a2a.message", { from: "coordinator", to: "benchmark", kind: "brief", text: "next" }));
    rerender(<AgentBoard state={s} />);
    expect(container.querySelector(".crew-packet-down")!.closest("li")!.querySelector(".crew-name")!.textContent).toBe("benchmark");
    // The coordinator gestures when it sends a brief.
    expect(container.querySelector('svg[data-role="coordinator"]')!.getAttribute("data-gesture")).toBe("true");
    // A non-a2a event clears nothing by itself, but scrubbing (live=false) never fires a packet.
    cleanup();
    const r = render(<AgentBoard state={s} live={false} />);
    r.rerender(<AgentBoard state={foldEvent(s, ev("a2a.message", { from: "benchmark", to: "coordinator", kind: "result", text: "x" }))} live={false} />);
    expect(r.container.querySelector(".crew-packet")).toBeNull();
    cleanup();

    setReducedMotion(true);
    const q = render(<AgentBoard state={s} />);
    q.rerender(<AgentBoard state={foldEvent(s, ev("a2a.message", { from: "benchmark", to: "coordinator", kind: "result", text: "y" }))} />);
    expect(q.container.querySelector(".crew-packet")).toBeNull();
    expect(q.container.querySelector(".crew-pulse")).toBeNull();
  });

  it("reduced motion: static poses (CSS), packets and pulses removed", () => {
    const block = CSS.slice(CSS.indexOf("@media (prefers-reduced-motion: reduce)"));
    expect(block).toMatch(/\.crew-packet/);
    expect(block).toMatch(/\.crew-pulse/);
    expect(block).toMatch(/\.bootseq/);
    // The failed slump is a base style (not an animation), so it survives reduced motion.
    expect(CSS).toMatch(/\.plush\[data-pose="failed"\] \.p-hop \{ transform: translateY\(5px\) rotate\(-7deg\); \}/);
    // At most three infinite animations per pose.
    for (const pose of ["asleep", "standby", "idle", "working", "done", "failed"]) {
      const rules = [...CSS.matchAll(new RegExp(`\\.plush\\[data-pose="${pose}"\\][^{]*\\{([^}]*)\\}`, "g"))];
      const infinite = rules.filter((m) => /animation:[^;]*infinite/.test(m[1])).length;
      expect(infinite, pose).toBeLessThanOrEqual(3);
    }
  });

  it("sample replay: crew labelled SIMULATED, trophy unlit after a REJECT", () => {
    const view = { state: foldAll(SAMPLE), mode: "REPLAY" as const, conn: "OFFLINE" as const, source: "sample" as const, detail: null, history: { base: emptyState(), events: SAMPLE } };
    render(<DashboardView view={view} runId="sample" />);
    const board = screen.getByRole("region", { name: /agents/i });
    within(board).getByText("SIMULATED");
    expect(board.querySelector('.p-trophy')!.getAttribute("data-lit")).toBe("false");
    expect(poseIn("training_executor")).toBe("done");
    expect(poseIn("champion_manager")).toBe("standby"); // replied ok, but no cheer
    cleanup();
    // Not labelled outside the sample.
    render(<AgentBoard state={foldAll(SAMPLE)} />);
    expect(within(screen.getByRole("region", { name: /agents/i })).queryByText("SIMULATED")).toBeNull();
  });

  it("the trophy lights only after a PROMOTE verdict", () => {
    render(<AgentBoard state={foldAll([ev("gate.verdict", { decision: "PROMOTE", reasons: [] })])} />);
    expect(document.querySelector(".p-trophy")!.getAttribute("data-lit")).toBe("true");
  });
});

describe("gauges", () => {
  it("ratioOf clamps and refuses non-numbers", () => {
    expect(ratioOf(0.5)).toBe(0.5);
    expect(ratioOf(-2)).toBe(0);
    expect(ratioOf(7)).toBe(1);
    expect(ratioOf(null)).toBeNull();
    expect(ratioOf(Number.NaN)).toBeNull();
    expect(ratioOf(5, 0, 10)).toBe(0.5);
    expect(ratioOf(5, 1, 1)).toBeNull();
  });

  it("missing data says no data, never zero", () => {
    const { container } = render(
      <>
        <Dial value={null} label="Error rate" tone="lime" text="" />
        <ProgressRing done={null} total={100} label="steps" />
        <BarGauge ratio={null} tone="cyan" label="x" min={0} max={1} now={null} />
        <RadialGauge rings={[{ ratio: null, tone: "cyan", label: "c" }]} center={NO_DATA} label="g" />
      </>,
    );
    expect(screen.getAllByText(NO_DATA)).toHaveLength(4);
    expect(container.textContent).not.toMatch(/0%|0\.0/);
    expect(container.querySelector(".gauge-value, .dial-value, .ring-value, .bar-fill")).toBeNull();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("values draw clamped, the real number stays in the text and aria", () => {
    render(<Dial value={1.7} label="Error rate" tone="red" text="170.0%" />);
    const meter = screen.getByRole("meter", { name: "Error rate" });
    expect(meter.getAttribute("aria-valuenow")).toBe("1.7");
    within(meter).getByText("170.0%");
    expect((meter.querySelector(".dial-needle") as SVGElement).style.transform).toBe("rotate(90deg)");
    cleanup();

    render(<ProgressRing done={150} total={100} label="Training steps" />);
    const ring = screen.getByRole("progressbar", { name: "Training steps" });
    expect(ring.getAttribute("aria-valuenow")).toBe("150");
    within(ring).getByText("100%");
    cleanup();

    render(<BarGauge ratio={0.25} tone="amber" label="Budget spent" min={0} max={20} now={5} />);
    const bar = screen.getByRole("meter", { name: "Budget spent" });
    expect((bar.querySelector(".bar-fill") as HTMLElement).style.transform).toBe("scaleX(0.2500)");
  });

  it("gate gauge shows candidate and champion; colour follows the verdict, not the scores", () => {
    const s = foldAll([ev("gate.verdict", { decision: "REJECT", reasons: [], champion_score: 0.3, candidate_score: 0.5, relative_improvement: 0.66 })]);
    render(<ScoreGate state={s} />);
    const gauge = screen.getByRole("img", { name: /gate scores/i });
    expect(gauge.getAttribute("aria-label")).toMatch(/candidate 0\.50.*champion 0\.30/);
    expect(gauge.querySelector(".gauge-ring")!.getAttribute("class")).toMatch(/tone-red/);
    within(screen.getByRole("region", { name: /gate/i })).getByText("REJECT");
    cleanup();
    // No gate yet: nothing drawn, the reactor stays neutral.
    render(<ScoreGate state={emptyState()} />);
    expect(screen.queryByRole("img", { name: /gate scores/i })).toBeNull();
    screen.getByText("no verdict yet");
  });

  it("job ring says no data for the sample's null steps; telemetry dial and token counters", () => {
    const s = foldAll(SAMPLE);
    render(<JobPanel state={s} />);
    within(screen.getByRole("region", { name: /job/i })).getByText(NO_DATA);
    cleanup();
    render(<Telemetry state={s} history={{ base: emptyState(), events: SAMPLE }} />);
    const tele = screen.getByRole("region", { name: /telemetry/i });
    const dial = within(tele).getByRole("meter", { name: "Error rate" });
    expect(dial.getAttribute("aria-valuenow")).toBe("0");
    expect(tele.querySelector(".tokens")!.textContent).toMatch(/TOKENS IN.*12.*TOKENS OUT.*8/);
    cleanup();
    render(<Budget state={s} />);
    screen.getByRole("meter", { name: "Budget spent" });
  });
});
