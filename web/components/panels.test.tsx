// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import type { RunView } from "../lib/connection";
import type { RunState } from "../lib/events";
import { emptyState, foldEvent } from "../lib/reducer";
import { AgentBoard } from "./AgentBoard";
import { ApprovalBanner } from "./ApprovalBanner";
import { Budget } from "./Budget";
import { DashboardView } from "./Dashboard";
import { ScoreGate } from "./ScoreGate";
import { StatusBar } from "./StatusBar";
import { Telemetry } from "./Telemetry";

const SAMPLE: unknown[] = readFileSync(join(__dirname, "..", "public", "sample-run.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));

const seqOf = (ev: unknown): number => (ev as { seq: number }).seq;
const foldAll = (events: readonly unknown[], base: RunState = emptyState()): RunState =>
  events.reduce<RunState>(foldEvent, base);
const foldUpTo = (seq: number): RunState => foldAll(SAMPLE.filter((e) => seqOf(e) <= seq));

let n = 1000;
const ev = (type: string, data: Record<string, unknown>) => ({ seq: ++n, ts: 1_760_000_000 + n, run: "t", type, data });

function viewOf(over: Partial<RunView> = {}, events: readonly unknown[] = SAMPLE): RunView {
  const base = emptyState();
  return {
    state: foldAll(events, base),
    mode: "REPLAY",
    conn: "OFFLINE",
    source: "sample",
    detail: "no API configured",
    history: { base, events },
    ...over,
  };
}

// jsdom has no 2D canvas: getContext returns null (the path the sparkline must survive), without
// jsdom's "not implemented" console noise. One test swaps in a recording fake context.
let getContext: MockInstance;
beforeEach(() => {
  getContext = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
});
afterEach(() => {
  cleanup();
  getContext.mockRestore();
});

describe("panels", () => {
  it("approval_banner_shows_only_when_pending_and_has_cli_hint", () => {
    // Before the pending event and after it was resolved: nothing rendered.
    const { container, rerender } = render(<ApprovalBanner state={foldUpTo(8)} />);
    expect(container.innerHTML).toBe("");
    rerender(<ApprovalBanner state={foldUpTo(SAMPLE.length)} />);
    expect(container.innerHTML).toBe("");

    rerender(<ApprovalBanner state={foldUpTo(9)} />);
    const banner = screen.getByRole("alert", { name: /approval required/i });
    within(banner).getByText("posttrain approve --thread sample-simulated-run");
    within(banner).getByText(/launch_training/);
    // Display-only: no approve/reject controls.
    expect(within(banner).queryAllByRole("button")).toHaveLength(0);
    expect(within(banner).queryAllByRole("link")).toHaveLength(0);
  });

  it("explanation_scores_are_badged_simulated", () => {
    const state = foldUpTo(SAMPLE.length);
    render(<ScoreGate state={state} />);
    const scores = screen.getByRole("region", { name: /score/i });
    // Both sample scores carry evidence EXPLANATION.
    expect(within(scores).getAllByText("SIMULATED").length).toBeGreaterThanOrEqual(2);
    cleanup();

    // A measured score shows its evidence and is not badged simulated.
    const live = foldEvent(emptyState(), ev("score", { split: "heldout", checkpoint: "ft:m", value: 0.62, n: 50, evidence: "LIVE" }));
    render(<ScoreGate state={live} />);
    screen.getByText("LIVE");
    expect(screen.queryByText("SIMULATED")).toBeNull();
  });

  it("active_agents_show_slot_labels", () => {
    let s = emptyState();
    s = foldEvent(s, ev("agent.start", { role: "evaluator", slot: 1 }));
    s = foldEvent(s, ev("agent.start", { role: "benchmark", slot: 2 }));
    s = foldEvent(s, ev("agent.start", { role: "researcher", slot: 1 }));
    s = foldEvent(s, ev("agent.end", { role: "researcher", slot: 1, ok: true }));
    render(<AgentBoard state={s} />);
    const board = screen.getByRole("region", { name: /agents/i });
    const items = within(board).getAllByRole("listitem");
    // coordinator + the eight roles
    expect(items).toHaveLength(9);
    const evaluator = items.find((li) => li.textContent?.includes("evaluator"))!;
    const benchmark = items.find((li) => li.textContent?.includes("benchmark"))!;
    const researcher = items.find((li) => li.textContent?.includes("researcher"))!;
    expect(evaluator.getAttribute("data-active")).toBe("true");
    within(evaluator).getByText("slot 1");
    within(benchmark).getByText("slot 2");
    expect(researcher.getAttribute("data-active")).toBe("false");
    expect(within(researcher).queryByText(/slot/)).toBeNull();
    within(board).getByText("2/2");
  });

  it("gate_verdict_renders_reasons", () => {
    render(<ScoreGate state={foldUpTo(SAMPLE.length)} />);
    const gate = screen.getByRole("region", { name: /gate/i });
    within(gate).getByText("REJECT");
    within(gate).getByText("champion evidence None is not verified");
    within(gate).getByText("candidate evidence None is not verified");
  });

  it("gate_inherits_simulated_evidence_from_the_run", () => {
    // Sample: scores are EXPLANATION, so the gate readout is simulated.
    render(<ScoreGate state={foldUpTo(SAMPLE.length)} />);
    within(screen.getByRole("region", { name: /gate/i })).getByText("SIMULATED");
    cleanup();

    // A gate with no scores at all: evidence undetermined, still badged.
    const verdict = ev("gate.verdict", { decision: "PROMOTE", reasons: [], champion_score: 0.5, candidate_score: 0.62, relative_improvement: 0.24 });
    render(<ScoreGate state={foldEvent(emptyState(), verdict)} />);
    within(screen.getByRole("region", { name: /gate/i })).getByText("SIMULATED");
    cleanup();

    // One measured and one simulated score: still simulated.
    const mixed = foldAll([
      ev("score", { split: "heldout", checkpoint: "base", value: 0.5, n: 50, evidence: "LIVE" }),
      ev("score", { split: "heldout", checkpoint: "ft:m", value: 0.62, n: 50, evidence: "EXPLANATION" }),
      ev("gate.verdict", { decision: "PROMOTE", reasons: [] }),
    ]);
    render(<ScoreGate state={mixed} />);
    within(screen.getByRole("region", { name: /gate/i })).getByText("SIMULATED");
  });

  it("gate_with_all_measured_scores_has_no_simulated_badge", () => {
    const measured = foldAll([
      ev("score", { split: "heldout", checkpoint: "base", value: 0.5, n: 50, evidence: "PRIOR_VERIFIED_RUN" }),
      ev("score", { split: "heldout", checkpoint: "ft:m", value: 0.62, n: 50, evidence: "LIVE" }),
      ev("gate.verdict", { decision: "PROMOTE", reasons: ["candidate beats champion"], champion_score: 0.5, candidate_score: 0.62, relative_improvement: 0.24 }),
    ]);
    render(<ScoreGate state={measured} />);
    const gate = screen.getByRole("region", { name: /gate/i });
    within(gate).getByText("PROMOTE");
    expect(within(gate).queryByText("SIMULATED")).toBeNull();
  });

  it("long_gate_reasons_are_clipped", () => {
    const long = "x".repeat(1000);
    render(<ScoreGate state={foldEvent(emptyState(), ev("gate.verdict", { decision: "REJECT", reasons: [long] }))} />);
    const li = within(screen.getByRole("list", { name: /gate reasons/i })).getByRole("listitem");
    expect(li.textContent!.length).toBeLessThanOrEqual(200);
    expect(li.getAttribute("title")).toBe(long);
  });

  it("scrubbing_shows_historical_state_and_live_button_returns", () => {
    render(<DashboardView view={viewOf()} />);
    screen.getByText("REJECT"); // live: the gate verdict (seq 29) is shown
    const slider = screen.getByRole("slider", { name: /timeline/i });

    fireEvent.change(slider, { target: { value: "9" } });
    screen.getByText("HISTORICAL · seq 9");
    expect(screen.queryByText("REJECT")).toBeNull();
    // the approval was pending at seq 9
    screen.getByText("posttrain approve --thread sample-simulated-run");

    act(() => {
      // REPLAY: nothing is live, so the return button says LATEST
      const back = screen.getByRole("button", { name: /return to latest/i });
      expect(back.textContent).toBe("LATEST ▸");
      fireEvent.click(back);
    });
    expect(screen.queryByText(/HISTORICAL/)).toBeNull();
    screen.getByText("REJECT");
    expect(screen.queryByText("posttrain approve --thread sample-simulated-run")).toBeNull();
  });

  it("replay_mode_shows_sample_data_label", () => {
    const { rerender } = render(<StatusBar view={viewOf()} state={viewOf().state} />);
    screen.getByText("REPLAY · SAMPLE DATA");

    // Server-side replay (REPLAY_FILE) is recorded, but not the bundled sample.
    rerender(<StatusBar view={viewOf({ source: "api", conn: "LIVE", detail: null })} state={viewOf().state} />);
    screen.getByText("REPLAY");
    expect(screen.queryByText(/SAMPLE DATA/)).toBeNull();
  });

  it("reconnecting_keeps_last_state_and_is_labelled", () => {
    render(<DashboardView view={viewOf({ mode: "LIVE", source: "api", conn: "RECONNECTING", detail: "connection lost" })} />);
    screen.getByText("RECONNECTING");
    screen.getByText(/connection lost/);
    screen.getByText("REJECT"); // last state stays on screen
    expect(screen.queryByText(/SAMPLE DATA/)).toBeNull();
    expect(screen.getByRole("button", { name: /return to live/i }).textContent).toBe("LIVE ▸");
  });

  it("auth_failed_and_offline_are_distinct_and_empty_state_is_never_shown_as_live", () => {
    const empty = viewOf({ mode: "LIVE", source: "api", conn: "AUTH_FAILED", detail: "invalid read token (401)" }, []);
    const { rerender } = render(<DashboardView view={empty} />);
    screen.getByText("AUTH FAILED");
    screen.getByText(/invalid read token/);
    screen.getByText(/no events/i);
    expect(screen.queryByText("● LIVE")).toBeNull();

    rerender(<DashboardView view={{ ...empty, conn: "CONNECTING", detail: null }} />);
    screen.getByText("CONNECTING");
    screen.getByText(/no events/i);

    rerender(<DashboardView view={{ ...empty, mode: "REPLAY", source: "sample", conn: "OFFLINE", detail: "sample data unavailable" }} />);
    screen.getByText("OFFLINE");
    screen.getByText(/sample data unavailable/);
  });

  it("budget_with_unknown_price_says_so_instead_of_zero", () => {
    const s = foldEvent(emptyState(), ev("budget.update", { spent_usd: null, cap_usd: 25, estimate_usd: null, price_known: false }));
    render(<Budget state={s} />);
    screen.getByText(/price unknown/i);
    expect(screen.queryByText("$0.00")).toBeNull();
  });

  it("job_panel_badges_simulated_job_and_telemetry_renders_without_canvas", () => {
    render(<DashboardView view={viewOf()} />);
    const job = screen.getByRole("region", { name: /job stub-job-sample/i });
    within(job).getByText("SIMULATED");
    const tele = screen.getByRole("region", { name: /telemetry/i });
    within(tele).getByText(/p50 120ms/);
    within(tele).getByText(/p95 480ms/);
  });

  it("telemetry_sparkline_handles_zero_one_and_many_points", () => {
    const many = [1, 2, 3].map((i) => ev("telemetry.snapshot", { p50_ms: 100 * i, p95_ms: 300 * i, error_rate: 0.01 * i }));
    const ctx = { clearRect: vi.fn(), fillRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn() };
    for (const fake of [null, ctx]) {
      getContext.mockImplementation(() => fake);
      for (const events of [[], many.slice(0, 1), many]) {
        const state = foldAll(events);
        const { unmount } = render(<Telemetry state={state} history={{ base: emptyState(), events }} />);
        const tele = screen.getByRole("region", { name: /telemetry/i });
        expect(within(tele).queryAllByRole("img")).toHaveLength(events.length === 0 ? 0 : 3);
        unmount();
      }
    }
    expect(ctx.fillRect).toHaveBeenCalled(); // one point: a dot
    expect(ctx.lineTo).toHaveBeenCalled(); // many points: a line
  });
});
