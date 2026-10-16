// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import type { RunView } from "../lib/connection";
import type { RunState } from "../lib/events";
import { emptyState, foldEvent } from "../lib/reducer";
import { A2APanel, COLLAPSE_LINES, displayText } from "./A2APanel";
import { DashboardView } from "./Dashboard";

const SAMPLE: unknown[] = readFileSync(join(__dirname, "..", "public", "sample-run.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));

const foldAll = (events: readonly unknown[], base: RunState = emptyState()): RunState => events.reduce<RunState>(foldEvent, base);
let n = 9000;
const a2a = (data: Record<string, unknown>) => ({ seq: ++n, ts: 1_760_000_000 + n, run: "t", type: "a2a.message", data });
const panel = () => screen.getByRole("region", { name: "Agent-to-agent conversation" });

function viewOf(events: readonly unknown[] = SAMPLE): RunView {
  const base = emptyState();
  return { state: foldAll(events, base), mode: "REPLAY", conn: "OFFLINE", source: "sample", detail: "no API configured", history: { base, events } };
}

let getContext: MockInstance;
beforeEach(() => {
  getContext = vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
});
afterEach(() => {
  cleanup();
  getContext.mockRestore();
  vi.restoreAllMocks();
});

describe("A2APanel", () => {
  it("empty state says there are no messages and offers no export", () => {
    render(<A2APanel state={emptyState()} />);
    within(panel()).getByText("no agent-to-agent messages yet");
    expect(within(panel()).queryByRole("listitem")).toBeNull();
    expect(screen.getByRole("button", { name: /export conversation as json/i })).toHaveProperty("disabled", true);
  });

  it("renders from -> to, kind badges, ok/fail and the text, coordinator and subagents on opposite sides", () => {
    const s = foldAll([
      a2a({ from: "coordinator", to: "evaluator", kind: "brief", text: "Evaluate ft:abc on heldout", text_len: 26 }),
      a2a({ from: "evaluator", to: "coordinator", kind: "result", ok: true, text: "done", text_len: 4 }),
      a2a({ from: "benchmark", to: "coordinator", kind: "result", ok: false, text: "tool failed", text_len: 11 }),
    ]);
    render(<A2APanel state={s} />);
    const items = within(panel()).getAllByRole("listitem");
    expect(items).toHaveLength(3);
    // newest at the bottom: DOM order is oldest first
    expect(items.map((li) => li.querySelector(".a2a-from")!.textContent)).toEqual(["coordinator", "evaluator", "benchmark"]);
    expect(items[0].textContent).toMatch(/coordinator → to evaluator/);
    within(items[0]).getByText("BRIEF");
    within(items[1]).getByText("RESULT");
    within(items[1]).getByText(/✓ ok/);
    within(items[2]).getByText(/✗ fail/);
    expect(within(items[0]).queryByText(/ok|fail/)).toBeNull();
    expect(items[0].className).toMatch(/a2a-coord/);
    expect(items[1].className).toMatch(/a2a-sub/);
    expect(items[0].querySelector("pre")!.textContent).toBe("Evaluate ft:abc on heldout");
  });

  it("long text starts collapsed and expands/collapses with an aria-expanded button", () => {
    const lines = Array.from({ length: 12 }, (_, i) => `line ${i + 1}`);
    render(<A2APanel state={foldAll([a2a({ from: "failure_analyst", to: "coordinator", kind: "result", ok: true, text: lines.join("\n") })])} />);
    const pre = panel().querySelector("pre")!;
    expect(pre.textContent).toContain(`line ${COLLAPSE_LINES}`);
    expect(pre.textContent).not.toContain(`line ${COLLAPSE_LINES + 1}`);
    const toggle = screen.getByRole("button", { name: /expand \(12 lines\)/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(toggle.getAttribute("aria-controls")).toBe(pre.id);
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(pre.textContent).toBe(lines.join("\n"));
    fireEvent.click(screen.getByRole("button", { name: /collapse/ }));
    expect(pre.textContent).not.toContain("line 12");

    // A single very long line collapses too; a short message has no toggle.
    cleanup();
    render(<A2APanel state={foldAll([a2a({ from: "a", to: "b", kind: "brief", text: "x".repeat(2000) }), a2a({ from: "a", to: "b", kind: "brief", text: "short" })])} />);
    expect(screen.getAllByRole("button", { name: /expand/ })).toHaveLength(1);
  });

  it("shows a cut note when text_len is larger than the text (code points, like Python len)", () => {
    const text = "y".repeat(2000);
    render(
      <A2APanel
        state={foldAll([
          a2a({ from: "evaluator", to: "coordinator", kind: "result", ok: true, text, text_len: 5311 }),
          a2a({ from: "coordinator", to: "evaluator", kind: "brief", text: "😀 ok", text_len: 4 }),
        ])}
      />,
    );
    const items = within(panel()).getAllByRole("listitem");
    within(items[0]).getByText("cut: showing 2000 of 5311 chars");
    // "😀 ok" is 4 code points (5 UTF-16 units): not cut.
    expect(within(items[1]).queryByText(/cut:/)).toBeNull();
  });

  it("pretty-prints a text that is a whole JSON object, as plain text only", () => {
    const evil = '{"html":"<img src=x onerror=alert(1)>","n":1}';
    render(
      <A2APanel
        state={foldAll([
          a2a({ from: "evaluator", to: "coordinator", kind: "result", ok: true, text: evil }),
          a2a({ from: "a", to: "b", kind: "brief", text: '{"cut": "mid-obj' }),
          a2a({ from: "a", to: "b", kind: "brief", text: "[1, 2]" }),
        ])}
      />,
    );
    const pres = [...panel().querySelectorAll("pre")];
    expect(pres[0].textContent).toBe(JSON.stringify(JSON.parse(evil), null, 2));
    expect(pres[0].querySelector("img")).toBeNull();
    expect(panel().querySelector("img")).toBeNull();
    expect(pres[1].textContent).toBe('{"cut": "mid-obj');
    expect(pres[2].textContent).toBe("[1, 2]");
    expect(displayText("null")).toEqual({ text: "null", json: false });
    expect(displayText(' {"a":1} ')).toEqual({ text: '{\n  "a": 1\n}', json: true });
  });

  it("is a focusable, named scroll region and never a live region", () => {
    render(<A2APanel state={foldAll(SAMPLE)} />);
    const scroll = screen.getByRole("region", { name: "Agent-to-agent messages (scrollable)" });
    expect(scroll.getAttribute("tabindex")).toBe("0");
    expect(panel().querySelectorAll("[aria-live]")).toHaveLength(0);
    expect(panel().querySelectorAll('[role="log"], [role="status"], [role="alert"]')).toHaveLength(0);
    expect(panel().closest("[aria-live]")).toBeNull();
  });

  it("tolerates a malformed a2a value from an untrusted snapshot", () => {
    const bad = { ...emptyState(), a2a: { not: "a list" } } as unknown as RunState;
    render(<A2APanel state={bad} />);
    within(panel()).getByText("no agent-to-agent messages yet");
    cleanup();
    const mixed = { ...emptyState(), a2a: [null, { from: 1 }, { from: "a", to: "b", kind: "brief", text: 3 }] } as unknown as RunState;
    render(<A2APanel state={mixed} />);
    expect(within(panel()).getAllByRole("listitem")).toHaveLength(1);
    within(panel()).getByText("(no text)");
  });

  it("export JSON downloads {run, messages}", async () => {
    const created: Blob[] = [];
    const create = vi.fn((b: Blob) => {
      created.push(b);
      return "blob:a2a";
    });
    Object.defineProperty(URL, "createObjectURL", { configurable: true, writable: true, value: create });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, writable: true, value: vi.fn() });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const state = foldAll([a2a({ from: "coordinator", to: "evaluator", kind: "brief", text: "go", text_len: 2 })]);
    render(<A2APanel state={state} runId="run/1" />);
    fireEvent.click(screen.getByRole("button", { name: /export conversation as json/i }));
    expect(click).toHaveBeenCalledTimes(1);
    expect(click.mock.contexts[0]).toMatchObject({ download: "a2a-run_1.json" });
    expect(created).toHaveLength(1);
    expect(JSON.parse(await created[0].text())).toStrictEqual({
      run: "run/1",
      messages: [{ from: "coordinator", to: "evaluator", kind: "brief", text: "go", text_len: 2 }],
    });
  });
});

describe("A2APanel in the dashboard", () => {
  it("shows the simulated sample conversation, labelled SIMULATED", () => {
    render(<DashboardView view={viewOf()} runId="sample" />);
    const p = panel();
    expect(within(p).getAllByRole("listitem")).toHaveLength(8);
    within(p).getByText("SIMULATED");
    // the sample's evaluator refuses to claim an improvement
    within(p).getByText(/✗ fail/);
  });

  it("scrubbing shows the conversation folded up to that seq", () => {
    render(<DashboardView view={viewOf()} runId="sample" />);
    const slider = screen.getByRole("slider", { name: /timeline/i });
    const firstA2A = (SAMPLE as { seq: number; type: string }[]).find((e) => e.type === "a2a.message")!.seq;

    fireEvent.change(slider, { target: { value: String(firstA2A - 1) } });
    within(panel()).getByText("no agent-to-agent messages yet");

    fireEvent.change(slider, { target: { value: String(firstA2A + 2) } });
    const items = within(panel()).getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items.at(-1)!.querySelector(".a2a-to")!.textContent).toBe("failure_analyst");

    fireEvent.click(screen.getByRole("button", { name: /return to latest/i }));
    expect(within(panel()).getAllByRole("listitem")).toHaveLength(8);
  });
});
