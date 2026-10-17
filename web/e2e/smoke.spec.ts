import { expect, type Page, test } from "@playwright/test";

/**
 * Smoke test of the production build with no NEXT_PUBLIC_API_URL: /run/<id> replays the bundled
 * simulated sample (web/public/sample-run.jsonl, ~38 events over ~30 s) and must say so.
 */

const RUN_PATH = "/run/sample";

/** The live seq counter in the status bar ("seq N"). */
async function liveSeq(page: Page): Promise<number> {
  const text = (await page.locator(".statusbar-seq").textContent()) ?? "";
  const m = /seq (\d+)/.exec(text);
  return m ? Number(m[1]) : -1;
}

/** Every element (and ::before / ::after) whose computed animation-name is not "none". */
async function animatedElements(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const out: string[] = [];
    const describe = (el: Element) => `${el.tagName.toLowerCase()}.${[...el.classList].join(".")}`;
    for (const el of document.querySelectorAll("*")) {
      for (const pseudo of [null, "::before", "::after"]) {
        const name = getComputedStyle(el, pseudo).animationName;
        if (name && name !== "none") out.push(`${describe(el)}${pseudo ?? ""} -> ${name}`);
      }
    }
    return out;
  });
}

test("sample replay: honesty label, events flow, SIMULATED badges, scrubber", async ({ page }) => {
  await page.goto(RUN_PATH);

  const status = page.getByRole("status").filter({ hasText: "REPLAY" }).first();
  await expect(status).toContainText("REPLAY · SAMPLE DATA");
  await expect(page.locator(".dashboard")).toHaveAttribute("data-mode", "REPLAY");
  // Never called LIVE: the sample connection is OFFLINE with its reason.
  await expect(page.locator(".statusbar-conn")).not.toContainText("● LIVE");
  await expect(page.locator(".statusbar-conn")).toContainText("no API configured");

  // Events flow in: the seq advances and stream rows appear.
  await expect.poll(() => liveSeq(page)).toBeGreaterThan(0);
  const firstSeq = await liveSeq(page);
  await expect.poll(() => liveSeq(page)).toBeGreaterThan(firstSeq);
  await expect.poll(() => page.locator(".stream-row").count()).toBeGreaterThan(1);

  // Motion is on without reduced-motion (so the reduced-motion test below is not vacuous).
  expect(await animatedElements(page)).not.toHaveLength(0);

  // The plush crew: coordinator + eight specialists, labelled SIMULATED in the sample; the boot
  // sequence has removed itself.
  const crew = page.getByRole("region", { name: "Agents" });
  await expect(crew.locator("svg.plush")).toHaveCount(9);
  await expect(crew.locator(".crew-sim")).toHaveText("SIMULATED");
  await expect(page.locator(".bootseq")).toHaveCount(0);
  // training_executor starts at seq 13 and ends ok at seq 19: it works, then is done.
  const executor = crew.locator('svg.plush[data-role="training_executor"]');
  await expect(executor).toHaveAttribute("data-pose", "done");
  // Never-started roles that took part in the conversation are on standby, never cheering.
  await expect(crew.locator('svg.plush[data-role="champion_manager"]')).toHaveAttribute("data-pose", "standby");
  await expect(crew.locator('svg.plush[data-role="researcher"]')).toHaveAttribute("data-pose", "asleep");
  // Gauges: the gate gauge and the sample's REJECT reactor; the job ring has no step data.
  await expect(page.getByRole("img", { name: /Gate scores: candidate 0.50/ })).toBeVisible();
  await expect(page.locator(".reactor-red .gate-decision")).toHaveText("REJECT");
  await expect(page.locator(".ring-empty")).toContainText("no data");

  // Simulated scores are badged (they arrive ~18 s into the replay).
  await expect(page.locator(".badge-sim").first()).toBeVisible();
  await expect(page.locator(".badge-sim").first()).toHaveText("SIMULATED");

  // Scrub back: the screen is marked HISTORICAL at that seq.
  const scrubber = page.getByRole("slider", { name: "Timeline scrubber (event seq)" });
  await expect(scrubber).toBeEnabled();
  await scrubber.fill("5");
  await expect(page.locator(".statusbar-conn")).toContainText("HISTORICAL · seq 5");
  await expect(page.locator(".dashboard")).toHaveAttribute("data-historical", "true");
  await expect(page.locator(".timeline-pos")).toContainText("seq 5 /");

  // LATEST returns to the newest state.
  const latest = page.getByRole("button", { name: "Return to latest state" });
  await expect(latest).toHaveText(/LATEST/);
  await latest.click();
  await expect(page.locator(".statusbar-conn")).not.toContainText("HISTORICAL");
  await expect(page.locator(".dashboard")).toHaveAttribute("data-historical", "false");
  await expect(latest).toBeDisabled();
  // The latest state still carries the simulated badges.
  await expect(page.locator(".badge-sim").first()).toBeVisible();

  // The simulated A2A conversation arrives last and is labelled SIMULATED.
  const a2a = page.getByRole("region", { name: "Agent-to-agent conversation" });
  await expect(a2a.locator(".a2a-msg")).toHaveCount(8);
  await expect(a2a.locator(".a2a-sim")).toHaveText("SIMULATED");
});

test("phone width: the dashboard, crew and A2A panel do not scroll the page sideways", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(RUN_PATH);
  // The last A2A message arrives ~30 s into the replay, right at the default 30 s expect timeout.
  await expect(page.locator(".a2a-msg")).toHaveCount(8, { timeout: 60_000 });
  await page.getByRole("button", { name: /expand/ }).first().click();
  // Open a long crew bubble too (full model text inside a narrow card).
  await page.locator("button.crew-bubble").first().click();
  await expect(page.locator("button.crew-bubble").first()).toHaveAttribute("aria-expanded", "true");
  const width = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(width).toBeLessThanOrEqual(390);
  // Honesty labels stay visible at phone width.
  await expect(page.locator(".statusbar-conn")).toContainText("REPLAY · SAMPLE DATA");
  await expect(page.locator(".crew-sim")).toBeVisible();
  // Two crew cards per row: every card fits inside the viewport.
  const boxes = await page.locator(".crew-card").evaluateAll((els) => els.map((e) => e.getBoundingClientRect().right));
  expect(Math.max(...boxes)).toBeLessThanOrEqual(390);
});

test("reduced motion: no animation is active", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(RUN_PATH);

  await expect(page.getByRole("status").filter({ hasText: "REPLAY · SAMPLE DATA" }).first()).toBeVisible();
  // Let rows, the status chip and the banner render (all of them animate without reduce).
  await expect.poll(() => page.locator(".stream-row").count()).toBeGreaterThan(3);

  for (const sel of [".statusbar-conn .conn", ".stream-row", ".stream-cursor", ".boot-cover", "svg.plush .p-breathe", ".ticker-track", ".bg-mesh"]) {
    const el = page.locator(sel).first();
    if ((await el.count()) === 0) continue;
    expect(await el.evaluate((e) => getComputedStyle(e).animationName), sel).toBe("none");
  }
  expect(await animatedElements(page)).toEqual([]);
  expect(await page.evaluate(() => document.getAnimations().length)).toBe(0);
  // The CRT sweep band is removed, not just paused.
  expect(await page.locator(".dashboard").evaluate((e) => getComputedStyle(e, "::before").display)).toBe("none");
  // No boot overlay and no message packets under reduced motion.
  await expect(page.locator(".bootseq")).toHaveCount(0);
  await expect(page.locator(".crew-packet")).toHaveCount(0);
});
