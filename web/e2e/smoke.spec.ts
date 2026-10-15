import { expect, type Page, test } from "@playwright/test";

/**
 * Smoke test of the production build with no NEXT_PUBLIC_API_URL: /run/<id> replays the bundled
 * simulated sample (web/public/sample-run.jsonl, ~30 events over ~25 s) and must say so.
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
});

test("reduced motion: no animation is active", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(RUN_PATH);

  await expect(page.getByRole("status").filter({ hasText: "REPLAY · SAMPLE DATA" }).first()).toBeVisible();
  // Let rows, the status chip and the banner render (all of them animate without reduce).
  await expect.poll(() => page.locator(".stream-row").count()).toBeGreaterThan(3);

  for (const sel of [".statusbar-conn .conn", ".stream-row", ".stream-cursor", ".boot-cover"]) {
    const el = page.locator(sel).first();
    if ((await el.count()) === 0) continue;
    expect(await el.evaluate((e) => getComputedStyle(e).animationName), sel).toBe("none");
  }
  expect(await animatedElements(page)).toEqual([]);
  expect(await page.evaluate(() => document.getAnimations().length)).toBe(0);
  // The CRT sweep band is removed, not just paused.
  expect(await page.locator(".dashboard").evaluate((e) => getComputedStyle(e, "::before").display)).toBe("none");
});
