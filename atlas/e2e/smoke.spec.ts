import { expect, test, type Page } from "@playwright/test";

const SAMPLE = "branch=atlas%2Fvector-dispatch-64b";

function trackErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  return errors;
}

test("canvas shows conformance and the inspector explains a missing port", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto(`/canvas?${SAMPLE}&focus=coral-core`, { waitUntil: "networkidle" });
  await page.click("[data-block=dtcm-arbiter]");
  const inspector = page.getByRole("complementary", { name: "Inspector" });
  await expect(inspector).toContainText("missing in RTL");
  await expect(inspector).toContainText("io_debug_* absent");
  await expect(page).toHaveURL(/sel=block%3Adtcm-arbiter/);
  expect(errors).toEqual([]);
});

test("double-click opens a block and the board fades back in", async ({ page }) => {
  await page.goto(`/canvas?${SAMPLE}&focus=coral-core`, { waitUntil: "networkidle" });
  await page.dblclick("[data-block=compute-cluster]");
  await expect(page).toHaveURL(/focus=compute-cluster/);
  await expect(page.locator("[data-block=vector-engine]")).toBeVisible();
  await expect(page.locator(".board-content")).toHaveClass("board-content");
});

test("branch picker switches branch without losing the board", async ({ page }) => {
  await page.goto("/canvas?branch=atlas%2Faxi-burst-fix&focus=coral-core", { waitUntil: "networkidle" });
  await expect(page.getByText("RTL extract pending.")).toBeVisible();
  await page.click(".branch-btn");
  await page.getByRole("option", { name: /atlas\/vector-dispatch-64b/ }).click();
  await expect(page).toHaveURL(/vector-dispatch-64b/);
  await expect(page.locator(".board-content")).toHaveClass("board-content");
  await expect(page.getByRole("radio", { name: "Conformance" })).toHaveAttribute("aria-checked", "true");
});

test("a violation moves the shared cycle, and other views follow", async ({ page }) => {
  const errors = trackErrors(page);
  await page.goto(`/runs?${SAMPLE}&run=212&cycle=0`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /no-drop/ }).click();
  await expect(page.getByLabel("Cycle", { exact: true }).first()).toHaveValue("640");
  await expect(page.locator(".log-row.on")).toContainText("drop");
  await page.getByRole("link", { name: "Process", exact: true }).click();
  await expect(page).toHaveURL(/\/process\?.*cycle=640/);
  await expect(page.locator(".tok.drop:not(.out)")).toContainText("i118");
  expect(errors).toEqual([]);
});

test("incomplete run says which file is missing", async ({ page }) => {
  await page.goto(`/runs?${SAMPLE}&run=206`, { waitUntil: "networkidle" });
  await expect(page.getByText("No waveform for this run")).toBeVisible();
  await expect(page.locator(".file.absent", { hasText: "signals.json" })).toBeVisible();
});

test("doc graph traces a protocol to its rules", async ({ page }) => {
  await page.goto(`/spec?${SAMPLE}&node=proto%3Adecoupled`, { waitUntil: "networkidle" });
  await expect(page.getByRole("complementary", { name: "Details" })).toContainText("Whenever valid && !ready, valid must hold on the next cycle.");
});

test("empty states", async ({ page }) => {
  await page.goto("/canvas?branch=atlas%2Fempty-spec", { waitUntil: "networkidle" });
  await expect(page.getByText("No spec on atlas/empty-spec")).toBeVisible();
  await page.goto("/process", { waitUntil: "networkidle" });
  await expect(page.getByText("No process section in this spec")).toBeVisible();
});
