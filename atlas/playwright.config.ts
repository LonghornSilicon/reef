import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: { baseURL: "http://localhost:3100", viewport: { width: 1600, height: 960 } },
  webServer: { command: "npm run dev", url: "http://localhost:3100/canvas", reuseExistingServer: true, timeout: 120_000 },
});
