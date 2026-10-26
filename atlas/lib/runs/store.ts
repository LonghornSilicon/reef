import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { RUN_FILES, type RunMeta } from "./types";

const RUNS = path.join(/*turbopackIgnore: true*/ process.cwd(), "fixtures", "runs");

/** Runs for a branch, newest first. Fixtures today; a bucket prefix listing in M3. */
export function listRuns(branch: string): RunMeta[] {
  if (!existsSync(RUNS)) return [];
  return readdirSync(RUNS)
    .filter((id) => /^\d+$/.test(id) && existsSync(path.join(RUNS, id, "run.json")))
    .map((id) => JSON.parse(readFileSync(path.join(RUNS, id, "run.json"), "utf8")) as RunMeta)
    .filter((r) => r.branch === branch)
    .sort((a, b) => Number(b.id) - Number(a.id));
}

export function readRunFile(id: string, file: string): string | null {
  if (!/^\d+$/.test(id) || !(RUN_FILES as readonly string[]).includes(file)) return null;
  const p = path.join(RUNS, id, file);
  return existsSync(p) ? readFileSync(p, "utf8") : null;
}
