import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { parse } from "yaml";
import { validateModel } from "./v1-validate";
import { applyOverlay, migrate, v2Issues } from "./migrate";
import type { SpecModel } from "./types";

/** The repository that holds the spec. Atlas lives in ./atlas, so the default is the parent directory. */
export const REPO_ROOT = path.resolve(/*turbopackIgnore: true*/ process.env.ATLAS_REPO_ROOT ?? path.join(/*turbopackIgnore: true*/ process.cwd(), ".."));
export const SPEC_PATH = "architecture/soc.yaml";
const FIXTURES = path.join(/*turbopackIgnore: true*/ process.cwd(), "fixtures");

export type BranchInfo = {
  name: string;
  commit: string;
  /** "repo": the spec on disk; "none": the branch has no spec. */
  spec: "repo" | "none";
  overlay?: string;
  /** Commit of the RTL extract, "pending" while CI is still running, or null when the branch has no RTL. */
  extract: string | null;
  sample: boolean;
  note: string;
};

export class SpecMissingError extends Error {}
export class SpecInvalidError extends Error {
  constructor(readonly issues: string[]) { super(`${SPEC_PATH} failed validation:\n${issues.slice(0, 12).join("\n")}`); }
}

function headCommit(): string {
  try {
    return execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: REPO_ROOT, encoding: "utf8" }).trim();
  } catch {
    return "working-tree";
  }
}

export function listBranches(): BranchInfo[] {
  const sample = JSON.parse(readFileSync(path.join(FIXTURES, "branches.json"), "utf8")) as BranchInfo[];
  return [
    { name: "main", commit: headCommit(), spec: "repo", extract: null, sample: false, note: "The spec on disk. Every block stays planned until the RTL lands." },
    ...sample,
  ];
}

export function branchInfo(name: string | undefined): BranchInfo {
  const all = listBranches();
  return all.find((b) => b.name === name) ?? all[0];
}

let cache: { mtime: number; base: SpecModel; text: string } | null = null;
const overlayCache = new Map<string, SpecModel>();

function baseModel(): SpecModel {
  const file = path.join(REPO_ROOT, SPEC_PATH);
  if (!existsSync(file)) throw new SpecMissingError(`${SPEC_PATH} was not found in ${REPO_ROOT}`);
  const mtime = statSync(file).mtimeMs;
  if (cache && cache.mtime === mtime) return cache.base;
  const text = readFileSync(file, "utf8");
  let raw: unknown;
  try { raw = parse(text); } catch (e) { throw new SpecInvalidError([(e as Error).message.split("\n")[0]]); }
  const result = validateModel(raw);
  if (!result.ok) throw new SpecInvalidError(result.issues);
  const base = migrate(result.model, text, { path: SPEC_PATH, branch: "main", commit: headCommit() });
  cache = { mtime, base, text };
  overlayCache.clear();
  return base;
}

/** The v2 spec for a branch. Throws SpecMissingError for a branch with no spec. */
export function loadSpec(branch: BranchInfo): SpecModel {
  if (branch.spec === "none") throw new SpecMissingError(`${SPEC_PATH} was not found on ${branch.name}`);
  const base = baseModel();
  if (!branch.overlay) return { ...base, source: { ...base.source, branch: branch.name, commit: branch.commit } };
  const key = `${branch.name}@${cache!.mtime}`;
  let model = overlayCache.get(key);
  if (!model) {
    const text = readFileSync(path.join(FIXTURES, branch.overlay), "utf8");
    model = applyOverlay(base, text, `atlas/fixtures/${branch.overlay}`);
    model.source = { ...model.source, branch: branch.name, commit: branch.commit };
    const issues = v2Issues(model);
    if (issues.length) throw new SpecInvalidError(issues);
    overlayCache.set(key, model);
  }
  return model;
}
