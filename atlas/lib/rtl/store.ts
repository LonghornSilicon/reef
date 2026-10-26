import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { BranchInfo } from "../spec/load";
import type { RtlExtract } from "./types";

const RTL_DIR = path.join(/*turbopackIgnore: true*/ process.cwd(), "fixtures", "rtl");

/** CI has not uploaded the extract for this commit yet. */
export class ExtractPendingError extends Error {
  constructor(readonly commit: string) { super(`The RTL extract for ${commit} is still being built in CI.`); }
}

/**
 * The extract for a branch. Today it reads fixtures/rtl/<sha>/extract.json; in M1 proper it reads
 * rtl/<sha>/extract.json from the bucket for the nearest ancestor commit and reports staleBy.
 * Returns null when the branch has no RTL at all.
 */
export function loadExtract(branch: BranchInfo): { extract: RtlExtract; staleBy: number } | null {
  if (branch.extract === null) return null;
  if (branch.extract === "pending") throw new ExtractPendingError(branch.commit);
  const file = path.join(RTL_DIR, branch.extract, "extract.json");
  if (!existsSync(file)) throw new ExtractPendingError(branch.extract);
  const extract = JSON.parse(readFileSync(file, "utf8")) as RtlExtract;
  return { extract, staleBy: extract.commit === branch.commit ? 0 : 1 };
}
