import "server-only";
import { branchInfo, listBranches, loadSpec, SpecInvalidError, SpecMissingError, type BranchInfo } from "../spec/load";
import { ExtractPendingError, loadExtract } from "../rtl/store";
import { compare } from "../rtl/compare";
import { listRuns } from "../runs/store";
import type { CompareReport } from "../rtl/types";
import type { SpecModel } from "../spec/types";
import type { RunMeta } from "../runs/types";

export type RtlState = { kind: "none" } | { kind: "pending"; message: string } | { kind: "ready"; commit: string; staleBy: number };

export type BranchContext =
  | { kind: "ok"; branch: BranchInfo; spec: SpecModel; report: CompareReport | null; rtl: RtlState; runs: RunMeta[] }
  | { kind: "no-spec"; branch: BranchInfo; message: string }
  | { kind: "invalid"; branch: BranchInfo; issues: string[] };

const reports = new Map<string, CompareReport>();

/** Everything a view needs for one branch: the v2 spec, the compare report (cached per spec+extract pair) and its runs. */
export function loadContext(name: string | string[] | undefined): BranchContext {
  const branch = branchInfo(Array.isArray(name) ? name[0] : name);
  let spec: SpecModel;
  try {
    spec = loadSpec(branch);
  } catch (e) {
    if (e instanceof SpecMissingError) return { kind: "no-spec", branch, message: e.message };
    if (e instanceof SpecInvalidError) return { kind: "invalid", branch, issues: e.issues };
    throw e;
  }
  let rtl: RtlState = { kind: "none" };
  let report: CompareReport | null = null;
  try {
    const loaded = loadExtract(branch);
    if (loaded) {
      const key = `${branch.name}:${spec.source.commit}:${loaded.extract.commit}`;
      report = reports.get(key) ?? compare(spec, loaded.extract, loaded.staleBy);
      reports.set(key, report);
      rtl = { kind: "ready", commit: loaded.extract.commit, staleBy: loaded.staleBy };
    }
  } catch (e) {
    if (!(e instanceof ExtractPendingError)) throw e;
    rtl = { kind: "pending", message: e.message };
  }
  return { kind: "ok", branch, spec, report, rtl, runs: listRuns(branch.name) };
}

export function shellData() {
  const branches = listBranches();
  return { branches, runs: branches.flatMap((b) => listRuns(b.name)) };
}

/** Link to a file in the spec repository, when the commit is a real one. */
export function repoUrl(branch: BranchInfo, file: string, line?: number): string | null {
  if (branch.sample) return null;
  const repo = process.env.ATLAS_GITHUB_REPO ?? "140oo041/LHS-SOC";
  return `https://github.com/${repo}/blob/${branch.commit}/${file}${line ? `#L${line}` : ""}`;
}
