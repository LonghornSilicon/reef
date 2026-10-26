import type { Status } from "../spec/types";

export type SrcLoc = { file: string; line: number };

/** `atlas.rtl-extract/1` — written by tools/atlas/extract_rtl.py (pyslang) in the RTL repository's CI. */
export type RtlPort = { name: string; dir: "in" | "out" | "inout"; width: number; src?: SrcLoc };
export type RtlModule = {
  sv: SrcLoc;
  src?: SrcLoc;
  doc?: string;
  ports: RtlPort[];
  instances?: Array<{ name: string; module: string; conns: Record<string, string> }>;
};
export type RtlExtract = {
  schema: "atlas.rtl-extract/1";
  commit: string;
  tools: { firtool: string; slang: string; yosys?: string };
  top: string;
  modules: Record<string, RtlModule>;
};

export type Result = "match" | "different" | "missing" | "unchecked";

export type ExpectedSignal = { signal: string; rtlName: string; width?: number; dir?: "in" | "out" };
export type SignalDiff = { signal: string; rtlName: string; kind: "absent" | "width" | "dir"; expected?: string | number; actual?: string | number };

export type PortReport = {
  result: Result;
  status: Status;
  expected: ExpectedSignal[];
  found: RtlPort[];
  diffs: SignalDiff[];
  /** True when differences only warn (status `modified`). */
  warnOnly: boolean;
};

export type BlockReport = {
  status: Status;
  module: string | null;
  result: Result;
  sv?: SrcLoc;
  src?: SrcLoc;
  doc?: string;
  ports: Record<string, PortReport>;
  extraRtlPorts: RtlPort[];
};

export type CompareReport = {
  schema: "atlas.compare/1";
  spec: { branch: string; commit: string };
  rtl: { commit: string; staleBy: number; tools: RtlExtract["tools"] };
  blocks: Record<string, BlockReport>;
  connections: Record<string, { result: Result; status: Status; mode: "instances" | "endpoints-only"; warnOnly: boolean }>;
  summary: Record<Result, number>;
  /** Mismatches that fail CI: different/missing on implemented or verified entries. */
  failing: number;
};
