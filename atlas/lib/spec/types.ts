import type { Side } from "./v1-schema";

/** Spec v2 (plan.md §M1.1): one status tag on blocks, ports and connections. */
export const STATUS = ["planned", "in-progress", "implemented", "modified", "verified"] as const;
export type Status = (typeof STATUS)[number];

/** Statuses that must exist in RTL and are compared against the extract. */
export const RTL_CHECKED: ReadonlySet<Status> = new Set(["implemented", "modified", "verified"]);
export const isPlanned = (status: Status) => !RTL_CHECKED.has(status);

/** The lower of two statuses, used when a port or connection inherits from its endpoints. */
export function lowerStatus(a: Status, b: Status): Status {
  return STATUS.indexOf(a) <= STATUS.indexOf(b) ? a : b;
}

export type SignalDef = { name: string; width: number | `param:${string}`; dir: "fwd" | "bwd" };

export type RuleKind = "implies" | "next" | "stable_while" | "within";
export type TimingRule = {
  id: string;
  kind: RuleKind;
  when: string;
  then?: string;
  signals?: string[];
  cycles?: number;
  severity: "error" | "warn";
  summary?: string;
};

export type SpecProtocol = {
  id: string;
  name: string;
  color: string;
  abbreviation: string;
  purpose: string;
  roles: string[];
  signals?: SignalDef[];
  timingRules?: TimingRule[];
  line?: number;
};

export type SpecInterface = {
  id: string;
  protocol: string;
  params: Record<string, number>;
  roleNaming: "decoupled" | "flat";
  line?: number;
};

export type SpecBlock = {
  id: string;
  name: string;
  kind: string;
  summary: string;
  parent: string | null;
  owner: string;
  status: Status;
  responsibilities: string[];
  documents: Array<{ title: string; path?: string; url?: string; anchor?: string; page?: number }>;
  rtl: { module: string; instance?: string } | null;
  childIds: string[];
  /** Line of the block's entry in the spec file, for source links. */
  line?: number;
};

export type SpecPort = {
  id: string;
  blockId: string;
  label: string;
  protocol: string;
  side: Side;
  offset: number;
  status: Status;
  interface?: string;
  role?: string;
  rtl?: { prefix?: string; map?: Record<string, string> };
};

export type SpecConnection = {
  id: string;
  from: string;
  fromPort: string;
  to: string;
  toPort: string;
  protocol: string;
  label: string;
  bidirectional: boolean;
  status: Status;
};

export type SpecLayout = {
  canvas: { width: number; height: number };
  nodes: Record<string, { x: number; y: number; w: number; h: number }>;
};

export type ProcessStage = {
  key: string;
  name: string;
  block: string;
  ordering: "pipeline" | "fifo" | "reorder";
  capacity: number;
  /** Signals whose handshake moves tokens into / out of the stage; used for stall detection. */
  enq?: { valid: string; ready: string };
  deq?: { valid: string; ready: string };
  /** Spec explanation shown when the stage stalls. */
  stallReason?: string;
};

export type SpecProcess = { id: string; name: string; stages: ProcessStage[] };

export type SpecModel = {
  product: { name: string; description: string; defaultFocus: string };
  protocols: Record<string, SpecProtocol>;
  interfaces: Record<string, SpecInterface>;
  blocks: Record<string, SpecBlock>;
  rootIds: string[];
  ports: Record<string, SpecPort[]>;
  connections: SpecConnection[];
  layouts: Record<string, SpecLayout>;
  flows: Array<{ id: string; name: string; summary: string; steps: Array<{ title: string; detail: string; blocks: string[]; protocol: string }> }>;
  process: SpecProcess[];
  counts: Record<Status, number>;
  /** Where this model came from. `path` is relative to the repository root. */
  source: { path: string; branch: string; commit: string; overlay?: string };
};
