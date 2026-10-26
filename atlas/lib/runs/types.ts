/** Files written by sim/run.py in the RTL repository (plan.md §M3). */

export type RunStatus = "passed" | "failed" | "running" | "incomplete";

/** run.json */
export type RunMeta = {
  id: string;
  branch: string;
  commit: string;
  test: string;
  simulator: string;
  seed: number;
  cycles: number;
  dirty: boolean;
  status: RunStatus;
  startedAt: string;
  durationS: number;
  /** Files the run uploaded, relative to the run folder. */
  files: string[];
  summary?: string;
};

/** One line of events.jsonl */
export type RunEvent = { cycle: number; block: string; type: string; details: string };

/**
 * signals.json — value changes for the signals the spec maps (fst_compact.py).
 * Keys are `<instance>.<rtl name>`; `changes` is [cycle, value] sorted by cycle.
 */
export type SignalTrace = { width: number; changes: Array<[number, number]> };
export type SignalsFile = { schema: "atlas.signals/1"; cycles: number; signals: Record<string, SignalTrace> };

/** transactions.json — one token per instruction, with the cycles it spent in each process stage (inclusive). */
export type Transaction = {
  id: string;
  label: string;
  pc: string;
  stages: Record<string, [number, number]>;
  dropped?: boolean;
};
export type TransactionsFile = { schema: "atlas.transactions/1"; process: string; transactions: Transaction[] };

/** Files a run folder may hold. Anything else is refused by the API. */
export const RUN_FILES = ["run.json", "events.jsonl", "signals.json", "transactions.json"] as const;
export type RunFile = (typeof RUN_FILES)[number];
