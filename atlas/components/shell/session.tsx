"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import type { BranchInfo } from "@/lib/spec/load";
import type { RunMeta } from "@/lib/runs/types";

/**
 * Shared branch / run / cycle (plan.md "User flow and shared state").
 * The URL is the source of truth for links and reloads; React state keeps scrubbing fast;
 * a BroadcastChannel keeps other open tabs on the same cycle.
 */
type Session = {
  branch: BranchInfo;
  branches: BranchInfo[];
  runs: RunMeta[];
  run: RunMeta | null;
  cycle: number;
  setRun: (id: string | null) => void;
  setCycle: (c: number | ((prev: number) => number)) => void;
  /** Query string carrying the session, for links between views. */
  query: (extra?: Record<string, string | null>) => string;
};

const Ctx = createContext<Session | null>(null);
export const useSession = () => {
  const s = useContext(Ctx);
  if (!s) throw new Error("useSession outside SessionProvider");
  return s;
};

/** A run is "playable" when it has signal data. */
export const hasFile = (run: RunMeta | null | undefined, f: string) => !!run?.files.includes(f);
export function defaultRun(runs: RunMeta[]): RunMeta | null {
  return runs.find((r) => hasFile(r, "signals.json")) ?? runs[0] ?? null;
}

const TAB = Math.random().toString(36).slice(2);

export function SessionProvider({ branches, allRuns, children }: { branches: BranchInfo[]; allRuns: RunMeta[]; children: ReactNode }) {
  const params = useSearchParams();
  const branchName = params.get("branch") ?? "main";
  const branch = branches.find((b) => b.name === branchName) ?? branches[0];
  const runs = useMemo(() => allRuns.filter((r) => r.branch === branch.name), [allRuns, branch.name]);

  const pick = useCallback((id: string | null) => runs.find((r) => r.id === id) ?? null, [runs]);
  const [runId, setRunId] = useState<string | null>(() => pick(params.get("run"))?.id ?? defaultRun(runs)?.id ?? null);
  const [cycle, setCycleRaw] = useState<number>(() => Math.max(0, Number(params.get("cycle")) || 0));
  const run = pick(runId);
  const maxCycle = Math.max(0, (run?.cycles ?? 1) - 1);

  // A new branch brings its own runs: take the URL's run if it belongs, otherwise the default.
  const lastBranch = useRef(branch.name);
  useEffect(() => {
    if (lastBranch.current === branch.name) return;
    lastBranch.current = branch.name;
    const fromUrl = pick(params.get("run"));
    setRunId(fromUrl?.id ?? defaultRun(runs)?.id ?? null);
    setCycleRaw(fromUrl ? Math.max(0, Number(params.get("cycle")) || 0) : 0);
  }, [branch.name, params, pick, runs]);

  const setCycle = useCallback((c: number | ((p: number) => number)) => {
    setCycleRaw((prev) => {
      const next = Math.round(typeof c === "function" ? c(prev) : c);
      return Math.min(Math.max(0, next), maxCycle);
    });
  }, [maxCycle]);
  const setRun = useCallback((id: string | null) => { setRunId(id); setCycleRaw(0); }, []);

  // Mirror to the URL (throttled; replaceState never refetches server components).
  const remote = useRef<string | null>(null);
  const channel = useRef<BroadcastChannel | null>(null);
  const pending = useRef<(() => void) | null>(null);
  useEffect(() => {
    const write = () => {
      pending.current = null;
      const u = new URL(window.location.href);
      if (runId) u.searchParams.set("run", runId); else u.searchParams.delete("run");
      if (runId) u.searchParams.set("cycle", String(cycle)); else u.searchParams.delete("cycle");
      if (u.href !== window.location.href) window.history.replaceState(null, "", u);
      const key = `${branch.name}|${runId}|${cycle}`;
      if (remote.current !== key) channel.current?.postMessage({ tab: TAB, branch: branch.name, run: runId, cycle });
      remote.current = null;
    };
    pending.current = write;
    const t = setTimeout(write, 120);
    return () => { clearTimeout(t); pending.current = null; };
  }, [branch.name, runId, cycle]);
  // A late write during a navigation would make Next treat the old URL as a new one and cancel it,
  // so a link click flushes the pending write first (links already carry run and cycle).
  useEffect(() => {
    const flush = (e: MouseEvent) => { if ((e.target as HTMLElement).closest?.("a[href]")) pending.current?.(); };
    document.addEventListener("click", flush, true);
    return () => document.removeEventListener("click", flush, true);
  }, []);

  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return;
    const ch = new BroadcastChannel("atlas-session");
    channel.current = ch;
    ch.onmessage = (e: MessageEvent<{ tab: string; branch: string; run: string | null; cycle: number }>) => {
      const m = e.data;
      if (m.tab === TAB || m.branch !== lastBranch.current) return;
      remote.current = `${m.branch}|${m.run}|${m.cycle}`;
      setRunId(m.run);
      setCycleRaw(m.cycle);
    };
    return () => { ch.close(); channel.current = null; };
  }, []);

  const query = useCallback((extra: Record<string, string | null> = {}) => {
    const q = new URLSearchParams();
    if (branch.name !== "main") q.set("branch", branch.name);
    if (runId) { q.set("run", runId); q.set("cycle", String(cycle)); }
    for (const [k, v] of Object.entries(extra)) { if (v === null) q.delete(k); else q.set(k, v); }
    const s = q.toString();
    return s ? `?${s}` : "";
  }, [branch.name, runId, cycle]);

  const value = useMemo<Session>(() => ({ branch, branches, runs, run, cycle: Math.min(cycle, maxCycle), setRun, setCycle, query }), [branch, branches, runs, run, cycle, maxCycle, setRun, setCycle, query]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
