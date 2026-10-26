"use client";

import { useCallback, useEffect, useState } from "react";
import type { RunEvent, SignalsFile, TransactionsFile } from "@/lib/runs/types";

export type Loadable<T> =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "ready"; data: T }
  | { state: "missing"; message: string }
  | { state: "error"; message: string };

type Files = { events: RunEvent[]; signals: SignalsFile; transactions: TransactionsFile };
const FILE: Record<keyof Files, string> = { events: "events.jsonl", signals: "signals.json", transactions: "transactions.json" };

/** One fetch per run file per page load, shared by every view that asks for it. */
const cache = new Map<string, Promise<unknown>>();

class Missing extends Error {}

function fetchFile<K extends keyof Files>(run: string, kind: K): Promise<Files[K]> {
  const key = `${run}/${kind}`;
  let p = cache.get(key) as Promise<Files[K]> | undefined;
  if (!p) {
    p = fetch(`/api/runs/${run}/${FILE[kind]}`).then(async (r) => {
      if (r.status === 404) throw new Missing(((await r.json()) as { error: string }).error);
      if (!r.ok) throw new Error(`${FILE[kind]}: HTTP ${r.status}`);
      const text = await r.text();
      if (kind === "events") return text.split("\n").filter(Boolean).map((l) => JSON.parse(l)) as Files[K];
      return JSON.parse(text) as Files[K];
    });
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return p;
}

/** A run file as loading → ready | missing | error, with retry. `run` null → idle. */
export function useRunFile<K extends keyof Files>(run: string | null, kind: K, enabled = true): Loadable<Files[K]> & { retry: () => void } {
  const [s, set] = useState<Loadable<Files[K]>>({ state: "idle" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!run || !enabled) { set({ state: "idle" }); return; }
    let live = true;
    set({ state: "loading" });
    fetchFile(run, kind).then(
      (data) => live && set({ state: "ready", data }),
      (e: Error) => live && set(e instanceof Missing ? { state: "missing", message: e.message } : { state: "error", message: e.message }),
    );
    return () => { live = false; };
  }, [run, kind, enabled, attempt]);
  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { ...s, retry };
}
