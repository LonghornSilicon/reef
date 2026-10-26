import type { SignalTrace } from "../runs/types";

/** Value of a traced signal at `cycle` (binary search over value changes). Before the first change → 0. */
export function valueAt(trace: SignalTrace | undefined, cycle: number): number {
  if (!trace || !trace.changes.length) return 0;
  const ch = trace.changes;
  let lo = 0, hi = ch.length - 1;
  if (cycle < ch[0][0]) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ch[mid][0] <= cycle) lo = mid; else hi = mid - 1;
  }
  return ch[lo][1];
}

/** Value changes inside [from, to), with the value entering the window first. */
export function windowChanges(trace: SignalTrace | undefined, from: number, to: number): Array<[number, number]> {
  if (!trace) return [];
  const out: Array<[number, number]> = [[from, valueAt(trace, from)]];
  for (const [c, v] of trace.changes) if (c > from && c < to) out.push([c, v]);
  return out;
}

export const hex = (v: number, width: number) => (width <= 1 ? String(v) : `0x${v.toString(16).padStart(Math.ceil(width / 4) > 8 ? 8 : Math.ceil(width / 4), "0")}`);
