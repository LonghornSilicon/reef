"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { SpecProcess } from "@/lib/spec/types";
import type { SignalsFile, Transaction } from "@/lib/runs/types";
import { valueAt } from "@/lib/sim/signals";

/** A stage is stalled when its output handshake offers (valid) and is refused (!ready). */
export function stallsAt(process: SpecProcess, signals: SignalsFile | null, cycle: number): Map<string, string> {
  const out = new Map<string, string>();
  if (!signals) return out;
  for (const s of process.stages) {
    if (!s.deq) continue;
    if (valueAt(signals.signals[s.deq.valid], cycle) && !valueAt(signals.signals[s.deq.ready], cycle)) out.set(s.key, s.stallReason ?? `${s.name} cannot hand off`);
  }
  return out;
}

/** Cycles where some stage starts stalling, for "next stall" jumps. */
export function stallStarts(process: SpecProcess, signals: SignalsFile | null): number[] {
  if (!signals) return [];
  const out: number[] = [];
  let prev = false;
  for (let c = 0; c < signals.cycles; c++) {
    const now = stallsAt(process, signals, c).size > 0;
    if (now && !prev) out.push(c);
    prev = now;
  }
  return out;
}

/** Which stage a token is in at `cycle`, or null. */
export function stageOf(t: Transaction, cycle: number): string | null {
  for (const [k, [a, b]] of Object.entries(t.stages)) if (cycle >= a && cycle <= b) return k;
  return null;
}

/** Cycles where the token changes stage. */
export function hops(t: Transaction): number[] {
  return [...new Set(Object.values(t.stages).flatMap(([a, b]) => [a, b + 1]))].sort((x, y) => x - y);
}

type Pos = { x: number; y: number; w: number };

type Props = {
  process: SpecProcess;
  tx: Transaction[];
  signals: SignalsFile | null;
  cycle: number;
  selected: string | null;
  onToken: (id: string) => void;
  onStage?: (key: string) => void;
  stageHref?: (blockId: string) => string;
  compact?: boolean;
};

/**
 * Stages and slots are laid out once; tokens live in one overlay and only their
 * --x/--y custom properties change, so moving between stages is a transform transition.
 */
export function Pipeline({ process, tx, signals, cycle, selected, onToken, onStage, stageHref, compact }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [slots, setSlots] = useState<Record<string, Pos>>({});

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => {
      const base = el.getBoundingClientRect();
      const next: Record<string, Pos> = {};
      el.querySelectorAll<HTMLElement>("[data-slot]").forEach((s) => {
        const r = s.getBoundingClientRect();
        next[s.dataset.slot!] = { x: r.left - base.left, y: r.top - base.top, w: r.width };
      });
      setSlots(next);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [process, compact]);

  const stalled = useMemo(() => stallsAt(process, signals, cycle), [process, signals, cycle]);

  // Tokens near the cursor; ones outside their stages fade at the first or last stage they touch.
  const tokens = useMemo(() => {
    const order = process.stages.map((s) => s.key);
    const near = tx.filter((t) => {
      const all = Object.values(t.stages);
      return all.length && Math.min(...all.map((s) => s[0])) <= cycle + 4 && Math.max(...all.map((s) => s[1])) >= cycle - 4;
    });
    const byStage = new Map<string, Transaction[]>();
    for (const t of near) {
      const s = stageOf(t, cycle);
      if (s) byStage.set(s, [...(byStage.get(s) ?? []), t]);
    }
    for (const list of byStage.values()) list.sort((a, b) => a.stages[stageOf(a, cycle)!][0] - b.stages[stageOf(b, cycle)!][0] || a.id.localeCompare(b.id));
    return near.map((t) => {
      const s = stageOf(t, cycle);
      const cap = process.stages.find((x) => x.key === s)?.capacity ?? 1;
      if (s) return { t, stage: s, slot: Math.min(cap - 1, byStage.get(s)!.indexOf(t)), out: false };
      const keys = Object.keys(t.stages).sort((a, b) => order.indexOf(a) - order.indexOf(b));
      const before = cycle < t.stages[keys[0]][0];
      return { t, stage: before ? keys[0] : keys[keys.length - 1], slot: 0, out: true };
    });
  }, [tx, cycle, process.stages]);

  return (
    <div className={`pipe-wrap${compact ? " compact" : ""}`} ref={wrap}>
      <div className="pipe">
        {process.stages.map((s) => (
          <div key={s.key} className={`stage${s.capacity > 1 ? " q" : ""}${stalled.has(s.key) ? " stalled" : ""}`}>
            <div className="stage-hd">
              <b title={s.name}>{s.name}{s.capacity > 1 && <span style={{ fontWeight: 400, letterSpacing: 0, textTransform: "none" }}> ×{s.capacity}</span>}</b>
              {stageHref ? <a href={stageHref(s.block)}>{s.block}</a> : <span>{s.block}</span>}
            </div>
            <div className="stage-box" onClick={() => onStage?.(s.key)} style={{ cursor: onStage ? "pointer" : undefined }}>
              {Array.from({ length: s.capacity }, (_, i) => <div key={i} className="slot" data-slot={`${s.key}:${i}`} />)}
            </div>
            <div className="stage-note" aria-live="polite">{stalled.get(s.key) ?? ""}</div>
          </div>
        ))}
      </div>
      <div className="toks">
        {tokens.map(({ t, stage, slot, out }) => {
          const p = slots[`${stage}:${slot}`];
          if (!p) return null;
          return (
            <button key={t.id} type="button" tabIndex={out ? -1 : 0}
              className={`tok${out ? " out" : ""}${t.id === selected ? " f" : ""}${t.dropped ? " drop" : ""}`}
              style={{ ["--x" as string]: `${p.x}px`, ["--y" as string]: `${p.y}px`, width: p.w }}
              onClick={() => onToken(t.id)} aria-label={`${t.id} ${t.label}`}>
              <b>{t.id}{t.dropped ? " ✕" : ""}</b>
              <small>{t.label}</small>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Play/pause at a steady rate; stops at the end. */
export function usePlayer(cycle: number, max: number, setCycle: (f: (c: number) => number) => void) {
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setCycle((c) => c + 1), 160);
    return () => clearInterval(id);
  }, [playing, setCycle]);
  useEffect(() => { if (playing && cycle >= max) setPlaying(false); }, [cycle, max, playing]);
  return { playing, toggle: () => setPlaying((p) => !p), stop: () => setPlaying(false) };
}

/**
 * Whole-run overview: one strip per stage, shaded by occupancy, amber where the stage stalls.
 * Clicking moves the shared cycle; the cursor slides by transform.
 */
export function OccupancyStrip({ process, tx, signals, cycle, cycles, onCycle }: { process: SpecProcess; tx: Transaction[]; signals: SignalsFile | null; cycle: number; cycles: number; onCycle: (c: number) => void }) {
  const box = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const ROWH = 14;
  const bins = Math.max(1, Math.floor(w / 3));
  const per = cycles / bins;
  const data = useMemo(() => process.stages.map((s) => {
    const diff = new Int32Array(cycles + 1);
    for (const t of tx) { const iv = t.stages[s.key]; if (iv) { diff[iv[0]] += 1; diff[Math.min(cycles, iv[1] + 1)] -= 1; } }
    const occ = new Float32Array(bins), stall = new Uint8Array(bins);
    let run = 0;
    for (let c = 0; c < cycles; c++) {
      run += diff[c];
      const b = Math.min(bins - 1, Math.floor(c / per));
      occ[b] = Math.max(occ[b], run / s.capacity);
      if (s.deq && signals && valueAt(signals.signals[s.deq.valid], c) && !valueAt(signals.signals[s.deq.ready], c)) stall[b] = 1;
    }
    return { s, occ, stall };
  }), [process, tx, signals, cycles, bins, per]);

  const pick = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onCycle(Math.round(((e.clientX - r.left - 120) / (r.width - 120)) * (cycles - 1)));
  };
  const dragging = useRef(false);
  const plot = w - 120;
  return (
    <div className="stack" style={{ gap: 4 }}>
      <div className="row"><span className="cap grow">Occupancy over the run</span><span className="faint" style={{ fontSize: 11 }}>darker = fuller · amber = stalled · click or drag to move</span></div>
      <div ref={box} style={{ position: "relative", cursor: "ew-resize", touchAction: "none" }}
        onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); pick(e); }}
        onPointerMove={(e) => dragging.current && pick(e)}
        onPointerUp={() => { dragging.current = false; }}>
        <svg width={w} height={data.length * (ROWH + 4)} aria-label="Stage occupancy across the run" role="img">
          {data.map(({ s, occ, stall }, i) => (
            <g key={s.key} transform={`translate(0 ${i * (ROWH + 4)})`}>
              <text x={0} y={ROWH - 3} style={{ font: "10px var(--mono)", fill: "var(--muted)" }}>{s.name}</text>
              <rect x={120} y={0} width={Math.max(0, plot)} height={ROWH} rx={3} fill="var(--subtle)" />
              {Array.from(occ, (o, b) => (o > 0 || stall[b]) && (
                <rect key={b} x={120 + (b / bins) * plot} y={0} width={plot / bins + 0.5} height={ROWH}
                  fill={stall[b] ? "var(--diff)" : "var(--accent)"} opacity={stall[b] ? 0.75 : 0.15 + 0.6 * Math.min(1, o)} />
              ))}
            </g>
          ))}
        </svg>
        <div className="wave-cursor" style={{ ["--x" as string]: `${120 + (cycle / Math.max(1, cycles - 1)) * plot}px` }} />
      </div>
    </div>
  );
}
