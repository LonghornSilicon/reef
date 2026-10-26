"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { SignalsFile } from "@/lib/runs/types";
import type { Violation } from "@/lib/sim/check";
import type { SignalGroup } from "@/lib/sim/groups";
import { hex, valueAt, windowChanges } from "@/lib/sim/signals";

const ROW = 24;
const STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];

type Props = {
  signals: SignalsFile;
  groups: SignalGroup[];
  violations: Violation[];
  cycle: number;
  onCycle: (c: number) => void;
  /** Trace keys to emphasize (a selected violation or a stall). */
  emphasis?: Set<string>;
  span?: number;
  labelWidth?: number;
};

/**
 * Waveform strip. Stand-in for the Surfer embed (plan.md §M4) with the same contract:
 * jump to a cycle, grouped by spec interface, violation markers. The cursor moves by transform;
 * paths only redraw when the visible window moves.
 */
export function WaveView({ signals, groups, violations, cycle, onCycle, emphasis, span: initialSpan = 64, labelWidth = 248 }: Props) {
  const tracks = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  const [span, setSpan] = useState(initialSpan);
  const [start, setStart] = useState(() => Math.max(0, cycle - Math.floor(initialSpan / 4)));
  const maxStart = Math.max(0, signals.cycles - span);

  useEffect(() => {
    const el = tracks.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Keep the cursor in view: page the window when it leaves.
  useEffect(() => {
    if (cycle < start || cycle >= start + span) {
      setStart(Math.min(maxStart, Math.max(0, cycle - Math.floor(span / 4))));
    }
  }, [cycle, start, span, maxStart]);

  const px = w / span;
  const rows = useMemo(() => {
    const out: Array<{ kind: "group"; title: string } | { kind: "sig"; key: string }> = [];
    for (const g of groups) { out.push({ kind: "group", title: g.title }); for (const k of g.keys) out.push({ kind: "sig", key: k }); }
    return out;
  }, [groups]);
  const height = (rows.length + 2) * ROW;
  const end = start + span;
  const step = STEPS.find((s) => s * px >= 56) ?? 1000;

  const drawn = useMemo(() => rows.map((r, i) => {
    if (r.kind === "group") return null;
    const t = signals.signals[r.key];
    const y = (i + 1) * ROW;
    const ch = windowChanges(t, start, end);
    const x = (c: number) => (c - start) * px;
    if (t.width === 1) {
      let d = "";
      const fills: Array<[number, number]> = [];
      ch.forEach(([c, v], j) => {
        const c1 = j + 1 < ch.length ? ch[j + 1][0] : end;
        const yy = y + (v ? 5 : 19);
        d += j === 0 ? `M ${x(c)} ${yy}` : ` V ${yy}`;
        d += ` H ${x(c1)}`;
        if (v) fills.push([x(c), x(c1)]);
      });
      return { key: r.key, y, d, fills, bus: [] as Array<{ x0: number; x1: number; text: string }> };
    }
    const bus = ch.map(([c, v], j) => ({ x0: x(c), x1: x(j + 1 < ch.length ? ch[j + 1][0] : end), text: hex(v, t.width) }));
    return { key: r.key, y, d: "", fills: [] as Array<[number, number]>, bus };
  }), [rows, signals, start, end, px]);

  const pick = (clientX: number) => {
    const r = tracks.current!.getBoundingClientRect();
    onCycle(Math.min(signals.cycles - 1, Math.max(0, start + Math.floor((clientX - r.left) / px))));
  };
  const drag = useRef(false);

  useEffect(() => {
    const el = tracks.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const d = Math.round((e.shiftKey ? e.deltaY : e.deltaX) / px);
        setStart((s) => Math.min(Math.max(0, signals.cycles - span), Math.max(0, s + d)));
        return;
      }
      const r = el.getBoundingClientRect();
      const at = start + (e.clientX - r.left) / px;
      const next = Math.min(1024, Math.max(16, Math.round(span * Math.exp(e.deltaY * 0.002))));
      setSpan(next);
      setStart(Math.min(Math.max(0, signals.cycles - next), Math.max(0, Math.round(at - ((e.clientX - r.left) / w) * next))));
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, [px, span, start, signals.cycles, w]);

  // When the window itself moved, the cursor jumps with it instead of sliding.
  const shownWindow = useRef(`${start}:${span}`);
  const instant = shownWindow.current !== `${start}:${span}`;
  useEffect(() => { shownWindow.current = `${start}:${span}`; });
  const ticks: number[] = [];
  for (let c = Math.ceil(start / step) * step; c < end; c += step) ticks.push(c);
  const vRow = (rows.length + 1) * ROW;

  return (
    <div className="wave" style={{ ["--label-w" as string]: `${labelWidth}px` }}>
      <div className="wave-labels">
        <div className="grp">cycle</div>
        {rows.map((r, i) => r.kind === "group"
          ? <div key={`g${i}`} className="grp ellipsis" title={r.title}>{r.title}</div>
          : (
            <div key={r.key} className={emphasis?.has(r.key) ? "em" : ""} title={r.key}>
              <span className="grow ellipsis">{r.key.split(".").slice(1).join(".")}</span>
              <span className="mono faint" style={{ paddingRight: 8 }}>{hex(valueAt(signals.signals[r.key], cycle), signals.signals[r.key].width)}</span>
            </div>
          ))}
        <div className="viol">violations · {violations.length}</div>
      </div>
      <div className={`wave-tracks${drag.current ? " drag" : ""}`} ref={tracks} role="slider" aria-label="Cycle" aria-valuemin={0} aria-valuemax={signals.cycles - 1} aria-valuenow={cycle} tabIndex={0}
        onPointerDown={(e) => { drag.current = true; e.currentTarget.setPointerCapture(e.pointerId); pick(e.clientX); }}
        onPointerMove={(e) => { if (drag.current) pick(e.clientX); }}
        onPointerUp={(e) => { drag.current = false; e.currentTarget.releasePointerCapture(e.pointerId); }}>
        <svg width={w} height={height} aria-hidden>
          {ticks.map((c) => (
            <g key={c}>
              <line className="tickl" x1={(c - start) * px} x2={(c - start) * px} y1={ROW - 4} y2={height} />
              <text className="tick" x={(c - start) * px + 4} y={16}>{c}</text>
            </g>
          ))}
          {drawn.map((s) => s && (
            <g key={s.key} className={emphasis?.has(s.key) ? "em" : ""}>
              <rect className="wrow-bg" x={0} y={s.y} width={w} height={ROW} />
              {s.fills.map(([a, b], j) => <rect key={j} className="hi-fill" x={a} y={s.y + 5} width={Math.max(0, b - a)} height={14} />)}
              {s.d && <path className="hi" d={s.d} />}
              {s.bus.map((b, j) => {
                const x0 = b.x0 + 1, x1 = Math.max(x0 + 2, b.x1 - 1), m = Math.min(3, (x1 - x0) / 2);
                return (
                  <g key={j}>
                    <path className="bus" d={`M ${x0} ${s.y + 12} L ${x0 + m} ${s.y + 4} H ${x1 - m} L ${x1} ${s.y + 12} L ${x1 - m} ${s.y + 20} H ${x0 + m} Z`} />
                    {x1 - x0 > 64 && <text className="bus-t" x={x0 + 6} y={s.y + 16}>{b.text}</text>}
                  </g>
                );
              })}
            </g>
          ))}
          {violations.filter((v) => v.cycle >= start && v.cycle < end).map((v, i) => (
            <rect key={i} className={`vm${v.severity === "warn" ? " warn" : ""}`} x={(v.cycle - start + 0.5) * px - 4} y={vRow + 8} width={8} height={8}
              transform={`rotate(45 ${(v.cycle - start + 0.5) * px} ${vRow + 12})`}>
              <title>{`${v.rule} @ ${v.cycle}: ${v.message}`}</title>
            </rect>
          ))}
        </svg>
        <div className={`wave-cursor${instant ? " instant" : ""}`} style={{ ["--x" as string]: `${(cycle - start + 0.5) * px}px` }}><b>{cycle}</b></div>
      </div>
    </div>
  );
}
