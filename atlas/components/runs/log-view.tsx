"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { RunEvent } from "@/lib/runs/types";
import { Empty } from "../ui";

const ROW = 28;

/**
 * Virtualized event log. Rows are absolutely placed by transform and reused by index,
 * so filtering and scrolling never rebuild the list. Clicking a row moves the shared cycle.
 */
export function LogView({ events, cycle, onCycle, compact }: { events: RunEvent[]; cycle: number; onCycle: (c: number) => void; compact?: boolean }) {
  const blocks = useMemo(() => [...new Set(events.map((e) => e.block))].sort(), [events]);
  const types = useMemo(() => [...new Set(events.map((e) => e.type))].sort(), [events]);
  const [block, setBlock] = useState("");
  const [off, setOff] = useState<Set<string>>(new Set());
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return events.filter((e) => (!block || e.block === block) && !off.has(e.type) && (!needle || e.details.toLowerCase().includes(needle) || String(e.cycle) === needle));
  }, [events, block, off, q]);

  // Current row: the last event at or before the cursor.
  const current = useMemo(() => {
    let lo = 0, hi = rows.length - 1, ans = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (rows[m].cycle <= cycle) { ans = m; lo = m + 1; } else hi = m - 1; }
    return ans;
  }, [rows, cycle]);

  const scroller = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ top: 0, h: 400 });
  const clicked = useRef(false);
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setView({ top: el.scrollTop, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // Follow the cursor when it moves from elsewhere (wave, process, keyboard).
  useEffect(() => {
    const el = scroller.current;
    if (!el || current < 0) return;
    if (clicked.current) { clicked.current = false; return; }
    const y = current * ROW;
    if (y < el.scrollTop || y + ROW > el.scrollTop + el.clientHeight) el.scrollTop = Math.max(0, y - el.clientHeight / 2);
  }, [current]);

  const first = Math.max(0, Math.floor(view.top / ROW) - 8);
  const last = Math.min(rows.length, Math.ceil((view.top + view.h) / ROW) + 8);

  return (
    <div className={`log${compact ? " compact-log" : ""}`}>
      {!compact && (
        <div className="log-filters">
          <select className="field sans" value={block} onChange={(e) => setBlock(e.target.value)} aria-label="Block">
            <option value="">All blocks</option>
            {blocks.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
          <div className="row" style={{ gap: 4, flexWrap: "wrap" }} role="group" aria-label="Event types">
            {types.map((t) => (
              <button key={t} type="button" className={`tchip t-${t}${off.has(t) ? " off" : ""}`} aria-pressed={!off.has(t)}
                onClick={() => setOff((o) => { const n = new Set(o); if (n.has(t)) n.delete(t); else n.add(t); return n; })}>{t}</button>
            ))}
          </div>
          <span className="grow" />
          <input className="field sans" placeholder="Search details or cycle" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search the log" style={{ width: 200 }} />
          <span className="muted mono" style={{ fontSize: 11 }}>{rows.length}/{events.length}</span>
        </div>
      )}
      <div className="log-scroll" ref={scroller} onScroll={(e) => { const t = e.currentTarget.scrollTop; setView((v) => (v.top === t ? v : { ...v, top: t })); }}>
        <div style={{ height: rows.length * ROW, position: "relative" }} role="list" aria-label="Events">
          {rows.slice(first, last).map((e, i) => {
            const idx = first + i;
            return (
              <button key={idx} type="button" role="listitem" className={`log-row${idx === current ? " on" : ""}`} style={{ transform: `translateY(${idx * ROW}px)` }}
                onClick={() => { clicked.current = true; onCycle(e.cycle); }}>
                <span className="lc">{e.cycle}</span>
                <span className="lb">{e.block}</span>
                <span className={`lt t-${e.type}`}>{e.type}</span>
                <span className="ld">{e.details}</span>
              </button>
            );
          })}
        </div>
        {!rows.length && <Empty title="No events match">Clear a filter or the search to see events again.</Empty>}
      </div>
    </div>
  );
}
