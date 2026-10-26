"use client";

import Link from "next/link";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import type { SpecModel } from "@/lib/spec/types";
import type { CompareReport } from "@/lib/rtl/types";
import type { RunEvent } from "@/lib/runs/types";
import { checkRun } from "@/lib/sim/check";
import { signalGroups } from "@/lib/sim/groups";
import { useSession, hasFile } from "../shell/session";
import { Empty, Note } from "../ui";
import { buildBoard, visibleAncestor } from "../canvas/geometry";
import { BoardView } from "../canvas/board-view";
import { useCamera } from "../canvas/use-camera";
import { LogView } from "../runs/log-view";
import { useRunFile } from "../runs/use-run";
import { WaveView } from "../runs/wave-view";
import { Pipeline, stallsAt, usePlayer } from "../process/pipeline";

/** The layout that shows every block a process touches, with the most of them as separate nodes. */
function pickLayout(spec: SpecModel, blocks: string[]): string | null {
  let best: string | null = null, score = -1;
  for (const [id, l] of Object.entries(spec.layouts)) {
    const visible = new Set(Object.keys(l.nodes));
    const hits = blocks.map((b) => visibleAncestor(spec, b, visible));
    if (hits.some((h) => !h)) continue;
    const distinct = new Set(hits).size;
    if (distinct > score) { score = distinct; best = id; }
  }
  return best;
}

/**
 * C1 synced workspace: canvas, process, log and waveform on one shared cycle.
 * Each pane is built once; moving the cycle only changes classes, transforms and the visible window.
 */
export function Workspace({ spec, report }: { spec: SpecModel; report: CompareReport | null }) {
  const { run, runs, cycle, setCycle, setRun, query } = useSession();
  const process = spec.process[0];
  const events = useRunFile(run?.id ?? null, "events", hasFile(run, "events.jsonl"));
  const signals = useRunFile(run?.id ?? null, "signals", hasFile(run, "signals.json"));
  const tx = useRunFile(run?.id ?? null, "transactions", hasFile(run, "transactions.json"));
  const sig = signals.state === "ready" ? signals.data : null;
  const max = Math.max(0, (run?.cycles ?? 1) - 1);
  const player = usePlayer(cycle, max, setCycle);
  const check = useMemo(() => (sig ? checkRun(spec, sig) : null), [spec, sig]);
  const groups = useMemo(() => (sig ? signalGroups(spec, sig) : []), [spec, sig]);
  const stalls = useMemo(() => (process ? stallsAt(process, sig, cycle) : new Map<string, string>()), [process, sig, cycle]);

  const focus = useMemo(() => (process ? pickLayout(spec, process.stages.map((s) => s.block)) : null) ?? spec.product.defaultFocus, [process, spec]);
  const board = useMemo(() => buildBoard(spec, focus), [spec, focus]);

  // Canvas overlay: blocks with events in the last few cycles, and blocks whose stage is stalled.
  const overlay = useMemo(() => {
    const active = new Map<string, string>(), stalled = new Map<string, string>();
    if (!board) return { active, stalled };
    const visible = new Set(board.nodes.map((n) => n.id));
    if (events.state === "ready") {
      for (const e of recent(events.data, cycle, 3)) {
        const at = visibleAncestor(spec, e.block, visible);
        if (at) active.set(at, `${e.type} · ${e.details}`);
      }
    }
    for (const [k, why] of stalls) {
      const s = process?.stages.find((x) => x.key === k);
      const at = s && visibleAncestor(spec, s.block, visible);
      if (at) stalled.set(at, `${s!.name} stalled: ${why}`);
    }
    return { active, stalled };
  }, [board, events, cycle, spec, stalls, process]);

  if (!run) {
    return (
      <div className="page" style={{ display: "grid", placeItems: "center" }}>
        <Empty icon="▦" title="No runs on this branch">
          The workspace replays one run across the canvas, process, log and waveform. Switch to <code>atlas/vector-dispatch-64b</code> to see one.
        </Empty>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="ws-bar float">
        <button type="button" className="btn primary" style={{ minWidth: 72 }} onClick={player.toggle} aria-pressed={player.playing} disabled={!hasFile(run, "signals.json")}>{player.playing ? "Pause" : "Play"}</button>
        <div className="scrub grow">
          <input type="range" min={0} max={max} value={cycle} onChange={(e) => { player.stop(); setCycle(Number(e.target.value)); }} aria-label="Cycle" />
        </div>
        <span className="mono">cycle <b>{cycle}</b></span>
        <span className="muted">run <b className="mono">#{run.id}</b> · {run.test}</span>
        {!hasFile(run, "signals.json") && runs.some((r) => hasFile(r, "signals.json")) && (
          <button type="button" className="btn sm" onClick={() => setRun(runs.find((r) => hasFile(r, "signals.json"))!.id)}>Open a run with a waveform</button>
        )}
      </div>
      <div className="ws">
        <section className="ws-pane float" aria-label="Canvas">
          <div className="panel-hd"><h3>Canvas</h3><span className="muted mono" style={{ fontSize: 11 }}>{spec.blocks[focus]?.name}</span><span className="grow" /><Link className="btn ghost sm" href={`/canvas${query({ focus })}`}>Open ›</Link></div>
          {board ? <MiniCanvas spec={spec} report={report} board={board} active={overlay.active} stalled={overlay.stalled} /> : <Empty title="No layout to draw" />}
        </section>
        <section className="ws-pane float" aria-label="Process">
          <div className="panel-hd"><h3>Process</h3><span className="muted">{process?.name ?? "—"}</span><span className="grow" /><Link className="btn ghost sm" href={`/process${query()}`}>Open ›</Link></div>
          <div className="pad" style={{ overflow: "auto", flex: 1 }}>
            {!process ? <Empty title="No process section">This spec has no <code>process:</code> section.</Empty>
              : tx.state === "ready" ? <Pipeline compact process={process} tx={tx.data.transactions} signals={sig} cycle={cycle} selected={null} onToken={() => {}} />
              : tx.state === "loading" ? <div className="stack">{Array.from({ length: 3 }, (_, i) => <div key={i} className="sk sk-block" />)}</div>
              : tx.state === "idle" ? <Empty title="No transactions">This run has no transactions.json.</Empty>
              : <Note kind="err" action={<button type="button" className="btn sm" onClick={tx.retry}>Retry</button>}>{tx.message}</Note>}
          </div>
        </section>
        <section className="ws-pane float" aria-label="Log">
          <div className="panel-hd"><h3>Log</h3><span className="grow" /><Link className="btn ghost sm" href={`/runs${query()}`}>Open ›</Link></div>
          <div style={{ flex: 1, minHeight: 0 }}>
            {events.state === "ready" ? <LogView compact events={events.data} cycle={cycle} onCycle={(c) => { player.stop(); setCycle(c); }} />
              : events.state === "loading" ? <div className="pad stack">{Array.from({ length: 6 }, (_, i) => <div key={i} className="sk sk-line" />)}</div>
              : events.state === "idle" ? <Empty title="No log">This run has no events.jsonl.</Empty>
              : <div className="pad"><Note kind="err" action={<button type="button" className="btn sm" onClick={events.retry}>Retry</button>}>{events.message}</Note></div>}
          </div>
        </section>
        <section className="ws-pane float" aria-label="Waveform">
          <div className="panel-hd"><h3>Waveform</h3>{check && <span className={check.violations.some((v) => v.severity === "error") ? "r-missing" : "muted"} style={{ fontSize: 12 }}>{check.violations.length} violation{check.violations.length === 1 ? "" : "s"}</span>}<span className="grow" /><Link className="btn ghost sm" href={`/runs${query()}`}>Open ›</Link></div>
          <div style={{ flex: 1, overflow: "auto" }}>
            {sig ? <WaveView signals={sig} groups={groups} violations={check?.violations ?? []} cycle={cycle} onCycle={(c) => { player.stop(); setCycle(c); }} span={48} labelWidth={200} />
              : signals.state === "loading" ? <div className="pad stack">{Array.from({ length: 6 }, (_, i) => <div key={i} className="sk sk-line" />)}</div>
              : <Empty title="No waveform">This run has no signals.json.</Empty>}
          </div>
        </section>
      </div>
    </div>
  );
}

/** Events in (cycle - window, cycle], newest last. */
function recent(events: RunEvent[], cycle: number, window: number): RunEvent[] {
  let lo = 0, hi = events.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (events[m].cycle <= cycle) lo = m + 1; else hi = m; }
  const out: RunEvent[] = [];
  for (let i = lo - 1; i >= 0 && events[i].cycle > cycle - window; i--) out.unshift(events[i]);
  return out;
}

function MiniCanvas({ spec, report, board, active, stalled }: { spec: SpecModel; report: CompareReport | null; board: NonNullable<ReturnType<typeof buildBoard>>; active: Map<string, string>; stalled: Map<string, string> }) {
  const vp = useRef<HTMLDivElement>(null);
  const el = useRef<HTMLDivElement>(null);
  const cam = useCamera(vp, el);
  const fit = () => cam.fitRect({ x: 0, y: 0, w: board.w, h: board.h }, { top: 8, right: 8, bottom: 8, left: 8 }, false, 1);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(() => fit(), [board]);
  useEffect(() => {
    const v = vp.current;
    if (!v) return;
    const ro = new ResizeObserver(fit);
    ro.observe(v);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [board]);
  return (
    <div className="viewport lens-conformance" ref={vp}>
      <div className="board" ref={el} style={{ width: board.w, height: board.h }}>
        <div className="board-content">
          <BoardView board={board} spec={spec} report={report} lens="conformance" selected={null} hover={null} active={active} stalled={stalled} />
        </div>
      </div>
    </div>
  );
}
