"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { SpecModel } from "@/lib/spec/types";
import { useSession, hasFile } from "../shell/session";
import { Empty, Note } from "../ui";
import { useRunFile } from "../runs/use-run";
import { hops, OccupancyStrip, Pipeline, stageOf, stallsAt, stallStarts, usePlayer } from "./pipeline";

export function ProcessView({ spec }: { spec: SpecModel }) {
  const { run, runs, cycle, setCycle, setRun, query } = useSession();
  const process = spec.process[0];
  const tx = useRunFile(run?.id ?? null, "transactions", hasFile(run, "transactions.json"));
  const signals = useRunFile(run?.id ?? null, "signals", hasFile(run, "signals.json"));
  const sig = signals.state === "ready" ? signals.data : null;
  const [selected, setSelected] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const max = Math.max(0, (run?.cycles ?? 1) - 1);
  const player = usePlayer(cycle, max, setCycle);
  const starts = useMemo(() => (process ? stallStarts(process, sig) : []), [process, sig]);
  const stalls = useMemo(() => (process ? stallsAt(process, sig, cycle) : new Map<string, string>()), [process, sig, cycle]);

  if (!process) {
    return (
      <div className="page" style={{ display: "grid", placeItems: "center" }}>
        <Empty icon="⇉" title="No process section in this spec" action={<Link className="btn primary" href="/process?branch=atlas%2Fvector-dispatch-64b">Open the sample branch</Link>}>
          The process view draws the stages a spec’s <code>process:</code> section declares: registers, queue slots and the handshakes between them.
          {" "}Try the sample branch <code>atlas/vector-dispatch-64b</code>.
        </Empty>
      </div>
    );
  }
  const playable = runs.filter((r) => hasFile(r, "transactions.json"));
  const token = tx.state === "ready" ? tx.data.transactions.find((t) => t.id === selected) ?? null : null;
  const stageDef = process.stages.find((s) => s.key === stage);
  const hopList = token ? hops(token) : [];
  const nextStall = starts.find((c) => c > cycle);
  const prevStall = [...starts].reverse().find((c) => c < cycle);

  return (
    <div className="page">
      <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1fr) 336px", gridTemplateRows: "auto minmax(0, 1fr)" }}>
        <section className="float pad stack" style={{ gridColumn: "1 / -1" }}>
          <div className="row" style={{ flexWrap: "wrap" }}>
            <h1>{process.name}</h1>
            <span className="muted">{process.stages.length} stages · from the spec’s <code>process:</code> section</span>
            <span className="grow" />
            {run && <span className="muted">run <b className="mono">#{run.id}</b> · {run.test}</span>}
          </div>
          <div className="scrub">
            <button type="button" className="btn primary" style={{ minWidth: 72 }} onClick={player.toggle} disabled={!run || tx.state !== "ready"} aria-pressed={player.playing}>{player.playing ? "Pause" : "Play"}</button>
            <button type="button" className="btn sm" disabled={prevStall === undefined} onClick={() => { player.stop(); setCycle(prevStall!); }}>‹ stall</button>
            <input type="range" min={0} max={max} value={cycle} onChange={(e) => { player.stop(); setCycle(Number(e.target.value)); }} aria-label="Cycle" disabled={!run} />
            <button type="button" className="btn sm" disabled={nextStall === undefined} onClick={() => { player.stop(); setCycle(nextStall!); }}>stall ›</button>
            <span className="mono" style={{ minWidth: 96, textAlign: "right" }}>cycle <b>{cycle}</b></span>
          </div>
        </section>

        <section className="float pad" style={{ overflow: "auto" }} aria-label="Pipeline">
          {!run || !hasFile(run, "transactions.json") ? (
            <Empty icon="⇉" title={run ? `Run #${run.id} has no transactions` : "No run selected"}
              action={playable.length ? <div className="row">{playable.map((r) => <button key={r.id} type="button" className="btn" onClick={() => setRun(r.id)}>Open #{r.id}</button>)}</div> : undefined}>
              The tracer writes <code>transactions.json</code> in CI. {playable.length ? "These runs have one:" : "No run on this branch has one yet."}
            </Empty>
          ) : tx.state === "ready" ? (
            <div className="fade-in stack" style={{ gap: 24 }}>
              <Pipeline process={process} tx={tx.data.transactions} signals={sig} cycle={cycle} selected={selected}
                onToken={(id) => { setSelected(id === selected ? null : id); setStage(null); }}
                onStage={(k) => { setStage(k); setSelected(null); }}
                stageHref={(b) => `/canvas${query({ focus: spec.blocks[b]?.parent ?? null, sel: `block:${b}` })}`} />
              {signals.state === "missing" && <Note kind="warn">signals.json is missing for this run, so stalls cannot be detected.</Note>}
              <OccupancyStrip process={process} tx={tx.data.transactions} signals={sig} cycle={cycle} cycles={run.cycles} onCycle={(c) => { player.stop(); setCycle(c); }} />
            </div>
          ) : tx.state === "missing" || tx.state === "error" ? (
            <Note kind="err" action={<button type="button" className="btn sm" onClick={tx.retry}>Retry</button>}><b>transactions.json could not load.</b> {tx.message}</Note>
          ) : (
            <div className="pipe" aria-busy="true">
              {process.stages.map((s) => (
                <div key={s.key} className={`stage${s.capacity > 1 ? " q" : ""}`}>
                  <div className="sk sk-line" style={{ width: "60%" }} />
                  <div className="stage-box">{Array.from({ length: s.capacity }, (_, i) => <div key={i} className="sk" style={{ height: 44 }} />)}</div>
                </div>
              ))}
            </div>
          )}
        </section>

        <aside className="float panel" aria-label="Details">
          {token ? (
            <>
              <div className="panel-hd"><h2 className="mono">{token.id}</h2>{token.dropped && <span className="rs rs-failed">dropped</span>}<span className="grow" /><button type="button" className="btn ghost icon sm" aria-label="Clear" onClick={() => setSelected(null)}>✕</button></div>
              <div className="panel-body pad stack" key={token.id}>
                <div className="mono fade-in" style={{ fontSize: 14 }}>{token.label}</div>
                <div className="muted mono">pc {token.pc}</div>
                {token.dropped && <Note kind="err">This command never reached the queue: <b>vcmd.valid</b> fell while the queue was full. See the <b>no-drop</b> violation on the Runs page.</Note>}
                <div className="row">
                  <button type="button" className="btn sm" disabled={!hopList.some((h) => h < cycle)} onClick={() => setCycle([...hopList].reverse().find((h) => h < cycle)!)}>‹ previous hop</button>
                  <button type="button" className="btn sm primary" disabled={!hopList.some((h) => h > cycle)} onClick={() => setCycle(hopList.find((h) => h > cycle)!)}>next hop ›</button>
                </div>
                <div className="timeline">
                  <span className="cap">stage</span><span className="cap">in</span><span className="cap">out</span><span className="cap">cycles</span>
                  {process.stages.filter((s) => token.stages[s.key]).map((s) => {
                    const [a, b] = token.stages[s.key];
                    const here = stageOf(token, cycle) === s.key;
                    return [
                      <button key={`${s.key}n`} type="button" className="orow" style={{ minHeight: 24, padding: "0 4px", fontWeight: here ? 600 : 400 }} onClick={() => setCycle(a)}>{here ? "▸ " : ""}{s.name}</button>,
                      <span key={`${s.key}a`}>{a}</span>, <span key={`${s.key}b`}>{b}</span>, <span key={`${s.key}c`} className={b - a + 1 > 3 ? "r-different" : ""}>{b - a + 1}</span>,
                    ];
                  })}
                </div>
                <Link className="btn sm" href={`/runs${query({ cycle: String(Object.values(token.stages)[0][0]) })}`}>Open in log</Link>
              </div>
            </>
          ) : stageDef ? (
            <>
              <div className="panel-hd"><h2>{stageDef.name}</h2><span className="grow" /><button type="button" className="btn ghost icon sm" aria-label="Clear" onClick={() => setStage(null)}>✕</button></div>
              <div className="panel-body pad stack" key={stageDef.key}>
                <dl className="kv fade-in">
                  <dt>block</dt><dd className="mono">{stageDef.block}</dd>
                  <dt>ordering</dt><dd>{stageDef.ordering}</dd>
                  <dt>capacity</dt><dd>{stageDef.capacity}</dd>
                  {stageDef.enq && <><dt>enq</dt><dd className="mono">{stageDef.enq.valid} / {stageDef.enq.ready}</dd></>}
                  {stageDef.deq && <><dt>deq</dt><dd className="mono">{stageDef.deq.valid} / {stageDef.deq.ready}</dd></>}
                </dl>
                {stalls.has(stageDef.key) ? <Note kind="warn"><b>Stalled at {cycle}.</b> {stalls.get(stageDef.key)}</Note> : <p className="muted">Not stalled at cycle {cycle}.</p>}
                {(stageDef.deq || stageDef.enq) && (
                  <Link className="btn sm" href={`/runs${query({ em: [stageDef.deq?.valid, stageDef.deq?.ready, stageDef.enq?.valid, stageDef.enq?.ready].filter(Boolean).join(",") })}`}>Show signals in the waveform</Link>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="panel-hd"><h2>At cycle {cycle}</h2></div>
              <div className="panel-body pad stack">
                {stalls.size ? [...stalls].map(([k, why]) => {
                  const s = process.stages.find((x) => x.key === k)!;
                  return <button key={k} type="button" className="orow" style={{ fontFamily: "var(--sans)", alignItems: "flex-start" }} onClick={() => setStage(k)}><span className="r-different">●</span><span className="grow"><b>{s.name}</b> stalled<br /><span className="muted">{why}</span></span></button>;
                }) : <p className="muted">No stage is stalled.</p>}
                <p className="faint" style={{ fontSize: 12 }}>Click a token to follow an instruction across stages, or a stage for its handshake signals. <kbd>←</kbd> <kbd>→</kbd> step one cycle, <kbd>Shift</kbd> for ten.</p>
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
