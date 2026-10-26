"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import type { SpecModel } from "@/lib/spec/types";
import { RUN_FILES, type RunMeta } from "@/lib/runs/types";
import { checkRun, type Violation } from "@/lib/sim/check";
import { signalGroups } from "@/lib/sim/groups";
import { duration, when } from "@/lib/format";
import { useSession } from "../shell/session";
import { Empty, Note } from "../ui";
import { LogView } from "./log-view";
import { useRunFile, type Loadable } from "./use-run";
import { WaveView } from "./wave-view";

export function RunsView({ spec }: { spec: SpecModel }) {
  const { runs, run, setRun, cycle, setCycle } = useSession();
  return (
    <div className="page">
      <div className="cols" style={{ gridTemplateColumns: "320px minmax(0, 1fr)" }}>
        <section className="panel float" aria-label="Run library">
          <div className="panel-hd"><h2>Runs</h2><span className="muted">{runs.length}</span></div>
          <div className="panel-body" style={{ padding: 4 }}>
            {!runs.length && <Empty icon="▶" title="No runs on this branch">Runs appear when <code>sim/run.py</code> uploads to this branch’s prefix.</Empty>}
            {runs.map((r) => (
              <button key={r.id} type="button" className={`orow run-card${r.id === run?.id ? " on" : ""}`} onClick={() => setRun(r.id)} aria-current={r.id === run?.id}>
                <span className="top">
                  <b className="mono">#{r.id}</b>
                  <span className="grow ellipsis">{r.test}</span>
                  <span className={`rs rs-${r.status}`}>{r.status}</span>
                </span>
                <small>{when(r.startedAt)} · {r.commit}{r.dirty ? " · dirty" : ""}</small>
                {r.summary && <small className="ellipsis">{r.summary}</small>}
              </button>
            ))}
          </div>
        </section>
        {run ? <RunDetail key={run.id} spec={spec} run={run} cycle={cycle} setCycle={setCycle} /> : (
          <section className="panel float"><Empty icon="▶" title="Pick a run">Select a run on the left to see its log, checks and waveform.</Empty></section>
        )}
      </div>
    </div>
  );
}

function Pending<T>({ file, l, children, rows = 6 }: { file: string; l: Loadable<T> & { retry: () => void }; children: (data: T) => React.ReactNode; rows?: number }) {
  if (l.state === "ready") return <div className="fade-in" style={{ height: "100%" }}>{children(l.data)}</div>;
  if (l.state === "missing") return <div className="pad"><Note kind="warn"><b>{file} not uploaded.</b> {l.message}. The run stopped before this file was written.</Note></div>;
  if (l.state === "error") return <div className="pad"><Note kind="err" action={<button type="button" className="btn sm" onClick={l.retry}>Retry</button>}><b>{file} failed to load.</b> {l.message}</Note></div>;
  return (
    <div className="pad stack" aria-busy="true">
      {Array.from({ length: rows }, (_, i) => <div key={i} className="sk sk-line" style={{ width: `${55 + ((i * 23) % 40)}%` }} />)}
    </div>
  );
}

function RunDetail({ spec, run, cycle, setCycle }: { spec: SpecModel; run: RunMeta; cycle: number; setCycle: (c: number) => void }) {
  const params = useSearchParams();
  const has = (f: string) => run.files.includes(f);
  const events = useRunFile(run.id, "events", has("events.jsonl"));
  const signals = useRunFile(run.id, "signals", has("signals.json"));
  const [picked, setPicked] = useState<Violation | null>(null);
  const check = useMemo(() => (signals.state === "ready" ? checkRun(spec, signals.data) : null), [spec, signals]);
  const groups = useMemo(() => (signals.state === "ready" ? signalGroups(spec, signals.data) : []), [spec, signals]);
  const emphasis = useMemo(() => new Set(picked?.signals ?? params.get("em")?.split(",") ?? []), [picked, params]);
  const errors = check?.violations.filter((v) => v.severity === "error").length ?? 0;
  const warns = (check?.violations.length ?? 0) - errors;

  return (
    <div style={{ minHeight: 0, overflow: "auto", gap: 8, display: "grid", gridTemplateRows: "auto auto minmax(280px, 1fr) auto" }}>
      <section className="float pad stack enter">
        <div className="row">
          <h1 className="mono">#{run.id}</h1>
          <span className={`rs rs-${run.status}`}>{run.status}</span>
          <span className="grow ellipsis" style={{ fontSize: 14 }}>{run.test}</span>
          <Link className="btn sm" href={`/process?${new URLSearchParams({ ...Object.fromEntries(params), run: run.id, cycle: String(cycle) })}`}>Process view</Link>
        </div>
        <div className="row muted" style={{ flexWrap: "wrap", gap: 16, fontSize: 12 }}>
          <span>commit <span className="mono">{run.commit}</span>{run.dirty && <span className="tag sample" style={{ marginLeft: 4 }}>dirty</span>}</span>
          <span>seed <span className="mono">{run.seed}</span></span>
          <span>{run.cycles ? `${run.cycles} cycles` : "no cycles yet"}</span>
          <span>{duration(run.durationS)}</span>
          <span>{run.simulator}</span>
          <span>{when(run.startedAt)}</span>
        </div>
        <div className="files">
          {RUN_FILES.map((f) => <span key={f} className={`file${has(f) ? "" : " absent"}`} title={has(f) ? "uploaded" : "not uploaded"}>{f}</span>)}
        </div>
        {run.status === "running" && <Note kind="info"><b>Still running.</b> Files appear here as the run uploads them.</Note>}
      </section>

      <section className="float panel" aria-label="Protocol checks">
        <div className="panel-hd">
          <h2>Checks</h2>
          {check && <span className={errors ? "r-missing" : "r-match"}>{errors ? `${errors} error${errors > 1 ? "s" : ""}` : "no errors"}</span>}
          {check && warns > 0 && <span className="r-different">{warns} warning{warns > 1 ? "s" : ""}</span>}
          <span className="grow" />
          {check && <span className="faint" style={{ fontSize: 11 }}>timing rules from the spec, evaluated over signals.json · {check.checked.map((c) => c.port).join(", ") || "no traced ports"}</span>}
        </div>
        <div style={{ maxHeight: 168, overflow: "auto", padding: 4 }}>
          {!has("signals.json") ? <p className="pad muted">No signals.json, so timing rules cannot be checked for this run.</p> : (
            <Pending file="signals.json" l={signals} rows={3}>
              {() => check && (check.violations.length ? check.violations.map((v, i) => (
                <button key={i} type="button" className={`orow viol-row${picked === v ? " on" : ""}`} onClick={() => { setPicked(v); setCycle(v.cycle); }}>
                  <span className={v.severity === "error" ? "r-missing" : "r-different"}>{v.severity}</span>
                  <span className="ellipsis"><b>{v.rule}</b> <span className="muted">{v.port}</span> · {v.message}</span>
                  <span className="mono" style={{ color: "var(--accent)" }}>@{v.cycle}</span>
                </button>
              )) : <Empty icon="✓" title="Every timing rule holds">{check.checked.length} port{check.checked.length === 1 ? "" : "s"} checked over {run.cycles} cycles.</Empty>)}
            </Pending>
          )}
          {check?.untraced.length ? <p className="faint" style={{ fontSize: 11, padding: "4px 8px" }}>Not traced in this run: {check.untraced.join(", ")}</p> : null}
        </div>
      </section>

      <section className="float panel" aria-label="Event log">
        <div className="panel-hd"><h2>Log</h2><span className="muted mono" style={{ fontSize: 11 }}>events.jsonl</span></div>
        <div className="panel-body" style={{ overflow: "hidden" }}>
          {!has("events.jsonl") ? <Empty icon="≡" title="No log for this run">{run.status === "running" ? "The log uploads when the run finishes." : "events.jsonl was not kept for this run."}</Empty> : (
            <Pending file="events.jsonl" l={events} rows={10}>{(data) => <LogView events={data} cycle={cycle} onCycle={setCycle} />}</Pending>
          )}
        </div>
      </section>

      <section className="float panel" aria-label="Waveform">
        <div className="panel-hd">
          <h2>Waveform</h2>
          <span className="faint" style={{ fontSize: 11 }}>drag to scrub · wheel to zoom · shift+wheel to pan</span>
          <span className="grow" />
          {picked && <button type="button" className="btn sm" onClick={() => setPicked(null)}>Clear highlight</button>}
        </div>
        {!has("signals.json") ? <Empty icon="∿" title="No waveform for this run">This run did not upload signals.json.</Empty> : (
          <Pending file="signals.json" l={signals} rows={8}>
            {(data) => <WaveView signals={data} groups={groups} violations={check?.violations ?? []} cycle={cycle} onCycle={setCycle} emphasis={emphasis} />}
          </Pending>
        )}
      </section>
    </div>
  );
}
