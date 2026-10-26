"use client";

import Link from "next/link";
import type { SpecModel } from "@/lib/spec/types";
import type { CompareReport, PortReport } from "@/lib/rtl/types";
import { expectedSignals } from "@/lib/rtl/names";
import { ResultDot, RESULT_LABEL, SourceLink, StatusChip } from "../ui";

export type Selection = { kind: "block" | "wire"; id: string } | null;

type Props = {
  spec: SpecModel;
  report: CompareReport | null;
  sel: Selection;
  links: { spec: (line?: number) => string | null; rtl: (file: string, line?: number) => string | null };
  query: (extra?: Record<string, string | null>) => string;
  onClose: () => void;
  onOpen: (id: string) => void;
  onSelect: (s: Selection) => void;
};

function diffText(p: PortReport): string | null {
  if (!p.diffs.length) return null;
  return p.diffs.map((d) => (d.kind === "absent" ? `${d.rtlName} absent` : `${d.rtlName}: ${d.kind} ${d.expected} → ${d.actual}`)).join("; ");
}

export function Inspector({ spec, report, sel, links, query, onClose, onOpen, onSelect }: Props) {
  return (
    <aside className={`inspector float${sel ? " open" : ""}`} aria-label="Inspector" aria-hidden={!sel} inert={!sel}>
      {sel?.kind === "block" && spec.blocks[sel.id] && <BlockPanel {...{ spec, report, id: sel.id, links, query, onClose, onOpen, onSelect }} />}
      {sel?.kind === "wire" && <WirePanel {...{ spec, report, id: sel.id, onClose, onSelect }} />}
    </aside>
  );
}

function Header({ title, sub, chip, onClose }: { title: string; sub: React.ReactNode; chip: React.ReactNode; onClose: () => void }) {
  return (
    <div className="insp-hd">
      <div className="row">
        <h2 className="mono grow ellipsis">{title}</h2>
        {chip}
        <button type="button" className="btn ghost icon sm" aria-label="Close inspector" onClick={onClose}>✕</button>
      </div>
      <div className="muted" style={{ fontSize: 12 }}>{sub}</div>
    </div>
  );
}

function BlockPanel({ spec, report, id, links, query, onClose, onOpen, onSelect }: Omit<Props, "sel"> & { id: string }) {
  const b = spec.blocks[id];
  const br = report?.blocks[id];
  const ports = spec.ports[id] ?? [];
  const wires = spec.connections.filter((c) => c.from === id || c.to === id);
  return (
    <>
      <Header title={b.name} sub={<>{b.kind} · owner {b.owner}</>} chip={<StatusChip status={b.status} />} onClose={onClose} />
      <div className="insp-body" key={id}>
        <section className="insp-sec fade-in">
          <p>{b.summary}</p>
          <div className="row" style={{ flexWrap: "wrap" }}>
            {spec.layouts[id] && <button type="button" className="btn primary sm" onClick={() => onOpen(id)}>Open inside ›</button>}
            <Link className="btn sm" href={`/spec${query({ node: `block:${id}` })}`}>Doc graph</Link>
            <SourceLink file={spec.source.path} line={b.line} href={links.spec(b.line)} />
          </div>
        </section>

        <section className="insp-sec">
          <div className="row"><span className="cap grow">RTL</span>{br && <span className={`r-${br.result} row`} style={{ gap: 4 }}><ResultDot result={br.result} small />{RESULT_LABEL[br.result]}</span>}</div>
          {!report && <p className="muted">No RTL extract on this branch. Status comes from the spec only.</p>}
          {report && !b.rtl && <p className="muted">No RTL module named in the spec{b.status === "planned" ? " (planned)" : ""}.</p>}
          {report && b.rtl && (
            <dl className="kv">
              <dt>module</dt><dd className="mono">{b.rtl.module}{b.rtl.instance && <span className="muted"> · {b.rtl.instance}</span>}</dd>
              <dt>SystemVerilog</dt><dd>{br?.sv ? <SourceLink file={br.sv.file} line={br.sv.line} href={links.rtl(br.sv.file, br.sv.line)} /> : <span className="r-missing">not in extract</span>}</dd>
              <dt>Chisel</dt><dd>{br?.src ? <SourceLink file={br.src.file} line={br.src.line} href={links.rtl(br.src.file, br.src.line)} /> : "—"}</dd>
            </dl>
          )}
          {br?.doc && (
            <div className="doc-pair">
              <div><span className="cap">spec</span><p>{b.summary}</p></div>
              <div><span className="cap">ScalaDoc</span><p>{br.doc}</p></div>
            </div>
          )}
        </section>

        <section className="insp-sec">
          <span className="cap">Ports · spec vs RTL</span>
          {!ports.length ? <p className="muted">No ports.</p> : (
            <table>
              <thead><tr><th></th><th>port</th><th>interface</th><th>status</th></tr></thead>
              <tbody>
                {ports.map((p) => {
                  const pr = br?.ports[p.id];
                  const iface = p.interface ? spec.interfaces[p.interface] : undefined;
                  const exp = expectedSignals(p, iface, iface ? spec.protocols[iface.protocol] : undefined);
                  const diff = pr && diffText(pr);
                  return (
                    <tr key={p.id}>
                      <td><ResultDot result={pr?.result ?? "unchecked"} small /></td>
                      <td className="mono">
                        {p.id}
                        {diff && <div className={`diffs${pr?.result === "missing" ? " miss" : ""}`}>{diff}{pr?.warnOnly && " (warning: modified)"}</div>}
                        {exp && <div className="faint" style={{ fontSize: 11 }}>{exp.map((e) => e.rtlName).join(", ")}</div>}
                        {!exp && pr?.found.length ? <div className="faint" style={{ fontSize: 11 }}>{pr.found.length} RTL signals</div> : null}
                      </td>
                      <td className="mono">{p.interface ?? <span className="muted">{p.protocol}</span>}</td>
                      <td><StatusChip status={p.status} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          {!!br?.extraRtlPorts.length && (
            <div className="note note-warn">
              <div className="grow">
                <b>In RTL, not in the spec</b>
                <ul>{br.extraRtlPorts.map((p) => <li key={p.name}>{p.dir} [{p.width}] {p.name}</li>)}</ul>
              </div>
            </div>
          )}
        </section>

        {!!wires.length && (
          <section className="insp-sec">
            <span className="cap">Connections</span>
            {wires.map((c) => (
              <button key={c.id} type="button" className="orow" onClick={() => onSelect({ kind: "wire", id: c.id })}>
                <ResultDot result={report?.connections[c.id]?.result ?? "unchecked"} small />
                <span className="grow ellipsis">{c.from === id ? `→ ${c.to}` : `← ${c.from}`}</span>
                <span className="muted">{c.protocol}</span>
              </button>
            ))}
          </section>
        )}

        {!!b.responsibilities.length && (
          <section className="insp-sec">
            <span className="cap">Responsibilities</span>
            <ul style={{ margin: 0, paddingLeft: 16 }}>{b.responsibilities.map((r) => <li key={r}>{r}</li>)}</ul>
          </section>
        )}
        {!!b.documents.length && (
          <section className="insp-sec">
            <span className="cap">Documents</span>
            {b.documents.map((d) => <span key={d.title} className="mono" style={{ fontSize: 12 }}>{d.title}{d.page ? ` · p.${d.page}` : ""}</span>)}
          </section>
        )}
      </div>
    </>
  );
}

function WirePanel({ spec, report, id, onClose, onSelect }: { spec: SpecModel; report: CompareReport | null; id: string; onClose: () => void; onSelect: (s: Selection) => void }) {
  const c = spec.connections.find((x) => x.id === id);
  if (!c) return null;
  const r = report?.connections[id];
  const proto = spec.protocols[c.protocol];
  const ends = [{ block: c.from, port: c.fromPort, dir: "from" }, { block: c.to, port: c.toPort, dir: "to" }];
  return (
    <>
      <Header title={c.label} sub={<span className="mono">{c.id}</span>} chip={<StatusChip status={c.status} />} onClose={onClose} />
      <div className="insp-body" key={id}>
        <section className="insp-sec fade-in">
          <dl className="kv">
            <dt>protocol</dt><dd><span className="row" style={{ gap: 4 }}><i className="sdot" style={{ background: proto?.color }} />{proto?.name ?? c.protocol}</span></dd>
            <dt>direction</dt><dd>{c.bidirectional ? "bidirectional" : "one way"}</dd>
            <dt>RTL check</dt><dd>{r ? <span className={`r-${r.result}`}>{RESULT_LABEL[r.result]}{r.warnOnly && r.result !== "match" ? " (warning)" : ""} · {r.mode}</span> : <span className="muted">no extract</span>}</dd>
          </dl>
          {proto?.purpose && <p className="muted">{proto.purpose}</p>}
        </section>
        <section className="insp-sec">
          <span className="cap">Endpoints</span>
          {ends.map((e) => {
            const pr = report?.blocks[e.block]?.ports[e.port];
            const diff = pr && diffText(pr);
            return (
              <button key={e.dir} type="button" className="orow" onClick={() => onSelect({ kind: "block", id: e.block })}>
                <ResultDot result={pr?.result ?? "unchecked"} small />
                <span className="grow">
                  <span>{e.block}.{e.port}</span>
                  {diff && <span className={`diffs${pr?.result === "missing" ? " miss" : ""}`} style={{ display: "block" }}>{diff}</span>}
                </span>
                <span className="muted">{e.dir}</span>
              </button>
            );
          })}
          {r?.mode === "endpoints-only" && <p className="faint" style={{ fontSize: 11 }}>Checked at the endpoints only: the extract has no instance wiring for this connection yet.</p>}
        </section>
      </div>
    </>
  );
}
