"use client";

import Link from "next/link";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import type { SpecModel, TimingRule } from "@/lib/spec/types";
import { expectedSignals } from "@/lib/rtl/names";
import { useSession } from "../shell/session";
import { Empty, SourceLink, StatusChip, StatusDot } from "../ui";

const ROW = 40;
type Col = "block" | "iface" | "proto" | "rule";
type Node = { id: string; col: Col; label: string; y: number; planned?: boolean; color?: string; status?: string };
const COLS: Array<{ col: Col; title: string; empty: string }> = [
  { col: "block", title: "Blocks", empty: "No block has ports." },
  { col: "iface", title: "Interfaces", empty: "No interfaces yet. Spec v2 adds them to bind ports to signal-level protocols." },
  { col: "proto", title: "Protocols", empty: "No protocols." },
  { col: "rule", title: "Timing rules", empty: "No timing rules yet. Protocols with signals can declare them." },
];

/** The rule in words, generated from its kind so it can never drift from what the checker does. */
export function ruleText(r: TimingRule): string {
  switch (r.kind) {
    case "implies": return `Whenever ${r.when}, ${r.then} must hold in the same cycle.`;
    case "next": return `Whenever ${r.when}, ${r.then} must hold on the next cycle.`;
    case "stable_while": return `While ${r.when}, ${(r.signals ?? []).join(", ")} must not change.`;
    case "within": return `Once ${r.when}, ${r.then} must follow within ${r.cycles} cycles.`;
  }
}

/** Place each node near the average height of its neighbours in the previous column, without overlaps. */
function settle(nodes: Node[], want: Map<string, number>) {
  nodes.sort((a, b) => (want.get(a.id) ?? 0) - (want.get(b.id) ?? 0) || a.label.localeCompare(b.label));
  let y = -ROW;
  for (const n of nodes) { y = Math.max(want.get(n.id) ?? 0, y + ROW); n.y = y; }
}

export function DocGraph({ spec, specBase }: { spec: SpecModel; specBase: string | null }) {
  const params = useSearchParams();
  const { query } = useSession();
  const [q, setQ] = useState("");
  const sel = params.get("node");
  const [hover, setHover] = useState<string | null>(null);

  const { nodes, edges } = useMemo(() => {
    const edges: Array<[string, string]> = [];
    const needle = q.trim().toLowerCase();
    const order: string[] = [];
    const walk = (id: string) => { order.push(id); spec.blocks[id].childIds.forEach(walk); };
    spec.rootIds.forEach(walk);
    const blocks: Node[] = order
      .filter((id) => spec.ports[id]?.length && (!needle || id.includes(needle) || spec.blocks[id].name.toLowerCase().includes(needle)))
      .map((id, i) => ({ id: `block:${id}`, col: "block", label: spec.blocks[id].name, y: i * ROW, planned: spec.blocks[id].status === "planned" || spec.blocks[id].status === "in-progress", status: spec.blocks[id].status }));
    const shown = new Set(blocks.map((b) => b.id));
    for (const b of blocks) {
      const id = b.id.slice(6);
      for (const p of spec.ports[id]) {
        const to = p.interface ? `iface:${p.interface}` : `proto:${p.protocol}`;
        if (!edges.some(([a, c]) => a === b.id && c === to)) edges.push([b.id, to]);
      }
    }
    const ifaces: Node[] = Object.values(spec.interfaces).map((i) => ({ id: `iface:${i.id}`, col: "iface", label: i.id, y: 0, color: spec.protocols[i.protocol]?.color }));
    for (const i of Object.values(spec.interfaces)) edges.push([`iface:${i.id}`, `proto:${i.protocol}`]);
    const used = new Set(edges.map(([, t]) => t).filter((t) => t.startsWith("proto:")));
    const protos: Node[] = Object.values(spec.protocols).filter((p) => used.has(`proto:${p.id}`) || p.timingRules?.length).map((p) => ({ id: `proto:${p.id}`, col: "proto", label: p.name, y: 0, color: p.color }));
    const rules: Node[] = [];
    for (const p of Object.values(spec.protocols)) for (const r of p.timingRules ?? []) {
      rules.push({ id: `rule:${p.id}:${r.id}`, col: "rule", label: r.id, y: 0, color: r.severity === "error" ? "var(--miss)" : "var(--diff)" });
      edges.push([`proto:${p.id}`, `rule:${p.id}:${r.id}`]);
    }
    const live = edges.filter(([a]) => !a.startsWith("block:") || shown.has(a));
    const yOf = new Map<string, number>(blocks.map((b) => [b.id, b.y]));
    const mean = (target: string) => {
      const ys = live.filter(([, t]) => t === target).map(([s]) => yOf.get(s)).filter((v): v is number => v !== undefined);
      return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
    };
    for (const col of [ifaces, protos, rules]) {
      settle(col, new Map(col.map((n) => [n.id, mean(n.id)])));
      for (const n of col) yOf.set(n.id, n.y);
    }
    return { nodes: [...blocks, ...ifaces, ...protos, ...rules], edges: live };
  }, [spec, q]);

  // Everything upstream and downstream of the focused node.
  const focus = hover ?? sel;
  const related = useMemo(() => {
    if (!focus) return null;
    const out = new Set([focus]);
    const go = (dir: 0 | 1) => {
      const stack = [focus];
      while (stack.length) {
        const n = stack.pop()!;
        for (const e of edges) if (e[dir] === n && !out.has(e[1 - dir])) { out.add(e[1 - dir]); stack.push(e[1 - dir]); }
      }
    };
    go(0); go(1);
    return out;
  }, [focus, edges]);

  const select = (id: string | null) => {
    const u = new URL(window.location.href);
    if (id) u.searchParams.set("node", id); else u.searchParams.delete("node");
    window.history.replaceState(null, "", u);
  };

  // Column x positions, measured; rows are computed, so edges need no per-node measuring.
  const grid = useRef<HTMLDivElement>(null);
  const [xs, setXs] = useState<Array<[number, number]>>([]);
  const [HEAD, setHead] = useState(40);
  useLayoutEffect(() => {
    const el = grid.current;
    if (!el) return;
    const measure = () => {
      const base = el.getBoundingClientRect();
      setXs([...el.querySelectorAll<HTMLElement>("[data-col]")].map((c) => { const r = c.getBoundingClientRect(); return [r.left - base.left, r.right - base.left]; }));
      const body = el.querySelector<HTMLElement>(".dg-col-body");
      if (body) setHead(body.getBoundingClientRect().top - base.top);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const colIndex: Record<Col, number> = { block: 0, iface: 1, proto: 2, rule: 3 };
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const height = Math.max(...nodes.map((n) => n.y + ROW), 200);

  // Selecting from the URL on first load scrolls the node into view.
  useEffect(() => {
    if (!sel) return;
    const n = byId.get(sel);
    const scroller = grid.current?.parentElement;
    if (n && scroller && (n.y < scroller.scrollTop || n.y > scroller.scrollTop + scroller.clientHeight - 80)) scroller.scrollTop = Math.max(0, n.y - 120);
    // only on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="page">
      <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1fr) 384px" }}>
        <section className="float panel" aria-label="Doc graph">
          <div className="panel-hd">
            <h2>Doc graph</h2>
            <span className="muted">block → interface → protocol → timing rule</span>
            <span className="grow" />
            <input className="field sans" placeholder="Filter blocks" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter blocks" style={{ width: 200 }} />
          </div>
          <div className="panel-body">
            <div className="dg" ref={grid} style={{ minHeight: height + HEAD + 16 }} onClick={(e) => { if (e.target === e.currentTarget) select(null); }}>
              <svg className="dg-edges" width="100%" height={height + HEAD + 16} aria-hidden>
                {xs.length === 4 && edges.map(([a, b]) => {
                  const na = byId.get(a), nb = byId.get(b);
                  if (!na || !nb) return null;
                  const x1 = xs[colIndex[na.col]][1], x2 = xs[colIndex[nb.col]][0];
                  const y1 = HEAD + na.y + 16, y2 = HEAD + nb.y + 16, mx = (x1 + x2) / 2;
                  const on = related?.has(a) && related.has(b);
                  return <path key={`${a}>${b}`} className={on ? "on" : related ? "dim" : ""} d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`} />;
                })}
              </svg>
              <div className="dg-cols">
                {COLS.map(({ col, title, empty }) => {
                  const list = nodes.filter((n) => n.col === col);
                  return (
                    <div key={col} data-col={col}>
                      <div className="dg-col-hd"><span className="cap">{title}</span><span className="faint mono" style={{ fontSize: 11 }}>{list.length}</span></div>
                      <div className="dg-col-body" style={{ height }}>
                        {!list.length && <p className="faint" style={{ fontSize: 12, paddingTop: 8 }}>{col === "block" && q ? `No block matches “${q}”.` : empty}</p>}
                        {list.map((n) => (
                          <button key={n.id} type="button" className={`dg-node${n.planned ? " planned" : ""}${sel === n.id ? " on" : ""}${related && !related.has(n.id) ? " dim" : ""}`}
                            style={{ transform: `translateY(${n.y}px)`, top: 0 }}
                            onClick={() => select(sel === n.id ? null : n.id)} onPointerEnter={() => setHover(n.id)} onPointerLeave={() => setHover(null)}>
                            {n.col === "block" ? <span className={`st-${n.status}`}><StatusDot status={n.status as never} /></span> : <span className="sw" style={{ background: n.color }} />}
                            <span className="ellipsis">{n.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>
        <aside className="float panel" aria-label="Details">
          <Detail spec={spec} id={sel} specBase={specBase} query={query} onSelect={select} />
        </aside>
      </div>
    </div>
  );
}

function Detail({ spec, id, specBase, query, onSelect }: { spec: SpecModel; id: string | null; specBase: string | null; query: (e?: Record<string, string | null>) => string; onSelect: (id: string) => void }) {
  const link = (line?: number) => (specBase ? `${specBase}${spec.source.path}${line ? `#L${line}` : ""}` : null);
  if (!id) {
    return (
      <div className="panel-body">
        <Empty icon="◇" title="Select a node">
          The graph follows each block port to the interface it uses, the protocol behind that interface, and the timing rules the protocol declares.
          Hover to trace a path; click to pin it.
        </Empty>
      </div>
    );
  }
  const [kind, a, b] = id.split(":");
  if (kind === "block" && spec.blocks[a]) {
    const blk = spec.blocks[a];
    return (
      <>
        <div className="panel-hd"><h2 className="grow ellipsis">{blk.name}</h2><StatusChip status={blk.status} /></div>
        <div className="panel-body pad stack" key={id}>
          <p className="fade-in">{blk.summary}</p>
          <div className="row"><Link className="btn sm" href={`/canvas${query({ focus: spec.blocks[a].parent, sel: `block:${a}` })}`}>Show on canvas</Link><SourceLink file={spec.source.path} line={blk.line} href={link(blk.line)} /></div>
          <table>
            <thead><tr><th>port</th><th>uses</th><th>status</th></tr></thead>
            <tbody>{(spec.ports[a] ?? []).map((p) => (
              <tr key={p.id}><td className="mono">{p.id}</td>
                <td><button type="button" className="btn ghost sm mono" onClick={() => onSelect(p.interface ? `iface:${p.interface}` : `proto:${p.protocol}`)}>{p.interface ?? p.protocol}</button></td>
                <td><StatusChip status={p.status} /></td></tr>
            ))}</tbody>
          </table>
        </div>
      </>
    );
  }
  if (kind === "iface" && spec.interfaces[a]) {
    const i = spec.interfaces[a];
    const users = Object.values(spec.ports).flat().filter((p) => p.interface === a);
    return (
      <>
        <div className="panel-hd"><h2 className="mono grow">{i.id}</h2><span className="tag">interface</span></div>
        <div className="panel-body pad stack" key={id}>
          <dl className="kv fade-in">
            <dt>protocol</dt><dd><button type="button" className="btn ghost sm" onClick={() => onSelect(`proto:${i.protocol}`)}>{spec.protocols[i.protocol]?.name ?? i.protocol}</button></dd>
            <dt>parameters</dt><dd className="mono">{Object.entries(i.params).map(([k, v]) => `${k}=${v}`).join(", ") || "—"}</dd>
            <dt>RTL naming</dt><dd>{i.roleNaming}</dd>
          </dl>
          <span className="cap">Ports using it</span>
          {users.map((p) => {
            const exp = expectedSignals(p, i, spec.protocols[i.protocol]);
            return (
              <div key={`${p.blockId}.${p.id}`} className="rule-card">
                <div className="row"><button type="button" className="btn ghost sm mono" onClick={() => onSelect(`block:${p.blockId}`)}>{p.blockId}.{p.id}</button><span className="grow" /><span className="muted">{p.role}</span><StatusChip status={p.status} /></div>
                {exp && <div className="mono faint" style={{ fontSize: 11 }}>{exp.map((e) => `${e.rtlName}${e.width ? `[${e.width}]` : ""} ${e.dir ?? ""}`).join(" · ")}</div>}
              </div>
            );
          })}
        </div>
      </>
    );
  }
  if (kind === "proto" && spec.protocols[a]) {
    const p = spec.protocols[a];
    const users = Object.values(spec.ports).flat().filter((x) => (x.interface ? spec.interfaces[x.interface]?.protocol === a : x.protocol === a));
    return (
      <>
        <div className="panel-hd"><span className="sw" style={{ width: 12, height: 12, borderRadius: 3, background: p.color }} /><h2 className="grow">{p.name}</h2><span className="tag">{p.abbreviation}</span></div>
        <div className="panel-body pad stack" key={id}>
          <p className="fade-in">{p.purpose}</p>
          {!!p.roles.length && <p className="muted">Roles: {p.roles.join(" → ")}</p>}
          <SourceLink file={spec.source.path} line={p.line} href={link(p.line)} />
          <span className="cap">Signals</span>
          {p.signals?.length ? (
            <table>
              <thead><tr><th>signal</th><th>width</th><th>driven by</th></tr></thead>
              <tbody>{p.signals.map((s) => <tr key={s.name}><td className="mono">{s.name}</td><td className="mono">{s.width}</td><td>{s.dir === "fwd" ? p.roles[0] ?? "producer" : p.roles[1] ?? "consumer"}</td></tr>)}</tbody>
            </table>
          ) : <p className="muted">Informal protocol: no signal list, so its ports are checked for presence only.</p>}
          {!!p.timingRules?.length && <span className="cap">Timing rules</span>}
          {p.timingRules?.map((r) => <RuleCard key={r.id} r={r} onClick={() => onSelect(`rule:${a}:${r.id}`)} />)}
          <span className="cap">Used by {users.length} port{users.length === 1 ? "" : "s"}</span>
          <div className="stack" style={{ gap: 0 }}>
            {users.slice(0, 40).map((u) => <button key={`${u.blockId}.${u.id}`} type="button" className="orow" onClick={() => onSelect(`block:${u.blockId}`)}>{u.blockId}.{u.id}</button>)}
          </div>
        </div>
      </>
    );
  }
  if (kind === "rule") {
    const r = spec.protocols[a]?.timingRules?.find((x) => x.id === b);
    if (r) return (
      <>
        <div className="panel-hd"><h2 className="mono grow">{r.id}</h2><span className={r.severity === "error" ? "rs rs-failed" : "rs rs-incomplete"}>{r.severity}</span></div>
        <div className="panel-body pad stack" key={id}>
          <RuleCard r={r} />
          <dl className="kv">
            <dt>protocol</dt><dd><button type="button" className="btn ghost sm" onClick={() => onSelect(`proto:${a}`)}>{spec.protocols[a].name}</button></dd>
            <dt>kind</dt><dd className="mono">{r.kind}</dd>
          </dl>
          <p className="faint" style={{ fontSize: 12 }}>Checked against every run’s signals.json for each port whose interface uses this protocol. See the Checks panel on the Runs page.</p>
          <Link className="btn sm" href={`/runs${query()}`}>Open runs</Link>
        </div>
      </>
    );
  }
  return <div className="panel-body"><Empty title="Not in this spec">{id} does not exist on this branch.</Empty></div>;
}

function RuleCard({ r, onClick }: { r: TimingRule; onClick?: () => void }) {
  const body = (
    <>
      <div className="row"><b className="mono">{r.id}</b><span className="muted">{r.kind}</span><span className="grow" /><span className={r.severity === "error" ? "r-missing" : "r-different"}>{r.severity}</span></div>
      <div className="expr">{r.when}{r.then ? <> ⇒ {r.kind === "next" ? "next: " : r.kind === "within" ? `≤${r.cycles}: ` : ""}{r.then}</> : null}{r.signals ? <> · stable: {r.signals.join(", ")}</> : null}</div>
      <p style={{ fontSize: 12 }}>{ruleText(r)}</p>
      {r.summary && <p className="muted" style={{ fontSize: 12 }}>{r.summary}</p>}
    </>
  );
  return onClick ? <button type="button" className="rule-card press" style={{ textAlign: "left", background: "var(--card)", cursor: "pointer" }} onClick={onClick}>{body}</button> : <div className="rule-card">{body}</div>;
}
