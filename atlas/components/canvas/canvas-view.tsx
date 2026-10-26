"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { STATUS, type SpecModel, type Status } from "@/lib/spec/types";
import type { CompareReport } from "@/lib/rtl/types";
import type { RtlState } from "@/lib/server/context";
import { useSession } from "../shell/session";
import { Note, Seg, StatusDot } from "../ui";
import { ancestors, buildBoard, layoutFor } from "./geometry";
import { BoardView, type Lens } from "./board-view";
import { Inspector, type Selection } from "./inspector";
import { useCamera, type Inset } from "./use-camera";

type Props = {
  spec: SpecModel;
  report: CompareReport | null;
  rtl: RtlState;
  note: string | null;
  specBase: string | null;
  rtlBase: string | null;
};

const INSET_BASE: Inset = { top: 128, left: 288, right: 24, bottom: 72 };

function parseSel(v: string | null): Selection {
  if (!v) return null;
  const [kind, ...rest] = v.split(":");
  return (kind === "block" || kind === "wire") && rest.length ? { kind, id: rest.join(":") } : null;
}

/** Write focus/selection into the URL without a server round trip (Next syncs useSearchParams). */
function writeUrl(patch: Record<string, string | null>, push: boolean) {
  const u = new URL(window.location.href);
  for (const [k, v] of Object.entries(patch)) { if (v === null) u.searchParams.delete(k); else u.searchParams.set(k, v); }
  if (u.href === window.location.href) return;
  if (push) window.history.pushState(null, "", u); else window.history.replaceState(null, "", u);
}

export function CanvasView({ spec, report, rtl, note, specBase, rtlBase }: Props) {
  const params = useSearchParams();
  const { query } = useSession();
  const focus = spec.layouts[params.get("focus") ?? ""] ? params.get("focus")! : spec.layouts[spec.product.defaultFocus] ? spec.product.defaultFocus : Object.keys(spec.layouts)[0];
  const sel = parseSel(params.get("sel"));
  const [lens, setLens] = useState<Lens>(report ? "conformance" : "status");
  const [hover, setHover] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<Status>>(new Set());
  const [phase, setPhase] = useState<"idle" | "leaving" | "arriving">("idle");

  const viewport = useRef<HTMLDivElement>(null);
  const boardEl = useRef<HTMLDivElement>(null);
  const camera = useCamera(viewport, boardEl);
  const board = useMemo(() => buildBoard(spec, focus), [spec, focus]);

  // Read through a ref so fitting never depends on (and re-runs with) the selection.
  // One banner at a time, most important first; each can be dismissed for this visit.
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const banners: Array<{ key: string; kind: "warn" | "info"; body: React.ReactNode }> = [];
  if (rtl.kind === "pending") banners.push({ key: "pending", kind: "warn", body: <><b>RTL extract pending.</b> {rtl.message} Status comes from the spec; conformance dots appear when CI uploads the extract.</> });
  if (rtl.kind === "ready" && rtl.staleBy > 0) banners.push({ key: "stale", kind: "warn", body: <><b>Extract is {rtl.staleBy} commit{rtl.staleBy > 1 ? "s" : ""} behind.</b> Results are from {rtl.commit}.</> });
  if (note && rtl.kind !== "pending") banners.push({ key: "note", kind: "info", body: note });
  const banner = banners.find((b) => !dismissed.has(b.key));
  const hasBanner = useRef(!!banner);
  hasBanner.current = !!banner;
  const hasSel = useRef(!!sel);
  hasSel.current = !!sel;
  const inset = useCallback((): Inset => ({ ...INSET_BASE, top: INSET_BASE.top + (hasBanner.current ? 48 : 0), right: hasSel.current ? 408 : INSET_BASE.right }), []);
  const fit = useCallback((animate = true) => board && camera.fitRect({ x: 0, y: 0, w: board.w, h: board.h }, inset(), animate), [board, camera, inset]);

  // A new layout fades in from slightly small while the camera fits it instantly.
  const shown = useRef<string | null>(null);
  useLayoutEffect(() => {
    if (!board || shown.current === board.focus) return;
    const first = shown.current === null;
    shown.current = board.focus;
    fit(false);
    if (!first) setPhase("arriving");
  }, [board, fit]);
  // Let the browser paint the "arriving" state once, then transition to idle.
  useEffect(() => {
    if (phase !== "arriving") return;
    let inner = 0;
    const outer = requestAnimationFrame(() => { inner = requestAnimationFrame(() => setPhase("idle")); });
    return () => { cancelAnimationFrame(outer); cancelAnimationFrame(inner); };
  }, [phase]);
  // A branch that gains (or loses) a compare report switches to the lens that can show it.
  const hadReport = useRef(!!report);
  useEffect(() => {
    if (hadReport.current === !!report) return;
    hadReport.current = !!report;
    setLens(report ? "conformance" : "status");
  }, [report]);
  useEffect(() => {
    const vp = viewport.current;
    if (!vp) return;
    let size = `${vp.clientWidth}x${vp.clientHeight}`;
    const ro = new ResizeObserver(() => {
      const next = `${vp.clientWidth}x${vp.clientHeight}`;
      if (next !== size) { size = next; fit(false); }
    });
    ro.observe(vp);
    return () => ro.disconnect();
  }, [fit]);

  // A newly selected block slides out from under the inspector.
  const selKey = sel ? `${sel.kind}:${sel.id}` : null;
  useEffect(() => {
    if (sel?.kind !== "block" || phase !== "idle") return;
    const n = board?.nodes.find((x) => x.id === sel.id);
    if (n) camera.reveal(n.box, inset());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selKey, phase]);

  const select = useCallback((s: Selection) => writeUrl({ sel: s ? `${s.kind}:${s.id}` : null }, false), []);
  const goTo = useCallback((target: string, selectId: string | null = null) => {
    if (target === focus || !spec.layouts[target]) { select(selectId ? { kind: "block", id: selectId } : null); return; }
    const node = board?.nodes.find((n) => n.id === target);
    if (node) camera.fitRect(node.box, inset(), true, 4);
    setPhase("leaving");
    setTimeout(() => writeUrl({ focus: target, sel: selectId ? `block:${selectId}` : null }, true), 160);
  }, [board, camera, focus, inset, select, spec.layouts]);

  // Escape clears the selection; Backspace goes up one level.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, select")) return;
      if (e.key === "Escape" && sel) select(null);
      if (e.key === "Backspace" && spec.blocks[focus]?.parent) { e.preventDefault(); goTo(layoutFor(spec, focus) ?? focus, focus); }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [focus, goTo, sel, select, spec]);

  const dimmed = useMemo(() => {
    if (!board || !hidden.size) return undefined;
    return new Set(board.nodes.filter((n) => hidden.has(n.block.status)).map((n) => n.id));
  }, [board, hidden]);

  const crumbs = [...ancestors(spec, focus), focus];
  const counts = useMemo(() => {
    const c = Object.fromEntries(STATUS.map((s) => [s, 0])) as Record<Status, number>;
    for (const n of board?.nodes ?? []) c[n.block.status] += 1;
    return c;
  }, [board]);

  const links = useMemo(() => ({
    spec: (line?: number) => (specBase ? `${specBase}${spec.source.path}${line ? `#L${line}` : ""}` : null),
    rtl: (file: string, line?: number) => (rtlBase ? `${rtlBase}${file}${line ? `#L${line}` : ""}` : null),
  }), [rtlBase, spec.source.path, specBase]);

  const onBlock = useCallback((id: string) => {
    if (camera.dragged.current) return;
    select(sel?.kind === "block" && sel.id === id ? null : { kind: "block", id });
  }, [camera.dragged, sel, select]);
  const onWire = useCallback((id: string) => { if (!camera.dragged.current) select({ kind: "wire", id }); }, [camera.dragged, select]);

  if (!board) return null;
  return (
    <div className={`viewport lens-${lens}`} ref={viewport} onClick={(e) => { if (e.target === viewport.current && !camera.dragged.current) select(null); }}>
      <div className="board" ref={boardEl} style={{ width: board.w, height: board.h }}>
        <div className={`board-content${phase !== "idle" ? ` ${phase}` : ""}`} onClick={(e) => { if (e.target === e.currentTarget && !camera.dragged.current) select(null); }}>
          <BoardView board={board} spec={spec} report={report} lens={lens} selected={sel?.id ?? null} hover={hover} dimmed={dimmed}
            onBlock={onBlock} onBlockDouble={(id) => goTo(id)} onWire={onWire} onHover={setHover} />
        </div>
      </div>

      <Outline spec={spec} focus={focus} sel={sel} onPick={(id) => { const l = layoutFor(spec, id); if (l) goTo(l, id); }} onOpen={(id) => goTo(id)} />

      <div className="c-tools float" style={{ left: 288, top: 72 }}>
        <button type="button" className="btn ghost icon" aria-label="Up one level" disabled={!spec.blocks[focus]?.parent || !layoutFor(spec, focus)} onClick={() => goTo(layoutFor(spec, focus)!, focus)}>↑</button>
        <nav className="crumbs" aria-label="Hierarchy">
          {crumbs.length > 3 && <span className="row" style={{ gap: 0 }}><button type="button" title={crumbs.slice(0, -3).map((c) => spec.blocks[c]?.name ?? c).join(" › ")} onClick={() => goTo(crumbs[crumbs.length - 4], crumbs[crumbs.length - 3])}>…</button><i>›</i></span>}
          {crumbs.slice(-3).map((c, j, arr) => { const i = crumbs.length - arr.length + j; return (
            <span key={c} className="row" style={{ gap: 0 }}>
              {j > 0 && <i>›</i>}
              <button type="button" className={c === focus ? "cur" : ""} disabled={c === focus || !spec.layouts[c]} onClick={() => goTo(c, crumbs[i + 1] ?? null)}>{spec.blocks[c]?.name ?? c}</button>
            </span>
          ); })}
        </nav>
      </div>

      <div className="c-tools float" style={{ right: 16, top: 72, transform: sel ? "translateX(-392px)" : "none", transition: "transform var(--med) var(--ease)" }}>
        <Seg label="Lens" value={lens} onChange={setLens} options={[
          { value: "conformance", label: "Conformance" },
          { value: "status", label: "Status" },
          { value: "protocol", label: "Protocol" },
        ]} />
        <button type="button" className="btn ghost icon" aria-label="Zoom out" onClick={() => camera.zoomAt(1 / 1.25, undefined, undefined, true)}>−</button>
        <button type="button" className="btn ghost" onClick={() => fit(true)}>Fit</button>
        <button type="button" className="btn ghost icon" aria-label="Zoom in" onClick={() => camera.zoomAt(1.25, undefined, undefined, true)}>+</button>
      </div>

      <div style={{ position: "absolute", zIndex: 5, left: 288, right: sel ? 408 : 16, top: 124, display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-start", pointerEvents: "none" }}>
        {banner && (
          <div style={{ pointerEvents: "auto" }} key={banner.key}>
            <Note kind={banner.kind} action={<button type="button" className="btn ghost sm icon" aria-label="Dismiss" onClick={() => setDismissed((d) => new Set([...d, banner.key]))}>✕</button>}>{banner.body}</Note>
          </div>
        )}
      </div>

      <div className="legend float" style={{ maxWidth: `calc(100% - 288px - ${sel ? 408 : 16}px)` }}>
        <span className="cap">Status</span>
        {STATUS.map((s) => (
          <span key={s} className="lg">
            <button type="button" className={`st-${s}${hidden.has(s) ? " off" : ""}`} aria-pressed={!hidden.has(s)} title={hidden.has(s) ? `Show ${s}` : `Dim ${s}`}
              onClick={() => setHidden((h) => { const n = new Set(h); if (n.has(s)) n.delete(s); else n.add(s); return n; })}>
              <StatusDot status={s} />{s} <span className="muted mono">{counts[s]}</span>
            </button>
          </span>
        ))}
        <span className="sep" />
        {report ? (
          <>
            <span className="lg r-match"><i className="dot sm k-match" />{report.summary.match}</span>
            <span className="lg r-different"><i className="dot sm k-different" />{report.summary.different}</span>
            <span className="lg r-missing"><i className="dot sm k-missing" />{report.summary.missing}</span>
            <span className="lg muted"><i className="dot sm k-unchecked" />{report.summary.unchecked}</span>
            <span className={`lg ${report.failing ? "r-missing" : "r-match"}`} title={`Compared with the RTL extract of ${report.rtl.commit}`}><b>{report.failing ? `${report.failing} failing` : "CI passes"}</b></span>
          </>
        ) : <span className="lg muted">{rtl.kind === "pending" ? "conformance pending" : "no RTL on this branch"}</span>}
      </div>

      <Inspector spec={spec} report={report} sel={sel} links={links} query={query} onClose={() => select(null)} onOpen={(id) => goTo(id)} onSelect={(s) => {
        if (s?.kind === "block" && !board.nodes.some((n) => n.id === s.id)) { const l = layoutFor(spec, s.id); if (l) { goTo(l, s.id); return; } }
        select(s);
      }} />
    </div>
  );
}

function Outline({ spec, focus, sel, onPick, onOpen }: { spec: SpecModel; focus: string; sel: Selection; onPick: (id: string) => void; onOpen: (id: string) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<Set<string>>(() => new Set([...ancestors(spec, focus), focus]));
  useEffect(() => { setOpen((o) => new Set([...o, ...ancestors(spec, focus), focus])); }, [focus, spec]);

  const rows = useMemo(() => {
    const out: Array<{ id: string; depth: number }> = [];
    const needle = q.trim().toLowerCase();
    if (needle) {
      for (const b of Object.values(spec.blocks)) if (b.id.includes(needle) || b.name.toLowerCase().includes(needle)) out.push({ id: b.id, depth: 0 });
      return out.slice(0, 200);
    }
    const walk = (id: string, depth: number) => {
      out.push({ id, depth });
      if (open.has(id)) for (const c of spec.blocks[id].childIds) walk(c, depth + 1);
    };
    for (const r of spec.rootIds) walk(r, 0);
    return out;
  }, [open, q, spec]);

  return (
    <aside className="outline float" aria-label="Outline">
      <div className="outline-hd">
        <div className="row"><span className="cap grow">Outline</span><span className="faint mono" style={{ fontSize: 11 }}>{Object.keys(spec.blocks).length} blocks</span></div>
        <input className="field sans" placeholder="Find a block" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Find a block" />
      </div>
      <div className="outline-list" role="tree">
        {rows.map(({ id, depth }) => {
          const b = spec.blocks[id];
          const kids = b.childIds.length > 0;
          return (
            <div key={id} role="treeitem" aria-expanded={kids ? open.has(id) : undefined} aria-selected={sel?.id === id}
              className={`orow tree-row st-${b.status}${sel?.id === id ? " on" : ""}${focus === id ? " here" : ""}`} style={{ ["--depth" as string]: depth }}
              tabIndex={0} onClick={() => onPick(id)} onDoubleClick={() => spec.layouts[id] && onOpen(id)}
              onKeyDown={(e) => { if (e.key === "Enter") onPick(id); if (e.key === "ArrowRight" && kids) setOpen((o) => new Set([...o, id])); if (e.key === "ArrowLeft") setOpen((o) => { const n = new Set(o); n.delete(id); return n; }); }}>
              {kids && !q ? (
                <button type="button" tabIndex={-1} className={`tw btn ghost${open.has(id) ? " open" : ""}`} style={{ minHeight: 16, padding: 0, border: 0 }} aria-label={open.has(id) ? "Collapse" : "Expand"}
                  onClick={(e) => { e.stopPropagation(); setOpen((o) => { const n = new Set(o); if (n.has(id)) n.delete(id); else n.add(id); return n; }); }}>›</button>
              ) : <span className="tw" />}
              <StatusDot status={b.status} />
              <span className="nm">{b.name}</span>
              {spec.layouts[id] && focus !== id && (
                <button type="button" tabIndex={-1} className="btn ghost sm" style={{ minHeight: 20, padding: "0 4px", fontSize: 11 }} onClick={(e) => { e.stopPropagation(); onOpen(id); }} aria-label={`Open ${b.name}`}>open</button>
              )}
            </div>
          );
        })}
        {!rows.length && <div className="empty" style={{ padding: 16 }}><b>No match</b><p>No block id or name contains “{q}”.</p></div>}
      </div>
    </aside>
  );
}
