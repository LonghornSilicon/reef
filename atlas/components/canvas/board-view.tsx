"use client";

import { memo } from "react";
import { RTL_CHECKED, isPlanned, type SpecModel } from "@/lib/spec/types";
import type { CompareReport, Result } from "@/lib/rtl/types";
import type { Board, BoardNode } from "./geometry";

export type Lens = "conformance" | "status" | "protocol";

type Props = {
  board: Board;
  spec: SpecModel;
  report: CompareReport | null;
  lens: Lens;
  selected: string | null;
  hover: string | null;
  dimmed?: Set<string>;
  /** Run overlay (workspace): blocks with activity at the cursor, and stalled blocks with their reason. */
  active?: Map<string, string>;
  stalled?: Map<string, string>;
  onBlock?: (id: string, e: React.MouseEvent) => void;
  onBlockDouble?: (id: string) => void;
  onWire?: (id: string) => void;
  onHover?: (id: string | null) => void;
};

const WIRE_COLOR: Record<Result, string> = { match: "var(--match)", different: "var(--diff)", missing: "var(--miss)", unchecked: "var(--none)" };

/**
 * The drawing for one layout. Every element keeps its key across lens, selection and run changes,
 * so those only toggle classes and the transitions run (built once, updated in place).
 */
export const BoardView = memo(function BoardView({ board, spec, report, lens, selected, hover, dimmed, active, stalled, onBlock, onBlockDouble, onWire, onHover }: Props) {
  const related = (wireFrom: string, wireTo: string) => selected === wireFrom || selected === wireTo || hover === wireFrom || hover === wireTo;
  return (
    <>
      <svg className="wires" width={board.w} height={board.h} aria-hidden>
        {board.wires.map((w) => {
          const r = report?.connections[w.id];
          const result: Result = r?.result ?? "unchecked";
          const color = lens === "protocol" ? spec.protocols[w.conn.protocol]?.color : lens === "status" ? (RTL_CHECKED.has(w.conn.status) ? "#5d636b" : "var(--none)") : result === "different" && r?.warnOnly ? "var(--diff)" : WIRE_COLOR[result];
          const dim = dimmed && (dimmed.has(w.conn.from) || dimmed.has(w.conn.to));
          const on = selected === w.id || related(w.conn.from, w.conn.to);
          const bad = lens === "conformance" && (result === "missing" || result === "different");
          return (
            <g key={w.id} className={`wire-g${dim ? " dim" : ""}${on ? " on" : ""}${hover === w.id ? " hover" : ""}`} style={{ ["--wire" as string]: color }}>
              <path className="wire-sel" d={w.d} />
              <path className={`wire${isPlanned(w.conn.status) ? " dash" : ""}${bad ? " bad" : ""}`} d={w.d} />
              {w.arrow && <path className="arrow" d={w.arrow} />}
              {onWire && (
                <path className="wire-hit" d={w.d} style={{ pointerEvents: "stroke" }}
                  onClick={(e) => { e.stopPropagation(); onWire(w.id); }}
                  onPointerEnter={() => onHover?.(w.id)} onPointerLeave={() => onHover?.(null)} />
              )}
            </g>
          );
        })}
      </svg>
      {board.wires.map((w) => {
        const r = report?.connections[w.id];
        const dim = dimmed && (dimmed.has(w.conn.from) || dimmed.has(w.conn.to));
        const cls = lens === "conformance" && r && r.result !== "match" && r.result !== "unchecked" ? (r.warnOnly ? " warn" : " bad") : "";
        return (
          <span key={w.id} className={`wlabel${dim ? " dim" : ""}${cls}`} style={{ left: w.mid.x, top: w.mid.y }}>
            {w.conn.label}
          </span>
        );
      })}
      {board.nodes.map((n) => (
        <BlockNode key={n.id} node={n} spec={spec} report={report} lens={lens}
          sel={selected === n.id} hov={hover === n.id} dim={!!dimmed?.has(n.id)}
          act={active?.get(n.id)} stall={stalled?.get(n.id)}
          onBlock={onBlock} onBlockDouble={onBlockDouble} onHover={onHover} />
      ))}
    </>
  );
});

function BlockNode({ node, spec, report, lens, sel, hov, dim, act, stall, onBlock, onBlockDouble, onHover }: {
  node: BoardNode; spec: SpecModel; report: CompareReport | null; lens: Lens; sel: boolean; hov: boolean; dim: boolean; act?: string; stall?: string;
  onBlock?: Props["onBlock"]; onBlockDouble?: Props["onBlockDouble"]; onHover?: Props["onHover"];
}) {
  const { block, box } = node;
  const br = report?.blocks[node.id];
  const tally = { match: 0, different: 0, missing: 0 };
  for (const p of Object.values(br?.ports ?? {})) if (p.result !== "unchecked") tally[p.result] += 1;
  const planned = isPlanned(block.status);
  const ghosts = lens === "conformance" ? br?.extraRtlPorts ?? [] : [];
  const cls = ["blk", `st-${block.status}`, planned && "planned", sel && "sel", hov && "hover", dim && "dim", act && "act", stall && "stall"].filter(Boolean).join(" ");
  return (
    <div role="button" tabIndex={0} className={cls} data-block={node.id}
      aria-label={`${block.name}, ${block.status}${br ? `, RTL ${br.result}` : ""}`} aria-pressed={sel}
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
      onClick={(e) => onBlock?.(node.id, e)}
      onDoubleClick={() => node.drill && onBlockDouble?.(node.id)}
      onKeyDown={(e) => {
        if (e.key === "Enter" && node.drill && sel) onBlockDouble?.(node.id);
        else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onBlock?.(node.id, e as unknown as React.MouseEvent); }
      }}
      onPointerEnter={() => onHover?.(node.id)} onPointerLeave={() => onHover?.(null)}>
      <div className="blk-hd">
        <span className="blk-nm">{block.name}</span>
        <span className="blk-st">{block.status}</span>
      </div>
      {box.h >= 96 && <div className="blk-sum">{block.summary}</div>}
      <span className="blk-note">{stall ?? act ?? ""}</span>
      <div className="blk-foot">
        {br && br.result !== "unchecked" && (
          <span className="blk-badge" title="RTL conformance: match / different / missing ports">
            {tally.match > 0 && <span className="r-match"><i className="dot sm k-match" />{tally.match}</span>}
            {tally.different > 0 && <span className="r-different"><i className="dot sm k-different" />{tally.different}</span>}
            {tally.missing > 0 && <span className="r-missing"><i className="dot sm k-missing" />{tally.missing}</span>}
            {!Object.keys(br.ports).length && <span className="r-match">module ✓</span>}
          </span>
        )}
        {br && br.result === "missing" && !Object.keys(br.ports).length && <span className="r-missing">module missing</span>}
        {node.drill && <span className="blk-drill" aria-hidden>open ›</span>}
      </div>
      {node.ports.map((p) => {
        const pr = br?.ports[p.id];
        return (
          <span key={p.id} className={`prt s-${p.side}`} style={{ left: p.at.x, top: p.at.y, ["--pc" as string]: spec.protocols[p.protocol]?.color ?? "var(--none)" }}>
            <span className={`dot k-${pr?.result ?? "unchecked"}`} title={`${p.label}: ${pr?.result ?? "not checked"}`} />
            <span className="pd" />
            <span className="plabel">{p.label}</span>
          </span>
        );
      })}
      {ghosts.map((g, i) => (
        <span key={g.name} className="prt s-bottom ghost-prt" style={{ left: box.w - 24 - i * 120, top: box.h }}>
          <span className="dot" title={`${g.name}: in RTL, not in the spec`} />
          <span className="plabel">RTL only · {g.name}</span>
        </span>
      ))}
    </div>
  );
}
