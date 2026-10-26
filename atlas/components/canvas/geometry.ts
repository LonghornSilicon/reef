import { portPoint, routePath, type Box, type Point } from "@/lib/spec/routing";
import type { SpecBlock, SpecConnection, SpecModel, SpecPort } from "@/lib/spec/types";

export type BoardPort = SpecPort & { at: Point };
export type BoardNode = { id: string; box: Box; block: SpecBlock; ports: BoardPort[]; drill: boolean };
export type BoardWire = { id: string; conn: SpecConnection; d: string; mid: Point; arrow: string | null };
export type Board = { focus: string; w: number; h: number; nodes: BoardNode[]; wires: BoardWire[] };

function points(d: string): Point[] {
  const n = d.match(/-?\d+(\.\d+)?/g)?.map(Number) ?? [];
  const out: Point[] = [];
  for (let i = 0; i + 1 < n.length; i += 2) out.push({ x: n[i], y: n[i + 1] });
  return out;
}

/** Point halfway along a polyline, for the wire label. */
function midpoint(pts: Point[]): Point {
  const seg = pts.slice(1).map((p, i) => Math.hypot(p.x - pts[i].x, p.y - pts[i].y));
  let half = seg.reduce((a, b) => a + b, 0) / 2;
  for (let i = 0; i < seg.length; i++) {
    if (half <= seg[i]) {
      const t = seg[i] ? half / seg[i] : 0;
      return { x: pts[i].x + (pts[i + 1].x - pts[i].x) * t, y: pts[i].y + (pts[i + 1].y - pts[i].y) * t };
    }
    half -= seg[i];
  }
  return pts[0] ?? { x: 0, y: 0 };
}

/** Small triangle on the last segment, pointing into the destination port. */
function arrowHead(pts: Point[]): string | null {
  if (pts.length < 2) return null;
  const a = pts[pts.length - 2], b = pts[pts.length - 1];
  const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
  const ux = (b.x - a.x) / len, uy = (b.y - a.y) / len;
  const tip = { x: b.x - ux * 7, y: b.y - uy * 7 };
  const base = { x: tip.x - ux * 8, y: tip.y - uy * 8 };
  return `M ${tip.x} ${tip.y} L ${base.x - uy * 4.5} ${base.y + ux * 4.5} L ${base.x + uy * 4.5} ${base.y - ux * 4.5} Z`;
}

export function buildBoard(spec: SpecModel, focus: string): Board | null {
  const layout = spec.layouts[focus];
  if (!layout) return null;
  const nodes: BoardNode[] = Object.entries(layout.nodes)
    .filter(([id]) => spec.blocks[id])
    .map(([id, box]) => ({
      id, box, block: spec.blocks[id],
      drill: !!spec.layouts[id],
      ports: (spec.ports[id] ?? []).map((p) => {
        const abs = portPoint(box, p.side, p.offset);
        return { ...p, at: { x: abs.x - box.x, y: abs.y - box.y } };
      }),
    }));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const boxes = nodes.map((n) => n.box);
  const wires: BoardWire[] = [];
  for (const c of spec.connections) {
    const a = byId.get(c.from), b = byId.get(c.to);
    const pa = a?.ports.find((p) => p.id === c.fromPort), pb = b?.ports.find((p) => p.id === c.toPort);
    if (!a || !b || !pa || !pb) continue;
    const start = portPoint(a.box, pa.side, pa.offset), end = portPoint(b.box, pb.side, pb.offset);
    const d = routePath(start, end, pa.side, pb.side, { obstacles: boxes.filter((x) => x !== a.box && x !== b.box), width: layout.canvas.width, height: layout.canvas.height });
    const pts = points(d);
    wires.push({ id: c.id, conn: c, d, mid: midpoint(pts), arrow: c.bidirectional ? null : arrowHead(pts) });
  }
  return { focus, w: layout.canvas.width, h: layout.canvas.height, nodes, wires };
}

/** Ancestors from the root down to (not including) `id`. */
export function ancestors(spec: SpecModel, id: string): string[] {
  const out: string[] = [];
  for (let p = spec.blocks[id]?.parent; p; p = spec.blocks[p]?.parent) out.unshift(p);
  return out;
}

/** The layout that draws `id`: its parent's when it has one, otherwise any layout that lists it. */
export function layoutFor(spec: SpecModel, id: string): string | null {
  for (const a of [...ancestors(spec, id)].reverse()) if (spec.layouts[a]?.nodes[id]) return a;
  return Object.keys(spec.layouts).find((l) => spec.layouts[l].nodes[id]) ?? null;
}

/** Nearest block in `visible` that is `id` or contains it. */
export function visibleAncestor(spec: SpecModel, id: string, visible: Set<string>): string | null {
  for (let b: string | null | undefined = id; b; b = spec.blocks[b]?.parent) if (visible.has(b)) return b;
  return null;
}
