import { graph } from "./graph";
import type { Maturity, Side } from "./schema";

/**
 * The serialisable slice of the architecture model handed to the client.
 *
 * The explorer keeps selection, focus, lens and viewport on the client so that
 * clicking a block is instant and never costs a round trip. That is only
 * possible if the whole model travels once, so this stays deliberately
 * compact: no PDF evidence bodies, no duplicated derived indexes.
 */

export type ViewBlock = {
  id: string;
  name: string;
  kind: string;
  summary: string;
  parent: string | null;
  owner: string;
  maturity: Maturity;
  responsibilities: string[];
  documents: Array<{ title: string; path?: string; url?: string; anchor?: string; page?: number }>;
  issues: Array<{ url: string; title?: string }>;
  rtl: Array<{ repository: string; revision: string; path: string; symbol: string; url: string }>;
  childIds: string[];
  /** Authored canvas to draw when this block is selected. */
  focusId: string;
  /** No children and no authored canvas — intentionally opaque at this level. */
  sealed: boolean;
};

export type ViewPort = { id: string; label: string; protocol: string; side: Side; offset: number };

export type ViewConnection = {
  id: string;
  from: string;
  fromPort: string;
  to: string;
  toPort: string;
  protocol: string;
  label: string;
  bidirectional: boolean;
};

export type ViewLayout = {
  canvas: { width: number; height: number };
  nodes: Record<string, { x: number; y: number; w: number; h: number }>;
};

export type ViewSnapshot = {
  product: { name: string; description: string; defaultFocus: string };
  protocols: Record<string, { name: string; color: string; abbreviation: string; purpose: string; roles: string[] }>;
  owners: Record<string, string>;
  blocks: Record<string, ViewBlock>;
  rootIds: string[];
  ports: Record<string, ViewPort[]>;
  connections: ViewConnection[];
  layouts: Record<string, ViewLayout>;
  sequences: Array<{
    id: string;
    name: string;
    summary: string;
    kind: "boot" | "instruction";
    steps: Array<{ id: string; title: string; detail: string; blocks: string[]; protocol?: string; parallel?: boolean; branch?: boolean }>;
  }>;
  counts: {
    blocks: number;
    connections: number;
    ports: number;
    byMaturity: Record<string, number>;
    unassigned: number;
    withoutRtl: number;
  };
};

let cached: ViewSnapshot | null = null;

export function viewSnapshot(): ViewSnapshot {
  if (cached) return cached;
  const g = graph();
  const m = g.model;

  const blocks: Record<string, ViewBlock> = {};
  for (const block of g.blocks) {
    blocks[block.id] = {
      id: block.id,
      name: block.name,
      kind: block.kind,
      summary: block.summary,
      parent: block.parent,
      owner: block.owner,
      maturity: block.maturity,
      responsibilities: block.responsibilities,
      documents: block.documents,
      issues: block.issues,
      rtl: block.rtl,
      childIds: g.childrenOf(block.id).map((child) => child.id),
      focusId: g.canvasFocusFor(block.id),
      sealed: g.isSealed(block.id),
    };
  }

  const ports: Record<string, ViewPort[]> = {};
  for (const block of g.blocks) {
    const bindings = g.portsFor(block.id);
    if (bindings.length) {
      ports[block.id] = bindings.map(({ id, port }) => ({
        id,
        label: port.label,
        protocol: port.protocol,
        side: port.side,
        offset: port.offset,
      }));
    }
  }

  const layouts: Record<string, ViewLayout> = {};
  for (const [focusId, layout] of Object.entries(m.layouts)) {
    layouts[focusId] = {
      canvas: layout.canvas,
      nodes: Object.fromEntries(
        Object.entries(layout.nodes).map(([id, rect]) => [id, { x: rect.x, y: rect.y, w: rect.width, h: rect.height }]),
      ),
    };
  }

  cached = {
    product: { name: m.product.name, description: m.product.description, defaultFocus: m.product.default_focus },
    protocols: m.protocols,
    owners: Object.fromEntries(m.owners.map((owner) => [owner.id, owner.name])),
    blocks,
    rootIds: g.roots.map((block) => block.id),
    ports,
    connections: m.connections.map((connection) => ({
      id: connection.id,
      from: connection.from,
      fromPort: connection.from_port,
      to: connection.to,
      toPort: connection.to_port,
      protocol: connection.protocol,
      label: connection.label,
      bidirectional: connection.bidirectional,
    })),
    layouts,
    sequences: [
      ...m.behaviors.map((behavior) => ({
        id: behavior.id,
        name: behavior.name,
        summary: behavior.summary,
        kind: "boot" as const,
        steps: behavior.steps.map((step) => ({
          id: step.id,
          title: step.title,
          detail: step.detail,
          blocks: step.blocks,
          parallel: step.parallel,
          branch: step.branch,
        })),
      })),
      ...m.instruction_flows.map((flow) => ({
        id: flow.id,
        name: flow.name,
        summary: flow.summary,
        kind: "instruction" as const,
        steps: flow.steps.map((step, index) => ({
          id: `${flow.id}-${index}`,
          title: step.title,
          detail: step.detail,
          blocks: step.blocks,
          protocol: step.protocol,
        })),
      })),
    ],
    counts: {
      blocks: g.counts.blocks,
      connections: g.counts.connections,
      ports: g.counts.ports,
      byMaturity: g.counts.byMaturity,
      unassigned: g.counts.unassigned,
      withoutRtl: g.counts.withoutRtl,
    },
  };
  return cached;
}
