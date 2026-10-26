import { parse } from "yaml";
import type { ArchitectureModel, Maturity } from "./v1-schema";
import { parseExpr, identifiers } from "./rules";
import {
  STATUS,
  lowerStatus,
  type ProcessStage,
  type SignalDef,
  type SpecModel,
  type SpecPort,
  type Status,
  type TimingRule,
} from "./types";

/** proposal/specified → planned; everything else carries over (plan.md §M1.2). */
export function statusFromMaturity(m: Maturity): Status {
  return m === "proposal" || m === "specified" ? "planned" : m;
}

function lineIndex(text: string) {
  const lines = text.split("\n");
  return (pattern: RegExp) => {
    const i = lines.findIndex((l) => pattern.test(l));
    return i < 0 ? undefined : i + 1;
  };
}
const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** In-memory v1 → v2 migration. Ports and connections inherit the lower status of their endpoints. */
export function migrate(v1: ArchitectureModel, text: string, source: SpecModel["source"]): SpecModel {
  const find = lineIndex(text);
  const childIds: Record<string, string[]> = {};
  for (const b of v1.blocks) if (b.parent) (childIds[b.parent] ??= []).push(b.id);
  const byRank = new Map(v1.blocks.map((b) => [b.id, [b.rank, b.order] as const]));
  for (const list of Object.values(childIds)) list.sort((a, b) => byRank.get(a)![0] - byRank.get(b)![0] || byRank.get(a)![1] - byRank.get(b)![1]);

  const blocks: SpecModel["blocks"] = {};
  for (const b of v1.blocks) {
    blocks[b.id] = {
      id: b.id, name: b.name, kind: b.kind, summary: b.summary, parent: b.parent, owner: b.owner,
      status: statusFromMaturity(b.maturity),
      responsibilities: b.responsibilities, documents: b.documents,
      rtl: b.rtl[0] ? { module: b.rtl[0].symbol } : null,
      childIds: childIds[b.id] ?? [],
      line: find(new RegExp(`^\\s*- id: ${esc(b.id)}\\s*$`)),
    };
  }
  const protocols: SpecModel["protocols"] = {};
  for (const [id, p] of Object.entries(v1.protocols)) {
    protocols[id] = { id, ...p, line: find(new RegExp(`^  ${esc(id)}: \\{`)) };
  }
  const ports: SpecModel["ports"] = {};
  for (const [blockId, map] of Object.entries(v1.ports)) {
    ports[blockId] = Object.entries(map).map(([id, p]) => ({
      id, blockId, label: p.label, protocol: p.protocol, side: p.side, offset: p.offset,
      status: blocks[blockId]?.status ?? "planned",
    }));
  }
  const model: SpecModel = {
    product: { name: v1.product.name, description: v1.product.description, defaultFocus: v1.product.default_focus },
    protocols,
    interfaces: {},
    blocks,
    rootIds: v1.blocks.filter((b) => b.parent === null).map((b) => b.id),
    ports,
    connections: v1.connections.map((c) => ({
      id: c.id, from: c.from, fromPort: c.from_port, to: c.to, toPort: c.to_port,
      protocol: c.protocol, label: c.label, bidirectional: c.bidirectional, status: "planned",
    })),
    layouts: Object.fromEntries(Object.entries(v1.layouts).map(([id, l]) => [id, {
      canvas: l.canvas,
      nodes: Object.fromEntries(Object.entries(l.nodes).map(([n, r]) => [n, { x: r.x, y: r.y, w: r.width, h: r.height }])),
    }])),
    flows: v1.instruction_flows,
    process: [],
    counts: Object.fromEntries(STATUS.map((s) => [s, 0])) as Record<Status, number>,
    source,
  };
  return finalize(model);
}

/** Recompute derived status for connections and the per-status counts. */
export function finalize(model: SpecModel, explicit: Set<string> = new Set()): SpecModel {
  const portOf = (block: string, port: string) => model.ports[block]?.find((p) => p.id === port);
  for (const c of model.connections) {
    if (explicit.has(`connection:${c.id}`)) continue;
    const a = portOf(c.from, c.fromPort)?.status ?? model.blocks[c.from]?.status ?? "planned";
    const b = portOf(c.to, c.toPort)?.status ?? model.blocks[c.to]?.status ?? "planned";
    c.status = lowerStatus(a, b);
  }
  model.counts = Object.fromEntries(STATUS.map((s) => [s, 0])) as Record<Status, number>;
  for (const b of Object.values(model.blocks)) model.counts[b.status] += 1;
  return model;
}

type Overlay = {
  protocols?: Record<string, { name?: string; color?: string; abbreviation?: string; purpose?: string; roles?: string[]; signals?: SignalDef[]; timing_rules?: TimingRule[] }>;
  interfaces?: Record<string, { protocol: string; params?: Record<string, number>; role_naming?: "decoupled" | "flat" }>;
  blocks?: Record<string, { status?: Status; rtl?: { module: string; instance?: string } | null }>;
  ports?: Record<string, { status?: Status; interface?: string; role?: string; rtl?: SpecPort["rtl"] }>;
  connections?: Record<string, { status?: Status }>;
  process?: Array<{ id: string; name: string; stages: ProcessStage[] }>;
};

/**
 * Apply a branch overlay: the v2 fields a branch adds on top of the spec on disk.
 * Used for the sample branches until the spec itself moves to v2 in the RTL repository.
 */
export function applyOverlay(base: SpecModel, overlayText: string, overlayPath: string): SpecModel {
  const o = (parse(overlayText) ?? {}) as Overlay;
  const m: SpecModel = structuredClone(base);
  m.source = { ...m.source, overlay: overlayPath };
  const find = lineIndex(overlayText);

  for (const [id, p] of Object.entries(o.protocols ?? {})) {
    const prev = m.protocols[id];
    m.protocols[id] = {
      id, name: p.name ?? prev?.name ?? id, color: p.color ?? prev?.color ?? "#64748b",
      abbreviation: p.abbreviation ?? prev?.abbreviation ?? id.toUpperCase(), purpose: p.purpose ?? prev?.purpose ?? "",
      roles: p.roles ?? prev?.roles ?? [], signals: p.signals, timingRules: p.timing_rules,
      line: prev?.line,
    };
  }
  for (const [id, i] of Object.entries(o.interfaces ?? {})) {
    m.interfaces[id] = { id, protocol: i.protocol, params: i.params ?? {}, roleNaming: i.role_naming ?? "flat", line: find(new RegExp(`^  ${esc(id)}:`)) };
  }
  for (const [id, b] of Object.entries(o.blocks ?? {})) {
    const block = m.blocks[id];
    if (!block) throw new Error(`${overlayPath}: unknown block "${id}"`);
    if (b.status) block.status = b.status;
    if (b.rtl !== undefined) block.rtl = b.rtl;
  }
  // Ports follow their block unless the overlay says otherwise.
  const explicitPorts = new Set(Object.entries(o.ports ?? {}).filter(([, p]) => p.status).map(([k]) => k));
  for (const [blockId, list] of Object.entries(m.ports)) {
    for (const p of list) if (!explicitPorts.has(`${blockId}.${p.id}`)) p.status = m.blocks[blockId]?.status ?? p.status;
  }
  for (const [key, p] of Object.entries(o.ports ?? {})) {
    const [blockId, portId] = key.split(".");
    const port = m.ports[blockId]?.find((x) => x.id === portId);
    if (!port) throw new Error(`${overlayPath}: unknown port "${key}"`);
    if (p.status) port.status = p.status;
    if (p.interface) port.interface = p.interface;
    if (p.role) port.role = p.role;
    if (p.rtl) port.rtl = p.rtl;
  }
  const explicit = new Set<string>();
  for (const [id, c] of Object.entries(o.connections ?? {})) {
    const conn = m.connections.find((x) => x.id === id);
    if (!conn) throw new Error(`${overlayPath}: unknown connection "${id}"`);
    if (c.status) { conn.status = c.status; explicit.add(`connection:${id}`); }
  }
  m.process = o.process ?? [];
  return finalize(m, explicit);
}

/** v2 reference checks (plan.md §M1.1): interfaces resolve, RTL-checked blocks name a module, rule identifiers are signals. */
export function v2Issues(m: SpecModel): string[] {
  const issues: string[] = [];
  const checked = (s: Status) => s === "implemented" || s === "modified" || s === "verified";
  for (const b of Object.values(m.blocks)) {
    if (checked(b.status) && !b.rtl?.module) issues.push(`blocks.${b.id}: status ${b.status} needs rtl.module`);
  }
  for (const [blockId, list] of Object.entries(m.ports)) {
    for (const p of list) {
      if (p.interface && !m.interfaces[p.interface]) issues.push(`ports.${blockId}.${p.id}: unknown interface "${p.interface}"`);
      if (checked(p.status) && !checked(m.blocks[blockId]?.status ?? "planned")) issues.push(`ports.${blockId}.${p.id}: RTL-checked port on a planned block`);
    }
  }
  for (const i of Object.values(m.interfaces)) if (!m.protocols[i.protocol]) issues.push(`interfaces.${i.id}: unknown protocol "${i.protocol}"`);
  for (const p of Object.values(m.protocols)) {
    const names = new Set((p.signals ?? []).map((s) => s.name));
    for (const r of p.timingRules ?? []) {
      for (const field of [r.when, r.then].filter(Boolean) as string[]) {
        try {
          for (const id of identifiers(parseExpr(field))) if (!names.has(id)) issues.push(`protocols.${p.id}.${r.id}: "${id}" is not a signal of ${p.id}`);
        } catch (e) {
          issues.push(`protocols.${p.id}.${r.id}: ${(e as Error).message}`);
        }
      }
    }
  }
  return issues;
}
