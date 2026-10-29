import { loadModel } from "./load";
import type { ArchitectureModel, Block, Connection, Port, Protocol } from "./schema";

export type PortBinding = { id: string; port: Port; protocol: Protocol };

export type EdgeBinding = {
  connection: Connection;
  protocol: Protocol;
  /** The block at the other end, from the perspective of the queried block. */
  peer: Block;
  direction: "outbound" | "inbound";
  /** Port on the queried block. */
  localPortId: string;
  /** Port on the peer block. */
  peerPortId: string;
};

export type ArchitectureGraph = {
  model: ArchitectureModel;
  blocks: Block[];
  blockById: Map<string, Block>;
  /** Blocks with parent === null. There are deliberately two in this model. */
  roots: Block[];
  childrenOf(blockId: string): Block[];
  /** Nearest ancestor first, root last. Excludes the block itself. */
  ancestorsOf(blockId: string): Block[];
  /** Root first, block itself last — breadcrumb order. */
  trailTo(blockId: string): Block[];
  portsFor(blockId: string): PortBinding[];
  edgesFor(blockId: string): EdgeBinding[];
  protocolOf(id: string): Protocol;
  ownerName(ownerId: string): string;
  hasLayout(blockId: string): boolean;
  /**
   * Which authored canvas should be drawn when `blockId` is selected.
   *
   * Selection and focus are deliberately separate: selecting a leaf must never
   * fail, so we draw the nearest ancestor canvas that actually contains the
   * block rather than demanding the block have children of its own.
   */
  canvasFocusFor(blockId: string): string;
  isSealed(blockId: string): boolean;
  counts: {
    blocks: number;
    connections: number;
    ports: number;
    byMaturity: Record<string, number>;
    byOwner: Record<string, number>;
    unassigned: number;
    withoutRtl: number;
  };
};

let cached: ArchitectureGraph | null = null;

export function graph(): ArchitectureGraph {
  if (cached) return cached;
  const model = loadModel();

  const blockById = new Map(model.blocks.map((block) => [block.id, block]));
  const childIds = new Map<string, string[]>();
  for (const block of model.blocks) {
    if (block.parent === null) continue;
    const siblings = childIds.get(block.parent) ?? [];
    siblings.push(block.id);
    childIds.set(block.parent, siblings);
  }

  const byRankOrder = (a: Block, b: Block) => a.rank - b.rank || a.order - b.order || a.name.localeCompare(b.name);

  const edgeIndex = new Map<string, EdgeBinding[]>();
  const protocolOf = (id: string): Protocol =>
    model.protocols[id] ?? model.protocols.control ?? { name: id, color: "#64748b", abbreviation: id.toUpperCase(), purpose: "", roles: [] };

  for (const connection of model.connections) {
    const from = blockById.get(connection.from);
    const to = blockById.get(connection.to);
    if (!from || !to) continue;
    const protocol = protocolOf(connection.protocol);
    const push = (key: string, binding: EdgeBinding) => {
      const list = edgeIndex.get(key) ?? [];
      list.push(binding);
      edgeIndex.set(key, list);
    };
    push(connection.from, {
      connection,
      protocol,
      peer: to,
      direction: "outbound",
      localPortId: connection.from_port,
      peerPortId: connection.to_port,
    });
    push(connection.to, {
      connection,
      protocol,
      peer: from,
      direction: connection.bidirectional ? "outbound" : "inbound",
      localPortId: connection.to_port,
      peerPortId: connection.from_port,
    });
  }

  const owners = new Map(model.owners.map((owner) => [owner.id, owner.name]));

  const byMaturity: Record<string, number> = {};
  const byOwner: Record<string, number> = {};
  let ports = 0;
  for (const block of model.blocks) {
    byMaturity[block.maturity] = (byMaturity[block.maturity] ?? 0) + 1;
    byOwner[block.owner] = (byOwner[block.owner] ?? 0) + 1;
    ports += Object.keys(model.ports[block.id] ?? {}).length;
  }

  const hasLayout = (blockId: string) => Boolean(model.layouts[blockId]);

  const ancestorsOf = (blockId: string): Block[] => {
    const chain: Block[] = [];
    let current = blockById.get(blockId)?.parent ?? null;
    const guard = new Set<string>([blockId]);
    while (current && !guard.has(current)) {
      guard.add(current);
      const block = blockById.get(current);
      if (!block) break;
      chain.push(block);
      current = block.parent;
    }
    return chain;
  };

  cached = {
    model,
    blocks: model.blocks,
    blockById,
    roots: model.blocks.filter((block) => block.parent === null).sort(byRankOrder),
    childrenOf: (blockId) =>
      (childIds.get(blockId) ?? [])
        .map((id) => blockById.get(id))
        .filter((block): block is Block => Boolean(block))
        .sort(byRankOrder),
    ancestorsOf,
    trailTo: (blockId) => {
      const block = blockById.get(blockId);
      if (!block) return [];
      return [...ancestorsOf(blockId).reverse(), block];
    },
    portsFor: (blockId) =>
      Object.entries(model.ports[blockId] ?? {}).map(([id, port]) => ({
        id,
        port,
        protocol: protocolOf(port.protocol),
      })),
    edgesFor: (blockId) => edgeIndex.get(blockId) ?? [],
    protocolOf,
    ownerName: (ownerId) => owners.get(ownerId) ?? ownerId,
    hasLayout,
    canvasFocusFor: (blockId) => {
      const block = blockById.get(blockId);
      if (!block) return model.product.default_focus;
      // Prefer the canvas that contains this block as a node.
      for (const ancestor of [blockById.get(block.parent ?? "") , ...ancestorsOf(blockId)]) {
        if (ancestor && hasLayout(ancestor.id) && model.layouts[ancestor.id].nodes[blockId]) return ancestor.id;
      }
      for (const ancestor of ancestorsOf(blockId)) {
        if (hasLayout(ancestor.id)) return ancestor.id;
      }
      // A root, or a block whose parent authored no canvas: draw its own.
      return hasLayout(blockId) ? blockId : model.product.default_focus;
    },
    isSealed: (blockId) => (childIds.get(blockId) ?? []).length === 0 && !hasLayout(blockId),
    counts: {
      blocks: model.blocks.length,
      connections: model.connections.length,
      ports,
      byMaturity,
      byOwner,
      unassigned: byOwner.unassigned ?? 0,
      withoutRtl: model.blocks.filter((block) => block.rtl.length === 0).length,
    },
  };
  return cached;
}
