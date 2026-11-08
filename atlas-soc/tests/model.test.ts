import { describe, expect, it } from "vitest";
import { graph } from "@/lib/model/graph";
import { viewSnapshot } from "@/lib/model/view";

/**
 * These assertions encode the guarantees the explorer is built on. They are
 * deliberately structural rather than golden-value: the model is edited in Git
 * and is expected to grow.
 */
describe("architecture model", () => {
  const g = graph();

  it("parses and validates the checked-in model", () => {
    expect(g.counts.blocks).toBeGreaterThan(0);
    expect(g.counts.connections).toBeGreaterThan(0);
    expect(g.model.schema_version).toBe(1);
  });

  it("exposes every authored root, not just the default focus", () => {
    // The systolic-array reference hierarchy is a second root; the incumbent
    // viewer left it reachable only through search.
    expect(g.roots.length).toBeGreaterThanOrEqual(2);
    expect(g.roots.map((block) => block.id)).toContain(g.model.product.default_focus === "coral-soc" ? "coral-system" : g.roots[0].id);
  });

  it("gives every block a canvas that actually draws it", () => {
    // This is what makes selection total: no block can be selected into a view
    // that cannot show it, which is how the old drill-down dead-ended.
    const orphans = g.blocks.filter((block) => {
      const focus = g.canvasFocusFor(block.id);
      const layout = g.model.layouts[focus];
      if (!layout) return true;
      return block.parent !== null && !layout.nodes[block.id];
    });
    expect(orphans.map((block) => block.id)).toEqual([]);
  });

  it("resolves both endpoints and ports of every connection", () => {
    for (const connection of g.model.connections) {
      expect(g.blockById.has(connection.from), `${connection.id} from`).toBe(true);
      expect(g.blockById.has(connection.to), `${connection.id} to`).toBe(true);
      expect(g.model.ports[connection.from]?.[connection.from_port], `${connection.id} from_port`).toBeDefined();
      expect(g.model.ports[connection.to]?.[connection.to_port], `${connection.id} to_port`).toBeDefined();
    }
  });

  it("can draw every connection on some authored canvas", () => {
    const snapshot = viewSnapshot();
    const undrawn = snapshot.connections.filter(
      (connection) =>
        !Object.values(snapshot.layouts).some((layout) => layout.nodes[connection.from] && layout.nodes[connection.to]),
    );
    expect(undrawn.map((connection) => connection.id)).toEqual([]);
  });

  it("reports containment without cycles", () => {
    for (const block of g.blocks) {
      const chain = g.ancestorsOf(block.id);
      expect(new Set(chain.map((item) => item.id)).size).toBe(chain.length);
      expect(chain.map((item) => item.id)).not.toContain(block.id);
    }
  });

  it("surfaces leaf blocks as navigable through their connections", () => {
    // A leaf has no children, so its edges are the only way onward. If a leaf
    // with connections stopped reporting them, dead ends would come back.
    const leaf = g.blocks.find((block) => g.childrenOf(block.id).length === 0 && g.edgesFor(block.id).length > 0);
    expect(leaf).toBeDefined();
    expect(g.edgesFor(leaf!.id).every((edge) => g.blockById.has(edge.peer.id))).toBe(true);
  });
});
