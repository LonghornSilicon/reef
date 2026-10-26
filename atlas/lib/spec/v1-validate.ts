import { modelSchema, type ArchitectureModel } from "./v1-schema";

/**
 * The model's cross-references, checked the way scripts/validate_model.py
 * checks them.
 *
 * The schema alone accepts a model whose connections point at missing ports
 * or whose children have nowhere to be drawn. CI catches that for edits made
 * in a clone; Atlas runs this before it commits an edit of its own, so a
 * change request never starts out broken.
 *
 * One check is left to CI: whether a document `path` exists. Atlas does not
 * edit document links, and the files they name live outside the model.
 */
export function validateModel(raw: unknown): { ok: true; model: ArchitectureModel } | { ok: false; issues: string[] } {
  const parsed = modelSchema.safeParse(raw);
  if (!parsed.success) {
    return { ok: false, issues: parsed.error.issues.map((issue) => `${issue.path.join(".") || "<root>"}: ${issue.message}`) };
  }
  const issues = referenceIssues(parsed.data);
  return issues.length ? { ok: false, issues } : { ok: true, model: parsed.data };
}

function duplicates(ids: string[]) {
  const seen = new Set<string>();
  const repeated = new Set<string>();
  for (const id of ids) (seen.has(id) ? repeated : seen).add(id);
  return [...repeated];
}

export function referenceIssues(model: ArchitectureModel): string[] {
  const issues: string[] = [];
  const blockIds = new Set(model.blocks.map((block) => block.id));
  const ownerIds = new Set(model.owners.map((owner) => owner.id));
  const protocolIds = new Set(Object.keys(model.protocols));

  for (const [label, ids] of [
    ["owners", model.owners.map((item) => item.id)],
    ["blocks", model.blocks.map((item) => item.id)],
    ["connections", model.connections.map((item) => item.id)],
    ["behaviors", model.behaviors.map((item) => item.id)],
    ["instruction_flows", model.instruction_flows.map((item) => item.id)],
  ] as const) {
    for (const id of duplicates([...ids])) issues.push(`${label}: duplicate id '${id}'`);
  }

  if (!blockIds.has(model.product.default_focus)) issues.push(`product.default_focus: unknown block '${model.product.default_focus}'`);

  const parentOf = new Map(model.blocks.map((block) => [block.id, block.parent]));
  model.blocks.forEach((block, index) => {
    const label = `blocks[${index}] (${block.id})`;
    if (block.parent !== null && !blockIds.has(block.parent)) issues.push(`${label}.parent: unknown block '${block.parent}'`);
    if (block.parent === block.id) issues.push(`${label}.parent: block cannot contain itself`);
    if (!ownerIds.has(block.owner)) issues.push(`${label}.owner: unknown owner '${block.owner}'`);
    if (block.responsibilities.some((item) => !item.trim())) issues.push(`${label}.responsibilities: a responsibility is empty`);
    block.documents.forEach((document, documentIndex) => {
      if (!document.title) issues.push(`${label}.documents[${documentIndex}]: expected a title`);
      if (document.path && (document.path.startsWith("/") || document.path.split("/").includes(".."))) {
        issues.push(`${label}.documents[${documentIndex}]: path must stay in repository`);
      } else if (!document.path && !document.url) issues.push(`${label}.documents[${documentIndex}]: expected path or url`);
    });
    block.issues.forEach((issue, issueIndex) => {
      if (!issue.url) issues.push(`${label}.issues[${issueIndex}]: expected a url`);
    });
  });

  for (const id of blockIds) {
    const seen = new Set<string>();
    let current: string | null | undefined = id;
    while (current) {
      if (seen.has(current)) {
        issues.push(`blocks: containment cycle includes '${id}'`);
        break;
      }
      seen.add(current);
      current = parentOf.get(current);
    }
  }

  const childrenOf = new Map<string, string[]>();
  for (const block of model.blocks) {
    if (block.parent !== null) childrenOf.set(block.parent, [...(childrenOf.get(block.parent) ?? []), block.id]);
  }
  for (const [parentId, childIds] of childrenOf) {
    const layout = model.layouts[parentId];
    if (!layout) {
      issues.push(`layouts: composite block '${parentId}' has no authored layout`);
      continue;
    }
    const missing = childIds.filter((id) => !layout.nodes[id]).sort();
    if (missing.length) issues.push(`layouts.${parentId}.nodes: missing children ${missing.join(", ")}`);
  }
  for (const [focusId, layout] of Object.entries(model.layouts)) {
    for (const [nodeId, rect] of Object.entries(layout.nodes)) {
      if (!blockIds.has(nodeId)) {
        issues.push(`layouts.${focusId}.nodes: unknown block '${nodeId}'`);
        continue;
      }
      if (rect.x < 0 || rect.y < 0) issues.push(`layouts.${focusId}.nodes.${nodeId}: position and size must be positive`);
      if (rect.x + rect.width > layout.canvas.width || rect.y + rect.height > layout.canvas.height) {
        issues.push(`layouts.${focusId}.nodes.${nodeId}: block falls outside the canvas`);
      }
    }
  }

  for (const [blockId, ports] of Object.entries(model.ports)) {
    if (!blockIds.has(blockId)) {
      issues.push(`ports: unknown block '${blockId}'`);
      continue;
    }
    for (const [portId, port] of Object.entries(ports)) {
      if (!port.label.trim()) issues.push(`ports.${blockId}.${portId}.label: expected a visible label`);
      if (!protocolIds.has(port.protocol)) issues.push(`ports.${blockId}.${portId}.protocol: unknown protocol '${port.protocol}'`);
    }
  }

  model.connections.forEach((connection, index) => {
    const label = `connections[${index}] (${connection.id})`;
    if (!blockIds.has(connection.from)) issues.push(`${label}.from: unknown block '${connection.from}'`);
    if (!blockIds.has(connection.to)) issues.push(`${label}.to: unknown block '${connection.to}'`);
    if (!protocolIds.has(connection.protocol)) issues.push(`${label}.protocol: unknown protocol '${connection.protocol}'`);
    for (const [blockId, portId, field] of [
      [connection.from, connection.from_port, "from_port"],
      [connection.to, connection.to_port, "to_port"],
    ] as const) {
      const port = model.ports[blockId]?.[portId];
      if (!port) issues.push(`${label}.${field}: unknown port '${blockId}'.'${portId}'`);
      else if (port.protocol !== connection.protocol) issues.push(`${label}.${field}: port protocol does not match connection protocol`);
    }
  });

  model.behaviors.forEach((behavior, behaviorIndex) => {
    behavior.steps.forEach((step, stepIndex) => {
      for (const id of step.blocks) if (!blockIds.has(id)) issues.push(`behaviors[${behaviorIndex}].steps[${stepIndex}]: unknown block '${id}'`);
    });
  });

  if (!model.instruction_flows.length) issues.push("instruction_flows: expected at least one instruction path");
  model.instruction_flows.forEach((flow, flowIndex) => {
    if (!flow.name || !flow.summary) issues.push(`instruction_flows[${flowIndex}]: expected name and summary`);
    flow.steps.forEach((step, stepIndex) => {
      const label = `instruction_flows[${flowIndex}].steps[${stepIndex}]`;
      if (!step.title || !step.detail || !step.blocks.length) issues.push(`${label}: expected title, detail, and blocks`);
      if (!protocolIds.has(step.protocol)) issues.push(`${label}.protocol: unknown protocol '${step.protocol}'`);
      for (const id of step.blocks) if (!blockIds.has(id)) issues.push(`${label}: unknown block '${id}'`);
    });
  });

  return [...new Set(issues)].sort();
}
