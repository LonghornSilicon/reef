"use client";

import Link from "next/link";
import { useMemo } from "react";
import { Icon } from "@/components/icons";
import { aiIntegrationEnabled } from "@/lib/features";
import type { ViewSnapshot } from "@/lib/model/view";
import { resolveEvidence } from "./evidence";
import { MATURITY_COLOR, MATURITY_LABEL, lensHref, type Lens } from "./lens";

type Props = {
  snapshot: ViewSnapshot;
  selectedId: string | null;
  lens: Lens;
  onSelect(blockId: string): void;
};

/**
 * The single inspector.
 *
 * The incumbent build rendered two: a field log in the React shell showing
 * metadata fabricated from the URL slug, and the legacy viewer's own panel
 * clipped inside an overflow-hidden iframe. This one reads the real model and
 * is the only one.
 */
export function Inspector({ snapshot, selectedId, lens, onSelect }: Props) {
  const block = selectedId ? snapshot.blocks[selectedId] : null;

  const edges = useMemo(() => {
    if (!block) return [];
    return snapshot.connections
      .filter((connection) => connection.from === block.id || connection.to === block.id)
      .map((connection) => {
        const outbound = connection.from === block.id;
        const peerId = outbound ? connection.to : connection.from;
        return {
          connection,
          peer: snapshot.blocks[peerId],
          direction: outbound || connection.bidirectional ? ("outbound" as const) : ("inbound" as const),
          localPortId: outbound ? connection.fromPort : connection.toPort,
          color: snapshot.protocols[connection.protocol]?.color ?? "var(--muted)",
        };
      })
      .filter((edge) => Boolean(edge.peer));
  }, [block, snapshot]);

  if (!block) {
    const { byMaturity, blocks, connections, ports, unassigned } = snapshot.counts;
    return (
      <aside className="inspector" aria-label="Block record">
        <div className="inspector-orient">
          <span className="inspector-eyebrow">No block selected</span>
          <h2>Three ways in</h2>
          <ol className="orient-list">
            <li>
              <strong>Outline</strong>
              <span>Every one of the {blocks} blocks, both roots, expandable.</span>
            </li>
            <li>
              <strong>Diagram</strong>
              <span>Click any box. Double-click a composite to open it.</span>
            </li>
            <li>
              <strong>Filter</strong>
              <span>Search by name, ID, kind, or summary.</span>
            </li>
          </ol>
          <dl className="orient-readiness">
            <div>
              <dt>Model</dt>
              <dd>
                {blocks} blocks · {connections} connections · {ports} ports
              </dd>
            </div>
            <div>
              <dt>Maturity</dt>
              <dd>
                {Object.entries(byMaturity)
                  .map(([maturity, count]) => `${count} ${maturity}`)
                  .join(" · ")}
              </dd>
            </div>
            <div>
              <dt>Ownership</dt>
              <dd>{unassigned === blocks ? "every block unassigned" : `${unassigned} unassigned`}</dd>
            </div>
          </dl>
          <p className="inspector-authority">
            Git-backed YAML is canonical. Nothing here is inferred.
          </p>
        </div>
      </aside>
    );
  }

  const ports = snapshot.ports[block.id] ?? [];
  const parent = block.parent ? snapshot.blocks[block.parent] : null;
  const children = block.childIds.map((id) => snapshot.blocks[id]).filter(Boolean);

  return (
    <aside className="inspector" aria-label={`Record for ${block.name}`}>
      <header className="inspector-identity">
        <span className="inspector-eyebrow">{block.kind}</span>
        <h2>{block.name}</h2>
        <div className="identity-chips">
          <span className="maturity-chip" style={{ ["--chip" as string]: MATURITY_COLOR[block.maturity] }}>
            {MATURITY_LABEL[block.maturity]}
          </span>
          <span className={`owner-chip${block.owner === "unassigned" ? " unassigned" : ""}`}>
            <Icon name="person" aria-hidden="true" />
            {snapshot.owners[block.owner] ?? block.owner}
          </span>
        </div>
        <code className="identity-id">{block.id}</code>
      </header>

      <section className="inspector-block">
        <p className="inspector-summary">{block.summary}</p>
      </section>

      <section className="inspector-block">
        <h3>Responsibilities</h3>
        <ul className="plain-list">
          {block.responsibilities.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="inspector-block">
        <h3>
          Interfaces <span className="section-count">{ports.length}</span>
        </h3>
        {ports.length === 0 ? (
          <p className="inspector-absent">No interface is declared for this block in the model.</p>
        ) : (
          <ul className="port-list">
            {ports.map((port) => {
              const protocol = snapshot.protocols[port.protocol];
              return (
                <li key={port.id}>
                  <span className="port-swatch" style={{ background: protocol?.color ?? "var(--muted)" }} aria-hidden="true" />
                  <span className="port-label">{port.label}</span>
                  <span className="port-meta">
                    {protocol?.abbreviation ?? port.protocol} · {port.side}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section className="inspector-block">
        <h3>
          Connections <span className="section-count">{edges.length}</span>
        </h3>
        {edges.length === 0 ? (
          <p className="inspector-absent">No connection reaches this block in the model.</p>
        ) : (
          <ul className="edge-list">
            {edges.map(({ connection, peer, direction, color }) => (
              <li key={connection.id}>
                <button type="button" onClick={() => onSelect(peer.id)}>
                  <span className="edge-swatch" style={{ background: color }} aria-hidden="true" />
                  <span className="edge-direction">{direction === "outbound" ? "to" : "from"}</span>
                  <span className="edge-peer">{peer.name}</span>
                  <span className="edge-caption">{connection.label}</span>
                  <Icon name="arrow" aria-hidden="true" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="inspector-block">
        <h3>Containment</h3>
        <ul className="relation-list">
          {parent ? (
            <li>
              <button type="button" onClick={() => onSelect(parent.id)}>
                <Icon name="level-up" aria-hidden="true" />
                <span>{parent.name}</span>
                <small>parent</small>
              </button>
            </li>
          ) : (
            <li className="relation-static">
              <span>Top-level root in the model.</span>
            </li>
          )}
          {children.map((child) => (
            <li key={child.id}>
              <button type="button" onClick={() => onSelect(child.id)}>
                <Icon name="chevron" aria-hidden="true" />
                <span>{child.name}</span>
                <small>{child.kind}</small>
              </button>
            </li>
          ))}
          {children.length === 0 && (
            <li className="relation-static">
              <span>No modelled internals.</span>
            </li>
          )}
        </ul>
      </section>

      <section className="inspector-block">
        <h3>
          Evidence <span className="section-count">{block.documents.length}</span>
        </h3>
        {block.documents.length === 0 ? (
          <p className="inspector-absent">No evidence is linked to this block.</p>
        ) : (
          <ul className="evidence-list">
            {block.documents.map((document, index) => {
              const resolved = resolveEvidence(document);
              return (
                <li key={`${resolved.path}-${index}`}>
                  {resolved.href ? (
                    <Link href={resolved.href} target={resolved.kind === "wiki" ? undefined : "_blank"}>
                      <Icon name={resolved.kind === "wiki" ? "wiki" : "documents"} aria-hidden="true" />
                      <span>{resolved.title}</span>
                      <code>{resolved.path}</code>
                    </Link>
                  ) : (
                    <span className="evidence-unresolved">
                      <Icon name="documents" aria-hidden="true" />
                      <span>{resolved.title}</span>
                      <code>{resolved.path}</code>
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        )}

        {block.issues.length > 0 && (
          <ul className="evidence-list">
            {block.issues.map((issue) => (
              <li key={issue.url}>
                <a href={issue.url} target="_blank" rel="noreferrer">
                  <Icon name="external" aria-hidden="true" />
                  <span>{issue.title ?? "Linked issue"}</span>
                  <code>{issue.url}</code>
                </a>
              </li>
            ))}
          </ul>
        )}

        {block.rtl.length === 0 ? (
          <p className="inspector-absent">
            No RTL anchor recorded. The expected form is in <Link href="/wiki/editing-guide">the editing guide</Link>.
          </p>
        ) : (
          <ul className="evidence-list">
            {block.rtl.map((anchor) => (
              <li key={`${anchor.repository}-${anchor.symbol}`}>
                <a href={anchor.url} target="_blank" rel="noreferrer">
                  <Icon name="source" aria-hidden="true" />
                  <span>{anchor.symbol}</span>
                  <code>
                    {anchor.path}@{anchor.revision.slice(0, 9)}
                  </code>
                </a>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="inspector-actions">
        <Link className="action-primary" href={`/blocks/${block.id}`}>
          <Icon name="documents" aria-hidden="true" />
          <span>
            <strong>Full record</strong>
            <small>Every field, plus linked work</small>
          </span>
          <Icon name="arrow" aria-hidden="true" />
        </Link>
        {aiIntegrationEnabled() && (
          <Link href={`/copilot?blockId=${encodeURIComponent(block.id)}&blockName=${encodeURIComponent(block.name)}&sourcePath=architecture/soc.yaml`}>
            <Icon name="copilot" aria-hidden="true" />
            <span>Ask about this block</span>
            <Icon name="arrow" aria-hidden="true" />
          </Link>
        )}
        <Link href={`/tasks?block=${encodeURIComponent(block.id)}`}>
          <Icon name="tasks" aria-hidden="true" />
          <span>Related work</span>
          <Icon name="arrow" aria-hidden="true" />
        </Link>
        <Link href={`/documents?block=${encodeURIComponent(block.id)}`}>
          <Icon name="documents" aria-hidden="true" />
          <span>Attached documents</span>
          <Icon name="arrow" aria-hidden="true" />
        </Link>
      </section>

      <footer className="inspector-foot">
        <dl className="field-facts">
          <div>
            <dt>Source</dt>
            <dd>architecture/soc.yaml</dd>
          </div>
          <div>
            <dt>Authority</dt>
            <dd>Canonical model</dd>
          </div>
          {lens.kind !== "structure" && (
            <div>
              <dt>Lens</dt>
              <dd>{lens.kind}</dd>
            </div>
          )}
        </dl>
        <p className="inspector-authority">
          Generated explanations stay secondary to this cited Git source.{" "}
          <Link href={lensHref(block.id, lens)}>Permalink</Link>
        </p>
      </footer>
    </aside>
  );
}
