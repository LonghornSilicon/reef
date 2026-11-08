import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Icon } from "@/components/icons";
import { resolveEvidence } from "@/components/architecture/evidence";
import { MATURITY_COLOR, MATURITY_LABEL } from "@/components/architecture/lens";
import { graph } from "@/lib/model/graph";
import { listDocuments, listTasks } from "@/lib/repository";

export async function generateMetadata({ params }: { params: Promise<{ blockId: string }> }): Promise<Metadata> {
  const block = graph().blockById.get(decodeURIComponent((await params).blockId));
  return { title: block ? `${block.name} · Block record` : "Block record" };
}

/**
 * The full record for one block.
 *
 * The inspector rail answers "what is this" at a glance; this page is the same
 * content without a 300px constraint, plus the collaborative data that only
 * exists in the database. Both are addressable so either can be shared.
 */
export default async function BlockRecordPage({ params }: { params: Promise<{ blockId: string }> }) {
  const blockId = decodeURIComponent((await params).blockId);
  const g = graph();
  const block = g.blockById.get(blockId);
  if (!block) notFound();

  const trail = g.trailTo(blockId);
  const children = g.childrenOf(blockId);
  const ports = g.portsFor(blockId);
  const edges = g.edgesFor(blockId);
  const [tasks, documents] = await Promise.all([listTasks(), listDocuments()]);
  const linkedTasks = tasks.filter((task) => task.blockIds.includes(blockId));
  const linkedDocuments = documents.filter((document) => document.blockIds.includes(blockId));

  return (
    <article className="record-page">
      <nav className="record-trail" aria-label="Containment">
        {trail.map((entry, index) => (
          <span key={entry.id}>
            {index > 0 && <Icon name="chevron" aria-hidden="true" />}
            {entry.id === blockId ? <b aria-current="true">{entry.name}</b> : <Link href={`/blocks/${entry.id}`}>{entry.name}</Link>}
          </span>
        ))}
      </nav>

      <header className="record-head">
        <span className="inspector-eyebrow">{block.kind}</span>
        <h1>{block.name}</h1>
        <p className="record-summary">{block.summary}</p>
        <div className="identity-chips">
          <span className="maturity-chip" style={{ ["--chip" as string]: MATURITY_COLOR[block.maturity] }}>
            {MATURITY_LABEL[block.maturity]}
          </span>
          <span className={`owner-chip${block.owner === "unassigned" ? " unassigned" : ""}`}>
            <Icon name="person" aria-hidden="true" />
            {g.ownerName(block.owner)}
          </span>
          <code className="identity-id">{block.id}</code>
        </div>
        <Link className="record-open" href={`/architecture/${block.id}`}>
          <Icon name="architecture" aria-hidden="true" />
          Show on the diagram
        </Link>
      </header>

      <div className="record-grid">
        <section className="record-section">
          <h2>Responsibilities</h2>
          <ul className="plain-list">
            {block.responsibilities.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <section className="record-section">
          <h2>
            Interfaces <span className="section-count">{ports.length}</span>
          </h2>
          {ports.length === 0 ? (
            <p className="inspector-absent">No interface is declared for this block in the model.</p>
          ) : (
            <table className="record-table">
              <thead>
                <tr>
                  <th scope="col">Label</th>
                  <th scope="col">Protocol</th>
                  <th scope="col">Edge</th>
                </tr>
              </thead>
              <tbody>
                {ports.map(({ id, port, protocol }) => (
                  <tr key={id}>
                    <td>
                      <span className="port-swatch" style={{ background: protocol.color }} aria-hidden="true" />
                      {port.label}
                    </td>
                    <td>{protocol.name}</td>
                    <td>{port.side}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="record-section">
          <h2>
            Connections <span className="section-count">{edges.length}</span>
          </h2>
          {edges.length === 0 ? (
            <p className="inspector-absent">No connection reaches this block in the model.</p>
          ) : (
            <ul className="edge-list">
              {edges.map(({ connection, peer, direction, protocol }) => (
                <li key={connection.id}>
                  <Link href={`/blocks/${peer.id}`}>
                    <span className="edge-swatch" style={{ background: protocol.color }} aria-hidden="true" />
                    <span className="edge-direction">{direction === "outbound" ? "to" : "from"}</span>
                    <span className="edge-peer">{peer.name}</span>
                    <span className="edge-caption">{connection.label}</span>
                    <Icon name="arrow" aria-hidden="true" />
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="record-section">
          <h2>
            Contains <span className="section-count">{children.length}</span>
          </h2>
          {children.length === 0 ? (
            <p className="inspector-absent">
              No modelled internals.
            </p>
          ) : (
            <ul className="relation-list">
              {children.map((child) => (
                <li key={child.id}>
                  <Link href={`/blocks/${child.id}`}>
                    <Icon name="chevron" aria-hidden="true" />
                    <span>{child.name}</span>
                    <small>{child.kind}</small>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="record-section">
          <h2>Evidence</h2>
          {block.documents.length === 0 && <p className="inspector-absent">No evidence is linked to this block.</p>}
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

        <section className="record-section">
          <h2>
            Linked work <span className="section-count">{linkedTasks.length + linkedDocuments.length}</span>
          </h2>
          {linkedTasks.length === 0 && linkedDocuments.length === 0 ? (
            <p className="inspector-absent">
              No task or managed document references this block yet.{" "}
              <Link href={`/tasks?block=${encodeURIComponent(blockId)}`}>Open the board</Link>.
            </p>
          ) : (
            <ul className="relation-list">
              {linkedTasks.map((task) => (
                <li key={task.id}>
                  <Link href={`/tasks?block=${encodeURIComponent(blockId)}`}>
                    <Icon name="tasks" aria-hidden="true" />
                    <span>{task.title}</span>
                    <small>{task.status.replace("_", " ")}</small>
                  </Link>
                </li>
              ))}
              {linkedDocuments.map((document) => (
                <li key={document.id}>
                  <Link href={`/api/documents/${document.id}`} target="_blank">
                    <Icon name="documents" aria-hidden="true" />
                    <span>{document.title}</span>
                    <small>managed upload</small>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <footer className="record-foot">
        <dl className="field-facts">
          <div>
            <dt>Source</dt>
            <dd>architecture/soc.yaml</dd>
          </div>
          <div>
            <dt>Authority</dt>
            <dd>Canonical model</dd>
          </div>
        </dl>
        <p className="inspector-authority">
          Ownership and maturity are model data. Change them in Git so the edit stays reviewable —{" "}
          <Link href="/wiki/editing-guide">editing guide</Link>.
        </p>
      </footer>
    </article>
  );
}
