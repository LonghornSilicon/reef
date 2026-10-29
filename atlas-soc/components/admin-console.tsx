"use client";

import { useState } from "react";
import type { KnowledgeRevision, Role } from "@/lib/domain";
import { aiIntegrationEnabled } from "@/lib/features";
import { Icon } from "@/components/icons";

export function AdminConsole({ actorRole, initialRevision }: { actorRole: Role; initialRevision: KnowledgeRevision | null }) {
  const [revision, setRevision] = useState(initialRevision);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function ingest() {
    setBusy(true); setMessage("");
    const response = await fetch("/api/admin/ingestion", { method: "POST" });
    const data = await response.json();
    if (!response.ok) setMessage(data.error ?? "Ingestion failed.");
    else {
      setMessage(`Indexed ${data.sourceCount} sources from ${data.gitRevision.slice(0, 12)}.`);
      const status = await fetch("/api/admin/ingestion");
      if (status.ok) setRevision((await status.json()).revision);
    }
    setBusy(false);
  }

  if (actorRole !== "admin") return <div className="access-denied"><Icon name="settings" /><h1>Administrator access required</h1><p>Your workspace role does not allow identity, ingestion, or retention changes.</p></div>;
  return (
    <div className="admin-page">
      <header><h1>Workspace controls</h1><p>Operational controls are explicit and auditable. Atlas never treats generated content as canonical design input.</p></header>
      {aiIntegrationEnabled() ? (
        <section className="admin-section">
          <div><h2>Knowledge revision</h2><p>Upload the current Git-backed sources to the configured retrieval index.</p></div>
          <dl><div><dt>Status</dt><dd><span className={`sync-state ${revision?.status ?? "empty"}`} />{revision?.status ?? "Not indexed"}</dd></div><div><dt>Git revision</dt><dd><code>{revision?.gitRevision.slice(0, 12) ?? "—"}</code></dd></div><div><dt>Sources</dt><dd>{revision?.sourceCount ?? 0}</dd></div><div><dt>Completed</dt><dd>{revision?.completedAt ? new Date(revision.completedAt).toLocaleString() : "—"}</dd></div></dl>
          <button className="primary-button" type="button" onClick={() => void ingest()} disabled={busy}><Icon name="source" />{busy ? "Indexing…" : "Index current sources"}</button>
          {message && <p className="admin-message" role="status">{message}</p>}
        </section>
      ) : (
        <section className="admin-section">
          <div><h2>Knowledge revision</h2><p>No AI integration. Indexing exists only to feed a retrieval model, so there is nothing to index into and no provider is configured.</p></div>
        </section>
      )}
      <section className="admin-section policy-section"><div><h2>Access and retention</h2><p>These policies are enforced by the application and should be mirrored in company infrastructure.</p></div><ul><li><Icon name="check" /><span><strong>Organization SSO</strong>Google or Microsoft OIDC; no Atlas passwords</span></li><li><Icon name="check" /><span><strong>Personal conversation history</strong>Deleted after 30 days or sooner by its owner</span></li><li><Icon name="check" /><span><strong>Ephemeral generated visuals</strong>Image bytes are returned to the browser and not stored by Atlas</span></li></ul></section>
    </div>
  );
}
