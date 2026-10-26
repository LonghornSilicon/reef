"use client";

import { useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import type { DocumentAsset } from "@/lib/domain";
import { Icon } from "@/components/icons";

type WikiEntry = { slug: string; title: string; relativePath: string; summary: string; relatedBlocks: string[] };

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function DocumentLibrary({
  initialDocuments,
  wiki,
  canEdit,
  notice,
}: {
  initialDocuments: DocumentAsset[];
  wiki: WikiEntry[];
  canEdit: boolean;
  /** Explains an empty register when no durable storage is attached. */
  notice?: string;
}) {
  const params = useSearchParams();
  const initialBlock = params.get("block") ?? "";
  const [documents, setDocuments] = useState(initialDocuments);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [blockIds, setBlockIds] = useState(initialBlock);
  const fileRef = useRef<HTMLInputElement>(null);

  async function upload() {
    const file = fileRef.current?.files?.[0];
    if (!file) { setError("Choose a document first."); return; }
    setUploading(true); setError("");
    const form = new FormData();
    form.set("file", file); form.set("title", file.name.replace(/\.[^.]+$/, "")); form.set("blockIds", blockIds);
    const response = await fetch("/api/documents", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) setError(data.error ?? "Atlas could not upload this document.");
    else { setDocuments((current) => [data.document, ...current]); if (fileRef.current) fileRef.current.value = ""; setBlockIds(initialBlock); }
    setUploading(false);
  }

  return (
    <div className="documents-page">
      <header className="documents-heading"><div><h1>Evidence library</h1><p>Git-backed design sources stay canonical. Managed files supplement them with ownership, versions, and architecture links.</p></div><span className="authority-key"><i />Canonical in Git <i />Managed attachment</span></header>
      {canEdit && <section className="upload-bench"><div><Icon name="upload" /><span><strong>Add supporting evidence</strong><small>PDF, image, Markdown, text, Word, or PowerPoint · 10 MB maximum</small></span></div><label>File<input ref={fileRef} type="file" /></label><label>Block IDs<input value={blockIds} onChange={(event) => setBlockIds(event.target.value)} placeholder="Optional: dma, coral-core" /></label><button type="button" onClick={() => void upload()} disabled={uploading}>{uploading ? "Uploading…" : "Add document"}</button></section>}
      {notice && <p className="storage-notice" role="status">{notice}</p>}
      {error && <div className="page-error" role="alert">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
      <section className="document-register">
        <header><h2>Repository sources</h2><span>{wiki.length} current files</span></header>
        {wiki.map((source) => <a className="document-row" href={`/wiki/${source.slug}`} key={source.slug}><span className="document-kind canonical">MD</span><div><strong>{source.title}</strong><p>{source.summary}</p></div><div><code>{source.relativePath}</code><span>{source.relatedBlocks.length} block links</span></div><Icon name="arrow" /></a>)}
      </section>
      <section className="document-register">
        <header><h2>Managed attachments</h2><span>{documents.length} uploaded files</span></header>
        {documents.map((document) => <a className="document-row" href={`/api/documents/${document.id}`} target="_blank" key={document.id}><span className="document-kind managed">{document.filename.split(".").pop()?.slice(0, 4).toUpperCase()}</span><div><strong>{document.title}</strong><p>Revision {document.revision} · {document.ownerEmail ?? "Workspace member"}</p></div><div><code>{document.filename}</code><span>{formatBytes(document.byteSize)} · {document.blockIds.join(", ") || "No block link"}</span></div><Icon name="external" /></a>)}
        {!documents.length && <div className="register-empty"><Icon name="documents" /><span><strong>No managed attachments yet</strong>Upload supporting evidence without changing the canonical wiki.</span></div>}
      </section>
    </div>
  );
}
