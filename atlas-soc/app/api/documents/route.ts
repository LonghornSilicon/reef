import { randomUUID } from "node:crypto";
import { extname } from "node:path";
import { requireActor } from "@/lib/authz";
import { documentStorageConfigured, maxUploadBytes } from "@/lib/config";
import { createDocument, listDocuments } from "@/lib/repository";
import { putObject } from "@/lib/storage";

const ALLOWED_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/webp",
  "text/plain",
  "text/markdown",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
]);

export async function GET() {
  const access = await requireActor();
  if (!access.actor) return access.response;
  return Response.json({ documents: await listDocuments() });
}

export async function POST(request: Request) {
  const access = await requireActor(["admin", "editor"]);
  if (!access.actor) return access.response;
  // Checked before reading the body: without object storage the upload would
  // be accepted and then lost, which is worse than refusing it.
  if (!documentStorageConfigured()) {
    return Response.json(
      { error: "Document storage is not configured, so uploads are disabled." },
      { status: 503 },
    );
  }
  const form = await request.formData();
  const file = form.get("file");
  if (!(file instanceof File)) return Response.json({ error: "Choose a document to upload" }, { status: 400 });
  if (!ALLOWED_TYPES.has(file.type)) return Response.json({ error: "This document type is not allowed" }, { status: 415 });
  const limit = maxUploadBytes();
  if (file.size > limit) {
    const megabytes = Math.floor(limit / (1024 * 1024));
    return Response.json({ error: `Documents must be ${megabytes} MB or smaller` }, { status: 413 });
  }
  const title = String(form.get("title") || file.name).trim().slice(0, 180);
  const blockIds = String(form.get("blockIds") || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 20);
  const extension = extname(file.name).replace(/[^.a-zA-Z0-9]/g, "").slice(0, 10);
  const key = `coral/${new Date().toISOString().slice(0, 10)}/${randomUUID()}${extension}`;
  await putObject(key, new Uint8Array(await file.arrayBuffer()), file.type);
  const document = await createDocument(access.actor, {
    title,
    filename: file.name.slice(0, 240),
    mimeType: file.type,
    byteSize: file.size,
    storageKey: key,
    blockIds,
  });
  return Response.json({ document }, { status: 201 });
}
