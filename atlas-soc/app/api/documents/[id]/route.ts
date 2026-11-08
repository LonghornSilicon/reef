import { requireActor } from "@/lib/authz";
import { listDocuments } from "@/lib/repository";
import { getObject } from "@/lib/storage";

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const access = await requireActor();
  if (!access.actor) return access.response;
  const { id } = await context.params;
  const document = (await listDocuments()).find((item) => item.id === id);
  if (!document) return Response.json({ error: "Document not found" }, { status: 404 });
  try {
    const bytes = await getObject(document.storageKey);
    const body = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    return new Response(body, {
      headers: {
        "Content-Type": document.mimeType,
        "Content-Disposition": `inline; filename*=UTF-8''${encodeURIComponent(document.filename)}`,
        "Cache-Control": "private, no-store",
      },
    });
  } catch {
    return Response.json({ error: "Document content is unavailable" }, { status: 404 });
  }
}
