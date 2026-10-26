import { readRunFile } from "@/lib/runs/store";

/** Serves one run file. In M3 this becomes a redirect to a signed bucket URL. */
export async function GET(_req: Request, { params }: { params: Promise<{ id: string; file: string }> }) {
  const { id, file } = await params;
  const body = readRunFile(id, file);
  if (body === null) return Response.json({ error: `${file} was not uploaded for run ${id}` }, { status: 404 });
  const type = file.endsWith(".jsonl") ? "application/x-ndjson" : "application/json";
  return new Response(body, { headers: { "content-type": `${type}; charset=utf-8`, "cache-control": "private, max-age=60" } });
}
