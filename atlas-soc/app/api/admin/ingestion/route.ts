import { requireActor } from "@/lib/authz";
import { aiIntegrationEnabled, publicErrorMessage, secretMatches } from "@/lib/config";
import { ingestKnowledge } from "@/lib/ingestion";
import { latestKnowledgeRevision } from "@/lib/repository";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function GET() {
  const access = await requireActor(["admin"]);
  if (!access.actor) return access.response;
  if (!aiIntegrationEnabled()) {
    return Response.json({ error: "Atlas has no AI integration yet." }, { status: 503 });
  }
  return Response.json({ revision: await latestKnowledgeRevision() });
}

export async function POST(request: Request) {
  const secret = request.headers.get("x-atlas-ingestion-secret");
  const trustedAutomation = secretMatches(secret, process.env.ATLAS_INGESTION_SECRET);
  if (!trustedAutomation) {
    const access = await requireActor(["admin"]);
    if (!access.actor) return access.response;
  }
  if (!aiIntegrationEnabled()) {
    return Response.json({ error: "Atlas has no AI integration yet." }, { status: 503 });
  }
  try {
    return Response.json(await ingestKnowledge(), { status: 201 });
  } catch (error) {
    console.error("Atlas knowledge ingestion failed", error);
    return Response.json({ error: publicErrorMessage(error, "Ingestion failed") }, { status: 502 });
  }
}
