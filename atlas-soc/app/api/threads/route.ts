import { requireActor } from "@/lib/authz";
import { listThreads } from "@/lib/repository";

export async function GET() {
  const access = await requireActor();
  if (!access.actor) return access.response;
  return Response.json({ threads: await listThreads(access.actor) });
}
