import { requireActor } from "@/lib/authz";
import { deleteThread, getThread } from "@/lib/repository";

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const access = await requireActor();
  if (!access.actor) return access.response;
  const { id } = await context.params;
  const thread = await getThread(access.actor, id);
  return thread ? Response.json({ thread }) : Response.json({ error: "Conversation not found" }, { status: 404 });
}

export async function DELETE(_request: Request, context: { params: Promise<{ id: string }> }) {
  const access = await requireActor();
  if (!access.actor) return access.response;
  const { id } = await context.params;
  return (await deleteThread(access.actor, id))
    ? new Response(null, { status: 204 })
    : Response.json({ error: "Conversation not found" }, { status: 404 });
}
