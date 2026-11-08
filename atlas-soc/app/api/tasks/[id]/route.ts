import { z } from "zod";
import { requireActor } from "@/lib/authz";
import { TASK_STATUSES } from "@/lib/domain";
import { updateTaskStatus } from "@/lib/repository";

export async function PATCH(request: Request, context: { params: Promise<{ id: string }> }) {
  const access = await requireActor(["admin", "editor"]);
  if (!access.actor) return access.response;
  const parsed = z.object({ status: z.enum(TASK_STATUSES) }).safeParse(await request.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "Invalid task status" }, { status: 400 });
  const { id } = await context.params;
  const task = await updateTaskStatus(id, parsed.data.status);
  return task ? Response.json({ task }) : Response.json({ error: "Task not found" }, { status: 404 });
}
