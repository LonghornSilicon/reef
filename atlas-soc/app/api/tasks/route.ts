import { z } from "zod";
import { requireActor } from "@/lib/authz";
import { createTask, listTasks } from "@/lib/repository";
import { TASK_STATUSES } from "@/lib/domain";

const schema = z.object({
  title: z.string().trim().min(3).max(180),
  description: z.string().trim().max(2_000).default(""),
  status: z.enum(TASK_STATUSES).default("backlog"),
  priority: z.enum(["low", "medium", "high", "critical"]).default("medium"),
  blockIds: z.array(z.string().max(120)).max(20).default([]),
});

export async function GET() {
  const access = await requireActor();
  if (!access.actor) return access.response;
  return Response.json({ tasks: await listTasks() });
}

export async function POST(request: Request) {
  const access = await requireActor(["admin", "editor"]);
  if (!access.actor) return access.response;
  const parsed = schema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "Task fields are invalid", details: parsed.error.flatten() }, { status: 400 });
  return Response.json({ task: await createTask(access.actor, parsed.data) }, { status: 201 });
}
