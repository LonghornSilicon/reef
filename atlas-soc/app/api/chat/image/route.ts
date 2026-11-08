import { z } from "zod";
import { imageProvider } from "@/lib/ai";
import { requireActor } from "@/lib/authz";
import { aiIntegrationEnabled, publicErrorMessage } from "@/lib/config";
import { appendMessage, createThread, getThread } from "@/lib/repository";
import { takeRateLimit } from "@/lib/rate-limit";

export const runtime = "nodejs";
export const maxDuration = 90;

const schema = z.object({
  prompt: z.string().trim().min(3).max(2_000),
  threadId: z.string().uuid().optional(),
  context: z
    .object({
      blockId: z.string().max(120).optional(),
      blockName: z.string().max(180).optional(),
      sourcePath: z.string().max(300).optional(),
      sourceTitle: z.string().max(180).optional(),
    })
    .default({}),
});

export async function POST(request: Request) {
  if (!aiIntegrationEnabled()) {
    return Response.json({ error: "Atlas has no AI integration yet." }, { status: 503 });
  }
  const access = await requireActor();
  if (!access.actor) return access.response;
  const limited = takeRateLimit(`image:${access.actor.email}`, 4, 60 * 60_000);
  if (!limited.allowed) return Response.json({ error: "Image generation limit reached. Try again later." }, { status: 429 });
  const parsed = schema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "Enter a valid visual explanation request." }, { status: 400 });
  const existing = parsed.data.threadId ? await getThread(access.actor, parsed.data.threadId) : null;
  if (parsed.data.threadId && !existing) return Response.json({ error: "Conversation not found" }, { status: 404 });
  const thread = existing ?? (await createThread(access.actor, parsed.data.prompt, parsed.data.context));
  try {
    const image = await imageProvider().generate({ actor: access.actor, prompt: parsed.data.prompt, context: parsed.data.context });
    await appendMessage(access.actor, thread.id, {
      role: "user",
      content: parsed.data.prompt,
      citations: [],
    });
    await appendMessage(access.actor, thread.id, {
      role: "assistant",
      content: `Generated a conceptual visual for: ${parsed.data.prompt}`,
      citations: [],
      imageState: "expired",
    });
    return Response.json({ ...image, threadId: thread.id, ephemeral: true });
  } catch (error) {
    console.error("Atlas image generation failed", error);
    return Response.json({ error: publicErrorMessage(error, "Atlas could not generate this image.") }, { status: 422 });
  }
}
