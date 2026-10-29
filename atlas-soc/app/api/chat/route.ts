import { z } from "zod";
import { chatProvider, sse } from "@/lib/ai";
import { requireActor } from "@/lib/authz";
import { aiIntegrationEnabled, publicErrorMessage } from "@/lib/config";
import { appendMessage, createThread, getThread } from "@/lib/repository";
import { takeRateLimit } from "@/lib/rate-limit";
import type { Citation } from "@/lib/domain";

export const runtime = "nodejs";
export const maxDuration = 60;

const requestSchema = z.object({
  question: z.string().trim().min(2).max(4_000),
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
  const limited = takeRateLimit(`chat:${access.actor.email}`, 20, 60_000);
  if (!limited.allowed) {
    return Response.json({ error: "Atlas is receiving too many questions. Try again in a minute." }, { status: 429 });
  }

  const parsed = requestSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "Enter a shorter, valid question." }, { status: 400 });
  const { question, context } = parsed.data;
  const existing = parsed.data.threadId ? await getThread(access.actor, parsed.data.threadId) : null;
  if (parsed.data.threadId && !existing) return Response.json({ error: "Conversation not found" }, { status: 404 });
  const thread = existing ?? (await createThread(access.actor, question.replace(/\s+/g, " "), context));
  const effectiveContext = { ...thread.context, ...context };
  const history = thread.messages.map(({ role, content }) => ({ role, content }));
  await appendMessage(access.actor, thread.id, { role: "user", content: question, citations: [] });

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      controller.enqueue(sse("meta", { threadId: thread.id, expiresAt: thread.expiresAt }));
      let answer = "";
      let citations: Citation[] = [];
      try {
        for await (const event of chatProvider().stream({
          actor: access.actor!,
          question,
          context: effectiveContext,
          history,
        })) {
          if (event.type === "delta") {
            answer += event.text;
            controller.enqueue(sse("delta", { text: event.text }));
          } else if (event.type === "citations") {
            citations = event.citations;
            controller.enqueue(sse("citations", { citations }));
          } else {
            answer = event.text || answer;
            citations = event.citations;
          }
        }
        await appendMessage(access.actor!, thread.id, { role: "assistant", content: answer, citations });
        controller.enqueue(sse("done", { threadId: thread.id }));
      } catch (error) {
        console.error("Atlas chat stream failed", error);
        controller.enqueue(sse("error", { message: publicErrorMessage(error, "Atlas could not complete this answer.") }));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
