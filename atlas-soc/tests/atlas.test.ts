import { describe, expect, it } from "vitest";
import { canAdmin, canEdit } from "@/lib/domain";
import { searchLocalKnowledge } from "@/lib/knowledge";
import { appendMessage, createTask, createThread, deleteThread, getThread, updateTaskStatus } from "@/lib/repository";

const viewer = { email: "viewer-test@atlas.local", name: "Viewer Test", role: "viewer" as const };
const editor = { email: "editor-test@atlas.local", name: "Editor Test", role: "editor" as const };

describe("Atlas authorization", () => {
  it("keeps editing and administration distinct", () => {
    expect(canEdit("viewer")).toBe(false);
    expect(canEdit("editor")).toBe(true);
    expect(canAdmin("editor")).toBe(false);
    expect(canAdmin("admin")).toBe(true);
  });
});

describe("local evidence retrieval", () => {
  it("grounds DMA questions in repository sources", async () => {
    const results = await searchLocalKnowledge("How does the DMA move data to Coral Core memory?", { blockId: "dma" });
    expect(results.length).toBeGreaterThan(0);
    expect(results.some((result) => result.citation.path === "docs/dma.md")).toBe(true);
    expect(results.every((result) => result.citation.quote)).toBe(true);
  });

  it("returns no evidence for unrelated nonsense", async () => {
    expect(await searchLocalKnowledge("xylophonic zeppelins marzipan")).toEqual([]);
  });
});

describe("personal conversation ownership", () => {
  it("does not expose a thread to another user", async () => {
    const thread = await createThread(viewer, "Explain boot", { sourcePath: "docs/boot-flow.md" });
    await appendMessage(viewer, thread.id, { role: "user", content: "Explain boot", citations: [] });
    expect((await getThread(viewer, thread.id))?.messages).toHaveLength(1);
    expect(await getThread(editor, thread.id)).toBeNull();
    expect(await deleteThread(editor, thread.id)).toBe(false);
    expect(await deleteThread(viewer, thread.id)).toBe(true);
  });
});

describe("architecture task workflow", () => {
  it("creates and advances linked work", async () => {
    const task = await createTask(editor, {
      title: "Confirm arbiter fairness",
      description: "Record the selected arbitration behavior.",
      status: "backlog",
      priority: "high",
      blockIds: ["dtcm-arbiter"],
    });
    const updated = await updateTaskStatus(task.id, "review");
    expect(updated?.status).toBe("review");
    expect(updated?.blockIds).toContain("dtcm-arbiter");
  });
});
