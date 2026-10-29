import { readFile } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import {
  beginKnowledgeRevision,
  completeKnowledgeRevision,
  failKnowledgeRevision,
} from "@/lib/repository";
import { currentGitRevision, loadLocalKnowledge } from "@/lib/knowledge";
import { aiIntegrationEnabled } from "@/lib/config";
import type { KnowledgeSourceRecord } from "@/lib/domain";

async function openAI(pathname: string, init: RequestInit) {
  if (!aiIntegrationEnabled()) throw new Error("Atlas has no AI integration, so no request was sent.");
  const response = await fetch(`https://api.openai.com/v1${pathname}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init.headers,
    },
  });
  if (!response.ok) throw new Error(`OpenAI ingestion failed (${response.status}): ${(await response.text()).slice(0, 400)}`);
  return response;
}

async function uploadFile(relativePath: string, content?: Uint8Array) {
  const bytes = content ?? (await readFile(path.join(/* turbopackIgnore: true */ process.cwd(), relativePath)));
  const form = new FormData();
  form.set("purpose", "assistants");
  const fileBytes = Uint8Array.from(bytes).buffer;
  form.set("file", new File([fileBytes], relativePath.replaceAll("/", "__"), { type: relativePath.endsWith(".pdf") ? "application/pdf" : "text/plain" }));
  const response = await openAI("/files", { method: "POST", body: form });
  return (await response.json()) as { id: string; filename: string };
}

async function waitForBatch(vectorStoreId: string, batchId: string) {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const response = await openAI(`/vector_stores/${vectorStoreId}/file_batches/${batchId}`, { method: "GET" });
    const payload = (await response.json()) as { status: string; file_counts?: { failed?: number } };
    if (payload.status === "completed") return;
    if (payload.status === "failed" || payload.status === "cancelled" || payload.file_counts?.failed) {
      throw new Error("One or more Atlas sources failed vector-store processing");
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  throw new Error("Vector-store processing did not complete within 60 seconds");
}

export async function ingestKnowledge() {
  if (!aiIntegrationEnabled()) throw new Error("Atlas has no AI integration, so there is nothing to ingest into.");
  if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is required for ingestion");
  const gitRevision = await currentGitRevision();
  const revision = await beginKnowledgeRevision(gitRevision);
  try {
    const localSources = await loadLocalKnowledge();
    const evidenceFiles = ["reef_spec.pdf", "reef_spec2.pdf", "SoC presentation.pdf"];
    const vectorResponse = await openAI("/vector_stores", {
      method: "POST",
      body: JSON.stringify({ name: `Atlas Coral ${gitRevision.slice(0, 12)}` }),
    });
    const vectorStore = (await vectorResponse.json()) as { id: string };
    const records: KnowledgeSourceRecord[] = [];
    const vectorFiles: Array<{
      file_id: string;
      attributes: Record<string, string | number | boolean>;
    }> = [];

    for (const source of localSources) {
      const uploaded = await uploadFile(source.relativePath, new TextEncoder().encode(source.content));
      vectorFiles.push({
        file_id: uploaded.id,
        attributes: {
          relative_path: source.relativePath,
          category: typeof source.metadata.category === "string" ? source.metadata.category : "repository",
        },
      });
      records.push({
        relativePath: source.relativePath,
        title: source.title,
        providerFileId: uploaded.id,
        contentSha256: source.contentSha256,
        metadata: source.metadata,
      });
    }
    for (const relativePath of evidenceFiles) {
      const bytes = await readFile(path.join(/* turbopackIgnore: true */ process.cwd(), relativePath));
      const uploaded = await uploadFile(relativePath, bytes);
      vectorFiles.push({
        file_id: uploaded.id,
        attributes: { relative_path: relativePath, category: "evidence" },
      });
      records.push({
        relativePath,
        title: relativePath.replace(/\.[^.]+$/, ""),
        providerFileId: uploaded.id,
        contentSha256: createHash("sha256").update(bytes).digest("hex"),
        metadata: { category: "evidence", relative_path: relativePath },
      });
    }

    const batchResponse = await openAI(`/vector_stores/${vectorStore.id}/file_batches`, {
      method: "POST",
      body: JSON.stringify({ files: vectorFiles }),
    });
    const batch = (await batchResponse.json()) as { id: string };
    await waitForBatch(vectorStore.id, batch.id);
    await completeKnowledgeRevision(revision.id, vectorStore.id, records);
    return { revisionId: revision.id, vectorStoreId: vectorStore.id, sourceCount: records.length, gitRevision };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown ingestion failure";
    await failKnowledgeRevision(revision.id, message);
    throw error;
  }
}
