import { randomUUID } from "node:crypto";
import type {
  Actor,
  AtlasContext,
  AtlasTask,
  ChatMessage,
  ChatThread,
  Citation,
  DocumentAsset,
  KnowledgeRevision,
  KnowledgeSourceRecord,
  TaskPriority,
  TaskStatus,
  ThreadSummary,
} from "@/lib/domain";
import { WORKSPACE_SLUG } from "@/lib/domain";
import { hasDatabase, query, transaction } from "@/lib/db";

type MemoryStore = {
  threads: Map<string, ChatThread & { ownerEmail: string }>;
  tasks: AtlasTask[];
  documents: DocumentAsset[];
  revisions: KnowledgeRevision[];
  sources: Map<string, KnowledgeSourceRecord[]>;
};

declare global {
  var __atlasMemory: MemoryStore | undefined;
}

const now = () => new Date().toISOString();
const daysFromNow = (days: number) => new Date(Date.now() + days * 86_400_000).toISOString();

function memory(): MemoryStore {
  if (!globalThis.__atlasMemory) {
    const createdAt = now();
    globalThis.__atlasMemory = {
      threads: new Map(),
      documents: [],
      revisions: [],
      sources: new Map(),
      tasks: [
        {
          id: "task-dma-timeout",
          title: "Define DMA timeout and recovery behavior",
          description: "Turn the open DMA recovery question into an owned architecture decision.",
          status: "ready",
          priority: "high",
          blockIds: ["dma", "dma-control-fsm"],
          documentIds: [],
          createdAt,
          updatedAt: createdAt,
        },
        {
          id: "task-rtl-anchors",
          title: "Attach immutable RTL anchors",
          description: "Record repository, revision, path, and declaration for implemented blocks.",
          status: "backlog",
          priority: "medium",
          blockIds: ["coral-core"],
          documentIds: [],
          createdAt,
          updatedAt: createdAt,
        },
        {
          id: "task-boot-fallback",
          title: "Specify invalid-image boot fallback",
          description: "Resolve fallback between host-assisted and flash boot paths.",
          status: "in_progress",
          priority: "critical",
          assigneeEmail: "architect@atlas.local",
          blockIds: ["coral-soc", "flash-controller", "host-link"],
          documentIds: [],
          createdAt,
          updatedAt: createdAt,
        },
      ],
    };
  }
  return globalThis.__atlasMemory;
}

async function ensureActor(actor: Actor) {
  await query(
    `INSERT INTO users (email, name, avatar_url)
     VALUES ($1, $2, $3)
     ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name, avatar_url = EXCLUDED.avatar_url`,
    [actor.email, actor.name, actor.image ?? null],
  );
}

function mapThreadRow(row: Record<string, unknown>): ThreadSummary {
  return {
    id: String(row.id),
    title: String(row.title),
    context: (row.context ?? {}) as AtlasContext,
    messageCount: Number(row.message_count ?? 0),
    createdAt: new Date(String(row.created_at)).toISOString(),
    updatedAt: new Date(String(row.updated_at)).toISOString(),
    expiresAt: new Date(String(row.expires_at)).toISOString(),
  };
}

export async function listThreads(actor: Actor): Promise<ThreadSummary[]> {
  if (!hasDatabase()) {
    return [...memory().threads.values()]
      .filter((thread) => thread.ownerEmail === actor.email && thread.expiresAt > now())
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
      .map((thread) => ({
        id: thread.id,
        title: thread.title,
        context: thread.context,
        createdAt: thread.createdAt,
        updatedAt: thread.updatedAt,
        expiresAt: thread.expiresAt,
        messageCount: thread.messages.length,
      }));
  }
  await ensureActor(actor);
  const result = await query(
    `SELECT t.*, count(m.id)::int AS message_count
     FROM chat_threads t
     JOIN users u ON u.id = t.user_id
     LEFT JOIN chat_messages m ON m.thread_id = t.id
     WHERE u.email = $1 AND t.expires_at > now()
     GROUP BY t.id ORDER BY t.updated_at DESC`,
    [actor.email],
  );
  return result.rows.map(mapThreadRow);
}

export async function getThread(actor: Actor, threadId: string): Promise<ChatThread | null> {
  if (!hasDatabase()) {
    const thread = memory().threads.get(threadId);
    if (!thread || thread.ownerEmail !== actor.email || thread.expiresAt <= now()) return null;
    return structuredClone({
      id: thread.id,
      title: thread.title,
      context: thread.context,
      messages: thread.messages,
      createdAt: thread.createdAt,
      updatedAt: thread.updatedAt,
      expiresAt: thread.expiresAt,
    });
  }
  const threadResult = await query(
    `SELECT t.* FROM chat_threads t JOIN users u ON u.id = t.user_id
     WHERE t.id = $1 AND u.email = $2 AND t.expires_at > now()`,
    [threadId, actor.email],
  );
  if (!threadResult.rowCount) return null;
  const row = threadResult.rows[0] as Record<string, unknown>;
  const messagesResult = await query(
    `SELECT id, role, content, citations, image_state, created_at
     FROM chat_messages WHERE thread_id = $1 ORDER BY created_at`,
    [threadId],
  );
  return {
    ...mapThreadRow({ ...row, message_count: messagesResult.rowCount }),
    messages: messagesResult.rows.map((message) => ({
      id: String(message.id),
      role: message.role as ChatMessage["role"],
      content: String(message.content),
      citations: (message.citations ?? []) as Citation[],
      imageState: message.image_state as ChatMessage["imageState"],
      createdAt: new Date(String(message.created_at)).toISOString(),
    })),
  };
}

export async function createThread(actor: Actor, title: string, context: AtlasContext): Promise<ChatThread> {
  const thread: ChatThread = {
    id: randomUUID(),
    title: title.slice(0, 80),
    context,
    messages: [],
    createdAt: now(),
    updatedAt: now(),
    expiresAt: daysFromNow(30),
  };
  if (!hasDatabase()) {
    memory().threads.set(thread.id, { ...thread, ownerEmail: actor.email });
    return thread;
  }
  await ensureActor(actor);
  const workspace = await query<{ id: string }>("SELECT id FROM workspaces WHERE slug = $1", [WORKSPACE_SLUG]);
  const user = await query<{ id: string }>("SELECT id FROM users WHERE email = $1", [actor.email]);
  await query(
    `INSERT INTO chat_threads (id, workspace_id, user_id, title, context, created_at, updated_at, expires_at)
     VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)`,
    [thread.id, workspace.rows[0].id, user.rows[0].id, thread.title, JSON.stringify(context), thread.createdAt, thread.updatedAt, thread.expiresAt],
  );
  return thread;
}

export async function appendMessage(
  actor: Actor,
  threadId: string,
  message: Omit<ChatMessage, "id" | "createdAt">,
): Promise<ChatMessage | null> {
  const saved: ChatMessage = { ...message, id: randomUUID(), createdAt: now() };
  if (!hasDatabase()) {
    const thread = memory().threads.get(threadId);
    if (!thread || thread.ownerEmail !== actor.email) return null;
    thread.messages.push(saved);
    thread.updatedAt = saved.createdAt;
    thread.expiresAt = daysFromNow(30);
    return saved;
  }
  const owned = await query(
    `SELECT t.id FROM chat_threads t JOIN users u ON u.id = t.user_id WHERE t.id = $1 AND u.email = $2`,
    [threadId, actor.email],
  );
  if (!owned.rowCount) return null;
  await transaction(async (client) => {
    await client.query(
      `INSERT INTO chat_messages (id, thread_id, role, content, citations, image_state, created_at)
       VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)`,
      [saved.id, threadId, saved.role, saved.content, JSON.stringify(saved.citations), saved.imageState ?? null, saved.createdAt],
    );
    await client.query(
      `UPDATE chat_threads SET updated_at = $2, expires_at = $2::timestamptz + interval '30 days' WHERE id = $1`,
      [threadId, saved.createdAt],
    );
  });
  return saved;
}

export async function deleteThread(actor: Actor, threadId: string) {
  if (!hasDatabase()) {
    const thread = memory().threads.get(threadId);
    if (!thread || thread.ownerEmail !== actor.email) return false;
    return memory().threads.delete(threadId);
  }
  const result = await query(
    `DELETE FROM chat_threads t USING users u WHERE t.id = $1 AND t.user_id = u.id AND u.email = $2`,
    [threadId, actor.email],
  );
  return Boolean(result.rowCount);
}

export async function deleteExpiredThreads() {
  if (!hasDatabase()) {
    let count = 0;
    for (const [id, thread] of memory().threads) {
      if (thread.expiresAt <= now()) {
        memory().threads.delete(id);
        count += 1;
      }
    }
    return count;
  }
  const result = await query("DELETE FROM chat_threads WHERE expires_at <= now()");
  return result.rowCount ?? 0;
}

function mapTask(row: Record<string, unknown>): AtlasTask {
  return {
    id: String(row.id),
    title: String(row.title),
    description: String(row.description ?? ""),
    status: row.status as TaskStatus,
    priority: row.priority as TaskPriority,
    assigneeEmail: row.assignee_email ? String(row.assignee_email) : undefined,
    dueDate: row.due_date ? String(row.due_date) : undefined,
    blockIds: (row.block_ids ?? []) as string[],
    documentIds: (row.document_ids ?? []) as string[],
    externalIssueUrl: row.external_issue_url ? String(row.external_issue_url) : undefined,
    createdAt: new Date(String(row.created_at)).toISOString(),
    updatedAt: new Date(String(row.updated_at)).toISOString(),
  };
}

export async function listTasks(): Promise<AtlasTask[]> {
  if (!hasDatabase()) return structuredClone(memory().tasks);
  const result = await query(
    `SELECT t.* FROM tasks t JOIN workspaces w ON w.id = t.workspace_id
     WHERE w.slug = $1 ORDER BY t.updated_at DESC`,
    [WORKSPACE_SLUG],
  );
  return result.rows.map(mapTask);
}

export async function createTask(
  actor: Actor,
  input: Pick<AtlasTask, "title" | "description" | "status" | "priority" | "blockIds">,
): Promise<AtlasTask> {
  const task: AtlasTask = {
    id: randomUUID(),
    ...input,
    documentIds: [],
    createdAt: now(),
    updatedAt: now(),
  };
  if (!hasDatabase()) {
    memory().tasks.unshift(task);
    return task;
  }
  await ensureActor(actor);
  const result = await query(
    `INSERT INTO tasks (id, workspace_id, title, description, status, priority, block_ids, created_by)
     SELECT $1, w.id, $2, $3, $4, $5, $6, u.id FROM workspaces w, users u
     WHERE w.slug = $7 AND u.email = $8 RETURNING *`,
    [task.id, task.title, task.description, task.status, task.priority, task.blockIds, WORKSPACE_SLUG, actor.email],
  );
  return mapTask(result.rows[0]);
}

export async function updateTaskStatus(taskId: string, status: TaskStatus): Promise<AtlasTask | null> {
  if (!hasDatabase()) {
    const task = memory().tasks.find((item) => item.id === taskId);
    if (!task) return null;
    task.status = status;
    task.updatedAt = now();
    return structuredClone(task);
  }
  const result = await query("UPDATE tasks SET status = $2, updated_at = now() WHERE id = $1 RETURNING *", [taskId, status]);
  return result.rowCount ? mapTask(result.rows[0]) : null;
}

function mapDocument(row: Record<string, unknown>): DocumentAsset {
  return {
    id: String(row.id),
    title: String(row.title),
    filename: String(row.filename),
    mimeType: String(row.mime_type),
    byteSize: Number(row.byte_size),
    storageKey: String(row.storage_key),
    revision: Number(row.revision),
    ownerEmail: row.owner_email ? String(row.owner_email) : undefined,
    blockIds: (row.block_ids ?? []) as string[],
    taskIds: (row.task_ids ?? []) as string[],
    createdAt: new Date(String(row.created_at)).toISOString(),
  };
}

export async function listDocuments(): Promise<DocumentAsset[]> {
  if (!hasDatabase()) return structuredClone(memory().documents);
  const result = await query(
    `SELECT d.*, u.email AS owner_email FROM document_assets d
     JOIN workspaces w ON w.id = d.workspace_id LEFT JOIN users u ON u.id = d.owner_id
     WHERE w.slug = $1 ORDER BY d.created_at DESC`,
    [WORKSPACE_SLUG],
  );
  return result.rows.map(mapDocument);
}

export async function createDocument(
  actor: Actor,
  input: Omit<DocumentAsset, "id" | "revision" | "ownerEmail" | "createdAt" | "taskIds">,
): Promise<DocumentAsset> {
  const document: DocumentAsset = {
    ...input,
    id: randomUUID(),
    revision: 1,
    ownerEmail: actor.email,
    taskIds: [],
    createdAt: now(),
  };
  if (!hasDatabase()) {
    memory().documents.unshift(document);
    return document;
  }
  await ensureActor(actor);
  const result = await query(
    `INSERT INTO document_assets
       (id, workspace_id, title, filename, mime_type, byte_size, storage_key, owner_id, block_ids)
     SELECT $1, w.id, $2, $3, $4, $5, $6, u.id, $7 FROM workspaces w, users u
     WHERE w.slug = $8 AND u.email = $9 RETURNING *`,
    [document.id, document.title, document.filename, document.mimeType, document.byteSize, document.storageKey, document.blockIds, WORKSPACE_SLUG, actor.email],
  );
  return { ...mapDocument(result.rows[0]), ownerEmail: actor.email };
}

export async function beginKnowledgeRevision(gitRevision: string): Promise<KnowledgeRevision> {
  const revision: KnowledgeRevision = {
    id: randomUUID(),
    gitRevision,
    status: "running",
    sourceCount: 0,
    createdAt: now(),
  };
  if (!hasDatabase()) {
    memory().revisions.unshift(revision);
    return revision;
  }
  const result = await query(
    `INSERT INTO knowledge_revisions (id, workspace_id, git_revision, status)
     SELECT $1, id, $2, 'running' FROM workspaces WHERE slug = $3 RETURNING *`,
    [revision.id, gitRevision, WORKSPACE_SLUG],
  );
  return mapRevision(result.rows[0]);
}

function mapRevision(row: Record<string, unknown>): KnowledgeRevision {
  return {
    id: String(row.id),
    gitRevision: String(row.git_revision),
    vectorStoreId: row.vector_store_id ? String(row.vector_store_id) : undefined,
    status: row.status as KnowledgeRevision["status"],
    sourceCount: Number(row.source_count),
    errorMessage: row.error_message ? String(row.error_message) : undefined,
    createdAt: new Date(String(row.created_at)).toISOString(),
    completedAt: row.completed_at ? new Date(String(row.completed_at)).toISOString() : undefined,
  };
}

export async function completeKnowledgeRevision(
  revisionId: string,
  vectorStoreId: string,
  sources: KnowledgeSourceRecord[],
): Promise<void> {
  if (!hasDatabase()) {
    const revision = memory().revisions.find((item) => item.id === revisionId);
    if (revision) Object.assign(revision, { status: "ready", vectorStoreId, sourceCount: sources.length, completedAt: now() });
    memory().sources.set(revisionId, sources);
    return;
  }
  await transaction(async (client) => {
    for (const source of sources) {
      await client.query(
        `INSERT INTO knowledge_sources
          (revision_id, relative_path, title, provider_file_id, content_sha256, metadata)
         VALUES ($1, $2, $3, $4, $5, $6::jsonb)`,
        [revisionId, source.relativePath, source.title, source.providerFileId ?? null, source.contentSha256, JSON.stringify(source.metadata)],
      );
    }
    await client.query(
      `UPDATE knowledge_revisions SET status = 'ready', vector_store_id = $2,
       source_count = $3, completed_at = now() WHERE id = $1`,
      [revisionId, vectorStoreId, sources.length],
    );
  });
}

export async function failKnowledgeRevision(revisionId: string, message: string) {
  if (!hasDatabase()) {
    const revision = memory().revisions.find((item) => item.id === revisionId);
    if (revision) Object.assign(revision, { status: "failed", errorMessage: message.slice(0, 500), completedAt: now() });
    return;
  }
  await query(
    `UPDATE knowledge_revisions SET status = 'failed', error_message = $2, completed_at = now() WHERE id = $1`,
    [revisionId, message.slice(0, 500)],
  );
}

export async function latestKnowledgeRevision(): Promise<KnowledgeRevision | null> {
  if (!hasDatabase()) return memory().revisions[0] ?? null;
  const result = await query("SELECT * FROM knowledge_revisions ORDER BY created_at DESC LIMIT 1");
  return result.rowCount ? mapRevision(result.rows[0]) : null;
}

export async function findKnowledgeSourceByProviderId(providerFileId: string): Promise<KnowledgeSourceRecord | null> {
  if (!hasDatabase()) {
    for (const sources of memory().sources.values()) {
      const source = sources.find((item) => item.providerFileId === providerFileId);
      if (source) return source;
    }
    return null;
  }
  const result = await query(
    `SELECT relative_path, title, provider_file_id, content_sha256, metadata
     FROM knowledge_sources WHERE provider_file_id = $1 ORDER BY id DESC LIMIT 1`,
    [providerFileId],
  );
  if (!result.rowCount) return null;
  const row = result.rows[0];
  return {
    relativePath: String(row.relative_path),
    title: String(row.title),
    providerFileId: String(row.provider_file_id),
    contentSha256: String(row.content_sha256),
    metadata: (row.metadata ?? {}) as KnowledgeSourceRecord["metadata"],
  };
}
