export const WORKSPACE_SLUG = "coral";

/**
 * Identity of an unauthenticated visitor under public read access.
 *
 * A reserved .invalid address (RFC 2606) so it can never collide with a real
 * account, and so anything keyed by owner email is obviously not a person.
 */
export const PUBLIC_VISITOR_EMAIL = "public@atlas.invalid";

export type Role = "admin" | "editor" | "viewer";

export type Actor = {
  email: string;
  name: string;
  image?: string | null;
  role: Role;
};

export type AtlasContext = {
  blockId?: string;
  blockName?: string;
  sourcePath?: string;
  sourceTitle?: string;
};

export type Citation = {
  id: string;
  title: string;
  path: string;
  page?: number;
  quote?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  imageState?: "expired" | "blocked" | "failed";
  createdAt: string;
};

export type ChatThread = {
  id: string;
  title: string;
  context: AtlasContext;
  messages: ChatMessage[];
  createdAt: string;
  updatedAt: string;
  expiresAt: string;
};

export type ThreadSummary = Omit<ChatThread, "messages"> & {
  messageCount: number;
};

export const TASK_STATUSES = ["backlog", "ready", "in_progress", "review", "done"] as const;
export type TaskStatus = (typeof TASK_STATUSES)[number];
export type TaskPriority = "low" | "medium" | "high" | "critical";

export type AtlasTask = {
  id: string;
  title: string;
  description: string;
  status: TaskStatus;
  priority: TaskPriority;
  assigneeEmail?: string;
  dueDate?: string;
  blockIds: string[];
  documentIds: string[];
  externalIssueUrl?: string;
  createdAt: string;
  updatedAt: string;
};

export type DocumentAsset = {
  id: string;
  title: string;
  filename: string;
  mimeType: string;
  byteSize: number;
  storageKey: string;
  revision: number;
  ownerEmail?: string;
  blockIds: string[];
  taskIds: string[];
  createdAt: string;
};

export type KnowledgeRevision = {
  id: string;
  gitRevision: string;
  vectorStoreId?: string;
  status: "running" | "ready" | "failed";
  sourceCount: number;
  errorMessage?: string;
  createdAt: string;
  completedAt?: string;
};

export type KnowledgeSourceRecord = {
  relativePath: string;
  title: string;
  providerFileId?: string;
  contentSha256: string;
  metadata: Record<string, string | number | string[]>;
};

export function canEdit(role: Role) {
  return role === "admin" || role === "editor";
}

export function canAdmin(role: Role) {
  return role === "admin";
}
