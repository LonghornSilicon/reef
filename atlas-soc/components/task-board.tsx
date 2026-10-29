"use client";

import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import type { AtlasTask, TaskPriority, TaskStatus } from "@/lib/domain";
import { TASK_STATUSES } from "@/lib/domain";
import { Icon } from "@/components/icons";

const statusLabels: Record<TaskStatus, string> = {
  backlog: "Backlog",
  ready: "Ready",
  in_progress: "In progress",
  review: "Review",
  done: "Done",
};

export function TaskBoard({
  initialTasks,
  canEdit,
  notice,
}: {
  initialTasks: AtlasTask[];
  canEdit: boolean;
  /** Explains an empty board when no durable storage is attached. */
  notice?: string;
}) {
  const searchParams = useSearchParams();
  const blockFilter = searchParams.get("block");
  const [tasks, setTasks] = useState(initialTasks);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const [draft, setDraft] = useState({ title: "", description: "", priority: "medium" as TaskPriority, blockIds: blockFilter ?? "" });
  const visible = useMemo(() => blockFilter ? tasks.filter((task) => task.blockIds.includes(blockFilter)) : tasks, [tasks, blockFilter]);

  async function move(task: AtlasTask, status: TaskStatus) {
    if (!canEdit || task.status === status) return;
    const previous = tasks;
    setTasks((current) => current.map((item) => item.id === task.id ? { ...item, status } : item));
    const response = await fetch(`/api/tasks/${task.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }) });
    if (!response.ok) { setTasks(previous); setError("Atlas could not move that task. Try again."); }
  }

  async function create() {
    setError("");
    const response = await fetch("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: draft.title, description: draft.description, priority: draft.priority, status: "backlog", blockIds: draft.blockIds.split(",").map((value) => value.trim()).filter(Boolean) }),
    });
    const data = await response.json();
    if (!response.ok) { setError(data.error ?? "Atlas could not create the task."); return; }
    setTasks((current) => [data.task, ...current]);
    setDraft({ title: "", description: "", priority: "medium", blockIds: blockFilter ?? "" });
    setCreating(false);
  }

  return (
    <div className="task-page">
      <header className="task-heading">
        <div><h1>Architecture work</h1><p>{blockFilter ? `Showing work linked to ${blockFilter}.` : "Move open design questions into owned, visible work without losing their architecture context."}</p></div>
        {canEdit && <button className="primary-button" type="button" onClick={() => setCreating((value) => !value)}><Icon name={creating ? "close" : "plus"} />{creating ? "Close" : "New task"}</button>}
      </header>
      {notice && <p className="storage-notice" role="status">{notice}</p>}
      {error && <div className="page-error" role="alert">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
      {creating && (
        <section className="task-composer">
          <label>Task title<input value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} placeholder="Name the engineering outcome" /></label>
          <label>Description<textarea value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} placeholder="What must be decided or delivered?" /></label>
          <label>Priority<select value={draft.priority} onChange={(event) => setDraft({ ...draft, priority: event.target.value as TaskPriority })}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option><option value="critical">Critical</option></select></label>
          <label>Architecture block IDs<input value={draft.blockIds} onChange={(event) => setDraft({ ...draft, blockIds: event.target.value })} placeholder="dma, coral-core" /></label>
          <button type="button" onClick={() => void create()} disabled={draft.title.trim().length < 3}>Create task</button>
        </section>
      )}
      <div className="task-board">
        {TASK_STATUSES.map((status) => {
          const column = visible.filter((task) => task.status === status);
          return (
            <section className="task-column" key={status}>
              <header><h2>{statusLabels[status]}</h2><span>{column.length}</span></header>
              <div className="task-stack">
                {column.map((task) => (
                  <article className={`task-ticket priority-${task.priority}`} key={task.id}>
                    <div className="task-ticket-meta"><span>{task.priority}</span>{task.dueDate && <time><Icon name="calendar" />{task.dueDate}</time>}</div>
                    <h3>{task.title}</h3><p>{task.description}</p>
                    {task.blockIds.length > 0 && <div className="block-tags">{task.blockIds.map((block) => <a href={`/architecture?selected=${block}`} key={block}>{block}</a>)}</div>}
                    <footer>
                      <span>{task.assigneeEmail ? <><Icon name="person" />{task.assigneeEmail.split("@")[0]}</> : "Unassigned"}</span>
                      {canEdit && <select aria-label={`Move ${task.title}`} value={task.status} onChange={(event) => void move(task, event.target.value as TaskStatus)}>{TASK_STATUSES.map((value) => <option value={value} key={value}>{statusLabels[value]}</option>)}</select>}
                    </footer>
                  </article>
                ))}
                {!column.length && <p className="column-empty">No work at this stage.</p>}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}
