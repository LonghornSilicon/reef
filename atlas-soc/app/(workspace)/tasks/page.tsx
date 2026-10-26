import type { Metadata } from "next";
import { currentActor } from "@/lib/authz";
import { hasDatabase } from "@/lib/db";
import { listTasks } from "@/lib/repository";
import { TaskBoard } from "@/components/task-board";

export const metadata: Metadata = { title: "Tasks" };

const NO_STORAGE =
  "No database is attached, so this board cannot hold work yet. Set DATABASE_URL to make tasks persist.";

export default async function TasksPage() {
  const actor = await currentActor();
  return (
    <TaskBoard
      initialTasks={await listTasks()}
      canEdit={actor?.role === "admin" || actor?.role === "editor"}
      notice={hasDatabase() ? undefined : NO_STORAGE}
    />
  );
}
