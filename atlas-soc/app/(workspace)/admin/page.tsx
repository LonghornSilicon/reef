import type { Metadata } from "next";
import { currentActor } from "@/lib/authz";
import { latestKnowledgeRevision } from "@/lib/repository";
import { AdminConsole } from "@/components/admin-console";

export const metadata: Metadata = { title: "Administration" };

export default async function AdminPage() {
  const actor = await currentActor();
  return <AdminConsole actorRole={actor?.role ?? "viewer"} initialRevision={await latestKnowledgeRevision()} />;
}
