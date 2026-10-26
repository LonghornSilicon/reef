import type { Metadata } from "next";
import { loadContext } from "@/lib/server/context";
import { ContextState } from "@/components/context-state";
import { Workspace } from "@/components/workspace/workspace";

export const metadata: Metadata = { title: "Workspace" };

export default async function WorkspacePage({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
  const ctx = loadContext((await searchParams).branch);
  if (ctx.kind !== "ok") return <ContextState ctx={ctx} />;
  return <Workspace spec={ctx.spec} report={ctx.report} />;
}
