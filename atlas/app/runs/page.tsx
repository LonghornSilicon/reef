import type { Metadata } from "next";
import { loadContext } from "@/lib/server/context";
import { ContextState } from "@/components/context-state";
import { RunsView } from "@/components/runs/runs-view";

export const metadata: Metadata = { title: "Runs" };

export default async function RunsPage({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
  const ctx = loadContext((await searchParams).branch);
  if (ctx.kind !== "ok") return <ContextState ctx={ctx} />;
  return <RunsView spec={ctx.spec} />;
}
