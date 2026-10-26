import type { Metadata } from "next";
import { loadContext } from "@/lib/server/context";
import { ContextState } from "@/components/context-state";
import { ProcessView } from "@/components/process/process-view";

export const metadata: Metadata = { title: "Process" };

export default async function ProcessPage({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
  const ctx = loadContext((await searchParams).branch);
  if (ctx.kind !== "ok") return <ContextState ctx={ctx} />;
  return <ProcessView spec={ctx.spec} />;
}
