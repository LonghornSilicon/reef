import type { Metadata } from "next";
import { loadContext, repoUrl } from "@/lib/server/context";
import { ContextState } from "@/components/context-state";
import { DocGraph } from "@/components/spec/doc-graph";

export const metadata: Metadata = { title: "Spec" };

export default async function SpecPage({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
  const ctx = loadContext((await searchParams).branch);
  if (ctx.kind !== "ok") return <ContextState ctx={ctx} />;
  return <DocGraph spec={ctx.spec} specBase={repoUrl(ctx.branch, "")} />;
}
