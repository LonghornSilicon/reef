import type { Metadata } from "next";
import { loadContext, repoUrl } from "@/lib/server/context";
import { ContextState } from "@/components/context-state";
import { CanvasView } from "@/components/canvas/canvas-view";

export const metadata: Metadata = { title: "Canvas" };

export default async function CanvasPage({ searchParams }: { searchParams: Promise<Record<string, string | undefined>> }) {
  const sp = await searchParams;
  const ctx = loadContext(sp.branch);
  if (ctx.kind !== "ok") return <ContextState ctx={ctx} />;
  return (
    <CanvasView
      spec={ctx.spec}
      report={ctx.report}
      rtl={ctx.rtl}
      note={ctx.branch.sample ? ctx.branch.note : null}
      specBase={repoUrl(ctx.branch, "")}
      rtlBase={null}
    />
  );
}
