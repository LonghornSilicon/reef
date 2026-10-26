import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Explorer } from "@/components/architecture/explorer";
import { lensFromParams } from "@/components/architecture/lens";
import { viewSnapshot } from "@/lib/model/view";

type Params = { selection?: string[] };
type Search = Record<string, string | string[] | undefined>;

function selectedIdFrom(selection: string[] | undefined): string | null {
  if (!selection || selection.length === 0) return null;
  return decodeURIComponent(selection[0]);
}

export async function generateMetadata({ params }: { params: Promise<Params> }): Promise<Metadata> {
  const selectedId = selectedIdFrom((await params).selection);
  if (!selectedId) return { title: "Architecture" };
  const block = viewSnapshot().blocks[selectedId];
  return { title: block ? `${block.name} · Architecture` : "Architecture" };
}

/**
 * The explorer route.
 *
 * An optional catch-all so /architecture and /architecture/<blockId> are the
 * same surface: a deep link selects a block without remounting the canvas or
 * sending the visitor to a different page.
 */
export default async function ArchitecturePage({
  params,
  searchParams,
}: {
  params: Promise<Params>;
  searchParams: Promise<Search>;
}) {
  const snapshot = viewSnapshot();
  const selectedId = selectedIdFrom((await params).selection);
  if (selectedId && !snapshot.blocks[selectedId]) notFound();

  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(await searchParams)) {
    if (typeof value === "string") search.set(key, value);
    else if (Array.isArray(value) && value[0]) search.set(key, value[0]);
  }

  return <Explorer snapshot={snapshot} initialSelectedId={selectedId} initialLens={lensFromParams(search)} />;
}
