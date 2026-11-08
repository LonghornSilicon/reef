import type { ViewBlock } from "@/lib/model/view";

/**
 * Where a model evidence reference actually resolves to in the app.
 *
 * Markdown specifications are first-class wiki pages. Diagrams, presentations
 * and PDFs are copied into public/evidence by scripts/sync-evidence.mjs and
 * served as files. Anything else is reported as unresolved rather than
 * rendered as a broken link.
 */
export type ResolvedEvidence = {
  title: string;
  href: string | null;
  path: string;
  kind: "wiki" | "file" | "external" | "unresolved";
};

export function resolveEvidence(document: ViewBlock["documents"][number]): ResolvedEvidence {
  if (document.url) {
    return { title: document.title, href: document.url, path: document.url, kind: "external" };
  }
  const path = document.path;
  if (!path) {
    return { title: document.title, href: null, path: "no path recorded", kind: "unresolved" };
  }
  const wiki = path.match(/^docs\/(.+)\.md$/);
  if (wiki) {
    return {
      title: document.title,
      href: `/wiki/${wiki[1]}${document.anchor ? `#${document.anchor}` : ""}`,
      path,
      kind: "wiki",
    };
  }
  return { title: document.title, href: `/evidence/${encodeURIComponent(path)}`, path, kind: "file" };
}
