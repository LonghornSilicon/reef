import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

export type WikiPage = {
  slug: string;
  title: string;
  summary: string;
  content: string;
  relativePath: string;
  updatedLabel: string;
  relatedBlocks: string[];
};

const blockMap: Record<string, string[]> = {
  "architecture-overview": ["coral-system", "coral-soc", "coral-core"],
  "boot-flow": ["coral-soc", "flash-controller", "host-link", "itcm"],
  "coral-core": ["coral-core", "compute-cluster", "dtcm", "itcm"],
  dma: ["dma", "dma-control-fsm", "dma-address-gen"],
  "editing-guide": ["coral-system"],
};

function parsePage(slug: string, content: string): WikiPage {
  const title = content.match(/^#\s+(.+)$/m)?.[1]?.trim() ?? slug.replaceAll("-", " ");
  const body = content.replace(/^#\s+.+$/m, "").trim();
  const summary = body
    .split(/\n\s*\n/)[0]
    .replace(/[*_`#>]/g, "")
    .replace(/\s+/g, " ")
    .slice(0, 220);
  return {
    slug,
    title,
    summary,
    content,
    relativePath: `docs/${slug}.md`,
    updatedLabel: "Git-backed source",
    relatedBlocks: blockMap[slug] ?? [],
  };
}

export async function listWikiPages() {
  const names = (await readdir(path.join(process.cwd(), "docs"))).filter((name) => name.endsWith(".md"));
  return Promise.all(
    names.map(async (name) => {
      const slug = name.slice(0, -3);
      const content = await readFile(path.join(process.cwd(), "docs", name), "utf8");
      return parsePage(slug, content);
    }),
  );
}

export async function getWikiPage(slug: string) {
  if (!/^[a-z0-9-]+$/.test(slug)) return null;
  try {
    const content = await readFile(path.join(process.cwd(), "docs", `${slug}.md`), "utf8");
    return parsePage(slug, content);
  } catch {
    return null;
  }
}
