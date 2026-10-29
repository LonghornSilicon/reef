import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Icon } from "@/components/icons";
import { aiIntegrationEnabled } from "@/lib/features";
import { getWikiPage, listWikiPages } from "@/lib/wiki";

export async function generateStaticParams() {
  return (await listWikiPages()).map((page) => ({ slug: page.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const page = await getWikiPage((await params).slug);
  return { title: page?.title ?? "Wiki" };
}

export default async function WikiArticlePage({ params }: { params: Promise<{ slug: string }> }) {
  const page = await getWikiPage((await params).slug);
  if (!page) notFound();
  const askParams = new URLSearchParams({ sourcePath: page.relativePath, sourceTitle: page.title });
  return (
    <div className="reading-workbench">
      <article className="wiki-article">
        <nav className="article-breadcrumbs" aria-label="Breadcrumb"><Link href="/wiki">Wiki</Link><span>/</span><span>{page.title}</span></nav>
        <header>
          <h1>{page.title}</h1>
          <p>{page.summary}</p>
        </header>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{page.content.replace(/^#\s+.+$/m, "")}</ReactMarkdown>
      </article>
      <aside className="reading-rail">
        <div className="source-stamp"><Icon name="source" /><span><strong>Canonical source</strong><code>{page.relativePath}</code></span></div>
        {aiIntegrationEnabled() && (
          <Link className="primary-field-action" href={`/copilot?${askParams}`}>
            <Icon name="copilot" />
            <span><strong>Ask about this page</strong><small>Answers stay grounded here</small></span>
            <Icon name="arrow" />
          </Link>
        )}
        <section>
          <h2>Related architecture</h2>
          {page.relatedBlocks.length ? page.relatedBlocks.map((block) => (
            <Link href={`/architecture?selected=${block}`} key={block}><span>{block}</span><Icon name="arrow" /></Link>
          )) : <p>No block links recorded yet.</p>}
        </section>
        <p className="field-note">Change the Markdown and architecture model in the same pull request when a decision affects both.</p>
      </aside>
    </div>
  );
}
