import type { Metadata } from "next";
import Link from "next/link";
import { Icon } from "@/components/icons";
import { aiIntegrationEnabled } from "@/lib/features";
import { listWikiPages } from "@/lib/wiki";

export const metadata: Metadata = { title: "Wiki" };

export default async function WikiIndexPage() {
  const pages = await listWikiPages();
  return (
    <div className="library-page">
      <header className="library-heading">
        <div>
          <h1>The design record</h1>
          <p>Read the editable specification beside the architecture it describes. Every explanation can be traced back to a repository source.</p>
        </div>
        {aiIntegrationEnabled() && (
          <Link className="primary-button" href="/copilot"><Icon name="copilot" />Ask across the wiki</Link>
        )}
      </header>
      <div className="source-register" role="list">
        {pages.map((page) => (
          <Link href={`/wiki/${page.slug}`} className="source-row" key={page.slug} role="listitem">
            <span className="source-index">{String(page.relatedBlocks.length).padStart(2, "0")}</span>
            <div>
              <h2>{page.title}</h2>
              <p>{page.summary}</p>
            </div>
            <div className="source-meta">
              <span>{page.updatedLabel}</span>
              <code>{page.relativePath}</code>
            </div>
            <Icon name="arrow" />
          </Link>
        ))}
      </div>
    </div>
  );
}
