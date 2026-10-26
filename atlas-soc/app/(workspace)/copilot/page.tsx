import type { Metadata } from "next";
import { Suspense } from "react";
import Link from "next/link";
import { CopilotWorkspace } from "@/components/copilot-workspace";
import { Icon } from "@/components/icons";
import { aiIntegrationEnabled } from "@/lib/config";

export const metadata: Metadata = { title: "Copilot" };

/**
 * Says the integration is absent rather than offering a prompt box that
 * cannot answer. The architecture and wiki surfaces carry the same evidence
 * this would have cited, so they are the useful next step.
 */
function NoIntegration() {
  return (
    <main className="surface-empty">
      <Icon name="copilot" aria-hidden="true" />
      <h1>No AI integration</h1>
      <p>
        Atlas does not have an AI integration yet, so there is nothing here to ask. No model provider is
        configured and no request leaves this deployment.
      </p>
      <p>
        The evidence a copilot would have quoted is already browsable: every block records its own
        documents and sources, and the wiki carries the written design notes.
      </p>
      <div className="surface-empty-actions">
        <Link className="primary-button" href="/architecture">
          <Icon name="architecture" aria-hidden="true" />
          Browse the architecture
        </Link>
        <Link className="secondary-button" href="/wiki">
          <Icon name="wiki" aria-hidden="true" />
          Read the wiki
        </Link>
      </div>
    </main>
  );
}

export default function CopilotPage() {
  if (!aiIntegrationEnabled()) return <NoIntegration />;
  return (
    <Suspense fallback={<div className="surface-loading">Opening the evidence desk…</div>}>
      <CopilotWorkspace />
    </Suspense>
  );
}
