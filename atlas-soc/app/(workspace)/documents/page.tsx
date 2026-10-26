import type { Metadata } from "next";
import { currentActor } from "@/lib/authz";
import { hasDatabase } from "@/lib/db";
import { listDocuments } from "@/lib/repository";
import { listWikiPages } from "@/lib/wiki";
import { DocumentLibrary } from "@/components/document-library";

export const metadata: Metadata = { title: "Documents" };

const NO_STORAGE =
  "No database is attached, so managed attachments cannot be kept yet. The repository sources below are " +
  "read straight from Git and are unaffected. Set DATABASE_URL and S3_* to enable uploads.";

export default async function DocumentsPage() {
  const actor = await currentActor();
  const [documents, wiki] = await Promise.all([listDocuments(), listWikiPages()]);
  return (
    <DocumentLibrary
      initialDocuments={documents}
      wiki={wiki.map(({ slug, title, relativePath, summary, relatedBlocks }) => ({ slug, title, relativePath, summary, relatedBlocks }))}
      canEdit={actor?.role === "admin" || actor?.role === "editor"}
      notice={hasDatabase() ? undefined : NO_STORAGE}
    />
  );
}
