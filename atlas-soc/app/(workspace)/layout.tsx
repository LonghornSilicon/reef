import { redirect } from "next/navigation";
import { AppShell } from "@/components/app-shell";
import { currentActor, isPublicVisitor } from "@/lib/authz";
import { signInAvailable } from "@/lib/config";

/**
 * Every page under this layout depends on who is asking, so none of them may
 * be prerendered.
 *
 * Next decides a route is dynamic when it observes cookies() or headers().
 * Under public read access there is no session to read, so currentActor()
 * consults only process.env — nothing Next recognises as request-dependent.
 * It therefore prerendered these pages and baked in whatever the build-time
 * environment decided, including this layout's redirect to /signin, which no
 * runtime variable could later undo.
 */
export const dynamic = "force-dynamic";

export default async function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  const actor = await currentActor();
  if (!actor) redirect("/signin");
  return (
    <AppShell actor={actor} anonymous={isPublicVisitor(actor)} canSignIn={signInAvailable()}>
      {children}
    </AppShell>
  );
}
