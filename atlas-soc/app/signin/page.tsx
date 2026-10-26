import Link from "next/link";
import Image from "next/image";
import { redirect } from "next/navigation";
import { auth, authProviderIds, signIn } from "@/auth";
import { configurationErrors, demoAuthEnabled, publicReadAccessEnabled, signInAvailable } from "@/lib/config";

// Reads the session and the live configuration; see the note in
// app/(workspace)/layout.tsx for why that is not enough to avoid prerendering.
export const dynamic = "force-dynamic";

export default async function SignInPage() {
  // With public read access and nothing to sign in with, this page has
  // nothing to offer; send the visitor to the thing they came for.
  if (!signInAvailable() && publicReadAccessEnabled()) redirect("/architecture");
  const session = signInAvailable() ? await auth() : null;
  if (session) redirect("/architecture");
  const demoAvailable = demoAuthEnabled();
  // Reasons stay in the server log; the page only admits that sign-in is
  // unavailable, so a misconfigured deployment is diagnosable without
  // publishing which variable is missing.
  const misconfigured = configurationErrors().length > 0;
  return (
    <main className="signin-page">
      <section className="signin-story">
        <div className="signin-brand"><Image src="/logo.png" width={48} height={48} alt="" /><span>Atlas</span></div>
        <h1>Find the design decision behind the diagram.</h1>
        <p>Architecture, evidence, explanations, project work, and documents stay attached to the same stable engineering concepts.</p>
        <div className="signin-route" aria-hidden="true">
          <span>Question</span><i /><span>Evidence</span><i /><span>Decision</span><i /><span>Work</span>
        </div>
      </section>
      <section className="signin-panel">
        <div>
          <h2>Enter Coral SoC</h2>
          <p>Use your organization identity. Atlas does not manage passwords.</p>
          {misconfigured && (
            <p className="inline-error">
              Atlas is not finished being set up, so sign-in is unavailable. An administrator can find the
              specific cause in the server log.
            </p>
          )}
          {!misconfigured && authProviderIds.includes("google") && (
            <form action={async () => { "use server"; await signIn("google", { redirectTo: "/architecture" }); }}>
              <button className="signin-action" type="submit">Continue with Google</button>
            </form>
          )}
          {!misconfigured && authProviderIds.includes("microsoft-entra-id") && (
            <form action={async () => { "use server"; await signIn("microsoft-entra-id", { redirectTo: "/architecture" }); }}>
              <button className="signin-action" type="submit">Continue with Microsoft</button>
            </form>
          )}
          {!misconfigured && !authProviderIds.length && !demoAvailable && <p className="inline-error">No identity provider is configured. Ask an Atlas administrator to configure organization SSO.</p>}
          {demoAvailable && <Link className="signin-action demo" href="/architecture">Open local development workspace</Link>}
          <small>Access is limited to approved organization domains. Activity may be recorded for security and support.</small>
        </div>
      </section>
    </main>
  );
}
