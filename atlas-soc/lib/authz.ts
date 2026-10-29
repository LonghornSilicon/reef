import { auth } from "@/auth";
import {
  demoAuthEnabled,
  deploymentIsMisconfigured,
  publicReadAccessEnabled,
  signInAvailable,
} from "@/lib/config";
import { PUBLIC_VISITOR_EMAIL, type Actor, type Role } from "@/lib/domain";

export async function currentActor(): Promise<Actor | null> {
  // A deployment whose sign-in is half-configured denies everyone rather than
  // falling through to one of the actors below.
  if (deploymentIsMisconfigured()) return null;

  // Skipped entirely when no provider and no secret exist: next-auth would
  // log MissingSecret on every request of an otherwise healthy public site.
  if (signInAvailable()) {
    const session = await auth();
    if (session?.user?.email) {
      return {
        email: session.user.email,
        name: session.user.name ?? session.user.email.split("@")[0],
        image: session.user.image,
        role: session.user.role,
      };
    }
  }

  if (demoAuthEnabled()) {
    return {
      email: "architect@atlas.local",
      name: "Avery Chen",
      role: "admin",
    };
  }

  // Public read access. `viewer` is what makes this safe: every mutating
  // route requires admin or editor, so a visitor can read everything and
  // change nothing.
  if (publicReadAccessEnabled()) {
    return {
      email: PUBLIC_VISITOR_EMAIL,
      name: "Public visitor",
      role: "viewer",
    };
  }

  return null;
}

/** True for the anonymous actor that public read access produces. */
export function isPublicVisitor(actor: Actor | null) {
  return actor?.email === PUBLIC_VISITOR_EMAIL;
}

export async function requireActor(allowed: Role[] = ["admin", "editor", "viewer"]) {
  const actor = await currentActor();
  if (!actor) return { actor: null, response: Response.json({ error: "Authentication required" }, { status: 401 }) };
  if (!allowed.includes(actor.role)) {
    return { actor: null, response: Response.json({ error: "Insufficient permission" }, { status: 403 }) };
  }
  return { actor, response: null };
}
