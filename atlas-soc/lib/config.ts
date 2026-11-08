import { createHash, timingSafeEqual } from "node:crypto";

/**
 * Deployment posture, and the single place that decides it.
 *
 * Every development convenience in Atlas used to be written as
 * `process.env.NODE_ENV !== "production"`. That reads as "be strict in
 * production", but it is really "be permissive unless this exact string
 * matches", and it fails *open*: an unset, misspelled, or host-overridden
 * NODE_ENV simultaneously enabled the anonymous admin actor, signed sessions
 * with a secret committed to this repository, accepted every email domain, and
 * returned raw upstream error text to the browser.
 *
 * So the polarity is inverted here. `next dev` is the only thing that sets
 * NODE_ENV to "development", and a hardened posture is the default for
 * everything else — including a blank environment.
 */
export function isLocalDevelopment() {
  return process.env.NODE_ENV === "development";
}

/**
 * The unauthenticated admin actor used while developing against no IdP.
 *
 * Opt-in, not opt-out. The old `!== "false"` test meant that forgetting the
 * variable granted the backdoor; now forgetting it closes the backdoor.
 */
export function demoAuthEnabled() {
  return isLocalDevelopment() && process.env.ATLAS_DEMO_AUTH === "true";
}

/**
 * Public read-only access: anyone may read Atlas without signing in.
 *
 * Opt-in by exact value, so losing the variable closes access rather than
 * opening it. "Read-only" is not enforced here — it falls out of the role
 * model. An unauthenticated visitor becomes a `viewer`, and every mutating
 * route already requires `admin` or `editor`, so writes answer 403 and the
 * edit controls never render. Nothing extra is needed to keep a public
 * visitor from changing anything.
 */
export function publicReadAccessEnabled() {
  return process.env.ATLAS_PUBLIC_ACCESS === "true";
}

function identityProviderConfigured() {
  const google = Boolean(process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET);
  const entra = Boolean(
    process.env.AUTH_MICROSOFT_ENTRA_ID_ID &&
      process.env.AUTH_MICROSOFT_ENTRA_ID_SECRET &&
      process.env.AUTH_MICROSOFT_ENTRA_ID_ISSUER,
  );
  return google || entra;
}

/**
 * Whether signing in is possible at all.
 *
 * Callers use this to avoid invoking next-auth when no provider and no secret
 * exist, which would otherwise log MissingSecret on every request of an
 * otherwise healthy public deployment.
 */
export function signInAvailable() {
  return identityProviderConfigured() && Boolean(process.env.AUTH_SECRET || isLocalDevelopment());
}

// Lives in lib/features so client components can read it too; re-exported
// here because the server-side callers already import from this module.
export { aiIntegrationEnabled } from "@/lib/features";

/**
 * Whether uploaded documents have somewhere durable to live.
 *
 * Mirrors the check in lib/storage so a surface can say storage is missing
 * instead of failing inside a request.
 */
export function documentStorageConfigured() {
  return Boolean(process.env.S3_BUCKET && process.env.S3_REGION) || isLocalDevelopment();
}

/**
 * Largest upload the platform will carry.
 *
 * Serverless request bodies are capped well below the 10 MB the document
 * route used to advertise, and the platform rejects an oversized body before
 * any of our code runs, which surfaces as an opaque failure. Keeping the
 * number in one place lets the route and its form agree on it.
 */
export function maxUploadBytes() {
  const configured = Number(process.env.ATLAS_MAX_UPLOAD_BYTES);
  if (Number.isFinite(configured) && configured > 0) return Math.floor(configured);
  return 4 * 1024 * 1024;
}

/**
 * The message a caller is allowed to see when something upstream fails.
 *
 * Upstream text can carry request URLs, model names, quota details, and
 * occasionally fragments of the prompt, so it stays in the server log and only
 * reaches a browser on a developer's own machine.
 */
export function publicErrorMessage(error: unknown, fallback: string) {
  if (isLocalDevelopment()) return error instanceof Error ? error.message : fallback;
  return fallback;
}

/**
 * Constant-time shared-secret check for machine callers.
 *
 * Compares SHA-256 digests rather than the raw strings: `timingSafeEqual`
 * throws on a length mismatch, and the digest keeps both sides at 32 bytes so
 * the secret's length does not leak through that throw.
 */
export function secretMatches(provided: string | null | undefined, expected: string | undefined) {
  if (!provided || !expected) return false;
  const a = createHash("sha256").update(provided).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b);
}

/**
 * Configuration that a reachable deployment must have, checked at request
 * time rather than at import time so a build never depends on runtime secrets.
 */
export function configurationErrors(): string[] {
  if (isLocalDevelopment()) return [];
  const errors: string[] = [];

  // Sign-in is optional: a public deployment needs none of it. What is not
  // optional is that a half-configured sign-in must never be reachable, so
  // these are required only once a provider exists.
  if (identityProviderConfigured()) {
    if (!process.env.AUTH_SECRET) {
      errors.push("AUTH_SECRET is missing. Generate one with `openssl rand -base64 32`.");
    }

    const admits =
      process.env.ATLAS_ALLOWED_EMAILS ||
      process.env.ATLAS_ALLOWED_EMAIL_DOMAINS ||
      process.env.ATLAS_ADMIN_EMAILS ||
      process.env.ATLAS_EDITOR_EMAILS;
    if (!admits) {
      errors.push(
        "An identity provider is configured but no one is admitted. Set ATLAS_ALLOWED_EMAILS to a " +
          "comma-separated list of addresses, or ATLAS_ALLOWED_EMAIL_DOMAINS to admit whole domains.",
      );
    }
  } else if (!publicReadAccessEnabled()) {
    errors.push(
      "Nobody can reach Atlas: no identity provider is configured and ATLAS_PUBLIC_ACCESS is not set. " +
        "Either configure AUTH_GOOGLE_* or AUTH_MICROSOFT_ENTRA_ID_*, or set ATLAS_PUBLIC_ACCESS=true " +
        "for public read-only access.",
    );
  }

  if (process.env.ATLAS_DEMO_AUTH === "true") {
    errors.push("ATLAS_DEMO_AUTH=true is set outside local development. Remove it.");
  }

  return errors;
}

let warned = false;

/**
 * True when the deployment is not safe to serve. Logs the reasons once.
 *
 * Callers deny access on a true result: a misconfigured Atlas locks everyone
 * out and says why in the log, instead of letting anyone in as an admin.
 */
export function deploymentIsMisconfigured() {
  const errors = configurationErrors();
  if (!errors.length) return false;
  if (!warned) {
    warned = true;
    console.error(
      `Atlas refused to serve authenticated requests because of ${errors.length} configuration problem(s):\n` +
        errors.map((error) => `  - ${error}`).join("\n"),
    );
  }
  return true;
}
