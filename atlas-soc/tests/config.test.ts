import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  aiIntegrationEnabled,
  configurationErrors,
  demoAuthEnabled,
  documentStorageConfigured,
  isLocalDevelopment,
  maxUploadBytes,
  publicErrorMessage,
  publicReadAccessEnabled,
  secretMatches,
  signInAvailable,
} from "@/lib/config";

const GUARDED = [
  "NODE_ENV",
  "ATLAS_DEMO_AUTH",
  "AUTH_SECRET",
  "AUTH_GOOGLE_ID",
  "AUTH_GOOGLE_SECRET",
  "ATLAS_ALLOWED_EMAIL_DOMAINS",
  "ATLAS_ALLOWED_EMAILS",
  "ATLAS_ADMIN_EMAILS",
  "ATLAS_EDITOR_EMAILS",
  "OPENAI_API_KEY",
  "ATLAS_MAX_UPLOAD_BYTES",
  "S3_BUCKET",
  "S3_REGION",
  "ATLAS_PUBLIC_ACCESS",
] as const;

// NODE_ENV is declared read-only on ProcessEnv, and these tests exist
// precisely to vary it.
const env = process.env as Record<string, string | undefined>;

let saved: Record<string, string | undefined> = {};

beforeEach(() => {
  saved = Object.fromEntries(GUARDED.map((key) => [key, env[key]]));
});

afterEach(() => {
  for (const key of GUARDED) {
    if (saved[key] === undefined) delete env[key];
    else env[key] = saved[key];
  }
});

function setEnv(values: Partial<Record<(typeof GUARDED)[number], string | undefined>>) {
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) delete env[key];
    else env[key] = value;
  }
}

/** A deployment posture with nothing auth-related configured. */
const NO_AUTH_AT_ALL = {
  NODE_ENV: undefined,
  ATLAS_DEMO_AUTH: undefined,
  AUTH_SECRET: undefined,
  AUTH_GOOGLE_ID: undefined,
  AUTH_GOOGLE_SECRET: undefined,
  ATLAS_ALLOWED_EMAIL_DOMAINS: undefined,
  ATLAS_ALLOWED_EMAILS: undefined,
  ATLAS_ADMIN_EMAILS: undefined,
  ATLAS_EDITOR_EMAILS: undefined,
  ATLAS_PUBLIC_ACCESS: undefined,
} as const;

/**
 * These assertions exist because the original code expressed every
 * development convenience as `NODE_ENV !== "production"`, which grants the
 * convenience whenever the variable is absent or misspelled. A deployment is
 * far more likely to lose NODE_ENV than to set it to "development", so the
 * default must be the strict posture.
 */
describe("deployment posture defaults to hardened", () => {
  it("treats a blank environment as a deployment, not as local development", () => {
    setEnv({ NODE_ENV: undefined });
    expect(isLocalDevelopment()).toBe(false);
  });

  it("treats an unexpected NODE_ENV as a deployment", () => {
    setEnv({ NODE_ENV: "prod" });
    expect(isLocalDevelopment()).toBe(false);
    setEnv({ NODE_ENV: "staging" });
    expect(isLocalDevelopment()).toBe(false);
  });

  it("only recognizes the value next dev sets", () => {
    setEnv({ NODE_ENV: "development" });
    expect(isLocalDevelopment()).toBe(true);
  });
});

describe("the demo actor is opt-in", () => {
  it("stays closed when NODE_ENV is missing even if the flag is set", () => {
    setEnv({ NODE_ENV: undefined, ATLAS_DEMO_AUTH: "true" });
    expect(demoAuthEnabled()).toBe(false);
  });

  it("stays closed in development unless explicitly requested", () => {
    setEnv({ NODE_ENV: "development", ATLAS_DEMO_AUTH: undefined });
    expect(demoAuthEnabled()).toBe(false);
  });

  it("opens only for an explicit flag in development", () => {
    setEnv({ NODE_ENV: "development", ATLAS_DEMO_AUTH: "true" });
    expect(demoAuthEnabled()).toBe(true);
  });
});

describe("upstream error text", () => {
  // Deliberately not shaped like a real credential, so a secret scanner has
  // no reason to flag this file.
  const error = new Error("OpenAI request failed (401): Incorrect API key [redacted-example]");

  it("never reaches a caller outside local development", () => {
    setEnv({ NODE_ENV: undefined });
    expect(publicErrorMessage(error, "Atlas could not answer.")).toBe("Atlas could not answer.");
  });

  it("is available while developing locally", () => {
    setEnv({ NODE_ENV: "development" });
    expect(publicErrorMessage(error, "Atlas could not answer.")).toContain("401");
  });
});

describe("machine secret comparison", () => {
  it("accepts an exact match", () => {
    expect(secretMatches("s3cret-value", "s3cret-value")).toBe(true);
  });

  it("rejects a mismatch, a prefix, and a length difference", () => {
    expect(secretMatches("s3cret-valuf", "s3cret-value")).toBe(false);
    expect(secretMatches("s3cret", "s3cret-value")).toBe(false);
    expect(secretMatches("s3cret-value-and-more", "s3cret-value")).toBe(false);
  });

  it("rejects a missing header or a missing configured secret", () => {
    expect(secretMatches(null, "s3cret-value")).toBe(false);
    expect(secretMatches("s3cret-value", undefined)).toBe(false);
    expect(secretMatches(null, undefined)).toBe(false);
    expect(secretMatches("", "")).toBe(false);
  });
});

describe("configuration report", () => {
  const NOTHING_SET = {
    ...NO_AUTH_AT_ALL,
    NODE_ENV: undefined,
    AUTH_SECRET: undefined,
    AUTH_GOOGLE_ID: undefined,
    AUTH_GOOGLE_SECRET: undefined,
    ATLAS_ALLOWED_EMAIL_DOMAINS: undefined,
    ATLAS_ALLOWED_EMAILS: undefined,
    ATLAS_ADMIN_EMAILS: undefined,
    ATLAS_EDITOR_EMAILS: undefined,
    ATLAS_DEMO_AUTH: undefined,
  } as const;

  const CONFIGURED = {
    ...NOTHING_SET,
    AUTH_SECRET: "generated-secret",
    AUTH_GOOGLE_ID: "client-id",
    AUTH_GOOGLE_SECRET: "client-secret",
  } as const;

  it("says nobody can reach a deployment with neither sign-in nor public access", () => {
    setEnv(NOTHING_SET);
    const errors = configurationErrors();
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain("Nobody can reach Atlas");
    expect(errors[0]).toContain("ATLAS_PUBLIC_ACCESS");
  });

  it("requires a secret and an allowlist once a provider exists", () => {
    setEnv({ ...NOTHING_SET, AUTH_GOOGLE_ID: "client-id", AUTH_GOOGLE_SECRET: "client-secret" });
    const errors = configurationErrors();
    expect(errors).toHaveLength(2);
    expect(errors.join("\n")).toContain("AUTH_SECRET");
    expect(errors.join("\n")).toContain("ATLAS_ALLOWED_EMAILS");
  });

  it("is silent once a deployment is fully configured", () => {
    setEnv({ ...CONFIGURED, ATLAS_ALLOWED_EMAIL_DOMAINS: "example.com" });
    expect(configurationErrors()).toEqual([]);
  });

  it("accepts a named-address allowlist with no domain allowlist", () => {
    setEnv({ ...CONFIGURED, ATLAS_ALLOWED_EMAILS: "lead@example.com,rtl@other.test" });
    expect(configurationErrors()).toEqual([]);
  });

  it("treats naming an admin or editor as admitting them", () => {
    setEnv({ ...CONFIGURED, ATLAS_ADMIN_EMAILS: "lead@example.com" });
    expect(configurationErrors()).toEqual([]);
    setEnv({ ...CONFIGURED, ATLAS_EDITOR_EMAILS: "rtl@example.com" });
    expect(configurationErrors()).toEqual([]);
  });

  it("refuses a deployment that leaves the demo actor enabled", () => {
    setEnv({ ...CONFIGURED, ATLAS_ALLOWED_EMAIL_DOMAINS: "example.com", ATLAS_DEMO_AUTH: "true" });
    expect(configurationErrors().join("\n")).toContain("ATLAS_DEMO_AUTH");
  });

  it("reports nothing while developing locally", () => {
    setEnv({ NODE_ENV: "development", AUTH_SECRET: undefined });
    expect(configurationErrors()).toEqual([]);
  });
});

describe("public read access", () => {
  it("is off unless asked for by exact value", () => {
    for (const value of [undefined, "", "false", "TRUE", "1", "yes"]) {
      setEnv({ ATLAS_PUBLIC_ACCESS: value });
      expect(publicReadAccessEnabled()).toBe(false);
    }
  });

  it("is on for the exact value", () => {
    setEnv({ ATLAS_PUBLIC_ACCESS: "true" });
    expect(publicReadAccessEnabled()).toBe(true);
  });

  it("satisfies the configuration report with no sign-in at all", () => {
    setEnv({ ...NO_AUTH_AT_ALL, ATLAS_PUBLIC_ACCESS: "true" });
    expect(configurationErrors()).toEqual([]);
  });

  it("does not excuse a half-configured sign-in", () => {
    setEnv({
      ...NO_AUTH_AT_ALL,
      ATLAS_PUBLIC_ACCESS: "true",
      AUTH_GOOGLE_ID: "client-id",
      AUTH_GOOGLE_SECRET: "client-secret",
    });
    // A provider is present, so the secret and allowlist are still required:
    // public reading must not quietly turn into an open sign-in.
    expect(configurationErrors().join("\n")).toContain("AUTH_SECRET");
  });
});

describe("sign-in availability", () => {
  it("is false with no provider, so next-auth is never invoked", () => {
    setEnv({ ...NO_AUTH_AT_ALL, ATLAS_PUBLIC_ACCESS: "true" });
    expect(signInAvailable()).toBe(false);
  });

  it("is false with a provider but no secret", () => {
    setEnv({ ...NO_AUTH_AT_ALL, AUTH_GOOGLE_ID: "client-id", AUTH_GOOGLE_SECRET: "client-secret" });
    expect(signInAvailable()).toBe(false);
  });

  it("is true once a provider and a secret exist", () => {
    setEnv({
      ...NO_AUTH_AT_ALL,
      AUTH_GOOGLE_ID: "client-id",
      AUTH_GOOGLE_SECRET: "client-secret",
      AUTH_SECRET: "generated-secret",
    });
    expect(signInAvailable()).toBe(true);
  });
});

/**
 * The integration is off by a constant rather than an environment flag, so a
 * key arriving in the environment must not be able to switch it on.
 */
describe("AI integration", () => {
  it("is off", () => {
    expect(aiIntegrationEnabled()).toBe(false);
  });

  it("stays off when an API key is present", () => {
    setEnv({ OPENAI_API_KEY: "any-value-at-all" });
    expect(aiIntegrationEnabled()).toBe(false);
  });

  it("stays off in local development", () => {
    setEnv({ NODE_ENV: "development", OPENAI_API_KEY: "any-value-at-all" });
    expect(aiIntegrationEnabled()).toBe(false);
  });
});

describe("document storage", () => {
  it("is unconfigured on a host with no bucket", () => {
    setEnv({ NODE_ENV: undefined, S3_BUCKET: undefined, S3_REGION: undefined });
    expect(documentStorageConfigured()).toBe(false);
  });

  it("is configured once a bucket and region exist", () => {
    setEnv({ NODE_ENV: undefined, S3_BUCKET: "atlas-docs", S3_REGION: "us-east-1" });
    expect(documentStorageConfigured()).toBe(true);
  });

  it("falls back to local disk only while developing", () => {
    setEnv({ NODE_ENV: "development", S3_BUCKET: undefined, S3_REGION: undefined });
    expect(documentStorageConfigured()).toBe(true);
  });
});

describe("upload ceiling", () => {
  it("defaults under the serverless request-body limit", () => {
    setEnv({ ATLAS_MAX_UPLOAD_BYTES: undefined });
    expect(maxUploadBytes()).toBe(4 * 1024 * 1024);
  });

  it("is raisable for a container host", () => {
    setEnv({ ATLAS_MAX_UPLOAD_BYTES: String(25 * 1024 * 1024) });
    expect(maxUploadBytes()).toBe(25 * 1024 * 1024);
  });

  it("ignores a value that is not a positive number", () => {
    for (const value of ["0", "-5", "lots", ""]) {
      setEnv({ ATLAS_MAX_UPLOAD_BYTES: value });
      expect(maxUploadBytes()).toBe(4 * 1024 * 1024);
    }
  });
});
