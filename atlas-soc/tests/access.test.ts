import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// lib/authz imports next-auth, which does not resolve under the node test
// environment. None of these cases reach it: signInAvailable() is false
// throughout, so currentActor() never calls auth(). Returning null keeps that
// explicit — if a future change did call it, these tests would still describe
// the no-session path truthfully.
vi.mock("@/auth", () => ({ auth: async () => null }));

const { currentActor, isPublicVisitor, requireActor } = await import("@/lib/authz");
const { PUBLIC_VISITOR_EMAIL } = await import("@/lib/domain");

const GUARDED = [
  "NODE_ENV",
  "ATLAS_DEMO_AUTH",
  "ATLAS_PUBLIC_ACCESS",
  "AUTH_SECRET",
  "AUTH_GOOGLE_ID",
  "AUTH_GOOGLE_SECRET",
  "ATLAS_ALLOWED_EMAILS",
] as const;

const env = process.env as Record<string, string | undefined>;
let saved: Record<string, string | undefined> = {};

beforeEach(() => {
  saved = Object.fromEntries(GUARDED.map((key) => [key, env[key]]));
  for (const key of GUARDED) delete env[key];
});

afterEach(() => {
  for (const key of GUARDED) {
    if (saved[key] === undefined) delete env[key];
    else env[key] = saved[key];
  }
});

/**
 * Public access is safe only because the anonymous actor is a `viewer` and
 * every mutating route demands admin or editor. That is the whole guarantee,
 * so it is asserted directly rather than inferred from the role table.
 */
describe("public read access", () => {
  it("produces an anonymous viewer, not a null actor", async () => {
    env.ATLAS_PUBLIC_ACCESS = "true";
    const actor = await currentActor();
    expect(actor).not.toBeNull();
    expect(actor?.role).toBe("viewer");
    expect(actor?.email).toBe(PUBLIC_VISITOR_EMAIL);
    expect(isPublicVisitor(actor)).toBe(true);
  });

  it("can read", async () => {
    env.ATLAS_PUBLIC_ACCESS = "true";
    const access = await requireActor();
    expect(access.response).toBeNull();
    expect(access.actor?.role).toBe("viewer");
  });

  it("cannot write: editor routes answer 403", async () => {
    env.ATLAS_PUBLIC_ACCESS = "true";
    const access = await requireActor(["admin", "editor"]);
    expect(access.actor).toBeNull();
    expect(access.response?.status).toBe(403);
  });

  it("cannot administer: admin routes answer 403", async () => {
    env.ATLAS_PUBLIC_ACCESS = "true";
    const access = await requireActor(["admin"]);
    expect(access.actor).toBeNull();
    expect(access.response?.status).toBe(403);
  });

  it("is not a person, so nothing can be owned by it", async () => {
    env.ATLAS_PUBLIC_ACCESS = "true";
    const actor = await currentActor();
    // .invalid can never resolve, so this address cannot collide with a real
    // account or be mistaken for one.
    expect(actor?.email.endsWith(".invalid")).toBe(true);
  });
});

describe("without public access", () => {
  it("has no actor at all, so every route answers 401", async () => {
    const access = await requireActor();
    expect(access.actor).toBeNull();
    expect(access.response?.status).toBe(401);
  });

  it("stays closed when the flag is not exactly \"true\"", async () => {
    for (const value of ["false", "TRUE", "1", "yes", ""]) {
      env.ATLAS_PUBLIC_ACCESS = value;
      expect(await currentActor()).toBeNull();
    }
  });
});
