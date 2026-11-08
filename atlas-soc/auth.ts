import NextAuth, { type NextAuthConfig } from "next-auth";
import Google from "next-auth/providers/google";
import MicrosoftEntraID from "next-auth/providers/microsoft-entra-id";
import { isLocalDevelopment } from "@/lib/config";
import type { Role } from "@/lib/domain";

const providers: NextAuthConfig["providers"] = [];

if (process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET) {
  providers.push(
    Google({
      clientId: process.env.AUTH_GOOGLE_ID,
      clientSecret: process.env.AUTH_GOOGLE_SECRET,
    }),
  );
}

if (
  process.env.AUTH_MICROSOFT_ENTRA_ID_ID &&
  process.env.AUTH_MICROSOFT_ENTRA_ID_SECRET &&
  process.env.AUTH_MICROSOFT_ENTRA_ID_ISSUER
) {
  providers.push(
    MicrosoftEntraID({
      clientId: process.env.AUTH_MICROSOFT_ENTRA_ID_ID,
      clientSecret: process.env.AUTH_MICROSOFT_ENTRA_ID_SECRET,
      issuer: process.env.AUTH_MICROSOFT_ENTRA_ID_ISSUER,
    }),
  );
}

function commaList(value?: string) {
  return new Set(
    (value ?? "")
      .split(",")
      .map((item) => item.trim().toLowerCase())
      .filter(Boolean),
  );
}

function roleForEmail(email?: string | null): Role {
  const normalized = email?.toLowerCase() ?? "";
  if (commaList(process.env.ATLAS_ADMIN_EMAILS).has(normalized)) return "admin";
  if (commaList(process.env.ATLAS_EDITOR_EMAILS).has(normalized)) return "editor";
  return "viewer";
}

export const authProviderIds = providers.map((provider) =>
  typeof provider === "function" ? "configured" : provider.id,
);

export const { handlers, auth, signIn, signOut } = NextAuth({
  // The fallback is a literal in a public repository, so it may only ever sign
  // sessions on a developer's own machine. Anywhere else, a missing
  // AUTH_SECRET leaves this undefined and lib/config reports it.
  secret: process.env.AUTH_SECRET ?? (isLocalDevelopment() ? "atlas-development-only-secret" : undefined),
  providers,
  pages: { signIn: "/signin" },
  session: { strategy: "jwt", maxAge: 8 * 60 * 60 },
  callbacks: {
    async signIn({ user, profile }) {
      const email = user.email ?? (typeof profile?.email === "string" ? profile.email : null);
      if (!email) return false;
      const normalized = email.toLowerCase();

      // Naming someone in any of these lists is a deliberate act of granting
      // them access, so each one admits as well as assigns a role. Otherwise
      // adding an admin without also adding them to the allowlist would lock
      // out the person you just promoted.
      const allowedEmails = new Set([
        ...commaList(process.env.ATLAS_ALLOWED_EMAILS),
        ...commaList(process.env.ATLAS_ADMIN_EMAILS),
        ...commaList(process.env.ATLAS_EDITOR_EMAILS),
      ]);
      if (allowedEmails.has(normalized)) return true;

      const allowedDomains = commaList(process.env.ATLAS_ALLOWED_EMAIL_DOMAINS);
      // An empty allowlist means no admission decision has been made, which
      // is not the same as admitting everyone.
      if (!allowedDomains.size) return isLocalDevelopment();
      return allowedDomains.has(normalized.split("@")[1]);
    },
    async jwt({ token, user }) {
      if (user?.email) token.role = roleForEmail(user.email);
      token.role ??= roleForEmail(token.email);
      return token;
    },
    async session({ session, token }) {
      session.user.role = (token.role as Role | undefined) ?? "viewer";
      return session;
    },
  },
});
