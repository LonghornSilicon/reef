import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  agentRules: false,
  allowedDevOrigins: ["127.0.0.1"],
  devIndicators: false,
  // `standalone` emits a self-contained server for the Dockerfile. A
  // serverless host builds its own entrypoint and does not want it, so this
  // is opt-in via the container build rather than always on.
  ...(process.env.ATLAS_BUILD_TARGET === "container" ? { output: "standalone" as const } : {}),
  poweredByHeader: false,
  // `npm run typecheck` is the release gate. Next 16.4.2 currently fails to
  // parse TypeScript 5.9's valid `--showConfig` output before route generation.
  typescript: { ignoreBuildErrors: true },
  // Every surface that reads the Git-backed model or wiki at request time must
  // declare it, or the standalone build ships without those files. The page
  // routes were missing before, which broke /wiki and /documents in the
  // container while the API routes worked.
  outputFileTracingIncludes: {
    "/**": ["./architecture/**/*", "./docs/**/*"],
    "/api/admin/ingestion": ["./docs/**/*", "./architecture/**/*", "./*.pdf"],
    "/api/chat": ["./docs/**/*", "./architecture/**/*", "./README.md"],
    "/api/chat/image": ["./docs/**/*", "./architecture/**/*", "./README.md"],
  },
  experimental: {
    // Kept at or below the serverless request-body limit so an oversized
    // upload is refused by the route with a readable message instead of by
    // the platform with an opaque one. lib/config's maxUploadBytes() is the
    // matching runtime check.
    serverActions: {
      bodySizeLimit: "4mb",
    },
  },
};

export default nextConfig;
