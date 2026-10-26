import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  poweredByHeader: false,
  // The spec is read from the parent repository (../architecture), so tracing must look there too.
  outputFileTracingRoot: path.join(import.meta.dirname, ".."),
  outputFileTracingIncludes: { "/**": ["../architecture/**/*", "./fixtures/**/*"] },
  turbopack: { root: path.join(import.meta.dirname, "..") },
};

export default nextConfig;
