import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { Suspense } from "react";
import { shellData } from "@/lib/server/context";
import { SessionProvider } from "@/components/shell/session";
import { Shell } from "@/components/shell/shell";
import "./globals.css";

const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-sans" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: { default: "Atlas", template: "%s · Atlas" },
  description: "The chip's single source of truth: spec, RTL conformance, runs, waveforms and process.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const { branches, runs } = shellData();
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <Suspense>
          <SessionProvider branches={branches} allRuns={runs}>
            <Shell>{children}</Shell>
          </SessionProvider>
        </Suspense>
      </body>
    </html>
  );
}
