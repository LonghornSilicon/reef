import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";

/**
 * IBM Plex: a technical grotesque with real engineering pedigree, replacing
 * Ubuntu Sans whose rounded humanist letterforms read as too soft for an
 * instrument. Self-hosted by next/font, so no third-party request at runtime.
 */
const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "Atlas · Coral SoC", template: "%s · Atlas" },
  description: "Evidence-linked architecture, explanations, tasks, and documents for the Coral SoC.",
  icons: { icon: "/logo.png" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <body>
        <template
          data-atlas-contract="ed7718f6"
          dangerouslySetInnerHTML={{
            __html: `<!--
THESIS: Atlas keeps the engineering artifact dominant while a persistent field log carries evidence, questions, ownership, and work; it refuses the generic dashboard of interchangeable cards.
OWN-WORLD: Bright lab drafting paper, ink-black structural rails, protocol color as status, ruled panels, compact engraved controls, and IBM Plex.
STORY: Enter the Coral workspace, inspect a source-linked artifact, ask from its exact context, then move the resulting uncertainty into owned work or evidence.
FIRST VIEWPORT: A slim black instrument rail and workspace bar frame one dominant artifact; its contextual actions stay attached at the right edge instead of sending the engineer to a dashboard.
FORM: Contextual workbench plus field log, grounded candidate 3, seed ed7718f6.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, and DESIGN.md
-->`,
          }}
        />
        {children}
      </body>
    </html>
  );
}
