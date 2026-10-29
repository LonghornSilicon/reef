---
version: 1
slug: "app-layout-tsx"
primary_target: "app/layout.tsx"
related_targets: ["app/globals.css","components","site/app.js"]
mode: "operate"
---

# Atlas product surface

## Intent

Atlas is an internal engineering workbench for understanding and operating the Coral SoC design record. It unifies the existing interactive architecture atlas with an editable wiki, an evidence-grounded copilot, native task tracking, managed documents, administration, and role-aware access.

## Visual language

Extend the incumbent Coral architecture atlas: bright drafting paper, ink-black instrument rails, precise rules, compact protocol colors, uppercase wayfinding, Ubuntu Sans, and diagram-first interaction. The shell should feel like engineering instrumentation rather than a generic SaaS dashboard.

## Surface structure

Use a persistent top workspace bar and narrow left instrument rail. Architecture gets a contextual field log. Wiki pages use an editorial source register. Copilot uses a conversation ledger and evidence desk. Tasks use a compact five-state field board. Documents distinguish Git-backed authority from managed uploads.

Architecture is the coordination layer: carry stable block context into Copilot, Tasks, and Documents. Keep source paths and authority visible at the point of use. Admin is a ruled operational record; sign-in is a split ink story and bright organization-SSO panel.

## Interaction and trust

Preserve keyboard focus, touch-sized controls, reduced motion, responsive drawer navigation, and strong source provenance. AI explanations and visuals are explicitly secondary to cited Git sources. Chat expires after 30 days; generated images are ephemeral.

At 820px and below, navigation and thread history become dismissible drawers, secondary rails stack after primary content, and forms collapse to one column. Preserve reading, chat, diagram pan/zoom, and all five task states; horizontal scrolling is acceptable where semantic order would otherwise be lost.

## Design contract

Concept seed `ed7718f6`, candidate 3: contextual workbench plus field log. The dotted sign-in field is intentional drafting-paper texture, and the legacy diagram animations are constrained to SVG strokes and fixed-height progress controls.

The root `DESIGN.md` is authoritative for durable tokens and patterns. The small Next.js development badge present in local review captures is not product UI.
