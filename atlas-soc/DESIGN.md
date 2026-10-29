---
name: Atlas
description: A source-linked engineering instrument for understanding and operating the Coral SoC design record.
colors:
  paper: "#e8edf0"
  paper-bright: "#f8fafb"
  panel: "#f1f5f6"
  ink: "#101c26"
  ink-soft: "#405260"
  muted: "#5d6f7a"
  rule: "#c6d1d6"
  rule-strong: "#91a3ad"
  teal: "#087f8c"
  teal-dark: "#065b65"
  red: "#d83d36"
  green: "#488b38"
  amber: "#b7650b"
  purple: "#6d28d9"
  blue: "#1764c0"
  focus: "#006dc7"
  danger: "#aa2b24"
typography:
  display:
    fontFamily: '"IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif'
    fontSize: "clamp(36px, 5.6vw, 72px)"
    fontWeight: 700
    lineHeight: 0.9
    letterSpacing: "-0.035em"
  headline:
    fontFamily: '"IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif'
    fontSize: "clamp(25px, 2.8vw, 36px)"
    fontWeight: 700
    lineHeight: 0.98
    letterSpacing: "-0.03em"
  title:
    fontFamily: '"IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif'
    fontSize: "20px"
    fontWeight: 700
    lineHeight: 1
  body:
    fontFamily: '"IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif'
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.65
  label:
    fontFamily: '"IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif'
    fontSize: "9px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.08em"
  data:
    fontFamily: '"IBM Plex Mono", ui-monospace, Consolas, "Liberation Mono", monospace'
    fontSize: "9px"
    fontWeight: 600
    lineHeight: 1.4
rounded:
  xs: "4px"
  sm: "5px"
  md: "8px"
  lg: "10px"
  xl: "12px"
  message: "14px"
  full: "50%"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  2xl: "32px"
  3xl: "48px"
  4xl: "64px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper-bright}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    padding: "0 16px"
    height: "42px"
  button-evidence:
    backgroundColor: "{colors.teal-dark}"
    textColor: "{colors.paper-bright}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: "0 14px"
    height: "40px"
  input:
    backgroundColor: "{colors.paper-bright}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "8px 10px"
    height: "40px"
  nav-active:
    backgroundColor: "{colors.paper-bright}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    padding: "7px 2px"
    height: "64px"
  task-ticket:
    backgroundColor: "{colors.paper-bright}"
    textColor: "{colors.ink}"
    rounded: "{rounded.xl}"
    padding: "11px"
  message-user:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper-bright}"
    typography: "{typography.body}"
    rounded: "{rounded.message}"
    padding: "14px 17px"
---

# Design System: Atlas

## Overview

**Creative North Star: "The Instrumented Field Notebook"**

Atlas is an Operate-mode engineering workbench: an explorable architecture instrument joined to a disciplined field log. The product helps engineers move from a hardware block or design question to source evidence, explanation, ownership, documents, and work without losing the original architecture context. Its visual world is inherited from the Coral Architecture Atlas—bright drafting paper, ink-black rails, exact rules, terse labels, and diagram-first interaction—rather than from a generic SaaS dashboard.

The durable design contract is concept seed `ed7718f6`, candidate 3: **contextual workbench plus field log**. Interfaces should feel candid, technical, and reviewable. Expressive moments come from scale, information geometry, and the architecture itself; chrome stays disciplined.

**Key Characteristics:**

- Instrument shell with a persistent workspace bar and narrow section rail.
- High-density engineering information separated by rules and tonal paper layers.
- Oversized uppercase orientation paired with compact metadata and monospaced source data.
- Architecture context carried into evidence, Copilot, tasks, and documents.
- Provenance and uncertainty visible at the point of action.

## Colors

The palette reads as blue-gray drafting paper marked with near-black ink. Teal identifies evidence-linked action and current context; the remaining hues communicate protocols, roles, priorities, and status only.

### Primary

- **Instrument Ink** (`ink`): rails, decisive actions, selected diagram blocks, and the strongest hierarchy rules.
- **Evidence Teal** (`teal`, `teal-dark`): contextual links, grounded states, source-aware actions, and diagram accents.

### Tertiary

- **Protocol and State Set** (`red`, `green`, `amber`, `purple`, `blue`): semantic protocol, priority, permission, and health states. Never use these as a decorative rainbow.
- **Focus Blue** (`focus`): the sole keyboard-focus outline.
- **Danger Red** (`danger`): destructive actions and failure messages, distinct from ordinary protocol red.

### Neutral

- **Drafting Paper** (`paper`): application ground and cooler structural regions.
- **Bright Sheet** (`paper-bright`): reading, diagram, card, and input surfaces.
- **Bench Panel** (`panel`): contextual rails, headers, composers, and supporting work areas.
- **Soft Ink** (`ink-soft`) and **Muted Annotation** (`muted`): explanatory copy and secondary metadata.
- **Rule** (`rule`) and **Strong Rule** (`rule-strong`): structure, grouping, and authority boundaries.

**The Semantic Color Rule.** Outside ink, paper, and teal, color must encode a named engineering or application state.

**The Authority Contrast Rule.** Canonical content uses ink; managed or generated material uses teal or a visibly secondary treatment.

## Typography

**Display Font:** IBM Plex Sans, with Segoe UI and Helvetica fallbacks  
**Body Font:** IBM Plex Sans, with Segoe UI and Helvetica fallbacks  
**Data Font:** IBM Plex Mono, with ui-monospace and Consolas fallbacks

**Character:** IBM Plex is a technical grotesque drawn for an engineering identity; its even, slightly mechanical letterforms give the system an instrument-panel voice without the softness of a rounded humanist face. Plex Sans and Plex Mono are siblings, so labels and source data share one skeleton. Weight, case, and scale create hierarchy; monospaced text is reserved for stable IDs, repository paths, revisions, counts, timestamps, protocol labels, and shortcuts.

### Hierarchy

- **Display** (700, `clamp(36px, 5.6vw, 72px)`, 0.9): heavy, tightly tracked, uppercase statements for page identity and purposeful empty states. Keep headings balanced and generally under 14 characters per line.
- **Headline** (700, `clamp(25px, 2.8vw, 36px)`, 0.98): large uppercase names for the selected architecture context and other signature objects.
- **Title** (700, 20px, 1): compact uppercase section and surface titles.
- **Body** (400, 13px, 1.65): sentence case for explanations and technical reading; long-form wiki copy stays near 72 characters per line.
- **Label** (700, 9px, `0.08em`): small, tracked, usually uppercase wayfinding and metadata keys.
- **Data** (500, 9px, 1.4): compact monospaced values for source truth and machine-readable identifiers.

**The Two-Register Rule.** Use uppercase sans for orientation and mono for evidence; never turn explanatory paragraphs into all caps or render ordinary prose as code.

## Layout

Atlas uses a persistent 64px workspace bar, an 84px instrument rail, and a scroll-contained main field. The information architecture is Architecture, Wiki, Copilot, Tasks, Documents, and Admin. Architecture is the primary navigation object: stable block context must survive deep links into the other workflows.

Desktop workbenches split the main field into a dominant artifact or conversation area and a 300–306px contextual rail. Lists use source-register rows rather than floating cards. Long-form reading centers an article up to 820px beside its source rail. Copilot pairs a 252px thread ledger with an evidence desk. Tasks deliberately retain five horizontally ordered states; horizontal scrolling is preferable to compressing their meaning.

Spacing follows an 8px rhythm, with 4px corrections for compact technical controls. Strong section changes use 24–48px; metadata groups use 4–12px. Dense does not mean cramped: reading surfaces preserve generous margins and 1.65–1.7 body line height.

At 1100px, secondary controls abbreviate and forms reflow. At 820px, the rail becomes a dismissible drawer, contextual sidebars stack after their primary content, forms become single-column, and Copilot threads become an overlay. At 520px, display type and gutters tighten while primary reading and chat functions remain present. The architecture field may retain a minimum-height canvas; preserve panning and zoom rather than shrinking the diagram into illegibility.

**The Context-Follows-Content Rule.** On narrow screens, primary evidence or conversation comes first and its field log follows; context may move, but it must not disappear.

## Elevation & Depth

Atlas is flat by default. Paper tones, one-pixel rules, and ink boundaries establish depth. Shadows are reserved for temporary overlays, drawers, the focused composer, and movable tickets—not for every container.

### Shadow Vocabulary

- **Overlay** (`box-shadow: 0 18px 42px rgba(16, 28, 38, 0.15)`): account menus and mobile drawers.
- **Composer** (`box-shadow: 0 12px 28px rgba(16, 28, 38, 0.10)`): the active Copilot input desk.
- **Ticket** (`box-shadow: 0 7px 18px rgba(16, 28, 38, 0.07)`): task objects that can move through workflow states.

**The Flat-by-Default Rule.** If a rule or paper-tone change explains the hierarchy, do not add a shadow.

## Shapes

The system uses crisp rectangular fields softened just enough for operation. Structural sheets and registers remain square; controls use 8–10px corners; tickets use 12px; the conversation composer and user messages use 14px. Circular forms are reserved for identity, status lights, numbered citations, and diagram orbit motifs. Diagram nodes remain compact engineered enclosures, not pill cards.

Rules are part of the form language: one-pixel separators establish local structure, two- or three-pixel ink rules mark strong boundaries, and dashed borders indicate expired or unavailable generated material. The dotted sign-in and diagram fields are intentional drafting-paper texture.

**The Engineered Edge Rule.** Do not apply one universal radius. Shape communicates whether an object is structure, control, movable work, or identity.

## Components

### Instrument Shell

The top bar identifies workspace and location; the ink rail changes sections. Active navigation reverses to bright paper with teal icon emphasis. Desktop labels are compact; mobile navigation becomes a 260px drawer with scrim, explicit close control, and the same section order. Account and role controls stay at the far edge.

### Architecture Field Log

The architecture canvas is dominant. A right field log reports model readiness, selected block name and summary, evidence-linked actions, stable ID, source, authority, and the reminder that generated explanation is secondary. Selection updates this log without replacing the diagram. Legacy diagram protocol colors and animations remain scoped to the model; animation is limited to SVG strokes, fixed-height progress, and brief spatial transitions.

### Wiki Source Register

The index is an editorial register: oversized title, source count, concise summary, Git status, repository path, and directional affordance. Article pages pair readable Markdown with a source stamp, contextual Copilot action, architecture links, and change guidance. Source location is never relegated to a tooltip.

### Copilot Evidence Desk

Copilot uses a conversation ledger, contextual source strip, evidence-first empty state, grounded marker, message transcript, citation register, and persistent composer. Assistant answers remain on paper with a teal evidence rule; user questions invert to ink. Citations show ordinal, title, and source path. Generated visuals require captions and explicit ephemeral or expired states. Personal threads announce their 30-day retention.

### Task Field Board

The board preserves Backlog, Ready, In progress, Review, and Done as a fixed semantic sequence. Tickets show priority through a restrained top edge, with architecture tags, owner, due date, and an explicit move control. Empty columns state that no work is present. Editing controls appear only for authorized roles.

### Document Authority Register

Repository sources and managed attachments are separate registers. Canonical Markdown uses ink; managed uploads use teal. Both expose source name, revision or path, owner/size metadata, and architecture links. Upload language must say that attachments supplement rather than replace Git authority.

### Admin and Authentication

Admin is an operational record, not a settings-card gallery: knowledge revision, source count, Git revision, completion state, and retention policies are grouped in ruled sections. Unauthorized users get an explicit access-denied state. Sign-in splits an ink, dotted product story from a bright SSO panel; organization identity is primary, and local-development access is visibly distinct.

### Controls and Interaction

Primary buttons are ink; evidence-specific submission may use dark teal. Inputs are bright paper with a strong rule and persistent labels. All interactive elements require a visible 3px focus outline, semantic disabled state, keyboard operation, and touch targets near 40–48px where space allows. Hover may shift border, paper tone, or a few pixels of padding; it must not be the only indication of state. Honor `prefers-reduced-motion` by reducing transitions and disabling continuous animation.

**The Evidence-at-the-Action Rule.** Every AI answer, generated visual, upload, and design decision must expose its provenance or authority exactly where a user acts on it.

## Do's and Don'ts

### Do:

- **Do** preserve architecture context across Wiki, Copilot, Tasks, and Documents via stable IDs and visible source paths.
- **Do** distinguish canonical Git truth, managed evidence, and generated explanation in both copy and visual treatment.
- **Do** use rules, paper tones, compact metadata, and uppercase wayfinding to create hierarchy before adding containers.
- **Do** keep keyboard focus, drawer dismissal, reduced motion, touch sizing, and readable narrow-screen order at WCAG 2.2 AA fundamentals.
- **Do** make uncertainty, missing evidence, retention, permissions, and failure states explicit.

### Don't:

- **Don't** turn Atlas into a generic card dashboard with gradients, glass, soft pastel decoration, oversized empty padding, or rounded panels everywhere.
- **Don't** use protocol, priority, role, or status colors decoratively or without a visible legend or label.
- **Don't** let AI output resemble canonical design truth; never hide citations, source paths, expiry, or generated status.
- **Don't** collapse the five task states or remove field-log context on mobile; reflow or allow deliberate scrolling instead.
- **Don't** invent customer claims, production metrics, finalized RTL anchors, or certainty the source record does not provide.
- **Don't** treat development overlays or framework badges visible in local screenshots as Atlas product UI.
