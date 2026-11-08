# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Next.js and TypeScript for the authenticated web application, with the existing Python architecture validation and artifact build pipeline retained. The production target is a container-capable company cloud with managed PostgreSQL and object storage.

## Users

Atlas is primarily for an internal engineering team: system architects, RTL engineers, technical leads, and project owners collaborating on one SoC program. Administrators manage workspace access and knowledge ingestion, editors manage project work and documents, and viewers inspect the architecture and use approved AI capabilities.

## Product Purpose

Atlas is the shared engineering workspace for understanding, explaining, and coordinating a complex hardware design. It connects the explorable architecture model to its supporting wiki, evidence, conversations, tasks, ownership, and documents. Success means an engineer can move from a block or design question to an evidence-backed explanation and the related project work without losing context.

## Positioning

Atlas makes the architecture itself the navigation and coordination layer. Explanations, source evidence, owners, tasks, and documents attach to stable architecture concepts instead of living as disconnected search results or project-management records.

## Operating Context

- The architecture model and technical wiki are reviewed as YAML and Markdown in Git.
- Engineers use the web application for shared understanding and project coordination, and a read-only VS Code companion for source navigation.
- Supporting evidence includes Markdown specifications, architecture diagrams, PDFs, presentations, issue links, and future immutable RTL anchors.
- Coral is the initial workspace and project inside the Atlas product.

## Capabilities and Constraints

- Git-backed YAML and Markdown remain canonical for architecture and technical wiki content.
- The first release adds organization SSO, grounded personal chat with source citations, contextual questions from wiki pages and architecture blocks, and ephemeral explanatory image generation.
- Personal chat threads expire after 30 days and can be deleted sooner by their owner. Generated image bytes are not retained by Atlas.
- OpenAI is the first AI provider, isolated behind application-owned chat, retrieval, and image interfaces.
- Later releases add native Atlas tasks and a versioned file library for PDFs, images, and office documents. Tasks and uploaded files can link to stable architecture blocks.
- The initial permission model is administrator, editor, and viewer. Public signup, sales CRM, customer pipelines, and real-time document co-editing are not part of the product.

## Brand Commitments

The product name is **Atlas**. Coral remains the name of the initial SoC workspace. Preserve the incumbent source-linked engineering character and the existing architecture diagrams, protocol language, and stable block identifiers. Product copy is direct, technical, and candid about unconfirmed design decisions.

## Evidence on Hand

- `architecture/soc.yaml` and `architecture/schema.json` define the current architecture model.
- `docs/` contains the editable technical wiki and engineering open decisions.
- The repository contains reference diagrams, source presentations, and PDFs used as architecture evidence.
- `site/` contains the established interactive architecture workbench.
- `tools/coral-architecture-atlas-vscode/` contains the read-only VS Code companion.
- No customer claims, production benchmarks, testimonials, or final RTL repository anchors are available and none should be fabricated.

## Product Principles

1. Ground explanations in inspectable engineering evidence.
2. Keep canonical design truth reviewable in Git.
3. Carry architecture context into every supporting workflow.
4. Make uncertainty, ownership, and missing evidence visible.
5. Keep generated explanations distinct from authoritative design artifacts.

## Accessibility & Inclusion

The web application should meet WCAG 2.2 AA fundamentals, support complete keyboard navigation, respect reduced-motion preferences, and remain usable on narrow screens without hiding core reading or chat functions.
