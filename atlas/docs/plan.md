# Atlas — implementation plan

Atlas is the chip's single source of truth (SSOT). For any branch it shows the design as a block diagram, checks the spec against the RTL, and shows what the RTL does on every cycle of a simulation run.

This plan covers four things:
- the architecture overview;
- the decisions made while scoping it;
- the UI mockups;
- the UX rules in [`ux-directions.md`](ux-directions.md).

It describes the full roadmap in phases. Only M0 and M1 are detailed enough to build from.

---

## 1. Guiding principles

1. **The truth is the repository and team decisions, never AI output.** Every fact links to a file and line in the repo, or to a reviewed spec entry.
2. **The spec-vs-RTL comparison is fully deterministic.** It uses third-party tools: firtool, Slang and Yosys. Our code only compares two JSON documents.
3. **AI can propose; humans approve.** AI drafts are tagged and enter the spec only through a change request (pull request).
4. **Plan and reality are always explicit.** Every block, port and connection carries a status tag, so the two can't be confused.
5. **The UI must feel smooth and native.** `ux-directions.md` is a hard requirement for every screen (see §4).

## 2. Decisions made

| Topic | Decision |
|---|---|
| Scope | Full roadmap in phases. Build the UI first using sample data (M0), then add the toolchain. |
| Where the spec lives | `architecture/` and `docs/` move into the existing **RTL repo** at the same paths. The GitHub App is installed there and `GITHUB_REPOSITORY` points at it. |
| Repositories | **Single repo** for now; users pick a branch only. Choosing between repos can come later. |
| Status tag | **One field, `status`:** `planned \| in-progress \| implemented \| modified \| verified`. It replaces `maturity`. `implemented`, `modified` and `verified` are checked against the RTL. `planned` and `in-progress` are drawn dashed and grey. |
| Canvas style | Status is shown on the **block border**: dashed and grey means planned or in-progress. Each port has a **conformance dot**: ● match, ◐ different, ○ missing. Each block has a summary badge. |
| Language split | **TypeScript** holds all spec logic: schema, comparison, rule checker and tracer. The app and CI share it. **Python** only wraps third-party tools (pyslang, pywellen). `scripts/validate_model.py` is retired. |
| Run storage | Use the existing S3/R2 bucket (`lib/storage.ts`), with no database. Files are uploaded straight to the bucket with presigned URLs; they never pass through Vercel functions. |
| Waveforms | **Surfer** (WASM) runs in the browser. The server never parses FST files. CI produces compact `signals.json` and `transactions.json` for everything else. |

## 3. User flow and shared state

1. Open Atlas and choose a branch. There is one repo for now.
2. Atlas reads the YAML spec from that branch and draws the canvas, showing how each block compares with the RTL.
3. From the canvas, move to the doc graph, run library, waveform or process view.
4. **Every view shares the same branch, run and cycle.** These live in URL search params (the same pattern as `lensFromParams`), and a `BroadcastChannel` keeps them in sync across tabs. Changing the cycle anywhere moves every view.

---

## 4. UI and UX

### 4.1 Rules (from `ux-directions.md`, mandatory)

- **Motion:**
  - Animate only `transform` and `opacity`. Never animate width, height, top, left, right, bottom, margin, padding or flex-basis.
  - Use the easing curve `cubic-bezier(0.25, 1, 0.5, 1)` with durations of 150–200ms.
  - Never use linear easing.
- **Pressing:** every clickable element responds the moment it is pressed with `scale(.98)` and `opacity .95`.
- **States:** every view includes:
  - a **loading** state, using a shimmer skeleton that matches the layout (no spinners);
  - an **empty** state, which fades in gently;
  - an **error** state, shown inline and without blocking the page.
- **Visual:**
  - Dense, information-first layouts.
  - Spacing on a strict 4/8px grid.
  - No neon gradients or large coloured shadows.
  - Respect `prefers-reduced-motion`.
- **Implementation rule learned from the first mockups:** build each view **once and update it in place** by toggling classes and setting transforms. Re-rendering a whole view on every interaction is what made the first mockups feel janky: nothing could animate, scroll positions jumped, and tokens teleported between slots.

### 4.2 Mockups

There are three directions, with two clickable screens each. They use block names from the Coral spec, but the commits, runs, logs and signal values are made up. Serve them locally with `python3 -m http.server 8123` from the mockup folder and open `index.html`.

| Direction | Screens | Character |
|---|---|---|
| **A · Workbench** | A1 canvas and RTL comparison · A2 process view and waveform | Dark and dense, like an EDA tool. Outline on the left, canvas in the middle, inspector on the right. |
| **B · Floating panels** | B1 canvas · B2 run library and log | Light. A full-screen canvas with outline and inspector panels floating over it, and blocks with tinted headers. Builds on the floating-panel layout picked on 2026-10-19. |
| **C · Synced split** | C1 four panes on one cycle · C2 doc graph | Neutral. Canvas, process view, waveform and log side by side, all following one cycle. |

**Open decision:** which direction, or which mix. My recommendation:
- **B** for the canvas and run library;
- **A2** as the full process view;
- **C1** as the "run workspace" layout;
- **C2** as the doc graph.

All three directions share the same interaction primitives (§4.3), so a mix costs no extra engineering.

### 4.3 Interaction primitives (prototyped in the mockups' `ui.js` and `ui.css`)

These become React components under `components/atlas/`. The mockup code is the reference for their behaviour.

| Primitive | Behaviour | App target |
|---|---|---|
| **Pan and zoom canvas** | Drag or scroll to pan; Ctrl/⌘+scroll or pinch to zoom. Fit and ± buttons. The background grid moves with the diagram. Only `translate3d`/`scale` change. A drag never counts as a click. | Extend the existing `components/architecture/use-camera.ts`, which already writes the transform once per animation frame. Add the drag-vs-click rule and an inset-aware Fit. |
| **Block and wire states** | Selection rings, the activity ring (active or stalled), dimming of unrelated blocks on hover, and status filters all fade with opacity on pseudo-elements. Blocks are created once. | `canvas-node.tsx`, `diagram-canvas.tsx` |
| **Lens control** | Segmented control whose indicator slides with `translateX` (equal-width cells). The lenses are Structure, Protocol, Conformance and Dataflow. | `lens-controls.tsx` |
| **Pipeline** | Fixed stages and slots. Each token (an instruction or data item) is one persistent element moved with transform between slot positions. New tokens fade in from the left and leaving ones fade out to the right. Slot positions are measured again with `ResizeObserver`. | new `components/atlas/pipeline.tsx` |
| **Waveform strip** | Rows are drawn once. A single cursor moves by `translateX(i × 100%)`, and dragging across the signals scrubs. When a stall is detected, the signals that caused it are highlighted. | new `waveform-strip.tsx`; the full viewer is Surfer (M4) |
| **Log list** | Rows are rendered once. Filters toggle `hidden`. The current line is highlighted with an opacity overlay and smoothly scrolled into view. ↑/↓ steps through lines. | new `run-log.tsx`, virtualised for real runs |
| **Cycle control** | Step buttons, Play, ← → and Space shortcuts, a slider, and a validated number field with the error shown inline below it. | new `cycle-bar.tsx` plus `lib/session/cycle-store.ts` |
| **States** | `boot` crossfades from skeleton to content; `swap` crossfades panel contents; `show` fades empty and error blocks in. | `components/atlas/states.tsx` |

### 4.4 Views (what each one must do)

- **Canvas**
  - Block diagram of the selected branch, drawn from the spec. Status is shown on the border and conformance on the port dots.
  - Clicking a block opens the inspector: source links to the Scala and SV files at the commit, a port-by-port spec-vs-RTL table, the YAML and RTL doc-comment descriptions side by side, and timing-rule results from the latest run.
  - Clicking a wire shows its protocol, the comparison result, and its timing rules with their formulas.
  - The Dataflow lens dims everything not connected to the selected block. Status filters hide planned work.
  - States: skeleton loading; "RTL extract pending" error with retry; "no spec on this branch" empty state.
- **Doc graph**
  - Columns for block → interface → protocol → timing rule. Selecting a node lights up its whole chain.
  - The detail panel shows the signal table, the rule YAML, its plain-language meaning, and results from the latest runs.
  - Empty state when a search matches nothing.
- **Run library and log**
  - Runs from CI, lab servers and local PCs, with each run's `run.json` details.
  - The log can be filtered by block and event type and searched. Each line starts with its cycle, and clicking it moves the cycle everywhere.
  - States: an error for an incomplete upload, with a retry; empty states for filters that match nothing.
- **Waveform:** Surfer, with signals grouped by spec interface, jump-to-cycle, and markers for transactions and violations.
- **Process view**
  - A strip of the blocks in the pipeline, a schematic per stage, and the cycle controls.
  - Tokens are placed from the real waveform. Users can follow one instruction through the pipeline.
  - A stall callout explains the cause and highlights the signals behind it.
  - The detail panel shows the instruction's path through the stages and links to the RTL for the current stage.

---

## 5. Roadmap

### M0 — UI foundation with sample data (build-level)

**Goal:** the chosen direction runs in the real Next.js app using fixture data, before any toolchain work. Every view has its loading, empty and error states, and everything follows §4.1.

1. **Pick the direction** (§4.2). Map its tokens into the app's theme. If the user also wants a dark theme, A's tokens become the dark theme.
2. **Shared motion layer.**
   - `app/atlas-motion.css`: easing and duration tokens, press feedback, the `enter` keyframe, the skeleton shimmer, and the reduced-motion override.
   - A lint check that fails on any `transition` that names a layout property.
3. **Build the primitives** listed in §4.3 as React components. React must keep the "update in place" behaviour:
   - Use stable `key`s and don't remount views on state changes.
   - Drive tokens and cursors through CSS variables or refs, not by re-rendering the tree.
4. **Fixture data.**
   - `tests/fixtures/atlas/`: compare report, run list, `events.jsonl`, `signals.json`, `transactions.json`.
   - Port these from the mockups' `data.js` into the formats defined in M1 and M3.
   - A `FixtureSource` stands in for the real data until each milestone lands.
5. **Routes:**
   - `app/(workspace)/architecture` (the canvas, upgraded)
   - `/spec/[[...node]]` (doc graph)
   - `/runs`
   - `/runs/[id]`
   - `/process`
   - `/runs/[id]/workspace` (the synced split layout)
6. **Shared cycle state:** `lib/session/cycle-store.ts`, synced through URL params and `BroadcastChannel`.

**Verification:**
- Playwright specs (Playwright is already installed but has no specs yet). For each view: wait for the skeleton to clear, then check the empty and error states, then check that selection and cycle changes keep the scroll position and focus.
- One test asserts that a token keeps the same DOM node from one cycle to the next. That is the "no re-render" guarantee.
- A test with reduced motion turned on.
- `npm run check` passes.

### M1 — Spec v2 and RTL comparison (build-level)

**Goal:** the spec lives in the RTL repo. CI extracts the RTL's modules and ports, and the canvas shows whether each block, port and connection matches the RTL, using the real comparison instead of fixtures.

1. **Schema v2** (`lib/model/schema.ts`, `architecture/schema.json`, `lib/model/validate.ts`)
   - `schema_version: 2`. `status` is required on blocks, ports and connections, and there is an `RTL_CHECKED` set.
   - Protocols get optional `signals: [{name, width: int | "param:X", dir: fwd|bwd}]` and `timing_rules`. Informal protocols (`containment`, `control`, …) stay as they are.
   - New `interfaces: {id: {protocol, params, role_naming: decoupled|flat}}`.
   - Ports get `interface` and an optional `rtl: {prefix, map}`.
   - Block `rtl: {module, instance?} | null` replaces the empty anchor arrays. It is required when the block's status is RTL-checked.
   - Timing rules have kinds `implies | next | stable_while | within` and use a small expression grammar (`! && || == !=`, literals, `$past`). The parser goes in `lib/model/rules.ts`.
   - `referenceIssues()` additionally checks that:
     - interface and protocol references resolve;
     - every RTL-checked block has `rtl.module`;
     - no RTL-checked port sits on a planned block;
     - no RTL-checked connection has a planned endpoint.
2. **Migration** (`scripts/migrate-spec-v2.ts`)
   - Uses the `yaml` Document API, as `lib/model/edit.ts` does, so comments are kept.
   - Maps `proposal`/`specified` → `planned`; everything else carries over. Ports and connections inherit the lower status of their endpoints.
   - Renames `maturity` → `status` in about 14 places: `edit.ts`, `view.ts`, `graph.ts`, `lens.ts`, `canvas-node.tsx`, `inspector.tsx`, `outline-tree.tsx`, `lens-controls.tsx`, `model-editor.tsx`, `canvas-popovers.tsx`, `blocks/[blockId]/page.tsx`, `scripts/build.py` and `tests/model-flow.test.ts`.
   - Adds an `updatePort` edit operation. `updateConnection` and `movePort` also accept `status`.
3. **Move the spec into the RTL repo**
   - Copy `architecture/`, `docs/` and the evidence files to the same paths in the RTL repo.
   - The Atlas repo keeps a copy as fixtures in `tests/fixtures/spec/`, read by `seedFromBundle()`.
   - Replace `scripts/sync-evidence.mjs` with `app/api/evidence/[sha]/[...path]` and add a binary `readBlobBytes`.
   - The branch picker hides branches without the `atlas/` prefix behind a toggle. The webhook only reacts to pushes that touch `architecture/` or `docs/`.
4. **Extraction CI in the RTL repo** (`.github/workflows/atlas.yml`)
   1. Run `firtool --split-verilog` with source locators.
   2. `tools/atlas/extract_rtl.py` (pyslang) writes `extract.json` (`atlas.rtl-extract/1`). For each module it records:
      - the SV and Scala file and line;
      - its ScalaDoc;
      - its ports `{name, dir, width}`;
      - its instances and their connections.
   3. Optionally cross-check with `yosys write_json`. Keep that output, because M5 needs it.
   4. `atlas-cli.ts validate && compare --fail-on implemented,verified`, run from Atlas checked out at a pinned tag. The result goes to the job summary.
   5. Upload `rtl/<sha>/{extract,yosys}.json` to the bucket.
5. **Comparison** (`lib/rtl/{names,compare,store}.ts`)
   - `names.ts` turns each spec port into the RTL signal names it expects:
     - decoupled ports: `io_<p>_{valid,ready}` and `io_<p>_bits_<sig>`;
     - flat ports: `<p>_<sig>`;
     - explicit `map` entries override both.
     - Golden tests cover these.
   - `compare.ts` produces `atlas.compare/1`. Each item is `match | missing | different | unchecked`, with the differences listed, any extra RTL ports, and a summary.
     - On a `modified` entry, differences warn instead of failing.
     - Connections are checked against how instances are wired.
   - The app runs the comparison and caches the result by `specBlobSha:extractSha`.
   - `store.ts` uses the extract from the nearest ancestor commit that has one and reports how far behind it is (`staleBy`). The UI shows a banner when it is stale and never guesses a result.
6. **Canvas wiring:** connect the M0 canvas to the real comparison report. `buildSnapshot` adds `status` to `ViewPort` and `ViewConnection`. The report is passed alongside the snapshot, not merged into the model.

**Verification:**
- Vitest:
  - the migration round-trip keeps structure and comments;
  - the rule parser accepts and rejects the right expressions;
  - the `names.ts` golden table;
  - `compare.ts` fixtures for match, width, direction, missing, extra and stale.
- A small Chisel fixture runs through the whole CI workflow and the canvas shows the expected dots.
- `npm run check` passes.

### M2 — Doc graph

- The M0 doc graph reads real interfaces, protocols and rules.
- Clicking a connection on the canvas opens its protocol and timing rules.
- `diffModels()` covers interfaces and protocols, so change requests show rule edits.
- SystemRDL is deferred.
- **Verify:** graph snapshot tests, and rule diffs showing on change-request pages.

### M3 — Simulation runs and run library

- **RTL repo:**
  - `sim/run.py` runs Verilator + cocotb with `--trace-fst`.
  - A cocotb hook writes `events.jsonl` lines `{cycle, block, type, details}`.
  - It also writes `run.json`: commit, branch, test, simulator version, seed, cycle count and a dirty flag.
  - `fst_compact.py` (pywellen) writes `signals.json` with only the signals the spec maps.
- **Upload:** `POST /api/runs` checks a per-machine run token and returns presigned PUT URLs. Files go to `runs/<branch>/<sha>/<runId>/`, and runs are listed by bucket prefix.
- **App:** the M0 run library and log view (virtualised) read the real bucket. Logs are streamed from signed URLs.
- **Verify:** a CI job uploads a fixture run, plus tests for listing and filtering.

### M4 — Waveforms, shared cycle, protocol checks

- **Surfer:**
  - Self-host the WASM build under `public/surfer/` and embed it in an iframe, controlled over postMessage/WCP (the Wave Control Protocol).
  - Supports jump-to-cycle, signal groups per interface (using the M1 name mapping), and markers for transactions and violations.
  - The bucket needs CORS so Surfer can load FST files from signed URLs.
- **Rule checking:** `lib/sim/check.ts` evaluates `timing_rules` over `signals.json`. It runs in `run.py` (writing `violations.json`) and again in the app when the spec changes.
- **Reference model (optional):** a diff against a cycle-level model that reports the first cycle where they diverge.
- **Risk:** the Surfer embedding API is the biggest unknown, so start with a 2-day spike that includes a memory test on a large FST.
- **Verify:** a Playwright test that clicking a log line moves the Surfer cursor; known-bad waveforms produce the expected violations.

### M5 — Dataflow

- **Static:** `lib/rtl/netlist.ts` builds a graph of what drives what from `yosys.json`. The canvas Dataflow lens overlays it.
- **Dynamic:**
  - `lib/sim/tracer.ts` infers transfers from `valid && ready` and follows FIFO and pipeline ordering, as declared per stage in the spec's `process:` section.
  - It cross-checks against simulation-only debug tags at the points where the pipeline reorders. Tags are an `ifdef` in SV and a strippable debug field in Chisel.
  - The tracer runs in CI and writes `transactions.json`.
- **Verify:** fixture FIFO and reorder-buffer designs checked against golden transaction files.

### M6 — Process view

- **Spec:** add a `process:` section that maps registers, queue slots and handshakes to signal paths, with the ordering of each stage.
- **App:**
  - The M0 process view places tokens from `transactions.json` and `signals.json`.
  - Users can follow an instruction across blocks.
  - Clicking a stalled token jumps to the signals that caused the stall in Surfer, through the shared cycle state.
- **Verify:** a Playwright walkthrough on a fixture run.

### Later — optional AI

- Turned on through `lib/features.ts`.
- Drafts are tagged `origin: ai`, arrive only as change-request branches, and are merged by a human.
- AI explanations sit next to the deterministic results and never replace them.

---

## 6. Cross-cutting risks

- **Vercel limits:** FST and JSONL files never pass through functions; they use signed URLs. Comparison results are cached by sha pair.
- **Chisel name drift** (`io_` prefixes, `_bits_` flattening, Vec suffixes): handled by `map` overrides and golden tests.
- **Extraction lagging behind spec edits:** shown with the `staleBy` banner.
- **Pinned Atlas version in the RTL repo's CI:** bumping the pin is a deliberate PR.
- **UX regressions:** the M0 Playwright checks ("no re-render", reduced motion, every state) run in CI, and the lint check blocks any transition on a layout property.

## 7. Open questions

- **Which UI direction** (§4.2). This is the only thing blocking M0.
- Where standard test runs execute: GitHub Actions, a lab server, or both.
- Which pipeline points reorder, and therefore need debug tags.
- Whether a cycle-level reference model will be built for planned blocks.
- Whether there should be a dark theme as well as the chosen direction.
