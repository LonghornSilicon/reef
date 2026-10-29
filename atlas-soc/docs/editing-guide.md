# Editing the Coral Architecture Atlas

The atlas is text-first: edit `architecture/soc.yaml`, run the validator, and review the generated website before opening a pull request.

## Add or change a block

1. Give the block a stable lowercase ID.
2. Set its parent. Keep `rank` and `order` for compatibility, but place it explicitly in `layouts.<parent>.nodes` using canvas coordinates and dimensions.
3. Record only confirmed ownership and maturity. Use `unassigned` when ownership is unknown.
4. Add Markdown, PDF-page, issue, and RTL references as evidence becomes available.
5. Declare every visible interface under `ports.<block-id>`. Give it a role label such as `AXI MASTER`, `AXI SLAVE`, `TL-UL HOST`, `TL-UL DEVICE`, `IBUS`, or `DBUS`, plus its edge and fractional offset.
6. Add connections using `from_port` and `to_port`. The validator requires both ports to exist and to use the connection's protocol.

Port sockets render flush to the inside of their owning block edge. Connections between immediate children are also rendered inside the parent composite, so the focused and nested views stay consistent without a second layout definition.

## Navigate the hierarchy

Select a composite block, then use the persistent **Down** control or Alt+Down to enter its authored child canvas. Use **Up** or Alt+Up to move to its parent; breadcrumbs jump directly to any ancestor. Down remains unavailable until the selected block has children. The detailed 4×4 Systolic Array reference is a standalone root reachable through search; the integrated 16×16 instance remains an intentionally solid black box.

## Fixed placement

Each composite block owns a fixed canvas under `layouts`. Coordinates are authored and reviewed; the viewer never rearranges them automatically. Add every immediate child to the parent's `nodes` map with `x`, `y`, `width`, and `height`; use the same process for the internals of DMA, Coral Core, or any future composite block.

Keep the active containment intact: Scalar Core, Vector Execution Engine, and the solid 16×16 Systolic Array instance belong to Scalar + Vector Core. The detailed 4×4 Systolic Array remains a standalone reference hierarchy. DTCM has four `128×512` bank children, ITCM has one, and the reference PE grid uses stable `pe-r<row>-c<column>` IDs for all 16 PE + WB cells.

### Draft positions in the viewer

1. Open Structure and focus the parent whose immediate children you want to arrange.
2. Outside Arrange mode, drag anywhere to pan in both axes. Wheel/trackpad scrolling and canvas-focused Arrow/Page keys also pan; middle-drag or Space+drag remains available.
3. Turn on **Arrange**. Drag a direct child, or keyboard-focus it and use the arrow keys. Arrows move 10px; Shift+Arrow moves 1px.
4. Review the live protocol rerouting and enclosure boundary while you move blocks. Positions remain clamped inside the authored canvas.
5. Use **Copy positions** to copy the complete `layouts.<focus>` YAML entry, including its canvas and every child rectangle. Paste it over the matching entry in `architecture/soc.yaml` and review the diff.
6. Use **Reset** to remove local edits for the current focus and restore the authored YAML positions.

Arrangement edits are local overlays, stored separately for each focus and retained across reloads (and in VS Code webview state). They do not change `architecture/soc.yaml` until you copy and commit the generated YAML. Leaving Arrange mode or navigating elsewhere does not discard them; Reset affects only the current focus.

## RTL references

RTL does not live in this planning repository. A future RTL entry should contain a public repository URL, immutable revision when possible, repository-relative path, and a declaration or module name. The VS Code extension opens a matching local workspace file when available and falls back to the repository URL.

Issue entries use `title` and `url`. Keep schedules and task details in the issue tracker; the atlas only exposes the architectural work associated with each block.

## Local preview

Run `python3 scripts/build.py`, then serve `dist/` with any static HTTP server. The checked-in tests and CI run the same validation before deployment.
