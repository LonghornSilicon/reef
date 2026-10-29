# Coral Architecture Atlas for VS Code

This local extension embeds the same Structure, Boot flow, Protocols, Instructions, and Ownership views used by the public atlas. Selecting a block opens its stable declaration in `architecture/soc.yaml`; moving the cursor through that file highlights the corresponding block.

The embedded Structure view has persistent **Up** and **Down** hierarchy controls plus clickable breadcrumbs. Select a composite and press Down or Alt+Down to enter it; press Up or Alt+Up to return to its parent. It supports two-axis drag, wheel/trackpad, and keyboard panning plus **Arrange** mode. Drag direct children or move the keyboard-focused block with arrows (Shift for 1px); obstacle-aware connections and boundaries update live.

## Develop locally

Run `python3 scripts/build.py` from the repository root, then press F5 using the checked-in Extension Host launch configuration. The Activity Bar contains **Coral Architecture Atlas Mental Model**.

Run `npm run check` after changing the extension shell. Validate `model.json` with the `project-mental-model` skill's anchor validator before packaging.

## Model maintenance

`model.json` and the packaged webview assets are generated from `architecture/soc.yaml` and the shared viewer. Source anchors use exact block declarations rather than line numbers; do not edit generated files by hand.
