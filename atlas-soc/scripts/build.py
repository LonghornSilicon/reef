#!/usr/bin/env python3
"""Validate the Coral architecture model and build its static/VS Code artifacts."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from validate_model import load_model, validate

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "architecture" / "soc.yaml"
SITE = ROOT / "site"
DIST = ROOT / "dist"
EXTENSION = ROOT / "tools" / "coral-architecture-atlas-vscode"


def section_for(block: dict[str, Any]) -> str:
    if block["kind"] in {"memory", "external-memory"}:
        return "memory"
    if block["kind"] in {"compute", "datapath", "engine"}:
        return "compute"
    if block["kind"] in {"interconnect", "bridge", "arbiter", "interface"}:
        return "interconnect"
    if block["kind"] in {"control", "controller", "peripheral", "peripheral-group"}:
        return "control"
    return "system"


def extension_model(model: dict[str, Any]) -> dict[str, Any]:
    sections = [
        {"id": "system", "name": "System", "role": "Own the Coral planning boundary", "summary": "Chip boundary, external memory, and system-level composition.", "color": "#101c26", "darkColor": "#cbd5e1"},
        {"id": "compute", "name": "Compute + movement", "role": "Execute and move working data", "summary": "Coral Core, its datapath, and DMA behavior.", "color": "#ef3434", "darkColor": "#ff7b7b"},
        {"id": "memory", "name": "Memory", "role": "Hold code and data", "summary": "Software-managed TCMs and external HyperRAM.", "color": "#c25700", "darkColor": "#ffad66"},
        {"id": "interconnect", "name": "Interconnect", "role": "Route requests and responses", "summary": "AXI, TL-UL, bridges, arbiters, and internal fabric.", "color": "#6d28d9", "darkColor": "#b79aff"},
        {"id": "control", "name": "Control + peripherals", "role": "Boot and expose the system", "summary": "Reset, boot sources, host link, and peripheral devices.", "color": "#087f8c", "darkColor": "#6ad4dc"},
    ]
    source_text = MODEL_PATH.read_text(encoding="utf-8")
    symbols = []
    for block in model["blocks"]:
        expanded_needle = f"  - id: {block['id']}"
        inline_needle = f"  - {{id: {block['id']},"
        needle = expanded_needle if expanded_needle in source_text else inline_needle
        anchor = {"file": "architecture/soc.yaml", "needle": needle}
        symbols.append({"id": block["id"], "name": block["name"], "kind": block["kind"], "section": section_for(block), "open": anchor, "matches": []})
    block_by_id = {block["id"]: block for block in model["blocks"]}
    overview_items = []
    for index, block in enumerate(block for block in model["blocks"] if block.get("parent") == "coral-soc"):
        overview_items.append({"id": f"overview-{block['id']}", "step": f"{index + 1:02d}", "title": block["name"], "description": block["summary"], "section": section_for(block), "symbols": [block["id"]], "note": f"{block['maturity'].replace('-', ' ').title()} · owner {block['owner']}"})
    boot = next(item for item in model["behaviors"] if item["id"] == "boot")
    boot_items = []
    for index, step in enumerate(boot["steps"]):
        sections_for_step = [section_for(block_by_id[item]) for item in step["blocks"] if item in block_by_id]
        boot_items.append({"id": step["id"], "step": f"{index + 1:02d}", "title": step["title"], "description": step["detail"], "section": sections_for_step[0] if sections_for_step else "system", "symbols": step["blocks"], "note": "Concurrent path" if step.get("parallel") else ("Alternative boot path" if step.get("branch") else "Ordered step")})
    ownership_items = []
    for section in sections:
        section_symbols = [symbol["id"] for symbol in symbols if symbol["section"] == section["id"]]
        ownership_items.append({"id": f"owners-{section['id']}", "step": str(len(section_symbols)), "title": section["name"], "description": section["summary"], "section": section["id"], "symbols": section_symbols, "note": "Assignments are intentionally unconfirmed until recorded in the source model."})
    protocol_items = []
    for protocol_id, protocol in model["protocols"].items():
        if protocol_id == "containment":
            continue
        links = [connection for connection in model["connections"] if connection["protocol"] == protocol_id]
        linked_symbols = list(dict.fromkeys(endpoint for connection in links for endpoint in (connection["from"], connection["to"])))
        protocol_items.append({"id": f"protocol-{protocol_id}", "step": str(len(links)), "title": protocol["name"], "description": protocol.get("purpose", "Architecture interface."), "section": "interconnect", "symbols": linked_symbols, "note": " → ".join(protocol.get("roles", []))})
    instruction_items = []
    for flow in model["instruction_flows"]:
        linked_symbols = list(dict.fromkeys(block_id for step in flow["steps"] for block_id in step["blocks"]))
        instruction_items.append({"id": f"instruction-{flow['id']}", "step": str(len(flow["steps"])), "title": flow["name"], "description": flow["summary"], "section": "compute", "symbols": linked_symbols, "note": "Architecture-level planned path"})
    return {
        "title": model["product"]["name"],
        "subtitle": "Source-linked structure, behavior, and ownership",
        "scopeHint": "Open architecture/soc.yaml or a linked Markdown specification to follow the model.",
        "atlas": model,
        "sections": sections,
        "symbols": symbols,
        "views": [
            {"id": "overview", "name": "Structure", "eyebrow": "Orientation", "title": "The SoC around Coral Core", "summary": "Enter a block to understand its role, maturity, evidence, and interfaces.", "items": overview_items},
            {"id": "boot", "name": "Boot flow", "eyebrow": "Behavior", "title": boot["name"], "summary": boot["summary"], "items": boot_items},
            {"id": "protocols", "name": "Protocols", "eyebrow": "Interfaces", "title": "Protocols and signal paths", "summary": "Decode the interface colors, endpoint roles, and instantiated links used by the architecture.", "items": protocol_items},
            {"id": "instructions", "name": "Instructions", "eyebrow": "Behavior", "title": "Instruction execution", "summary": "Follow planned scalar and vector instruction families from fetch through execution and writeback.", "items": instruction_items},
            {"id": "ownership", "name": "Ownership", "eyebrow": "Coordination", "title": "Make unassigned work visible", "summary": "Ownership and maturity live with the architecture; tasks remain in linked issues.", "items": ownership_items},
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    model = load_model(MODEL_PATH)
    errors = validate(model, ROOT)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    site_model = "window.__CORAL_MODEL__ = " + json.dumps(model, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c") + ";\n"
    (SITE / "model.js").write_text(site_model, encoding="utf-8")

    write_json(EXTENSION / "model.json", extension_model(model))
    media = EXTENSION / "media"
    media.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SITE / "app.js", media / "app.js")
    shutil.copy2(SITE / "styles.css", media / "styles.css")
    shutil.copy2(SITE / "index.html", media / "index.html")
    if (media / "assets").exists():
        shutil.rmtree(media / "assets")
    shutil.copytree(SITE / "assets", media / "assets")

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    for name in ("index.html", "styles.css", "app.js", "model.js"):
        shutil.copy2(SITE / name, DIST / name)
    shutil.copytree(SITE / "assets", DIST / "assets")
    shutil.copytree(ROOT / "docs", DIST / "docs")
    shutil.copytree(ROOT / "architecture", DIST / "architecture")
    for name in ("SoC_arch.png", "SoC presentation.pdf", "reef_spec.pdf", "reef_spec2.pdf"):
        shutil.copy2(ROOT / name, DIST / name)

    print(f"Built {len(model['blocks'])} blocks into {DIST.relative_to(ROOT)}/ and {EXTENSION.relative_to(ROOT)}/model.json")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
