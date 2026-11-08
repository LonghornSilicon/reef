#!/usr/bin/env python3
"""Validate Coral's editable architecture model and its local evidence links."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

MATURITY = {"proposal", "specified", "in-progress", "implemented", "verified"}


def load_model(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("model root must be a mapping")
    return data


def duplicate_ids(items: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        item_id = item.get("id")
        if item_id in seen:
            duplicates.add(str(item_id))
        seen.add(item_id)
    return duplicates


def validate(model: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    required_top = {"schema_version", "product", "owners", "protocols", "instruction_flows", "layouts", "ports", "blocks", "connections", "behaviors"}
    missing = sorted(required_top - model.keys())
    if missing:
        errors.append(f"model: missing top-level fields: {', '.join(missing)}")
        return errors
    if model["schema_version"] != 1:
        errors.append("model.schema_version: expected 1")

    owners = model.get("owners", [])
    blocks = model.get("blocks", [])
    connections = model.get("connections", [])
    behaviors = model.get("behaviors", [])
    protocols = model.get("protocols", {})
    instruction_flows = model.get("instruction_flows", [])
    layouts = model.get("layouts", {})
    ports = model.get("ports", {})
    for label, items in (("owners", owners), ("blocks", blocks), ("connections", connections), ("behaviors", behaviors)):
        if not isinstance(items, list):
            errors.append(f"{label}: must be a list")
            continue
        for duplicate in duplicate_ids(items):
            errors.append(f"{label}: duplicate id {duplicate!r}")

    owner_ids = {item.get("id") for item in owners}
    block_ids = {item.get("id") for item in blocks}
    protocol_ids = set(protocols)
    default_focus = model.get("product", {}).get("default_focus")
    if default_focus not in block_ids:
        errors.append(f"product.default_focus: unknown block {default_focus!r}")

    required_block = {"id", "name", "kind", "summary", "parent", "owner", "maturity", "rank", "order", "responsibilities", "documents", "issues", "rtl"}
    for index, block in enumerate(blocks):
        label = f"blocks[{index}]"
        missing_fields = sorted(required_block - block.keys())
        if missing_fields:
            errors.append(f"{label}: missing {', '.join(missing_fields)}")
        if block.get("parent") is not None and block.get("parent") not in block_ids:
            errors.append(f"{label}.parent: unknown block {block.get('parent')!r}")
        if block.get("parent") == block.get("id"):
            errors.append(f"{label}.parent: block cannot contain itself")
        if block.get("owner") not in owner_ids:
            errors.append(f"{label}.owner: unknown owner {block.get('owner')!r}")
        if block.get("maturity") not in MATURITY:
            errors.append(f"{label}.maturity: expected one of {sorted(MATURITY)}")
        for field in ("rank", "order"):
            if not isinstance(block.get(field), int) or block[field] < 0:
                errors.append(f"{label}.{field}: expected a non-negative integer")
        if not isinstance(block.get("responsibilities"), list) or not block.get("responsibilities"):
            errors.append(f"{label}.responsibilities: expected at least one responsibility")
        for doc_index, document in enumerate(block.get("documents", [])):
            if not isinstance(document, dict) or not document.get("title"):
                errors.append(f"{label}.documents[{doc_index}]: expected a titled document mapping")
                continue
            if document.get("path"):
                document_path = root / document["path"]
                if document_path.is_absolute() and root not in document_path.parents:
                    errors.append(f"{label}.documents[{doc_index}]: path must stay in repository")
                elif not document_path.is_file():
                    errors.append(f"{label}.documents[{doc_index}]: missing file {document['path']!r}")
            elif not document.get("url"):
                errors.append(f"{label}.documents[{doc_index}]: expected path or url")
        for rtl_index, reference in enumerate(block.get("rtl", [])):
            required_rtl = {"repository", "revision", "path", "symbol", "url"}
            if not isinstance(reference, dict) or not required_rtl.issubset(reference):
                errors.append(f"{label}.rtl[{rtl_index}]: expected repository, revision, path, symbol, and url")
        for issue_index, issue in enumerate(block.get("issues", [])):
            if not isinstance(issue, dict) or not issue.get("url"):
                errors.append(f"{label}.issues[{issue_index}]: expected a mapping with a url and optional title")

    for block_id in block_ids:
        seen: set[str] = set()
        current = block_id
        while current is not None:
            if current in seen:
                errors.append(f"blocks: containment cycle includes {block_id!r}")
                break
            seen.add(current)
            current_block = next((item for item in blocks if item.get("id") == current), None)
            current = current_block.get("parent") if current_block else None

    children_by_parent: dict[str, set[str]] = {}
    for block in blocks:
        parent = block.get("parent")
        if parent is not None:
            children_by_parent.setdefault(parent, set()).add(block.get("id"))

    for parent_id, child_ids in children_by_parent.items():
        if parent_id not in layouts:
            errors.append(f"layouts: composite block {parent_id!r} has no authored layout")
            continue
        layout = layouts[parent_id]
        canvas = layout.get("canvas", {}) if isinstance(layout, dict) else {}
        nodes = layout.get("nodes", {}) if isinstance(layout, dict) else {}
        width = canvas.get("width")
        height = canvas.get("height")
        if not isinstance(width, (int, float)) or width <= 0 or not isinstance(height, (int, float)) or height <= 0:
            errors.append(f"layouts.{parent_id}.canvas: width and height must be positive numbers")
            continue
        missing_children = sorted(child_ids - set(nodes))
        if missing_children:
            errors.append(f"layouts.{parent_id}.nodes: missing children {', '.join(missing_children)}")
        for node_id, position in nodes.items():
            if node_id not in block_ids:
                errors.append(f"layouts.{parent_id}.nodes: unknown block {node_id!r}")
                continue
            if not isinstance(position, dict):
                errors.append(f"layouts.{parent_id}.nodes.{node_id}: expected a position mapping")
                continue
            values = [position.get(field) for field in ("x", "y", "width", "height")]
            if not all(isinstance(value, (int, float)) for value in values):
                errors.append(f"layouts.{parent_id}.nodes.{node_id}: x, y, width, and height must be numbers")
                continue
            x, y, node_width, node_height = values
            if x < 0 or y < 0 or node_width <= 0 or node_height <= 0:
                errors.append(f"layouts.{parent_id}.nodes.{node_id}: position and size must be positive")
            if x + node_width > width or y + node_height > height:
                errors.append(f"layouts.{parent_id}.nodes.{node_id}: block falls outside the canvas")

    for block_id, block_ports in ports.items():
        if block_id not in block_ids:
            errors.append(f"ports: unknown block {block_id!r}")
            continue
        if not isinstance(block_ports, dict):
            errors.append(f"ports.{block_id}: expected a mapping")
            continue
        for port_id, port in block_ports.items():
            if not isinstance(port, dict):
                errors.append(f"ports.{block_id}.{port_id}: expected a mapping")
                continue
            missing_port = {"label", "protocol", "side", "offset"} - port.keys()
            if missing_port:
                errors.append(f"ports.{block_id}.{port_id}: missing {', '.join(sorted(missing_port))}")
            if not str(port.get("label", "")).strip():
                errors.append(f"ports.{block_id}.{port_id}.label: expected a visible label")
            if port.get("protocol") not in protocol_ids:
                errors.append(f"ports.{block_id}.{port_id}.protocol: unknown protocol {port.get('protocol')!r}")
            if port.get("side") not in {"left", "right", "top", "bottom"}:
                errors.append(f"ports.{block_id}.{port_id}.side: expected left, right, top, or bottom")
            if not isinstance(port.get("offset"), (int, float)) or not 0 <= port.get("offset", -1) <= 1:
                errors.append(f"ports.{block_id}.{port_id}.offset: expected a number from 0 to 1")

    required_connection = {"id", "from", "from_port", "to", "to_port", "protocol", "label", "bidirectional"}
    for index, connection in enumerate(connections):
        label = f"connections[{index}]"
        missing_fields = sorted(required_connection - connection.keys())
        if missing_fields:
            errors.append(f"{label}: missing {', '.join(missing_fields)}")
        if connection.get("from") not in block_ids:
            errors.append(f"{label}.from: unknown block {connection.get('from')!r}")
        if connection.get("to") not in block_ids:
            errors.append(f"{label}.to: unknown block {connection.get('to')!r}")
        if connection.get("protocol") not in protocol_ids:
            errors.append(f"{label}.protocol: unknown protocol {connection.get('protocol')!r}")
        for endpoint, port_field in (("from", "from_port"), ("to", "to_port")):
            block_id = connection.get(endpoint)
            port_id = connection.get(port_field)
            port = ports.get(block_id, {}).get(port_id)
            if port is None:
                errors.append(f"{label}.{port_field}: unknown port {block_id!r}.{port_id!r}")
            elif port.get("protocol") != connection.get("protocol"):
                errors.append(f"{label}.{port_field}: port protocol does not match connection protocol")

    for behavior_index, behavior in enumerate(behaviors):
        if not behavior.get("steps"):
            errors.append(f"behaviors[{behavior_index}].steps: expected at least one step")
        for step_index, step in enumerate(behavior.get("steps", [])):
            for block_id in step.get("blocks", []):
                if block_id not in block_ids:
                    errors.append(f"behaviors[{behavior_index}].steps[{step_index}]: unknown block {block_id!r}")

    if not isinstance(instruction_flows, list) or not instruction_flows:
        errors.append("instruction_flows: expected at least one instruction path")
    else:
        for duplicate in duplicate_ids(instruction_flows):
            errors.append(f"instruction_flows: duplicate id {duplicate!r}")
        for flow_index, flow in enumerate(instruction_flows):
            if not flow.get("name") or not flow.get("summary") or not flow.get("steps"):
                errors.append(f"instruction_flows[{flow_index}]: expected name, summary, and at least one step")
            for step_index, step in enumerate(flow.get("steps", [])):
                if not step.get("title") or not step.get("detail") or not step.get("blocks"):
                    errors.append(f"instruction_flows[{flow_index}].steps[{step_index}]: expected title, detail, and blocks")
                if step.get("protocol") not in protocol_ids:
                    errors.append(f"instruction_flows[{flow_index}].steps[{step_index}].protocol: unknown protocol {step.get('protocol')!r}")
                for block_id in step.get("blocks", []):
                    if block_id not in block_ids:
                        errors.append(f"instruction_flows[{flow_index}].steps[{step_index}]: unknown block {block_id!r}")

    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=Path("architecture/soc.yaml"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    model_path = args.model if args.model.is_absolute() else root / args.model
    try:
        model = load_model(model_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: cannot load {model_path}: {error}", file=sys.stderr)
        return 1
    errors = validate(model, root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"Validation failed with {len(errors)} error(s).")
        return 1
    print(f"Validated {len(model['blocks'])} blocks, {len(model['connections'])} connections, and {len(model['behaviors'])} behaviors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
