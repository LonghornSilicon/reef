"""Resolve the run config: tech, then design, then overrides. Later wins.

Writes run_config.json, the one file stages read, plus a flat run_config.tcl
for Tcl stages. Every key records the layer it came from.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema
import yaml

PD = Path(__file__).resolve().parent.parent
SCHEMA = PD / "flow" / "config.schema.json"


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one yaml file as a dict, empty if the file is empty."""
    return yaml.safe_load(path.read_text()) or {}


def read_cells(path: Path) -> list[str]:
    """One cell or glob per line, comments and blanks dropped."""
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def load_tech(tech_dir: Path) -> dict[str, Any]:
    """platform.yaml plus dont_use.txt. dont_use_file stays a path for synth."""
    cfg = load_yaml(tech_dir / "platform.yaml")
    cfg["dont_use"] = []
    if (tech_dir / "dont_use.txt").is_file():
        cfg["dont_use"] = read_cells(tech_dir / "dont_use.txt")
    return cfg


def merge(
    base: dict[str, Any],
    layer: dict[str, Any],
    name: str,
    sources: dict[str, str],
    prefix: str = "",
) -> None:
    """Deep-merge layer into base, recording the layer name per leaf key."""
    for key, value in layer.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            if not isinstance(base.get(key), dict):
                base[key] = {}
            merge(base[key], value, name, sources, f"{path}.")
        else:
            base[key] = value
            sources[path] = name


def expand_env(value: object) -> object:
    """Replace ${VAR} in strings, recursively. Unset variables are an error."""
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            var = m.group(1)
            if var not in os.environ:
                msg = f"config references ${{{var}}} but it is not set"
                raise SystemExit(msg)
            return os.environ[var]

        return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", sub, value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def to_tcl(cfg: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten to `set cfg(a.b) value` lines. Lists become Tcl lists."""
    lines = []
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            lines.extend(to_tcl(value, f"{path}."))
        elif isinstance(value, list):
            items = " ".join(f"{{{v}}}" for v in value)
            lines.append(f"set cfg({path}) [list {items}]")
        elif isinstance(value, bool):
            lines.append(f"set cfg({path}) {int(value)}")
        else:
            lines.append(f"set cfg({path}) {{{value}}}")
    return lines


def resolve(
    tech_dir: Path, design: Path, override: Path | None, run_id: str
) -> dict[str, Any]:
    """Merge the layers and validate the result."""
    cfg: dict[str, Any] = {}
    sources: dict[str, str] = {}
    merge(cfg, load_tech(tech_dir), f"tech:{tech_dir.name}", sources)
    merge(cfg, load_yaml(design), f"design:{design.parent.name}", sources)
    if override is not None:
        merge(cfg, load_yaml(override), f"override:{override.name}", sources)
    cfg = expand_env(cfg)
    cfg["run_id"] = run_id
    cfg["pd_root"] = str(PD)
    sources["run_id"] = sources["pd_root"] = "flow"
    schema = json.loads(SCHEMA.read_text())
    jsonschema.validate(cfg, schema)
    return {"config": cfg, "sources": sources}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tech", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--override", type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    resolved = resolve(args.tech, args.design, args.override, args.run_id)
    args.out.write_text(json.dumps(resolved, indent=2) + "\n")
    tcl = args.out.with_suffix(".tcl")
    tcl.write_text("\n".join(to_tcl(resolved["config"])) + "\n")
    for key, src in sorted(resolved["sources"].items()):
        print(f"{key:40s} {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
