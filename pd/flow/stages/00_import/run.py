"""Import stage: check the RTL handoff package and copy what later stages use.

Refuses a package whose schema version is not 1, whose manifest says dirty,
whose lint did not pass, whose checksums do not match, or that declares a
macro the design does not list. Copies filelist, RTL and SDC into results/.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_package(pkg: Path, design_macros: set[str]) -> list[str]:
    """Problems that make a package unusable. Empty means importable."""
    errs: list[str] = []
    manifest_path = pkg / "manifest.yaml"
    if not manifest_path.is_file():
        return ["no manifest.yaml"]
    m: dict[str, Any] = yaml.safe_load(manifest_path.read_text())
    if m.get("schema_version") != SCHEMA_VERSION:
        errs.append(
            f"schema_version {m.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    if m.get("dirty") is not False:
        errs.append("package was exported from a dirty tree")
    if m.get("checks", {}).get("lint") != "pass":
        errs.append(f"lint is {m.get('checks', {}).get('lint')!r}, not pass")
    for entry in m.get("files", []):
        f = pkg / entry["path"]
        if not f.is_file():
            errs.append(f"missing {entry['path']}")
        elif sha256(f) != entry["sha256"]:
            errs.append(f"sha256 mismatch: {entry['path']}")
    macros_path = pkg / "macros" / "macros.yaml"
    if macros_path.is_file():
        declared = {
            x["name"]
            for x in (yaml.safe_load(macros_path.read_text()) or {}).get(
                "macros", []
            )
        }
        for name in sorted(declared - design_macros):
            errs.append(
                f"macro {name} declared by RTL but not in the design's macros"
            )
    return errs


def main() -> int:
    """Stage entry point. The driver sets REEF_CONFIG and REEF_STAGE_DIR."""
    cfg = json.loads(Path(os.environ["REEF_CONFIG"]).read_text())["config"]
    stage = Path(os.environ["REEF_STAGE_DIR"])
    results = stage / "results"
    pkg = Path(cfg["pd_root"]) / cfg["handoff"]
    design_macros = {m["name"] for m in cfg.get("macros", [])}

    errs = check_package(pkg, design_macros)
    gate = {"package_valid": not errs}
    (results / "gate.json").write_text(json.dumps(gate) + "\n")
    if errs:
        for e in errs:
            print(f"import: {e}")
        return 1

    manifest = yaml.safe_load((pkg / "manifest.yaml").read_text())
    if manifest["top"] != cfg["top"]:
        print(
            f"import: package top {manifest['top']} != design top {cfg['top']}"
        )
        return 1
    shutil.copytree(pkg / "rtl", results / "rtl")
    shutil.copy2(pkg / "filelist.f", results / "filelist.f")
    shutil.copy2(pkg / "constraints" / "design.sdc", results / "design.sdc")
    shutil.copy2(pkg / "manifest.yaml", results / "handoff_manifest.yaml")
    (stage / "reports" / "import.txt").write_text(
        f"package {pkg}\nrtl sha {manifest['git_sha']}\n"
        f"files {len(manifest['files'])}\nclocks {manifest['clocks']}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
