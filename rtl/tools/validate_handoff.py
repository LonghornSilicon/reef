"""Validate handoff packages and target files.

Usage: ``uv run python tools/validate_handoff.py <path>...``. A directory is
checked as a handoff package (manifest against the schema, checksums, file
list). A ``.yaml`` file is checked as a ``targets/`` entry. Exit 1 on the
first failing path, after printing every problem found in it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import yaml

RTL = Path(__file__).resolve().parent.parent
SCHEMA = RTL / "handoff" / "schema" / "handoff.schema.json"
STATUSES = {"smoke", "experimental", "candidate", "frozen"}
TECH_HINTS = {"open", "sky130", "private-node"}
TARGET_KEYS = {
    "name",
    "top",
    "description",
    "filelist",
    "defines",
    "include_dirs",
    "constraints",
    "clocks",
    "macros",
    "technology_hint",
    "status",
    "owner",
}


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filelist_is_placeholder(path: Path) -> bool:
    """True if a .f file has only comments and blank lines."""
    return not any(
        line.strip() and not line.strip().startswith("//")
        for line in path.read_text().splitlines()
    )


def macro_catalog() -> set[str]:
    """Every macro name defined under macros/*.yaml."""
    names: set[str] = set()
    for path in (RTL / "macros").glob("*.yaml"):
        names.update((yaml.safe_load(path.read_text()) or {}).keys())
    return names


def check_target(path: Path) -> list[str]:
    """Problems with a targets/<name>.yaml file."""
    errs: list[str] = []
    t: dict[str, Any] = yaml.safe_load(path.read_text())
    missing = TARGET_KEYS - t.keys()
    extra = t.keys() - TARGET_KEYS - {"io_map", "models"}
    if missing:
        errs.append(f"missing keys: {sorted(missing)}")
    if extra:
        errs.append(f"unknown keys: {sorted(extra)}")
    if missing:
        return errs
    if t["name"] != path.stem:
        errs.append(f"name {t['name']!r} != file name {path.stem!r}")
    if t["status"] not in STATUSES:
        errs.append(f"status {t['status']!r} not in {sorted(STATUSES)}")
    if t["technology_hint"] not in TECH_HINTS:
        hint = t["technology_hint"]
        errs.append(f"technology_hint {hint!r} not in {sorted(TECH_HINTS)}")
    for key in ("filelist", "constraints"):
        if not (RTL / t[key]).is_file():
            errs.append(f"{key} {t[key]} does not exist")
    for m in t.get("models", []):
        if not (RTL / m).is_file():
            errs.append(f"model {m} does not exist")
    for d in t["include_dirs"]:
        if not (RTL / d).is_dir():
            errs.append(f"include_dir {d} does not exist")
    for c in t["clocks"]:
        if set(c) != {"name", "port", "period_ns"}:
            errs.append(f"clock {c} needs exactly name, port, period_ns")
    unknown = set(t["macros"]) - macro_catalog()
    if unknown:
        errs.append(f"macros not under macros/: {sorted(unknown)}")
    flist = RTL / t["filelist"]
    if t["status"] != "experimental" and flist.is_file():
        if filelist_is_placeholder(flist):
            errs.append("placeholder filelist but status is not experimental")
        if not t["clocks"]:
            errs.append("no clocks but status is not experimental")
    return errs


def check_package(pkg: Path) -> list[str]:
    """Problems with a handoff package directory."""
    errs: list[str] = []
    manifest_path = pkg / "manifest.yaml"
    if not manifest_path.is_file():
        return ["no manifest.yaml"]
    manifest = yaml.safe_load(manifest_path.read_text())
    schema = json.loads(SCHEMA.read_text())
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errs.extend(
        f"manifest: {e.json_path}: {e.message}"
        for e in sorted(validator.iter_errors(manifest), key=str)
    )
    if errs:
        return errs

    listed = {f["path"]: f["sha256"] for f in manifest["files"]}
    on_disk = {
        str(p.relative_to(pkg))
        for p in pkg.rglob("*")
        if p.is_file() and p.name not in {"manifest.yaml", "CHECKSUMS.sha256"}
    }
    for path in sorted(on_disk - listed.keys()):
        errs.append(f"file not in manifest: {path}")
    for path, digest in sorted(listed.items()):
        f = pkg / path
        if not f.is_file():
            errs.append(f"manifest lists missing file: {path}")
        elif sha256(f) != digest:
            errs.append(f"sha256 mismatch: {path}")

    checksums = pkg / "CHECKSUMS.sha256"
    if not checksums.is_file():
        errs.append("no CHECKSUMS.sha256")
    else:
        seen: set[str] = set()
        for line in checksums.read_text().splitlines():
            digest, _, rel = line.partition("  ")
            seen.add(rel)
            f = pkg / rel
            if not f.is_file():
                errs.append(f"CHECKSUMS lists missing file: {rel}")
            elif sha256(f) != digest:
                errs.append(f"CHECKSUMS mismatch: {rel}")
        for rel in sorted((on_disk | {"manifest.yaml"}) - seen):
            errs.append(f"file not in CHECKSUMS: {rel}")

    for required in (
        "filelist.f",
        "constraints/design.sdc",
        "io/io_map.yaml",
        "macros/macros.yaml",
        "reports/lint.txt",
        "reports/sim_smoke.txt",
    ):
        if not (pkg / required).is_file():
            errs.append(f"missing {required}")
    flist = pkg / "filelist.f"
    if flist.is_file():
        for line in flist.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("+incdir+"):
                continue
            if not (pkg / line).is_file():
                errs.append(f"filelist.f entry missing: {line}")
    return errs


def main(argv: list[str]) -> int:
    """CLI entry point."""
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        path = Path(arg)
        if path.is_dir():
            errs = check_package(path)
        elif path.suffix == ".yaml":
            errs = check_target(path)
        else:
            errs = ["not a package directory or target yaml"]
        if errs:
            print(f"FAIL {path}")
            for e in errs:
                print(f"  {e}")
            return 1
        print(f"ok {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
