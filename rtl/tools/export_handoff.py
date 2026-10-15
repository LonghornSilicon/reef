"""Build an RTL-to-PD handoff package for one synthesis target.

Usage: ``uv run python tools/export_handoff.py <target> [--allow-dirty]``.
Writes ``out/<target>/<sha8>/`` as described in ``docs/handoff-contract.md``.
Run from inside the Docker image; it calls verilator and verible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

RTL = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1


def run(cmd: list[str], cwd: Path = RTL) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output without raising."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def git(*args: str, cwd: Path = RTL) -> str:
    """Run git and return stripped stdout, raising on failure."""
    proc = run(["git", *args], cwd=cwd)
    if proc.returncode != 0:
        msg = f"git {' '.join(args)}: {proc.stderr.strip()}"
        raise RuntimeError(msg)
    return proc.stdout.strip()


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_target(name: str) -> dict[str, Any]:
    """Read targets/<name>.yaml."""
    path = RTL / "targets" / f"{name}.yaml"
    if not path.is_file():
        msg = f"no such target: {path}"
        raise FileNotFoundError(msg)
    return yaml.safe_load(path.read_text())


def parse_filelist(path: Path) -> tuple[list[str], list[str]]:
    """Split a .f file into (include dirs, source files), rtl-relative."""
    incdirs: list[str] = []
    files: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("+incdir+"):
            incdirs.append(line.removeprefix("+incdir+"))
        else:
            files.append(line)
    return incdirs, files


def tree_dirty() -> bool:
    """True if rtl/ has uncommitted changes."""
    return bool(git("status", "--porcelain", "--", "."))


def tool_version(tool: str, flag: str = "--version") -> str:
    """First line of a tool's version output, or 'missing'."""
    try:
        proc = run([tool, flag])
    except FileNotFoundError:
        return "missing"
    first = (proc.stdout or proc.stderr).strip().splitlines()
    return first[0] if first else "unknown"


def lint(
    target: dict[str, Any], filelist: Path, sources: list[str]
) -> tuple[str, str]:
    """Run verilator and verible lint. Return (status, report text)."""
    parts: list[str] = []
    ok = True
    defines = [f"+define+{d}" for d in target["defines"]]
    v = run(
        [
            "verilator",
            "--lint-only",
            "-Wall",
            *defines,
            "-f",
            str(filelist),
            "--top-module",
            target["top"],
        ]
    )
    parts.append(
        f"$ verilator --lint-only -Wall -f {filelist.name} "
        f"--top-module {target['top']}\nexit {v.returncode}\n{v.stderr}"
    )
    ok &= v.returncode == 0
    vl = run(
        ["verible-verilog-lint", "--rules_config=.rules.verible_lint", *sources]
    )
    parts.append(
        f"$ verible-verilog-lint {' '.join(sources)}\n"
        f"exit {vl.returncode}\n{vl.stdout}{vl.stderr}"
    )
    ok &= vl.returncode == 0
    return ("pass" if ok else "fail"), "\n".join(parts)


def sim_smoke(target: dict[str, Any]) -> tuple[str, str]:
    """Run <top>_tb.sv through make sim if one exists under src/."""
    tbs = list((RTL / "src").rglob(f"{target['top']}_tb.sv"))
    if not tbs:
        return "none", f"no {target['top']}_tb.sv under src/\n"
    proc = run(["make", "sim", f"TARGET={target['name']}"])
    status = "pass" if proc.returncode == 0 else "fail"
    text = (
        f"$ make sim TARGET={target['name']}\nexit {proc.returncode}\n"
        f"{proc.stdout}{proc.stderr}"
    )
    return status, text


def macro_entries(names: list[str]) -> list[dict[str, Any]]:
    """Look up macro names in macros/*.yaml. No catalog exists yet."""
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted((RTL / "macros").glob("*.yaml")):
        catalog.update(yaml.safe_load(path.read_text()) or {})
    out = []
    for name in names:
        if name not in catalog:
            msg = f"macro {name} not found under macros/"
            raise KeyError(msg)
        entry = catalog[name]
        out.append(
            {
                "name": name,
                "role": entry["role"],
                "blackbox": entry["blackbox"],
                "views_pd_must_supply": entry["views_pd_must_supply"],
            }
        )
    return out


def export(name: str, out_root: Path, *, allow_dirty: bool) -> Path:
    """Build the package and return its directory."""
    target = load_target(name)
    dirty = tree_dirty()
    if dirty and not allow_dirty:
        msg = "working tree is dirty; commit or pass --allow-dirty"
        raise SystemExit(msg)
    sha = git("rev-parse", "HEAD")
    pkg = out_root / name / sha[:8]
    if pkg.exists():
        shutil.rmtree(pkg)

    filelist = RTL / target["filelist"]
    incdirs, sources = parse_filelist(filelist)

    for src in sources:
        dst = pkg / "rtl" / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RTL / src, dst)
    for inc in incdirs:
        for hdr in sorted((RTL / inc).glob("*.svh")):
            rel = str(hdr.relative_to(RTL))
            dst = pkg / "rtl" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hdr, dst)

    lines = [f"+incdir+rtl/{d}" for d in incdirs] + [
        f"rtl/{s}" for s in sources
    ]
    (pkg / "filelist.f").write_text("\n".join(lines) + "\n")

    (pkg / "constraints").mkdir()
    shutil.copy2(
        RTL / target["constraints"], pkg / "constraints" / "design.sdc"
    )

    (pkg / "io").mkdir()
    io_map = {"ports": target.get("io_map", [])}
    (pkg / "io" / "io_map.yaml").write_text(
        yaml.safe_dump(io_map, sort_keys=False)
    )

    (pkg / "macros" / "blackbox").mkdir(parents=True)
    macros = macro_entries(target["macros"])
    (pkg / "macros" / "macros.yaml").write_text(
        yaml.safe_dump({"macros": macros}, sort_keys=False)
    )
    for m in macros:
        if m["blackbox"] and m["blackbox"] != "TBD":
            shutil.copy2(RTL / m["blackbox"], pkg / "macros" / "blackbox")

    (pkg / "reports").mkdir()
    lint_status, lint_text = lint(target, filelist, sources)
    (pkg / "reports" / "lint.txt").write_text(lint_text)
    sim_status, sim_text = sim_smoke(target)
    (pkg / "reports" / "sim_smoke.txt").write_text(sim_text)

    files = [
        {"path": str(p.relative_to(pkg)), "sha256": sha256(p)}
        for p in sorted(pkg.rglob("*"))
        if p.is_file()
    ]
    now = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target": name,
        "top": target["top"],
        "git_sha": sha,
        "dirty": dirty,
        "created_utc": now,
        "defines": list(target["defines"]),
        "clocks": list(target["clocks"]),
        "technology_hint": target["technology_hint"],
        "checks": {"lint": lint_status, "sim_smoke": sim_status},
        "tool_versions": {
            "verilator": tool_version("verilator"),
            "verible": tool_version("verible-verilog-lint"),
            "yosys": tool_version("yosys", "-V"),
        },
        "files": files,
    }
    (pkg / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )

    checksums = [
        f"{sha256(p)}  {p.relative_to(pkg)}"
        for p in sorted(pkg.rglob("*"))
        if p.is_file()
    ]
    (pkg / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n")
    return pkg


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", type=Path, default=RTL / "out")
    args = ap.parse_args(argv)
    pkg = export(args.target, args.out, allow_dirty=args.allow_dirty)
    print(pkg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
