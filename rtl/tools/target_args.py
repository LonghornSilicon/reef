"""Print the defines and models of a target as tool arguments.

Usage: ``uv run python tools/target_args.py <target> [--tool verilator|slang]``.
check.sh, the Makefile and export_handoff.py use it so every tool sees the
target's defines. Models are lint and sim only: slang blackboxes them and the
export never copies them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

RTL = Path(__file__).resolve().parent.parent


def target_args(target: dict[str, Any], tool: str = "verilator") -> list[str]:
    """Defines and model files of a loaded target, in the tool's syntax."""
    models = list(target.get("models", []))
    if tool == "slang":
        args = [f"-D{d}" for d in target["defines"]] + models
        for m in models:
            args += ["--blackboxed-module", Path(m).stem]
        return args
    return [f"+define+{d}" for d in target["defines"]] + models


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target")
    ap.add_argument(
        "--tool", choices=["verilator", "slang"], default="verilator"
    )
    args = ap.parse_args(argv)
    path = RTL / "targets" / f"{args.target}.yaml"
    print(" ".join(target_args(yaml.safe_load(path.read_text()), args.tool)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
