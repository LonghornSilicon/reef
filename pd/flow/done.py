"""Write a stage's DONE.json from its exit code and results/gate.json.

status is ok when the script exited 0 and every gate check passed, stub when
the stage script is still a placeholder. With gates.strict false, failed gate
checks are recorded but the status stays ok, so the next stage can run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def tool_versions() -> dict[str, str]:
    """First line of each tool's version output, or missing."""
    cmds = {
        "yosys": ["yosys", "-V"],
        "openroad": ["openroad", "-version"],
        "magic": ["magic", "--version"],
    }
    out = {}
    for name, cmd in cmds.items():
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            first = (proc.stdout or proc.stderr).strip().splitlines()
            out[name] = first[0] if first else "unknown"
        except FileNotFoundError:
            out[name] = "missing"
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--rc", type=int, required=True)
    ap.add_argument("--wall", type=int, required=True)
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text())["config"]
    gate_path = args.out / "results" / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
    failed = sorted(k for k, v in gate.items() if v is False)
    strict = cfg.get("gates", {}).get("strict", True)
    if gate.get("implemented") is False:
        status = "stub"
    elif args.rc != 0:
        status = "error"
    elif failed and strict:
        status = "gate_fail"
    else:
        status = "ok"
    done = {
        "stage": args.stage,
        "status": status,
        "exit_code": args.rc,
        "wall_s": args.wall,
        "gate": gate,
        "gate_failed": failed,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "tool_versions": tool_versions(),
    }
    (args.out / "results" / "DONE.json").write_text(
        json.dumps(done, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
