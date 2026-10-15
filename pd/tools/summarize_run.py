"""Summarize a run dir, one line per stage, as summary.json and summary.md.

Usage: ``uv run python tools/summarize_run.py runs/<target>/<run_id>``.
Reads only each stage's results/DONE.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(run_dir: Path) -> list[dict[str, object]]:
    """One record per stage directory, in stage order."""
    rows = []
    for stage in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        done = stage / "results" / "DONE.json"
        if done.is_file():
            d = json.loads(done.read_text())
            rows.append(
                {
                    "stage": stage.name,
                    "status": d["status"],
                    "wall_s": d["wall_s"],
                    "gate_failed": d.get("gate_failed", []),
                }
            )
        else:
            rows.append(
                {
                    "stage": stage.name,
                    "status": "not run",
                    "wall_s": 0,
                    "gate_failed": [],
                }
            )
    return rows


def main(argv: list[str]) -> int:
    """CLI entry point."""
    if len(argv) != 1:
        print(__doc__)
        return 2
    run_dir = Path(argv[0])
    rows = summarize(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    md = ["| stage | status | wall s | gate failed |", "|---|---|---:|---|"]
    for r in rows:
        failed = ", ".join(r["gate_failed"])
        md.append(
            f"| {r['stage']} | {r['status']} | {r['wall_s']} | {failed} |"
        )
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
