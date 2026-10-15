"""The private guard fails on a planted file and passes on the clean tree."""

from __future__ import annotations

import subprocess
from pathlib import Path

PD = Path(__file__).resolve().parents[2]
GUARD = PD / "tools" / "check_no_private.sh"


def run_guard() -> subprocess.CompletedProcess[str]:
    """Run the guard from pd/."""
    return subprocess.run(
        ["bash", str(GUARD)], capture_output=True, text=True, cwd=PD
    )


def test_clean_tree_passes() -> None:
    """Nothing private is committed today."""
    proc = run_guard()
    assert proc.returncode == 0, proc.stdout


def test_planted_lib_fails() -> None:
    """A .lib forced past .gitignore is caught by the guard, then removed."""
    planted = PD / "tech" / "sky130" / "planted_cell.lib"
    planted.write_text("library (planted) {}\n")
    try:
        subprocess.run(
            ["git", "add", "-N", "-f", str(planted)], check=True, cwd=PD
        )
        proc = run_guard()
        assert proc.returncode == 1
        assert "planted_cell.lib" in proc.stdout
    finally:
        subprocess.run(["git", "reset", "-q", "--", str(planted)], cwd=PD)
        planted.unlink()
