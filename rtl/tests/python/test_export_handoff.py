"""Tests for export_handoff.py and validate_handoff.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

RTL = Path(__file__).resolve().parents[2]
EXAMPLE = RTL / "handoff" / "example"
EXPORT = RTL / "tools" / "export_handoff.py"
VALIDATE = RTL / "tools" / "validate_handoff.py"


def validate(*paths: Path) -> subprocess.CompletedProcess[str]:
    """Run validate_handoff.py on the given paths."""
    return subprocess.run(
        [sys.executable, str(VALIDATE), *map(str, paths)],
        capture_output=True,
        text=True,
    )


def test_targets_validate() -> None:
    """Every committed target passes the target check."""
    for target in sorted((RTL / "targets").glob("*.yaml")):
        proc = validate(target)
        assert proc.returncode == 0, proc.stdout


def test_bad_target_fails(tmp_path: Path) -> None:
    """A target missing a required key is rejected."""
    t = yaml.safe_load((RTL / "targets" / "toy_counter.yaml").read_text())
    del t["top"]
    bad = tmp_path / "toy_counter.yaml"
    bad.write_text(yaml.safe_dump(t))
    proc = validate(bad)
    assert proc.returncode == 1
    assert "missing keys" in proc.stdout


def test_missing_model_fails(tmp_path: Path) -> None:
    """A target listing a model file that does not exist is rejected."""
    t = yaml.safe_load((RTL / "targets" / "toy_counter.yaml").read_text())
    t["models"] = ["ext/no_such_model.v"]
    bad = tmp_path / "toy_counter.yaml"
    bad.write_text(yaml.safe_dump(t))
    proc = validate(bad)
    assert proc.returncode == 1
    assert "model ext/no_such_model.v does not exist" in proc.stdout


def test_example_validates() -> None:
    """The committed example package passes validation."""
    proc = validate(EXAMPLE)
    assert proc.returncode == 0, proc.stdout


def test_tampered_example_fails(tmp_path: Path) -> None:
    """Changing one RTL byte is caught by the checksums."""
    pkg = tmp_path / "pkg"
    shutil.copytree(EXAMPLE, pkg)
    sdc = pkg / "constraints" / "design.sdc"
    sdc.write_text(sdc.read_text().replace("20.0", "10.0"))
    proc = validate(pkg)
    assert proc.returncode == 1
    assert "sha256 mismatch" in proc.stdout


@pytest.mark.skipif(
    shutil.which("verilator") is None
    or shutil.which("verible-verilog-lint") is None,
    reason="needs verilator and verible on PATH (run in the Docker image)",
)
def test_toy_export_validates(tmp_path: Path) -> None:
    """A fresh toy_counter export passes the same validation as the example."""
    proc = subprocess.run(
        [
            sys.executable,
            str(EXPORT),
            "toy_counter",
            "--allow-dirty",
            "--out",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=RTL,
    )
    assert proc.returncode == 0, proc.stderr
    pkg = Path(proc.stdout.strip())
    assert validate(pkg).returncode == 0
    manifest = yaml.safe_load((pkg / "manifest.yaml").read_text())
    assert manifest["checks"] == {"lint": "pass", "sim_smoke": "pass"}
