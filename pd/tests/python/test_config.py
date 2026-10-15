"""Tests for flow/config.py: layering, provenance, env expansion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PD = Path(__file__).resolve().parents[2]
CONFIG = PD / "flow" / "config.py"


def resolve(out: Path, override: Path | None = None) -> dict:
    """Run config.py for toy_counter on sky130 and load the result."""
    cmd = [
        sys.executable,
        str(CONFIG),
        "--tech",
        str(PD / "tech" / "sky130"),
        "--design",
        str(PD / "designs" / "toy_counter" / "design.yaml"),
        "--run-id",
        "test",
        "--out",
        str(out),
    ]
    if override is not None:
        cmd += ["--override", str(override)]
    env = {**os.environ, "PDK_ROOT": "/pdk"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=PD)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_layers_and_sources(tmp_path: Path) -> None:
    """Design values win over tech, and every key names its layer."""
    r = resolve(tmp_path / "c.json")
    assert r["config"]["node"] == "sky130"
    assert r["config"]["top"] == "toy_counter"
    assert r["sources"]["site"] == "tech:sky130"
    assert r["sources"]["floorplan.utilization"] == "design:toy_counter"


def test_override_wins(tmp_path: Path) -> None:
    """A run override is the last layer."""
    ov = tmp_path / "ov.yaml"
    ov.write_text(yaml.safe_dump({"floorplan": {"utilization": 0.55}}))
    r = resolve(tmp_path / "c.json", ov)
    assert r["config"]["floorplan"]["utilization"] == 0.55
    assert r["sources"]["floorplan.utilization"] == "override:ov.yaml"
    assert r["sources"]["floorplan.aspect_ratio"] == "design:toy_counter"


def test_pdk_root_expanded(tmp_path: Path) -> None:
    """${PDK_ROOT} is expanded and no literal placeholder survives."""
    r = resolve(tmp_path / "c.json")
    assert r["config"]["lef"][0].startswith("/pdk/")
    assert "${" not in json.dumps(r["config"])
    assert (tmp_path / "c.tcl").read_text().startswith("set cfg(")
