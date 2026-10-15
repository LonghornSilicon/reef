"""Tests for the import stage against the fixture package."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

PD = Path(__file__).resolve().parents[2]
FIXTURE = PD / "tests" / "fixtures" / "handoff_example"
EXPECTED_SCHEMA_VERSION = 1


def load_import() -> object:
    """Import flow/stages/00_import/run.py as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "import_stage", PD / "flow" / "stages" / "00_import" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pkg(tmp_path: Path) -> Path:
    """A writable copy of the fixture package."""
    dst = tmp_path / "pkg"
    shutil.copytree(FIXTURE, dst)
    return dst


def edit_manifest(pkg: Path, **fields: object) -> None:
    """Rewrite top-level manifest fields."""
    path = pkg / "manifest.yaml"
    m = yaml.safe_load(path.read_text())
    m.update(fields)
    path.write_text(yaml.safe_dump(m, sort_keys=False))


def rehash(pkg: Path, rel: str) -> None:
    """Update one file's sha256 in the manifest after editing it."""
    import hashlib

    path = pkg / "manifest.yaml"
    m = yaml.safe_load(path.read_text())
    for entry in m["files"]:
        if entry["path"] == rel:
            entry["sha256"] = hashlib.sha256(
                (pkg / rel).read_bytes()
            ).hexdigest()
    path.write_text(yaml.safe_dump(m, sort_keys=False))


def test_fixture_schema_version() -> None:
    """The fixture carries the schema version the import stage expects."""
    m = yaml.safe_load((FIXTURE / "manifest.yaml").read_text())
    assert m["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert load_import().SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION


def test_fixture_accepted() -> None:
    """The untouched fixture imports cleanly."""
    assert load_import().check_package(FIXTURE, set()) == []


def test_tampered_file_rejected(pkg: Path) -> None:
    """A changed RTL byte fails the checksum."""
    sv = pkg / "rtl" / "src" / "toy" / "toy_counter.sv"
    sv.write_text(sv.read_text().replace("WIDTH'(1)", "WIDTH'(2)"))
    errs = load_import().check_package(pkg, set())
    assert any("sha256 mismatch" in e for e in errs)


def test_dirty_rejected(pkg: Path) -> None:
    """dirty: true never gets past import."""
    edit_manifest(pkg, dirty=True)
    errs = load_import().check_package(pkg, set())
    assert any("dirty" in e for e in errs)


def test_unknown_schema_version_rejected(pkg: Path) -> None:
    """A future schema version is refused rather than guessed at."""
    edit_manifest(pkg, schema_version=2)
    errs = load_import().check_package(pkg, set())
    assert any("schema_version" in e for e in errs)


def test_undeclared_macro_rejected(pkg: Path) -> None:
    """A macro RTL declares must be in the design's macro list."""
    (pkg / "macros" / "macros.yaml").write_text(
        yaml.safe_dump({"macros": [{"name": "sram_x", "role": "sram"}]})
    )
    rehash(pkg, "macros/macros.yaml")
    errs = load_import().check_package(pkg, set())
    assert any("sram_x" in e for e in errs)
    assert load_import().check_package(pkg, {"sram_x"}) == []
