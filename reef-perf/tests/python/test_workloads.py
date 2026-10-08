"""Check that the run experiment's workloads assemble."""

from pathlib import Path

import pytest

from reef_perf.experiments.run.workloads import build_workload, list_workloads


@pytest.mark.unit
def test_lists_known_workloads() -> None:
    """The workload directory is found and contains the expected programs."""
    names = list_workloads()
    assert "alu_chain" in names
    assert "gemv_int8" in names


@pytest.mark.unit
def test_every_workload_builds(tmp_path: Path) -> None:
    """Each workload assembles and links into a RISC-V ELF."""
    for name in list_workloads():
        elf = build_workload(name, tmp_path)
        header = elf.read_bytes()[:20]
        assert header[:4] == b"\x7fELF", name
        assert header[4] == 1, f"{name}: not a 32-bit ELF"
        assert int.from_bytes(header[18:20], "little") == 243, name


@pytest.mark.unit
def test_rejects_unknown_workload(tmp_path: Path) -> None:
    """Asking for a workload that does not exist is an error."""
    with pytest.raises(FileNotFoundError):
        build_workload("no_such_workload", tmp_path)
