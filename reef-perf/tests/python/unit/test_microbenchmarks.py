"""Check that the sweep experiment's microbenchmarks assemble."""

from pathlib import Path

import pytest

from reef_perf.microbenchmarks import build_microbenchmark, list_microbenchmarks


@pytest.mark.unit
def test_lists_known_microbenchmarks() -> None:
    """The microbenchmark directory is found and holds the expected programs."""
    names = list_microbenchmarks()
    assert "alu_chain" in names
    assert "gemv_int8" in names


@pytest.mark.unit
def test_every_microbenchmark_builds(tmp_path: Path) -> None:
    """Each microbenchmark assembles and links into a RISC-V ELF."""
    for name in list_microbenchmarks():
        elf = build_microbenchmark(name, tmp_path)
        header = elf.read_bytes()[:20]
        assert header[:4] == b"\x7fELF", name
        assert header[4] == 1, f"{name}: not a 32-bit ELF"
        assert int.from_bytes(header[18:20], "little") == 243, name


@pytest.mark.unit
def test_rejects_unknown_microbenchmark(tmp_path: Path) -> None:
    """Asking for a microbenchmark that does not exist is an error."""
    with pytest.raises(FileNotFoundError):
        build_microbenchmark("no_such_microbenchmark", tmp_path)
