"""Fast tests for the M3 RTL microbenchmark parameter sweeps."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reef_perf.experiments.correlation.rtl_sweep import (
    DEFAULT_SWEEPS,
    Run,
    generate,
    load_sweeps,
    run_rtl,
)
from reef_perf.microbenchmarks import build_microbenchmark


@pytest.mark.unit
def test_sweeps_cover_every_source_and_are_disjoint() -> None:
    """Every benchmark has two configurations in each split, without overlap."""
    runs = load_sweeps()
    assert len(runs) == 80
    assert len({run.name for run in runs}) == 20
    assert sum(run.split == "train" for run in runs) == 40
    assert sum(run.split == "test" for run in runs) == 40


@pytest.mark.unit
def test_rejects_duplicate_configuration(tmp_path: Path) -> None:
    """Leakage between the training and test partitions is rejected."""
    sweeps = json.loads(DEFAULT_SWEEPS.read_text())
    sweeps["alu_chain"]["test"][0] = sweeps["alu_chain"]["train"][0]
    path = tmp_path / "sweeps.json"
    path.write_text(json.dumps(sweeps))
    with pytest.raises(ValueError, match="duplicate"):
        load_sweeps(path)


@pytest.mark.unit
def test_rejects_invalid_vector_length(tmp_path: Path) -> None:
    """GEMV should never silently truncate at the vector register limit."""
    sweeps = json.loads(DEFAULT_SWEEPS.read_text())
    sweeps["gemv_int8"]["test"][0]["COLS"] = 128
    path = tmp_path / "sweeps.json"
    path.write_text(json.dumps(sweeps))
    with pytest.raises(ValueError, match="COLS"):
        load_sweeps(path)


@pytest.mark.unit
def test_builder_supplies_assembler_symbols(tmp_path: Path) -> None:
    """Overrides reach GNU as rather than being substituted into source text."""
    calls = []

    def capture(args: list[str], **kwargs: object) -> None:
        calls.append(args)

    with patch("reef_perf.microbenchmarks.subprocess.run", side_effect=capture):
        build_microbenchmark(
            "alu_chain", tmp_path, {"ITERATIONS": 10, "ADDS": 4}
        )
    assert "--defsym=ADDS=4" in calls[0]
    assert "--defsym=ITERATIONS=10" in calls[0]
    assert len(calls) == 2
    with pytest.raises(ValueError, match="positive"):
        build_microbenchmark("alu_chain", tmp_path, {"ITERATIONS": 0})


@pytest.mark.unit
def test_generate_writes_two_csvs(tmp_path: Path) -> None:
    """Output contains the measured cycles and numeric parameter columns."""
    runs = [
        Run("alu_chain", "train", 0, {"ITERATIONS": 1000, "ADDS": 16}),
        Run("alu_chain", "test", 0, {"ITERATIONS": 750, "ADDS": 12}),
    ]
    simulator = tmp_path / "verilator"
    simulator.touch()
    with (
        patch(
            "reef_perf.experiments.correlation.rtl_sweep.build_microbenchmark"
        ) as build,
        patch("reef_perf.experiments.correlation.rtl_sweep.run_rtl") as measure,
    ):
        build.return_value = tmp_path / "bench.elf"
        measure.side_effect = [(123, 456), (78, 90)]
        paths = generate(runs, simulator, tmp_path / "results")
    assert set(paths) == {"train", "test"}
    with paths["train"].open(newline="") as handle:
        train = list(csv.DictReader(handle))
    with paths["test"].open(newline="") as handle:
        test = list(csv.DictReader(handle))
    assert train[0]["rtl_roi_cycles"] == "123"
    assert train[0]["ADDS"] == "16"
    assert test[0]["rtl_retired_insts"] == "90"
    assert test[0]["ITERATIONS"] == "750"


@pytest.mark.unit
def test_run_rtl_measures_trace(tmp_path: Path) -> None:
    """Harness measures ROI via the trace parser, not log timing."""
    def csr(reg: int) -> int:
        return (0xB00 << 20) | (2 << 12) | (reg << 7) | 0x73

    simulator = tmp_path / "simulator"
    simulator.touch()
    elf = tmp_path / "bench.elf"
    elf.touch()
    text = "\n".join(
        [
            f"0x00000000,0x{csr(26):08x},0x1a,0x00000064,trap=no",
            "0x00000004,0x00100093,0x01,0x00000001,trap=no",
            f"0x00000008,0x{csr(27):08x},0x1b,0x000000c8,trap=no",
            "0x0000000c,0x08000073,0x7f,0x00000000,trap=no",
        ]
    )

    def fake_run(args: list[str], **kwargs: object) -> None:
        assert args[1] == "--binary"
        assert args[-2:] == ["--backdoor_load", "--instr_trace"]
        kwargs["stdout"].write(text)

    with patch(
        "reef_perf.experiments.correlation.rtl_sweep.subprocess.run",
        side_effect=fake_run,
    ):
        assert run_rtl(simulator, elf, tmp_path / "trace.log", 60) == (100, 1)
