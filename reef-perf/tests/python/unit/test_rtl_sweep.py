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
@pytest.mark.parametrize(
    ("name", "params", "error"),
    [
        ("load_chain", {"NODES": 1}, "NODES must be at least 2"),
        ("load_chain", {"NODES": 2000}, "DTCM budget"),
        ("load_stream", {"BLOCKS": 751}, "DTCM budget"),
        ("vec_add_m1", {"ELEMENTS": 2001}, "DTCM budget"),
        ("vec_memcpy", {"BYTES": 3}, "multiple of 4"),
        ("gemv_int8", {"COLS": 65}, "COLS"),
        ("gemv_int8", {"ROWS": 400, "COLS": 64}, "DTCM budget"),
        ("alu_chain", {"ADDS": 129}, "unroll limit"),
        ("alu_chain", {"TYPO": 5}, "unknown assembler parameters"),
    ],
)
def test_builder_rejects_invalid_partial_overrides(
    tmp_path: Path, name: str, params: dict[str, int], error: str
) -> None:
    """Direct callers cannot bypass sweep constraints or mistype symbols."""
    with patch("reef_perf.microbenchmarks.subprocess.run") as assemble:
        with pytest.raises(ValueError, match=error):
            build_microbenchmark(name, tmp_path, params)
    assemble.assert_not_called()


@pytest.mark.unit
def test_builder_partial_overrides_use_source_defaults(tmp_path: Path) -> None:
    """Omitted symbols use assembly defaults, not missing dictionary keys."""
    with patch("reef_perf.microbenchmarks.subprocess.run") as assemble:
        build_microbenchmark("load_chain", tmp_path, {"ITERATIONS": 100})
    assert assemble.call_count == 2
    command = assemble.call_args_list[0].args[0]
    assert "--defsym=ITERATIONS=100" in command
    assert not any(arg.startswith("--defsym=NODES=") for arg in command)


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


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["compile", "timeout", "rtl_error"])
def test_generate_preserves_existing_csvs_on_failure(
    tmp_path: Path, failure: str
) -> None:
    """No original dataset may be deleted by a failed build or RTL run."""
    runs = [
        Run("alu_chain", "train", 0, {"ITERATIONS": 100, "ADDS": 4}),
        Run("alu_chain", "test", 0, {"ITERATIONS": 200, "ADDS": 8}),
    ]
    simulator = tmp_path / "simulator"
    simulator.touch()
    csv_dir = tmp_path / "custom_csv"
    csv_dir.mkdir()
    original = {
        "train": b"important existing training measurements\n",
        "test": b"important existing testing measurements\n",
    }
    for split, contents in original.items():
        (csv_dir / f"rtl_m3_{split}.csv").write_bytes(contents)

    def failed_build(*args: object, **kwargs: object) -> Path:
        if failure == "compile":
            raise FileNotFoundError("assembler missing")
        return tmp_path / "dummy.elf"

    def failed_measure(*args: object, **kwargs: object) -> tuple[int, int]:
        if args[2].parent.name == "test":
            if failure == "timeout":
                import subprocess
                raise subprocess.TimeoutExpired("simulator", 5)
            raise ValueError("RTL did not retire mpause")
        return 100, 20

    with (
        patch(
            "reef_perf.experiments.correlation.rtl_sweep.build_microbenchmark",
            side_effect=failed_build,
        ),
        patch(
            "reef_perf.experiments.correlation.rtl_sweep.run_rtl",
            side_effect=failed_measure,
        ),
    ):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            generate(runs, simulator, tmp_path / "build", csv_dir=csv_dir)

    for split, contents in original.items():
        assert (csv_dir / f"rtl_m3_{split}.csv").read_bytes() == contents
    assert not list(csv_dir.glob(".rtl_sweep_*"))


@pytest.mark.unit
def test_generate_replaces_existing_csvs_after_success(tmp_path: Path) -> None:
    """Successful runs replace both CSVs in an independently selected directory."""
    runs = [
        Run("alu_chain", "train", 0, {"ITERATIONS": 100, "ADDS": 4}),
        Run("alu_chain", "test", 0, {"ITERATIONS": 200, "ADDS": 8}),
    ]
    simulator = tmp_path / "simulator"
    simulator.touch()
    csv_dir = tmp_path / "custom_csv"
    csv_dir.mkdir()
    for split in ("train", "test"):
        (csv_dir / f"rtl_m3_{split}.csv").write_text("old data")

    with (
        patch(
            "reef_perf.experiments.correlation.rtl_sweep.build_microbenchmark",
            return_value=tmp_path / "dummy.elf",
        ),
        patch(
            "reef_perf.experiments.correlation.rtl_sweep.run_rtl",
            side_effect=[(111, 22), (222, 33)],
        ),
    ):
        result = generate(runs, simulator, tmp_path / "build", csv_dir=csv_dir)

    assert set(result) == {"train", "test"}
    for split, expected in (("train", "111"), ("test", "222")):
        with result[split].open(newline="") as handle:
            assert list(csv.DictReader(handle))[0]["rtl_roi_cycles"] == expected
    assert not list(csv_dir.glob(".rtl_sweep_*"))
