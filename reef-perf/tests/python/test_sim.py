"""End-to-end checks of the simulator through the Python interface."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reef_perf.sim import SimResult, SimulationError, run_sim

FETCH_INTERVAL = "top.core.fetch.params.fetch_interval"


@pytest.mark.integration
@pytest.mark.slow
def test_runs_workload_to_completion(
    cpp_build_dir: Path, workload_elfs: dict[str, Path]
) -> None:
    """Every instruction Spike executes is retired, in plausible time."""
    del cpp_build_dir  # built for its side effect
    result = run_sim(workload_elfs["alu_chain"])
    assert result.instructions == result.functional_instructions
    # 1000 iterations of 16 dependent adds plus loop overhead.
    assert result.instructions > 16_000
    assert 0.0 < result.ipc <= 4.0
    busy = result.dispatch["cycles_with_dispatch"]
    assert busy + sum(result.stall_cycles.values()) <= result.cycles
    #ADDED FOR ROI
    assert result.roi_complete is True
    assert result.roi_instructions == 18000
    assert result.roi_cycles > 0

@pytest.mark.integration
@pytest.mark.slow
def test_parameter_override_changes_timing(
    cpp_build_dir: Path, workload_elfs: dict[str, Path]
) -> None:
    """A -p override reaches the model: halving fetch rate costs cycles."""
    del cpp_build_dir
    elf = workload_elfs["alu_indep"]
    fast = run_sim(elf)
    slow = run_sim(elf, params={FETCH_INTERVAL: 2})
    assert slow.instructions == fast.instructions
    assert slow.cycles > fast.cycles


@pytest.mark.integration
@pytest.mark.slow
def test_parallel_runs_match_serial_runs(
    cpp_build_dir: Path, workload_elfs: dict[str, Path]
) -> None:
    """Simulator runs are deterministic and independent of each other.

    This is what makes the run experiment's parallel sweep valid.
    """
    del cpp_build_dir
    names = ["alu_chain", "branch_tight", "load_chain", "gemv_int8"]
    serial = [run_sim(workload_elfs[name]) for name in names]
    with ThreadPoolExecutor(max_workers=len(names) * 2) as pool:
        parallel = list(
            pool.map(lambda name: run_sim(workload_elfs[name]), names * 2)
        )
    assert parallel[: len(names)] == serial
    assert parallel[len(names) :] == serial


@pytest.mark.integration
@pytest.mark.slow
def test_missing_elf_raises(cpp_build_dir: Path, tmp_path: Path) -> None:
    """A bad workload is reported as a SimulationError, not a hang."""
    del cpp_build_dir
    with pytest.raises(SimulationError):
        run_sim(tmp_path / "missing.elf")

#Added for ROI tracking json files
@pytest.mark.unit # Testing python json parsing without running the actual C++ simulator
def test_sim_result_roi_json() -> None:
    """ROI measurements are correctly parsed and preserved."""

    data = {
        "workload": "alu_chain.elf",
        "instructions": 18009,
        "functional_instructions": 18009,
        "cycles": 16007,
        "ipc": 1.125,
        "taken_redirects": 999,
        "dispatch": {},
        "pools": {},
        "roi_complete": True,
        "roi_instructions": 18000,
        "roi_cycles": 16000,
    }

    result = SimResult.from_json(data)

    assert result.roi_complete is True
    assert result.roi_instructions == 18000
    assert result.roi_cycles == 16000

    # Verify serialization preserves the measurements.
    assert result.to_json() == data

    # Older JSON files without ROI fields should still work.
    old_data = {
        key: value
        for key, value in data.items()
        if not key.startswith("roi_")
    }

    old_result = SimResult.from_json(old_data)

    assert old_result.roi_complete is False
    assert old_result.roi_instructions == 0
    assert old_result.roi_cycles == 0