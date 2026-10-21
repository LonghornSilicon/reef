"""End-to-end checks of the simulator through the Python interface."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reef_perf.sim import SimulationError, run_sim

FETCH_INTERVAL = "top.frontend.params.fetch_interval"
PREDICTOR = "top.frontend.params.predictor"
TAKEN_BUBBLE = "top.frontend.params.taken_bubble"
REDIRECT_PENALTY = "top.frontend.params.redirect_penalty"
ALIGNED_BLOCKS = "top.frontend.params.aligned_blocks"


@pytest.mark.integration
@pytest.mark.slow
def test_runs_microbenchmark_to_completion(
    cpp_build_dir: Path, microbenchmark_elfs: dict[str, Path]
) -> None:
    """Every instruction Spike executes is retired, in plausible time."""
    del cpp_build_dir  # built for its side effect
    result = run_sim(microbenchmark_elfs["alu_chain"])
    assert result.instructions == result.functional_instructions
    # 1000 iterations of 16 dependent adds plus loop overhead.
    assert result.instructions > 16_000
    assert 0.0 < result.ipc <= 4.0
    busy = result.dispatch["cycles_with_dispatch"]
    assert busy + sum(result.stall_cycles.values()) <= result.cycles


@pytest.mark.integration
@pytest.mark.slow
def test_parameter_override_changes_timing(
    cpp_build_dir: Path, microbenchmark_elfs: dict[str, Path]
) -> None:
    """A -p override reaches the model: halving fetch rate costs cycles."""
    del cpp_build_dir
    elf = microbenchmark_elfs["alu_indep"]
    fast = run_sim(elf, params={FETCH_INTERVAL: 1})
    slow = run_sim(elf, params={FETCH_INTERVAL: 2})
    assert slow.instructions == fast.instructions
    assert slow.cycles > fast.cycles


@pytest.mark.integration
@pytest.mark.slow
def test_static_prediction_makes_loop_branches_free(
    cpp_build_dir: Path, microbenchmark_elfs: dict[str, Path]
) -> None:
    """With static_btfn a correctly predicted loop branch costs no extra."""
    del cpp_build_dir
    elf = microbenchmark_elfs["branch_tight"]
    common = {FETCH_INTERVAL: 2, TAKEN_BUBBLE: 0, REDIRECT_PENALTY: 1}
    none = run_sim(elf, params={**common, PREDICTOR: "none"})
    btfn = run_sim(elf, params={**common, PREDICTOR: "static_btfn"})
    assert btfn.instructions == none.instructions
    assert btfn.cycles < none.cycles


@pytest.mark.integration
@pytest.mark.slow
def test_aligned_blocks_slow_down_misaligned_loop(
    cpp_build_dir: Path, microbenchmark_elfs: dict[str, Path]
) -> None:
    """alu_indep's loop starts at slot 3, so aligned blocks cost a fetch."""
    del cpp_build_dir
    elf = microbenchmark_elfs["alu_indep"]
    free = run_sim(elf, params={ALIGNED_BLOCKS: "false"})
    aligned = run_sim(elf, params={ALIGNED_BLOCKS: "true"})
    assert aligned.instructions == free.instructions
    assert aligned.cycles > free.cycles


@pytest.mark.integration
@pytest.mark.slow
def test_parallel_runs_match_serial_runs(
    cpp_build_dir: Path, microbenchmark_elfs: dict[str, Path]
) -> None:
    """Simulator runs are deterministic and independent of each other.

    This is what makes the sweep experiment's parallel sweep valid.
    """
    del cpp_build_dir
    names = ["alu_chain", "branch_tight", "load_chain", "gemv_int8"]
    serial = [run_sim(microbenchmark_elfs[name]) for name in names]
    with ThreadPoolExecutor(max_workers=len(names) * 2) as pool:
        parallel = list(
            pool.map(lambda name: run_sim(microbenchmark_elfs[name]), names * 2)
        )
    assert parallel[: len(names)] == serial
    assert parallel[len(names) :] == serial


@pytest.mark.integration
@pytest.mark.slow
def test_missing_elf_raises(cpp_build_dir: Path, tmp_path: Path) -> None:
    """A bad microbenchmark is reported as a SimulationError, not a hang."""
    del cpp_build_dir
    with pytest.raises(SimulationError):
        run_sim(tmp_path / "missing.elf")
