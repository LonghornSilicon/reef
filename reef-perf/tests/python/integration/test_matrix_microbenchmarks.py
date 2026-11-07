"""The Zvt microbenchmarks run on the matrix module and give consistent results.

There are no RTL numbers for them yet (docs/matrix.md), so these tests check
the model against itself: every program runs without a deadlock, the matrix
module executes the expected instructions, and results respect bounds that
hold for any array design.
"""

from pathlib import Path

import pytest

from reef_perf.microbenchmarks import MICROBENCHMARK_DIR, build_microbenchmark
from reef_perf.sim import MODULE_ROOT, SimResult, SimulationError, run_sim

MATRIX_DIR = MICROBENCHMARK_DIR / "matrix"
OVERLAY_DIR = MODULE_ROOT / "configs" / "matrix"
VME_CONFIG = MODULE_ROOT / "configs" / "machines" / "m3_vme.yaml"
"""Turns Zvt on; applied after m3.yaml, before any array overlay."""

EXPECTED_OPS = {
    "mac_chain": {"mac": 200, "zero": 1},
    "mac_indep": {"mac": 200},
    "tile_load": {"tile_load": 320},
    "tile_store": {"tile_store": 320},
    "gemm_tile": {"mac": 16, "zero": 1, "move_to_vector": 16},
    "illegal_shapes": {"nop": 2},
}
"""Matrix instructions each program executes, by op."""


def _run(
    name: str, tmp_path: Path, overlays: tuple[Path, ...] = ()
) -> SimResult:
    """Build and run one matrix microbenchmark.

    Args:
        name: Program name in microbenchmarks/matrix/.
        tmp_path: Directory for the build outputs.
        overlays: Configuration overlays applied after m3.yaml and
            m3_vme.yaml.

    Returns:
        The simulator's summary.
    """
    (tmp_path / "matrix").mkdir(exist_ok=True)
    elf = build_microbenchmark(f"matrix/{name}", tmp_path)
    return run_sim(elf, overlay_configs=(VME_CONFIG, *overlays))


@pytest.mark.integration
def test_every_matrix_microbenchmark_is_listed() -> None:
    """EXPECTED_OPS covers exactly the programs in microbenchmarks/matrix/."""
    names = sorted(path.stem for path in MATRIX_DIR.glob("*.S"))
    assert names == sorted(EXPECTED_OPS)


@pytest.mark.integration
def test_zvt_traps_without_vme_config(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """With m3.yaml alone, Zvt is off and a Zvt program traps, as on M3."""
    del cpp_build_dir
    (tmp_path / "matrix").mkdir()
    elf = build_microbenchmark("matrix/mac_chain", tmp_path)
    with pytest.raises(SimulationError, match="trap"):
        run_sim(elf)


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(EXPECTED_OPS))
def test_matrix_microbenchmark_runs(
    name: str, cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Each program retires every instruction and runs the expected ops."""
    del cpp_build_dir  # Built for its side effect.
    result = _run(name, tmp_path)
    assert result.instructions == result.functional_instructions
    assert result.roi_cycles is not None
    ops = result.matrix["ops"]
    for op, count in EXPECTED_OPS[name].items():
        assert ops[op] == count, (name, op)
    assert sum(ops.values()) == sum(EXPECTED_OPS[name].values())


@pytest.mark.integration
def test_mac_chain_is_slower_than_independent_macs(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Dependent MACs take longer than the same MACs on rotating tiles.

    MACs into one accumulator wait for each other; MACs rotating over four
    accumulators overlap.
    """
    del cpp_build_dir
    chain = _run("mac_chain", tmp_path)
    indep = _run("mac_indep", tmp_path)
    assert chain.roi_cycles is not None and indep.roi_cycles is not None
    assert chain.roi_cycles > indep.roi_cycles


@pytest.mark.integration
@pytest.mark.parametrize(
    "overlay", sorted(OVERLAY_DIR.glob("*.yaml")), ids=lambda path: path.stem
)
def test_overlays_respect_compute_bound(
    overlay: Path, cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Every array preset runs and respects its peak MAC rate.

    Utilization stays at or below 100%, and the run lasts at least as long as
    the array was busy.
    """
    del cpp_build_dir
    result = _run("mac_indep", tmp_path, (overlay,))
    stats = result.matrix
    assert result.instructions == result.functional_instructions
    assert 0.0 < stats["mac_utilization"] <= 1.0
    assert stats["useful_macs"] == 200 * 16 * 16 * 4
    # The array cannot finish faster than its busy cycles allow. Total
    # cycles, not ROI cycles: the closing mcycle read issues while MACs are
    # still in flight.
    assert result.cycles >= stats["array_busy_cycles"]
