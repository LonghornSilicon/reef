"""The Zvt microbenchmarks run on the matrix module and give consistent results.

Most of these tests check the model against itself: every program runs
without a deadlock, the matrix module executes the expected instructions, and
results respect bounds that hold for any array design. The RTL comparison is
the matrix_diff experiment (microbenchmarks/matrix/rtl_vme.csv); the MAC rate
it measured is pinned here too.
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
    "mac_chain": {"mac": 200, "zero": 1, "move_to_vector": 1},
    "mac_indep": {"mac": 200, "move_to_vector": 1},
    "tile_load": {"tile_load": 320, "move_to_vector": 1},
    "tile_store": {"tile_store": 320},
    "tile_load_behind_mac": {
        "mac": 50,
        "tile_load": 400,
        "zero": 1,
        "move_to_vector": 1,
    },
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
def test_iters_symbol_sets_the_loop_count(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """--defsym ITERS changes the loop count, as matrix_diff relies on."""
    del cpp_build_dir
    (tmp_path / "matrix").mkdir()
    elf = build_microbenchmark("matrix/mac_chain", tmp_path, {"ITERS": 7})
    result = run_sim(elf, overlay_configs=(VME_CONFIG,))
    assert result.matrix["ops"]["mac"] == 7


def _cycles_per_mac(
    name: str,
    iters: int,
    macs_per_iter: int,
    tmp_path: Path,
    params: dict[str, object],
) -> float:
    """Per-MAC ROI cost of a MAC loop, by differencing N and 2N iterations.

    Args:
        name: Program name in microbenchmarks/matrix/.
        iters: N.
        macs_per_iter: MACs in one loop iteration.
        tmp_path: Directory for the build outputs.
        params: Model parameters (-p overrides), by tree path.

    Returns:
        (ROI(2N) - ROI(N)) / (N * macs_per_iter).
    """
    rois = []
    for n in (iters, 2 * iters):
        out = tmp_path / f"{name}_{n}"
        (out / "matrix").mkdir(parents=True)
        elf = build_microbenchmark(f"matrix/{name}", out, {"ITERS": n})
        result = run_sim(
            elf,
            params=params,
            overlay_configs=(VME_CONFIG,),
        )
        assert result.roi_cycles is not None
        rois.append(result.roi_cycles)
    return (rois[1] - rois[0]) / (iters * macs_per_iter)


@pytest.mark.integration
def test_dependent_macs_issue_as_fast_as_independent_ones(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """A MAC into the tile the previous MAC writes need not wait for it.

    On CoralNPU, hitRaw only holds a MAC while the previous MAC's first step
    is in the multiply stages, so a chain into one tile runs at the array's
    rate (4 cycles per full int8 tile, microbenchmarks/matrix/rtl_vme.csv).
    The ROB is the reference core's 16 entries; with M3's 8, the ROB limits
    the chain instead. With mtx_acc_raw_cycles 0, a MAC waits for the tile
    write and the chain is slower.
    """
    del cpp_build_dir
    rob: dict[str, object] = {"top.core.rob.params.rob_entries": 16}
    chain = _cycles_per_mac("mac_chain", 50, 1, tmp_path / "chain", rob)
    indep = _cycles_per_mac("mac_indep", 50, 4, tmp_path / "indep", rob)
    assert chain == pytest.approx(4.0)
    assert indep == pytest.approx(4.0)
    waiting = _cycles_per_mac(
        "mac_chain",
        50,
        1,
        tmp_path / "waiting",
        {**rob, "top.matrix.params.mtx_acc_raw_cycles": 0},
    )
    assert waiting > chain


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
