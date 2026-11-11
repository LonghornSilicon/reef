"""The matrix engine stays correlated with the CoralNPU VME RTL.

The matrix module's golden test. Each program below runs on the model at N
and 2N iterations with the reference core's settings
(matrix_diff.REFERENCE_CORE_PARAMS), and its per-op cost,
(ROI(2N) - ROI(N)) / ops, must stay within a tolerance of the RTL's, from
microbenchmarks/matrix/rtl_vme.csv (measured on CoralNPU 8a5e1b31;
docs/matrix.md, "Correlating it").

A program is either CORRELATED (it must match) or a KNOWN_GAP (its error is
outside the matrix module and is tracked elsewhere; matrix_diff compare still
reports it). A new matrix_diff program has to go in one of the two.
"""

import csv
from pathlib import Path

import pytest

from reef_perf.experiments.matrix_diff.main import (
    BENCHMARKS,
    REFERENCE_CORE_PARAMS,
    RTL_REFERENCE,
    VME_CONFIG,
    DiffBenchmark,
)
from reef_perf.microbenchmarks import build_microbenchmark
from reef_perf.sim import run_sim

CORRELATED = {
    "mac_chain": 0.02,
    "mac_indep": 0.02,
    "tile_load_behind_mac": 0.02,
    "tile_load_burst": 0.02,
    "tile_store_burst": 0.02,
    "tile_load32_burst": 0.02,
    "tile_store32_burst": 0.02,
    "mac_chain_tm4": 0.02,
    "mac_indep_tm4": 0.02,
    "mac_chain_tn4": 0.02,
    "mac_indep_tn4": 0.02,
    "mac_indep_tm8tn8": 0.02,
    "mac_chain_fp32": 0.02,
    "mac_indep_fp32": 0.02,
    "mac_chain_bf16": 0.02,
    "mac_indep_bf16": 0.02,
    # 5.83 against 6.0: ROB-bound, a little differently from the RTL
    # (docs/matrix.md, B2).
    "vtmm_misaligned": 0.05,
    # C4/C5 (docs/matrix.md): vtzero, moves, tile-write and completion paths.
    "zero_burst": 0.02,
    "move_tv_burst": 0.02,
    "move_vt_burst": 0.02,
    "load_then_mac": 0.02,
    "move_then_mac": 0.02,
    "mac_then_store": 0.02,
    "mac_rob": 0.02,
    "zero_rob": 0.02,
    "move_tv_rob": 0.02,
    "mac_queue": 0.02,
    "mac_rob_tm4": 0.02,
    "mac_rob_tn4": 0.02,
    "mac_rob_tm8tn8": 0.02,
}
"""Programs that must match the RTL, with their relative tolerance."""

KNOWN_GAP = {
    "tile_load": "frontend: one branch per row; the M3 frontend is slower",
    "tile_store": "frontend: one branch per row; the M3 frontend is slower",
    "gemm_tile": "vector loads on M3's LSU (docs/lsu-tile-beats.md)",
    "gemm_multi": "vector loads on M3's LSU (docs/lsu-tile-beats.md)",
    "gemm_longk": "vector loads on M3's LSU (docs/lsu-tile-beats.md)",
    "vec_load_burst": "vector reference: M3's LSU, not the newer one",
    "vec_store_burst": "vector reference: M3's LSU, not the newer one",
    "vec_load4_burst": "vector reference: M3's LSU, not the newer one",
    "vec_store4_burst": "vector reference: M3's LSU, not the newer one",
    "vtdiscard_after_mac": "core: vtdiscard's dispatch interlock (case 7)",
    "move_tv8_burst": "core: Dispatch's 4 matrix credits cap one-uop ops; "
    "and the frontend loop bubble",
    "move_vt8_burst": "core: Dispatch's 4 matrix credits cap one-uop ops; "
    "and the frontend loop bubble",
    "move_vt_use": "vector: vmv.x.s of a move result is 1 cycle later than "
    "on the RTL",
    "load_then_store": "core: a tile store holds the LSU while it waits for "
    "data (case 5)",
}
"""Programs with a known error outside the matrix module, and why."""

_BY_NAME = {bench.name: bench for bench in BENCHMARKS}


RTL_REFERENCE_PE128 = RTL_REFERENCE.with_name("rtl_vme_pe128.csv")
"""The same programs on a 128-PE CoralNPU build (docs/matrix.md, D4)."""

PE128_CONFIG = VME_CONFIG.parents[1] / "matrix" / "coral_128pe.yaml"
"""Model settings for the 128-PE build: 8 x 16 PEs, PROCESS_DELAY 2."""

CORRELATED_PE128 = {
    name: tol
    for name, tol in CORRELATED.items()
    if name not in ("mac_indep_tm4", "mac_queue")
}
"""Programs that must also match at 128 PEs.

mac_indep_tm4 (1.835 against 1.755) and mac_queue (15.36 against 15.0) are
ROB-bound 6- and 14-instruction loops there; the gap is not in the engine
(mac_rob_tm4 matches), most likely the frontend's loop cost.
"""


def _rtl_rows(
    bench: DiffBenchmark, reference: Path = RTL_REFERENCE
) -> list[dict[str, str]]:
    with reference.open() as f:
        rows = {(r["name"], int(r["iters"])): r for r in csv.DictReader(f)}
    return [rows[(bench.name, k)] for k in (bench.iters, 2 * bench.iters)]


def _rtl_cost(bench: DiffBenchmark, reference: Path = RTL_REFERENCE) -> float:
    roi = [int(row["roi_cycles"]) for row in _rtl_rows(bench, reference)]
    return (roi[1] - roi[0]) / (bench.iters * bench.ops_per_iter)


def _model_cost(
    bench: DiffBenchmark,
    tmp_path: Path,
    reference: Path = RTL_REFERENCE,
    overlays: tuple[Path, ...] = (),
) -> float:
    rtl_insts = [
        int(row["roi_instructions"]) for row in _rtl_rows(bench, reference)
    ]
    roi = []
    for n, insts in zip((bench.iters, 2 * bench.iters), rtl_insts, strict=True):
        out = tmp_path / str(n)
        (out / "matrix").mkdir(parents=True)
        elf = build_microbenchmark(f"matrix/{bench.name}", out, {"ITERS": n})
        result = run_sim(
            elf,
            params=REFERENCE_CORE_PARAMS,
            overlay_configs=(VME_CONFIG, *overlays),
        )
        assert result.instructions == result.functional_instructions
        assert result.roi_cycles is not None
        # Same work on both sides, or the cycle comparison means nothing (a
        # stale rtl_vme.csv row or a changed program).
        assert result.roi_instructions == insts, (
            f"{bench.name} at {n}: model {result.roi_instructions}, RTL "
            f"{insts} instructions; re-run matrix_diff rtl --record"
        )
        roi.append(result.roi_cycles)
    return (roi[1] - roi[0]) / (bench.iters * bench.ops_per_iter)


@pytest.mark.unit
def test_every_program_is_classified() -> None:
    """Each matrix_diff program is CORRELATED or a KNOWN_GAP, not both."""
    assert not set(CORRELATED) & set(KNOWN_GAP)
    assert set(CORRELATED) | set(KNOWN_GAP) == set(_BY_NAME)


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(CORRELATED))
def test_matches_rtl(name: str, cpp_build_dir: Path, tmp_path: Path) -> None:
    """The model's per-op cost stays within tolerance of the RTL's."""
    del cpp_build_dir
    bench = _BY_NAME[name]
    rtl = _rtl_cost(bench)
    model = _model_cost(bench, tmp_path)
    assert model == pytest.approx(rtl, rel=CORRELATED[name]), (
        f"{name}: model {model:.3f} against RTL {rtl:.3f} cycles per op"
    )


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(CORRELATED_PE128))
def test_matches_rtl_at_128_pes(
    name: str, cpp_build_dir: Path, tmp_path: Path
) -> None:
    """The Coral mapping, scaled to 128 PEs, matches a 128-PE RTL build."""
    del cpp_build_dir
    bench = _BY_NAME[name]
    rtl = _rtl_cost(bench, RTL_REFERENCE_PE128)
    model = _model_cost(bench, tmp_path, RTL_REFERENCE_PE128, (PE128_CONFIG,))
    assert model == pytest.approx(rtl, rel=CORRELATED_PE128[name]), (
        f"{name} at 128 PEs: model {model:.3f} against RTL {rtl:.3f}"
    )
