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
    # 6.25 against 6.0: the loop is ROB-bound in the model a little more
    # tightly than on the RTL (docs/matrix.md, B2).
    "vtmm_misaligned": 0.05,
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
}
"""Programs with a known error outside the matrix module, and why."""

_BY_NAME = {bench.name: bench for bench in BENCHMARKS}


def _rtl_rows(bench: DiffBenchmark) -> list[dict[str, str]]:
    with RTL_REFERENCE.open() as f:
        rows = {(r["name"], int(r["iters"])): r for r in csv.DictReader(f)}
    return [rows[(bench.name, k)] for k in (bench.iters, 2 * bench.iters)]


def _rtl_cost(bench: DiffBenchmark) -> float:
    roi = [int(row["roi_cycles"]) for row in _rtl_rows(bench)]
    return (roi[1] - roi[0]) / (bench.iters * bench.ops_per_iter)


def _model_cost(bench: DiffBenchmark, tmp_path: Path) -> float:
    rtl_insts = [int(row["roi_instructions"]) for row in _rtl_rows(bench)]
    roi = []
    for n, insts in zip((bench.iters, 2 * bench.iters), rtl_insts, strict=True):
        out = tmp_path / str(n)
        (out / "matrix").mkdir(parents=True)
        elf = build_microbenchmark(f"matrix/{bench.name}", out, {"ITERS": n})
        result = run_sim(
            elf, params=REFERENCE_CORE_PARAMS, overlay_configs=(VME_CONFIG,)
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
