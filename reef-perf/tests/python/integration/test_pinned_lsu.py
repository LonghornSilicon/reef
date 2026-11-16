"""The LSU microbenchmarks on the pinned-RTL config match the pinned RTL.

microbenchmarks/lsu/ and microbenchmarks/pinned/ measure CoralNPU 8a5e1b31 by
differencing N and 2N iterations, as the matrix programs do
(rtl_pinned.csv, from vme_core_mini_verification_axi_sim). The model runs
m3.yaml + configs/machines/coral_pinned.yaml, which turns lsu_v3 on.
"""

import csv
from pathlib import Path

import pytest

from reef_perf.experiments.correlation.pinned import PINNED_CONFIG
from reef_perf.microbenchmarks import MICROBENCHMARK_DIR, build_microbenchmark
from reef_perf.sim import run_sim

SUITES = ("lsu", "pinned")
"""Directories of differenced programs with an rtl_pinned.csv."""
TOLERANCE = 0.02
"""Allowed relative error in cycles per loop iteration."""

KNOWN_GAP = {
    "pinned/fdiv_chain": "core: the FP divider's time depends on its operands "
    "(18 cycles per divide here, 13 in fdiv_indep)",
    "pinned/fcvt_chain": "core: fcvt between an x and an f register takes 6 "
    "cycles on the RTL, 2 in the model",
    "pinned/flw_fsw": "core: an fsw of a just-loaded f register is a cycle "
    "later on the RTL (7.1 cycles per pair against 6.1)",
    "pinned/fetch_mispredict_rob": "frontend or core: a mispredict with the "
    "retirement buffer full takes 17 cycles on the RTL, 28 in the model; no "
    "waveform yet (fst2vcd aborts on this trace)",
}
"""Programs with a known error, and why (docs/pinned.md, "Coverage")."""


def _rtl() -> dict[str, list[tuple[int, int, int]]]:
    rows: dict[str, list[tuple[int, int, int]]] = {}
    for suite in SUITES:
        path = MICROBENCHMARK_DIR / suite / "rtl_pinned.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.setdefault(f"{suite}/{row['name']}", []).append(
                    (
                        int(row["iters"]),
                        int(row["roi_cycles"]),
                        int(row["roi_instructions"]),
                    )
                )
    return {name: sorted(runs) for name, runs in rows.items()}


PROGRAMS = sorted(
    f"{suite}/{path.stem}"
    for suite in SUITES
    for path in (MICROBENCHMARK_DIR / suite).glob("*.S")
)


@pytest.mark.unit
def test_every_lsu_program_has_rtl() -> None:
    """Each program has its N and 2N rows, and known gaps are programs."""
    rtl = _rtl()
    assert sorted(rtl) == PROGRAMS
    assert all(len(runs) == 2 for runs in rtl.values())
    assert set(KNOWN_GAP) <= set(PROGRAMS)


@pytest.mark.integration
@pytest.mark.parametrize("name", [n for n in PROGRAMS if n not in KNOWN_GAP])
def test_lsu_program_matches_pinned_rtl(
    name: str, cpp_build_dir: Path, tmp_path: Path
) -> None:
    """The model's cycles per loop iteration are within 2% of the RTL's."""
    del cpp_build_dir
    (low, rtl_low, _), (high, rtl_high, _) = _rtl()[name]
    model = []
    for iters, _, insts in _rtl()[name]:
        out = tmp_path / str(iters)
        (out / name).parent.mkdir(parents=True)
        elf = build_microbenchmark(name, out, {"ITERS": iters})
        result = run_sim(elf, overlay_configs=[PINNED_CONFIG])
        assert result.roi_instructions == insts, name
        assert result.roi_cycles is not None
        model.append(result.roi_cycles)
    rtl_cost = (rtl_high - rtl_low) / (high - low)
    model_cost = (model[1] - model[0]) / (high - low)
    assert model_cost == pytest.approx(rtl_cost, rel=TOLERANCE), (
        f"{name}: model {model_cost:.2f} against RTL {rtl_cost:.2f} cycles "
        "per iteration"
    )


@pytest.mark.integration
def test_lsu_v3_is_off_without_the_pinned_config(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """On m3.yaml a 4-register load still costs M3's 17 cycles, not 6."""
    del cpp_build_dir
    cost = {}
    for label, overlays in (("m3", []), ("pinned", [PINNED_CONFIG])):
        roi = []
        for iters in (20, 40):
            out = tmp_path / label / str(iters)
            (out / "lsu").mkdir(parents=True)
            elf = build_microbenchmark("lsu/vload_m4", out, {"ITERS": iters})
            roi.append(run_sim(elf, overlay_configs=overlays).roi_cycles)
        cost[label] = (roi[1] - roi[0]) / (20 * 16)
    assert cost["m3"] == pytest.approx(17.0, abs=0.1)
    assert cost["pinned"] == pytest.approx(6.0, abs=0.1)
