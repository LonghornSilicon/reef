"""The vector functional unit microbenchmarks match the M3 RTL."""

import csv
from pathlib import Path

import pytest

from reef_perf.microbenchmarks import MICROBENCHMARK_DIR, build_microbenchmark
from reef_perf.sim import MODULE_ROOT, run_sim

VECTOR_UNITS_DIR = MICROBENCHMARK_DIR / "vector_units"
UNIFIED_OVERLAY = (
    MODULE_ROOT / "configs" / "machines" / "m3_unified_vector.yaml"
)
"""Two general-purpose vector lanes instead of M3's per-type units."""
TOLERANCE = 0.10
"""Allowed relative ROI-cycle error (roadmap B15)."""


@pytest.mark.integration
@pytest.mark.slow
def test_vector_unit_microbenchmarks_match_rtl(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Every modeled vector unit microbenchmark is within 10% of the RTL."""
    del cpp_build_dir  # Built for its side effect.
    with (VECTOR_UNITS_DIR / "rtl_m3.csv").open(newline="") as handle:
        rtl = {row["microbenchmark"]: row for row in csv.DictReader(handle)}
    names = sorted(path.stem for path in VECTOR_UNITS_DIR.glob("*.S"))
    assert sorted(rtl) == names, "microbenchmarks and RTL rows differ"
    (tmp_path / "vector_units").mkdir()
    for name in names:
        if rtl[name]["notes"].startswith("not modeled"):
            continue
        result = run_sim(build_microbenchmark(f"vector_units/{name}", tmp_path))
        assert result.roi_instructions == int(rtl[name]["rtl_retired_insts"])
        assert result.roi_cycles is not None, name
        rtl_cycles = int(rtl[name]["rtl_roi_cycles"])
        assert abs(result.roi_cycles - rtl_cycles) <= TOLERANCE * rtl_cycles, (
            name
        )


@pytest.mark.integration
def test_unified_vector_overlay_changes_the_organization(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """On two shared lanes, reductions no longer queue for one unit."""
    del cpp_build_dir  # Built for its side effect.
    (tmp_path / "vector_units").mkdir()
    elf = build_microbenchmark("vector_units/vredsum_m4_indep", tmp_path)
    m3 = run_sim(elf)
    unified = run_sim(elf, overlay_configs=[UNIFIED_OVERLAY])
    assert m3.roi_cycles is not None and unified.roi_cycles is not None
    assert unified.roi_cycles < m3.roi_cycles
