"""The frontend calibration microbenchmarks match the M3 RTL."""

import csv
from pathlib import Path

import pytest

from reef_perf.microbenchmarks import MICROBENCHMARK_DIR, build_microbenchmark
from reef_perf.sim import run_sim

FRONTEND_DIR = MICROBENCHMARK_DIR / "frontend"
TOLERANCE = 0.01
"""Allowed relative ROI-cycle error; these loops calibrate the parameters."""


@pytest.mark.integration
@pytest.mark.slow
def test_frontend_microbenchmarks_match_rtl(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Every frontend microbenchmark is within 1% of its RTL ROI cycles."""
    del cpp_build_dir  # Built for its side effect.
    with (FRONTEND_DIR / "rtl_m3.csv").open(newline="") as handle:
        rtl = {row["microbenchmark"]: row for row in csv.DictReader(handle)}
    names = sorted(path.stem for path in FRONTEND_DIR.glob("*.S"))
    assert sorted(rtl) == names, "microbenchmarks and RTL rows differ"
    (tmp_path / "frontend").mkdir()
    for name in names:
        result = run_sim(build_microbenchmark(f"frontend/{name}", tmp_path))
        assert result.roi_instructions == int(rtl[name]["rtl_retired_insts"])
        assert result.roi_cycles is not None, name
        rtl_cycles = int(rtl[name]["rtl_roi_cycles"])
        assert abs(result.roi_cycles - rtl_cycles) <= TOLERANCE * rtl_cycles, (
            name
        )
