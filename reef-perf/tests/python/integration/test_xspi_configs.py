"""The XSPI link overlays in configs/memory/ time external accesses."""

from pathlib import Path

import pytest

from reef_perf.microbenchmarks import build_microbenchmark
from reef_perf.sim import run_sim

MEMORY_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "memory"


@pytest.mark.integration
def test_link_overlays_slow_external_loads(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """External loads cost more on the link, and STR is no faster than DTR."""
    del cpp_build_dir  # Built for its side effect.
    elf = build_microbenchmark("ext_load_stream", tmp_path)
    roi = {
        name: run_sim(
            elf, overlay_configs=[MEMORY_CONFIGS / f"{name}.yaml"]
        ).roi_cycles
        for name in ("xspi_dtr", "xspi_str")
    }
    off = run_sim(elf).roi_cycles
    assert off is not None
    assert roi["xspi_dtr"] > off
    assert roi["xspi_str"] >= roi["xspi_dtr"]
