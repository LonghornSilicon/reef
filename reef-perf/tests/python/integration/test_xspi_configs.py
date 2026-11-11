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


@pytest.mark.integration
def test_ram_overlay_times_its_own_address_range(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """Loads from the RAM's address range use the RAM link, not the flash's."""
    del cpp_build_dir  # Built for its side effect.
    elf = build_microbenchmark("ext_load_stream", tmp_path)
    flash = MEMORY_CONFIGS / "xspi_str.yaml"
    ram = MEMORY_CONFIGS / "psram_kv.yaml"
    flash_only = run_sim(elf, overlay_configs=[flash]).roi_cycles
    # The RAM file's own range starts at 0x24000000: the loads miss it.
    elsewhere = run_sim(elf, overlay_configs=[flash, ram]).roi_cycles
    # Moved onto the microbenchmark's buffer, the RAM link times the loads.
    on_buffer = run_sim(
        elf,
        overlay_configs=[flash, ram],
        params={"top.memory.params.ext_ram_base": 0x20000000},
    ).roi_cycles
    assert flash_only is not None
    assert elsewhere == flash_only
    # A 4-byte load costs 30 + 16 cycles on the RAM link and 68 + 4 on the
    # flash link, so these small loads get faster.
    assert on_buffer < flash_only
