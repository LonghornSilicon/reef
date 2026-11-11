"""A full DMA queue holds the core's GO store (memory/Dma.h)."""

from pathlib import Path

import pytest

from reef_perf.microbenchmarks import build_microbenchmark
from reef_perf.sim import run_sim

DEPTH = "top.memory.params.dma_queue_depth"


@pytest.mark.integration
def test_full_queue_delays_the_core(
    cpp_build_dir: Path, tmp_path: Path
) -> None:
    """With room for one transfer, the second GO waits for the first."""
    del cpp_build_dir  # Built for its side effect.
    (tmp_path / "memory").mkdir()
    elf = build_microbenchmark("memory/dma_queue", tmp_path)
    roi = {d: run_sim(elf, params={DEPTH: d}).roi_cycles for d in (0, 1, 2)}
    assert roi[0] is not None
    assert roi[2] == roi[0]  # Room for both: nothing waits.
    assert roi[1] > roi[0]  # The loop starts only after the first transfer.
