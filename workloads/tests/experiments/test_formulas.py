"""Closed-form costs used by the roofline experiment."""

import pytest

from experiments import roofline

pytestmark = pytest.mark.unit


def test_gemm_traffic_reads_each_operand_once_when_tiles_fit() -> None:
    sram = 1 << 20
    assert roofline.gemm_traffic((256, 512, 1024, 1), sram) == (
        256 * 1024 + 512 * 1024
    )


def test_gemm_traffic_rereads_operands_across_tiles() -> None:
    # 1 MB of bf16 is a 724-wide square tile: 2048 columns need three
    # passes over the left operand and 4096 rows six over the right.
    sram = 1 << 20
    assert roofline.gemm_traffic((4096, 2048, 64, 1), sram) == (
        4096 * 64 * 3 + 2048 * 64 * 6
    )


def test_latency_is_the_larger_of_compute_and_memory_time() -> None:
    # 1 TMAC/s and 1 GB/s: 1e9 MACs and 1e6 bytes are each one millisecond.
    totals = {"Linear": [1e9, 1e6], "Softmax": [1e6, 1e6]}
    compute, memory, total = roofline.latency(totals, 1.0, 1)
    assert compute == pytest.approx(1.001)
    assert memory == pytest.approx(2.0)
    assert total == pytest.approx(2.0)
