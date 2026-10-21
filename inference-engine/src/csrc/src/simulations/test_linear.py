"""The linear operator: matmul on both engines, then linear() on top.

ELF: bin/bench_linear.elf. Run: python sim.py test linear
"""

import cocotb
from bench import Bench, run_linear, run_matmul
from cocotb.handle import SimHandleBase

VME_TILE = (16, 512, 16)  # prefill tile
RVV_TILE = (1, 512, 32)  # decode tile


async def matmul_cases(bench: Bench) -> None:
    """VME and RVV tiles, the dispatcher, and the VME k sweep."""
    vn, vk, vm = VME_TILE
    rn, rk, rm = RVV_TILE
    # prefill tiles on both engines
    for op in [
        "vme_matmul_u8",
        "vme_matmul_s8u8",
        "matmul_rvv",
        "matmul_rvv_s8u8",
    ]:
        await run_matmul(
            bench, f"{op}_{vn}x{vk}x{vm}", op, vn, vk, vm, "prefill"
        )
    # decode tiles: matrix-vector on RVV
    for op in ["matmul_rvv", "matmul_rvv_s8u8"]:
        await run_matmul(
            bench, f"{op}_{rn}x{rk}x{rm}", op, rn, rk, rm, "decode"
        )
    await run_matmul(
        bench, f"matmul_rvv_1x{vk}x{vm}", "matmul_rvv", 1, vk, vm, "decode"
    )
    # dispatcher on both paths, and an odd shape on RVV
    await run_matmul(
        bench,
        "matmul_dispatch_vme",
        "matmul",
        vn,
        vk,
        vm,
        "prefill",
        signed=True,
    )
    await run_matmul(
        bench, "matmul_dispatch_rvv", "matmul", 1, vk, vm, "decode", signed=True
    )
    await run_matmul(
        bench, "matmul_rvv_odd_7x13x23", "matmul_rvv", 7, 13, 23, "check"
    )
    # VME k sweep at one 16x16 tile: the slope is cycles per matmul_step
    for k in [16, 32, 64, 128, 256, 512]:
        await run_matmul(
            bench, f"sweep_vme_k{k}", "vme_matmul_u8", 16, k, 16, "sweep"
        )


async def linear_cases(bench: Bench) -> None:
    """linear() at the matmul tiles, with and without bias."""
    vn, vk, vm = VME_TILE
    rn, rk, rm = RVV_TILE
    await run_linear(bench, f"linear_{vn}x{vk}x{vm}", vn, vk, vm, "prefill")
    await run_linear(bench, f"linear_{rn}x{rk}x{rm}", rn, rk, rm, "decode")
    await run_linear(bench, "linear_odd_7x13x23", 7, 13, 23, "check", False)


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """Matmul tiles and sweep, then linear()."""
    bench = await Bench.load(dut, "bench_linear")
    await matmul_cases(bench)
    await linear_cases(bench)
    bench.finish()
