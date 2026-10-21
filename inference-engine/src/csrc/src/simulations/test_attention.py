"""Matmul on both engines and block-flash attention on the matrix engine.

ELF: bin/bench_attention.elf. Run: python sim.py test attention
"""

import cocotb
import numpy as np
from bench import (
    KV_BLOCK,
    NEO_D_HEAD,
    NEO_WINDOW,
    SCORE_SCALE,
    Arena,
    Bench,
    run_chunked,
    run_matmul,
)
from cocotb.handle import SimHandleBase

VME_TILE = (16, 512, 16)  # prefill tile
RVV_TILE = (1, 512, 32)  # decode tile
NQ, D = 16, 64  # GPT-2 head


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


async def attention_cases(bench: Bench) -> None:
    """GPT-2 (d 64) and GPT-Neo 28M (d 32, global and local) streams."""
    # the once-per-layer exp table build, checked against the exact exp
    ot = Arena().take(1024)
    await bench.call(
        "softmax_lut_table",
        "softmax_lut_table",
        [SCORE_SCALE, ot],
        {},
        {"table": (ot, np.uint16, (512,))},
        {
            "kernel": "softmax_lut_table",
            "shape": [512],
            "phase": "both",
            "ops": 0,
        },
    )
    # 64 keys in 16-key blocks: ragged per-block masking
    await run_chunked(bench, "vme_block16", NQ, 64, D, 16, 48, "prefill")
    nk = 4 * KV_BLOCK
    for name, d in [("vme", D), ("vme_d32", NEO_D_HEAD)]:
        await run_chunked(
            bench, f"{name}_prefill", NQ, nk, d, KV_BLOCK, nk - NQ, "prefill"
        )
        # decode: nq = 1, so matmul() takes the vector unit inside the kernel
        await run_chunked(
            bench, f"{name}_decode", 1, nk, d, KV_BLOCK, nk - 1, "decode"
        )
    # a GPT-Neo local layer: its 256-key window skips the oldest blocks
    nk = 6 * KV_BLOCK
    await run_chunked(
        bench,
        "vme_d32_local",
        NQ,
        nk,
        NEO_D_HEAD,
        KV_BLOCK,
        nk - NQ,
        "prefill",
        window=NEO_WINDOW,
    )


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """Matmul tiles and sweep, then the attention streams."""
    bench = await Bench.load(dut, "bench_attention")
    await matmul_cases(bench)
    await attention_cases(bench)
    bench.finish()
