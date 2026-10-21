"""The attention operator: block-flash attention on the matrix engine.

ELF: bin/bench_attention.elf. Run: python sim.py test attention
"""

import cocotb
from bench import KV_BLOCK, NEO_D_HEAD, NEO_WINDOW, Bench, run_chunked
from cocotb.handle import SimHandleBase

NQ, D = 16, 64  # GPT-2 head


async def attention_cases(bench: Bench) -> None:
    """GPT-2 (d 64) and GPT-Neo 28M (d 32, global and local) streams."""
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
    """The attention streams."""
    bench = await Bench.load(dut, "bench_attention")
    await attention_cases(bench)
    bench.finish()
