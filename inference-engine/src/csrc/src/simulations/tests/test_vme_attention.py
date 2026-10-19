"""Block-flash attention with both matmuls on the matrix engine.

Streams one head's K/V through DTCM: vme_attention_block per 64-key block,
then the finish step. ELF: bin/bench_vme_attention.elf.
Run: python run_tests.py vme_attention
"""

import cocotb
from bench import KV_BLOCK, NEO_D_HEAD, NEO_WINDOW, Bench, run_chunked
from cocotb.handle import SimHandleBase

NQ, D = 16, 64  # GPT-2 head
VME = "vme_attention_block"


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """GPT-2 (d 64) and GPT-Neo 28M (d 32, global and local) streams."""
    bench = await Bench.load(dut, "bench_vme_attention")
    nk = 4 * KV_BLOCK
    # 64 keys in 16-key blocks: the attention_serial_prefill inputs
    await run_chunked(bench, "vme_vs_serial", NQ, 64, D, 16, 48, "prefill", VME)
    for name, d in [("vme", D), ("vme_d32", NEO_D_HEAD)]:
        await run_chunked(
            bench,
            f"{name}_prefill",
            NQ,
            nk,
            d,
            KV_BLOCK,
            nk - NQ,
            "prefill",
            VME,
        )
        # decode: nq = 1, so matmul() takes the vector unit inside the kernel
        await run_chunked(
            bench, f"{name}_decode", 1, nk, d, KV_BLOCK, nk - 1, "decode", VME
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
        VME,
        window=NEO_WINDOW,
    )
    bench.finish()
