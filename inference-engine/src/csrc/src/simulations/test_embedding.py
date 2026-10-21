"""The embedding operator: token lookup from table slices, position add.

ELF: bin/bench_embedding.elf. Run: python sim.py test embedding
"""

import cocotb
from bench import (
    NEO_D_MODEL,
    Bench,
    run_embedding_lookup,
    run_position_embeddings,
)
from cocotb.handle import SimHandleBase


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """GPT-Neo 28M's width, prefill and decode."""
    bench = await Bench.load(dut, "bench_embedding")
    d = NEO_D_MODEL
    # 16 tokens drawn from 64 vocab rows, streamed as 8-row (4 KB) slices
    await run_embedding_lookup(bench, "embed_prefill", 16, d, 64, 8, "prefill")
    await run_embedding_lookup(bench, "embed_decode", 1, d, 8, 8, "decode")
    # position rows: 8 prefill tokens from row 0; one decode token at row 4
    # of a 16-row slice (absolute position 100 in a slice starting at 96)
    await run_position_embeddings(
        bench, "position_prefill", 8, d, 8, 0, "prefill"
    )
    await run_position_embeddings(
        bench, "position_decode", 1, d, 16, 4, "decode"
    )
    bench.finish()
