"""The MLP operator: c_fc, GeLU and c_proj streamed over the inner width.

ELF: bin/bench_mlp.elf. Run: python sim.py test mlp
"""

import cocotb
from bench import NEO_D_MODEL, Bench, run_mlp
from cocotb.handle import SimHandleBase

CHUNK = 16  # inner-width columns per mlp_block call


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """Prefill on the matrix engine, decode on the vector unit."""
    bench = await Bench.load(dut, "bench_mlp")
    # 16 rows x d 64 is the largest prefill tile whose running sums fit in
    # DTCM next to its weights; d 512 needs c_proj's outputs split
    await run_mlp(bench, "mlp_prefill", 16, 64, 4 * CHUNK, CHUNK, "prefill")
    # GPT-Neo 28M's width at decode (1 row: the vector unit), in 8-wide
    # chunks so a chunk of both weights fits
    await run_mlp(bench, "mlp_d512_decode", 1, NEO_D_MODEL, 64, 8, "decode")
    bench.finish()
