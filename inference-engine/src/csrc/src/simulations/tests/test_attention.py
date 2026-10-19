"""Scalar attention: serial and flash (resident K/V), chunked flash (streamed).

ELF: bin/bench_attention.elf. Run: python run_tests.py attention
"""

import cocotb
import numpy as np
from bench import (
    KV_BLOCK,
    SCORE_SCALE,
    Arena,
    Bench,
    Q,
    attention_inputs,
    run_chunked,
)
from cocotb.handle import SimHandleBase

NQ, NK, D = 16, 64, 64  # one 16-query block over 64 resident keys


async def run_attention(
    bench: Bench,
    name: str,
    op: str,
    nq: int,
    nk: int,
    d: int,
    query_start: int,
    phase: str,
) -> None:
    """One serial or flash call over the whole (resident) K and V."""
    q, kt, v = attention_inputs(np.random.default_rng(nq * 7 + nk), nq, nk, d)
    arena = Arena()
    oq, ok, ov, oo = (
        arena.take(x) for x in (q.nbytes, kt.nbytes, v.nbytes, nq * d)
    )
    await bench.call(
        name,
        op,
        [
            nq,
            nk,
            d,
            query_start,
            nk,
            oq,
            ok,
            ov,
            oo,
            Q["k_zero"],
            Q["v_zero"],
            SCORE_SCALE,
            Q["v_scale"],
            Q["attn_out_scale"],
        ],
        {"q": (oq, q), "kt": (ok, kt), "v": (ov, v)},
        {"out": (oo, np.int8, (nq, d))},
        {
            "kernel": op,
            "shape": [nq, nk, d],
            "phase": phase,
            "query_start": query_start,
            "window": nk,
            "ops": 4 * nq * (query_start + (nq + 1) / 2) * d,
        },
    )


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """Serial and flash (resident KV) plus chunked flash (streamed KV)."""
    bench = await Bench.load(dut, "bench_attention")
    for op in ["attention_serial", "attention_flash"]:
        await run_attention(
            bench, f"{op}_prefill", op, NQ, NK, D, NK - NQ, "prefill"
        )
        await run_attention(
            bench, f"{op}_decode", op, 1, NK, D, NK - 1, "decode"
        )
    # same shape as serial, 16-key blocks
    await run_chunked(
        bench, "chunk_vs_serial", NQ, NK, D, 16, NK - NQ, "prefill"
    )
    # streamed context 4x what fits, in 64-key blocks
    nk = 4 * KV_BLOCK
    await run_chunked(
        bench, "chunk_prefill", NQ, nk, D, KV_BLOCK, nk - NQ, "prefill"
    )
    await run_chunked(
        bench, "chunk_decode", 1, nk, D, KV_BLOCK, nk - 1, "decode"
    )
    bench.finish()
