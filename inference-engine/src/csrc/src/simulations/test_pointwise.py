"""Pointwise kernels: GeLU, LayerNorm and residual add.

ELF: bin/bench_pointwise.elf. Run: python sim.py test pointwise
"""

import cocotb
import numpy as np
from bench import NEO_D_MODEL, Arena, Bench, Q
from cocotb.handle import SimHandleBase

GELU_N = 4096
LAYERNORM_ROWS, LAYERNORM_D = 4, 768
RESIDUAL_N = 4096


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """GeLU, LayerNorm and residual add at their tile sizes."""
    bench = await Bench.load(dut, "bench_pointwise")
    rng = np.random.default_rng(5)

    n = GELU_N
    x = rng.integers(-128, 128, n).astype(np.int8)
    arena = Arena()
    ox, ot, oy = arena.take(n), arena.take(256), arena.take(n)
    await bench.call(
        "gelu",
        "gelu",
        [n, ox, ot, oy, Q["gelu_in_scale"], Q["gelu_out_scale"]],
        {"x": (ox, x)},
        {"y": (oy, np.int8, (n,)), "table": (ot, np.int8, (256,))},
        {"kernel": "gelu", "shape": [n], "phase": "both", "ops": n},
    )

    # GPT-2's row width, then GPT-Neo 28M's
    rows = LAYERNORM_ROWS
    for name, d in [
        ("layernorm", LAYERNORM_D),
        ("layernorm_d512", NEO_D_MODEL),
    ]:
        x = rng.integers(-60, 60, (rows, d)).astype(np.int8)
        gamma = rng.integers(20, 70, d).astype(np.int8)
        beta = rng.integers(-30, 30, d).astype(np.int8)
        arena = Arena()
        ox, og, ob, oo = (arena.take(s) for s in (x.nbytes, d, d, x.nbytes))
        await bench.call(
            name,
            "layernorm",
            [
                rows,
                d,
                ox,
                og,
                ob,
                oo,
                Q["act_scale"],
                Q["ln_gamma_scale"],
                Q["ln_beta_scale"],
                Q["ln_out_scale"],
            ],
            {"x": (ox, x), "gamma": (og, gamma), "beta": (ob, beta)},
            {"out": (oo, np.int8, (rows, d))},
            {
                "kernel": "layernorm",
                "shape": [rows, d],
                "phase": "both",
                "ops": 8 * rows * d,
            },
        )

    n = RESIDUAL_N
    a = rng.integers(-128, 128, n).astype(np.int8)
    b = rng.integers(-128, 128, n).astype(np.int8)
    arena = Arena()
    oa, ob, oo = arena.take(n), arena.take(n), arena.take(n)
    await bench.call(
        "residual_add",
        "residual_add",
        [
            n,
            oa,
            ob,
            oo,
            Q["act_scale"],
            Q["act_scale"],
            Q["residual_out_scale"],
        ],
        {"a": (oa, a), "b": (ob, b)},
        {"out": (oo, np.int8, (n,))},
        {"kernel": "residual_add", "shape": [n], "phase": "both", "ops": 3 * n},
    )
    bench.finish()
