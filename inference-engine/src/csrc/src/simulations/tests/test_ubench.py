"""Microbenchmarks: load-use latency chains and host->DTCM write timing.

ELF: bin/bench_ubench.elf. Run: python run_tests.py ubench
"""

import json
import os

import cocotb
import numpy as np
from bench import OUT_DIR, Arena, Bench
from cocotb.handle import SimHandleBase


@cocotb.test()
async def test(dut: SimHandleBase) -> None:
    """Load-use latency chains and host->DTCM write timing."""
    bench = await Bench.load(dut, "bench_ubench")
    arena = Arena()
    op = arena.take(64)
    base = bench.fx.resolve_address("arena", offset=op)
    # every word holds its own address, so each load feeds the next
    self_ptr = np.full(16, base, dtype=np.uint32)
    for kind in ["scalar", "vector"]:
        for iters in [64, 256]:
            await bench.call(
                f"load_chain_{kind}_{iters}",
                f"load_chain_{kind}",
                [iters, op],
                {"p": (op, self_ptr)},
                {},
                {
                    "kernel": f"load_chain_{kind}",
                    "shape": [iters],
                    "phase": "ubench",
                    "loads": 4 * iters if kind == "scalar" else 2 * iters,
                    "ops": 0,
                },
            )
    # host->DTCM: time AXI writes of growing size while the core is halted
    sweep = {}
    for nbytes in [256, 1024, 4096, 16384]:
        sweep[nbytes] = await bench.write(0, np.zeros(nbytes, np.uint8))
        cocotb.log.info(f"axi write {nbytes} B: {sweep[nbytes]} cycles")
    with open(os.path.join(OUT_DIR, "axi_write_sweep.json"), "w") as f:
        json.dump({"clock_ns": bench.clock_ns, "cycles": sweep}, f)
