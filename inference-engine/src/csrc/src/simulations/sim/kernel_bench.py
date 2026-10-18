"""Cocotb bench: run Reef kernels on Coral's VME RTL model (VmeCoreMiniAxi).

Each job writes a kernel's inputs into the bench ELF's DTCM `arena` over the
AXI slave (timed in clock cycles: there is no DMA, so this is the host->DTCM
cost), runs one call to halt, and reads back the outputs plus `bench_cycles`
(mcycle around the kernel call only). Every job is saved as an .npz in the
Bazel undeclared-outputs dir; run_all.py checks correctness and reports.
Shapes and scales come from model.json (copied next to this file).
"""

import json
import math
import os

import cocotb
import numpy as np
from bazel_tools.tools.python.runfiles import runfiles
from cocotb.handle import SimHandleBase
from cocotb.utils import get_sim_time
from coralnpu_test_utils.sim_backends.verilator_test_fixture import (
    VerilatorTestFixture,
)

ARENA_BYTES = 20 * 1024  # bench_main.cc kArenaBytes
TIMEOUT_CYCLES = 50_000_000
RUNFILES = runfiles.Create()


def runfile(name: str) -> str:
    """Return the runfiles path of a file in this package."""
    return RUNFILES.Rlocation(f"coralnpu_hw/tests/cocotb/reef/{name}")


with open(runfile("model.json")) as f:
    MODEL = json.load(f)
Q, T = MODEL["quant"], MODEL["tiles"]
# same as config/gen_params.py derived(): GPT-2's 1/sqrt(d_head) folded in
SCORE_SCALE = Q["q_scale"] * Q["k_scale"] / math.sqrt(MODEL["model"]["d_head"])
OUT_DIR = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")

# bench_main.cc Op values
OPS = {
    "matmul_vme_u8": 1,
    "matmul_vme_s8u8": 2,
    "matmul_rvv": 3,
    "matmul_rvv_s8u8": 4,
    "matmul": 5,
    "attention_serial": 10,
    "attention_flash": 11,
    "attention_flash_chunk": 12,
    "attention_flash_finish": 13,
    "online_softmax": 20,
    "gelu": 21,
    "layernorm": 22,
    "residual_add": 23,
    "load_chain_scalar": 30,
    "load_chain_vector": 31,
}


def f32_bits(value: float) -> int:
    """IEEE-754 bits of a float32, as bench_main.cc's arg_f() reads them."""
    return int(np.float32(value).view(np.uint32))


class Arena:
    """Bump allocator for 16-byte-aligned offsets in the DTCM arena."""

    def __init__(self) -> None:
        """Start with an empty arena."""
        self.top = 0

    def take(self, nbytes: int) -> int:
        """Reserve nbytes and return their offset."""
        offset = self.top
        self.top += -(-nbytes // 16) * 16
        assert self.top <= ARENA_BYTES, f"arena overflow: {self.top} bytes"
        return offset


class Bench:
    """One loaded bench ELF on a fresh VmeCoreMiniAxi model."""

    def __init__(self, fixture: VerilatorTestFixture) -> None:
        """Wrap a fixture whose bench ELF is already loaded."""
        self.fx = fixture
        self.clock_ns = fixture.core_mini_axi.clock_ns

    @classmethod
    async def load(cls, dut: SimHandleBase, elf: str) -> "Bench":
        """Start a fresh model and load bench ELF `elf`."""
        fixture = await VerilatorTestFixture.Create(dut)
        await fixture.load_elf_and_lookup_symbols(
            runfile(f"{elf}.elf"), ["arena", "bench_args", "bench_cycles"]
        )
        return cls(fixture)

    async def write(self, offset: int, data: np.ndarray) -> int:
        """Write data into the arena; return the clock cycles it took."""
        start = get_sim_time(unit="ns")
        await self.fx.write("arena", np.ascontiguousarray(data), offset=offset)
        return round((get_sim_time(unit="ns") - start) / self.clock_ns)

    async def call(
        self,
        name: str,
        op: str,
        args: list,
        writes: dict,
        reads: dict,
        meta: dict,
        save: dict | None = None,
    ) -> dict:
        """Run one kernel call and save its inputs, outputs and cycles.

        writes: {name: (offset, array)} written before the run.
        reads:  {name: (offset, dtype, shape)} read back after it.
        save:   {name: array} saved with the record but not written (e.g.
                the full K/V of a streamed run, for the correctness check).
        """
        write_cycles = 0
        write_bytes = 0
        for offset, array in writes.values():
            write_cycles += await self.write(offset, array)
            write_bytes += array.nbytes
        words = [OPS[op]] + [
            f32_bits(a) if isinstance(a, float) else int(a) & 0xFFFFFFFF
            for a in args
        ]
        await self.fx.write("bench_args", np.array(words, dtype=np.uint32))
        halted = await self.fx.run_to_halt(timeout_cycles=TIMEOUT_CYCLES)
        cycles = int(await self.fx.read_word("bench_cycles"))
        outputs = {}
        for key, (offset, dtype, shape) in reads.items():
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            raw = await self.fx.read("arena", nbytes, offset=offset)
            outputs[key] = np.frombuffer(bytes(raw), dtype=dtype).reshape(shape)
        cocotb.log.info(
            f"{name}: halted at cycle {halted}, kernel {cycles} cycles, "
            f"host wrote {write_bytes} B in {write_cycles} cycles"
        )
        record = {
            "name": name,
            "op": op,
            "cycles": cycles,
            "halted": halted,
            "host_write_bytes": write_bytes,
            "host_write_cycles": write_cycles,
            **meta,
        }
        np.savez(
            os.path.join(OUT_DIR, f"{name}.npz"),
            meta=json.dumps(record),
            **{f"in_{k}": v for k, (_, v) in writes.items()},
            **{f"in_{k}": v for k, v in (save or {}).items()},
            **{f"out_{k}": v for k, v in outputs.items()},
        )
        return outputs


async def run_matmul(
    bench: Bench,
    name: str,
    op: str,
    n: int,
    k: int,
    m: int,
    phase: str,
    signed: bool | None = None,
) -> None:
    """One matmul call on random inputs (signed A for the s8u8 variants)."""
    rng = np.random.default_rng(n * 1_000_003 + k * 1009 + m)
    signed = op.endswith("s8u8") if signed is None else signed
    a = rng.integers(-128 if signed else 0, 128 if signed else 256, (n, k))
    a = a.astype(np.int8 if signed else np.uint8)
    b = rng.integers(0, 256, (k, m)).astype(np.uint8)
    arena = Arena()
    oa, ob, oc = (
        arena.take(a.nbytes),
        arena.take(b.nbytes),
        arena.take(n * m * 4),
    )
    await bench.call(
        name,
        op,
        [n, k, m, oa, ob, oc, int(signed)],
        {"a": (oa, a), "b": (ob, b)},
        {"c": (oc, np.int32, (n, m))},
        {
            "kernel": op,
            "shape": [n, k, m],
            "phase": phase,
            "signed": signed,
            "ops": 2 * n * k * m,
        },
    )


@cocotb.test()
async def matmul_test(dut: SimHandleBase) -> None:
    """VME and RVV matmul tiles, the dispatcher, and the VME k sweep."""
    bench = await Bench.load(dut, "bench_matmul")
    vn, vk, vm = (T["matmul_vme"][x] for x in "nkm")
    rn, rk, rm = (T["matmul_rvv"][x] for x in "nkm")
    # prefill tiles on both engines
    for op in [
        "matmul_vme_u8",
        "matmul_vme_s8u8",
        "matmul_rvv",
        "matmul_rvv_s8u8",
    ]:
        await run_matmul(
            bench, f"{op}_{vn}x{vk}x{vm}", op, vn, vk, vm, "prefill"
        )
    # decode tiles: matrix-vector on RVV, and on VME padded to 16 rows
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
            bench, f"sweep_vme_k{k}", "matmul_vme_u8", 16, k, 16, "sweep"
        )


def attention_inputs(
    rng: np.random.Generator, nq: int, nk: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random q (int8) and K^T, V (uint8 around their zero points)."""
    q = rng.integers(-64, 64, (nq, d)).astype(np.int8)
    kt = np.clip(rng.normal(Q["k_zero"], 40, (d, nk)), 0, 255).astype(np.uint8)
    v = np.clip(rng.normal(Q["v_zero"], 40, (nk, d)), 0, 255).astype(np.uint8)
    return q, kt, v


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


async def run_chunked(
    bench: Bench,
    name: str,
    nq: int,
    nk: int,
    d: int,
    block: int,
    query_start: int,
    phase: str,
) -> None:
    """Stream K and V through DTCM in `block`-key blocks, then finish."""
    q, kt, v = attention_inputs(np.random.default_rng(nq * 7 + nk), nq, nk, d)
    arena = Arena()
    oq, ok, ov = (
        arena.take(q.nbytes),
        arena.take(d * block),
        arena.take(block * d),
    )
    om, ol, oa = arena.take(nq * 4), arena.take(nq * 4), arena.take(nq * d * 4)
    oo = arena.take(nq * d)
    # q and the zeroed state go in once; only K/V blocks stream per call
    setup = {
        "q": (oq, q),
        "l": (ol, np.zeros(nq, np.float32)),
        "acc": (oa, np.zeros(nq * d, np.float32)),
    }
    for start in range(0, nk, block):
        kt_block = np.ascontiguousarray(kt[:, start : start + block])
        await bench.call(
            f"{name}_block{start // block}",
            "attention_flash_chunk",
            [
                nq,
                d,
                query_start,
                nk,
                start,
                block,
                oq,
                ok,
                ov,
                om,
                ol,
                oa,
                Q["k_zero"],
                SCORE_SCALE,
            ],
            {
                **(setup if start == 0 else {}),
                "kt": (ok, kt_block),
                "v": (ov, v[start : start + block]),
            },
            {},
            {
                "kernel": "attention_flash_chunk",
                "shape": [nq, block, d],
                "phase": phase,
                "stream": name,
                "kv_start": start,
                "ops": 4 * nq * block * d,
            },
        )
    await bench.call(
        f"{name}_finish",
        "attention_flash_finish",
        [nq, d, ol, oa, oo, Q["v_zero"], Q["v_scale"], Q["attn_out_scale"]],
        {},
        {"out": (oo, np.int8, (nq, d))},
        {
            "kernel": "attention_flash_finish",
            "shape": [nq, nk, d],
            "phase": phase,
            "stream": name,
            "query_start": query_start,
            "window": nk,
            "ops": 0,
        },
        save={"q": q, "kt": kt, "v": v},
    )


@cocotb.test()
async def attention_test(dut: SimHandleBase) -> None:
    """Serial and flash (resident KV) plus chunked flash (streamed KV)."""
    bench = await Bench.load(dut, "bench_attention")
    nq, nk, d = (T["attention_serial"][x] for x in ("nq", "nk", "d"))
    for op in ["attention_serial", "attention_flash"]:
        await run_attention(
            bench, f"{op}_prefill", op, nq, nk, d, nk - nq, "prefill"
        )
        await run_attention(
            bench, f"{op}_decode", op, 1, nk, d, nk - 1, "decode"
        )
    # same shape as serial, 16-key blocks: checked directly against serial
    await run_chunked(
        bench, "chunk_vs_serial", nq, nk, d, 16, nk - nq, "prefill"
    )
    # streamed context 4x what fits: 64-key blocks from the config
    block = T["attention_flash_chunk"]["kv_block"]
    await run_chunked(
        bench,
        "chunk_prefill",
        nq,
        4 * block,
        d,
        block,
        4 * block - nq,
        "prefill",
    )
    await run_chunked(
        bench, "chunk_decode", 1, 4 * block, d, block, 4 * block - 1, "decode"
    )


@cocotb.test()
async def pointwise_test(dut: SimHandleBase) -> None:
    """Softmax, GeLU, LayerNorm and residual add at their config tiles."""
    bench = await Bench.load(dut, "bench_pointwise")
    rng = np.random.default_rng(5)

    row = T["softmax"]["row"]
    scores = rng.integers(-40000, 40000, row).astype(np.int32)
    for name, lo, hi, phase in [
        ("online_softmax_row", 0, row - 1, "decode"),
        ("online_softmax_64", 0, 63, "prefill"),
    ]:
        arena = Arena()
        os_, op = arena.take(scores.nbytes), arena.take(row)
        await bench.call(
            name,
            "online_softmax",
            [lo, hi, os_, op, SCORE_SCALE],
            {"scores": (os_, scores)},
            {"probs": (op, np.uint8, (row,))},
            {
                "kernel": "online_softmax",
                "shape": [hi - lo + 1],
                "phase": phase,
                "lo": lo,
                "hi": hi,
                "ops": 6 * (hi - lo + 1),
            },
        )

    n = T["gelu"]["n"]
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

    rows, d = T["layernorm"]["rows"], T["layernorm"]["d"]
    x = rng.integers(-60, 60, (rows, d)).astype(np.int8)
    gamma = rng.integers(20, 70, d).astype(np.int8)
    beta = rng.integers(-30, 30, d).astype(np.int8)
    arena = Arena()
    ox, og, ob, oo = (arena.take(s) for s in (x.nbytes, d, d, x.nbytes))
    await bench.call(
        "layernorm",
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

    n = T["residual"]["n"]
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


@cocotb.test()
async def ubench_test(dut: SimHandleBase) -> None:
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
