"""Shared cocotb harness for the Reef kernel tests on VmeCoreMiniAxi.

Each test loads one bench ELF from bin/ (built by sim.py build), writes a
kernel's inputs into its DTCM `arena` over the AXI slave (timed in clock
cycles: there is no DMA, so this is the host->DTCM cost), runs one call to
halt, reads back the outputs plus `bench_cycles` (mcycle around the kernel
call only), and checks them against a NumPy reference. Every call is saved
as an .npz in the Bazel undeclared-outputs dir; sim.py test copies them to
out/<test>/. The call protocol is in bench_main.cc.
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
OUT_DIR = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")

# GPT-2 small quantization and tile shapes (analysis/config/model.json)
Q = {
    "act_scale": 0.05,
    "q_scale": 0.05,
    "k_scale": 0.05,
    "k_zero": 128,
    "v_scale": 0.05,
    "v_zero": 128,
    "attn_out_scale": 0.05,
    "gelu_in_scale": 0.05,
    "gelu_out_scale": 0.05,
    "ln_gamma_scale": 0.02,
    "ln_beta_scale": 0.02,
    "ln_out_scale": 0.05,
    "residual_out_scale": 0.05,
    "embedding_scale": 0.02,
}
D_HEAD = 64
LAYERNORM_EPS = 1e-5
SCORE_SCALE = Q["q_scale"] * Q["k_scale"] / math.sqrt(D_HEAD)
KV_BLOCK = 64  # keys per streamed K/V block
# GPT-Neo 28M head size, local window and row width
NEO_D_HEAD, NEO_WINDOW, NEO_D_MODEL = 32, 256, 512

# bench_main.cc Op values
OPS = {
    "vme_matmul_u8": 1,
    "vme_matmul_s8u8": 2,
    "matmul_rvv": 3,
    "matmul_rvv_s8u8": 4,
    "matmul": 5,
    "linear": 6,
    "attention_flash_finish": 13,
    "vme_attention_block": 14,
    "gelu": 21,
    "layernorm": 22,
    "residual_add": 23,
    "softmax_lut": 24,
    "softmax_lut_table": 25,
    "mlp_block": 40,
    "mlp_finish": 41,
    "embedding_lookup": 50,
    "add_position_embeddings": 51,
}
NULL = 0xFFFFFFFF  # bench_main.cc kNull: no buffer (e.g. no bias)
W_ZERO = 128  # weight zero point


def f32_bits(value: float) -> int:
    """IEEE-754 bits of a float32, as bench_main.cc's arg_f() reads them."""
    return int(np.float32(value).view(np.uint32))


def exp_table(score_scale: float) -> np.ndarray:
    """softmax_lut_table's Q15 exp(-score_scale * gap) table, in float32."""
    gap = np.arange(256, dtype=np.float32)
    scale = np.float32(score_scale)
    coarse = np.exp(-scale * np.float32(256) * gap)
    fine = np.exp(-scale * gap)
    q15 = np.float32(32768) * np.concatenate([coarse, fine]) + np.float32(0.5)
    return q15.astype(np.uint16)


# --------------------------------------------------------------------------
# NumPy references
# --------------------------------------------------------------------------


def to_int8(x: np.ndarray) -> np.ndarray:
    """The kernels' requantization: round half away from zero, clamp."""
    return np.clip(np.sign(x) * np.floor(np.abs(x) + 0.5), -128, 127)


def attention_ref(
    q: np.ndarray,
    kt: np.ndarray,
    v: np.ndarray,
    query_start: int,
    window: int,
) -> np.ndarray:
    """Float64 causal attention over dequantized inputs, in output LSBs."""
    q, kt, v = (np.asarray(x, np.float64) for x in (q, kt, v))
    scores = SCORE_SCALE * (q @ (kt - Q["k_zero"]))
    out = np.empty((q.shape[0], v.shape[1]))
    for row in range(q.shape[0]):
        pos = query_start + row
        lo = max(0, pos + 1 - window)
        s = scores[row, lo : pos + 1]
        p = np.exp(s - s.max())
        out[row] = (p / p.sum()) @ (v[lo : pos + 1] - Q["v_zero"])
    return Q["v_scale"] * out / Q["attn_out_scale"]


def attention_u8_ref(
    q: np.ndarray,
    kt: np.ndarray,
    v: np.ndarray,
    query_start: int,
    window: int,
    block: int,
) -> np.ndarray:
    """vme_attention_block's own algorithm in float64, in output LSBs.

    Streams `block`-key blocks with a running max and rounds each block's
    probabilities to uint8 (scale 1/255) against it, so it shows how far
    uint8 P alone is from attention_ref on these inputs.
    """
    scores = SCORE_SCALE * (
        q.astype(np.float64) @ (kt.astype(np.float64) - Q["k_zero"])
    )
    v = v.astype(np.float64)
    out = np.empty((q.shape[0], v.shape[1]))
    for row in range(q.shape[0]):
        pos = query_start + row
        lo = max(0, pos + 1 - window)
        m, total, acc = -np.inf, 0.0, np.zeros(v.shape[1])
        for start in range(lo // block * block, pos + 1, block):
            a, b = max(lo, start), min(pos + 1, start + block)
            s = scores[row, a:b]
            m_new = max(m, s.max())
            corr = np.exp(m - m_new)
            p = np.floor(255 * np.exp(s - m_new) + 0.5)
            acc = acc * corr + p @ v[a:b] / 255
            total, m = total * corr + p.sum() / 255, m_new
        out[row] = acc / total - Q["v_zero"]
    return Q["v_scale"] * out / Q["attn_out_scale"]


def linear_acc_ref(
    x: np.ndarray, wt: np.ndarray, bias: np.ndarray | None
) -> np.ndarray:
    """Exact int64 accumulator of linear_acc: x . (wt - W_ZERO) + bias."""
    acc = x.astype(np.int64) @ (wt.astype(np.int64) - W_ZERO)
    return acc if bias is None else acc + bias.astype(np.int64)


def gelu_table_ref(in_scale: float, out_scale: float) -> np.ndarray:
    """gelu_table's 256 entries: table[x + 128] = GeLU(x) in int8."""
    x = np.arange(-128, 128, dtype=np.float64)
    r = x * in_scale
    y = 0.5 * r * (1 + np.tanh(np.sqrt(2 / np.pi) * (r + 0.044715 * r**3)))
    return to_int8(y / out_scale).astype(np.int8)


def mlp_acc_ref(data: dict, fc_scale: float) -> np.ndarray:
    """The unchunked c_fc -> GeLU -> c_proj accumulator, without c_proj bias.

    The GeLU input is rounded in float32 like the kernel, so the result
    must match its int32 sums exactly.
    """
    fc = linear_acc_ref(data["in_x"], data["in_fc_wt"], data["in_fc_bias"])
    pre = to_int8(np.float32(fc_scale) * fc.astype(np.float32))
    h = data["in_gelu_table"][pre.astype(np.int64) + 128]
    return linear_acc_ref(h, data["in_proj_wt"], None)


def gelu_ref(x: np.ndarray) -> np.ndarray:
    """GPT-2's tanh GeLU, input and output in int8 LSBs."""
    r = x.astype(np.float64) * Q["gelu_in_scale"]
    y = 0.5 * r * (1 + np.tanh(np.sqrt(2 / np.pi) * (r + 0.044715 * r**3)))
    return np.clip(y / Q["gelu_out_scale"], -128, 127)


def layernorm_ref(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Float64 LayerNorm in output LSBs."""
    r = x.astype(np.float64) * Q["act_scale"]
    norm = (r - r.mean(1, keepdims=True)) / np.sqrt(
        r.var(1, keepdims=True) + LAYERNORM_EPS
    )
    y = norm * gamma * Q["ln_gamma_scale"] + beta * Q["ln_beta_scale"]
    return np.clip(y / Q["ln_out_scale"], -128, 127)


def check(job: dict, data: dict) -> tuple[float, float]:
    """Return (max abs error in output LSBs, tolerance) for one call."""
    op = job["op"]
    if op == "linear":
        acc = linear_acc_ref(data["in_x"], data["in_wt"], data.get("in_bias"))
        expected = np.clip(acc * job["acc_scale"] / job["out_scale"], -128, 127)
        return float(np.abs(data["out_y"] - expected).max()), 0.5 + 1e-3
    if op == "mlp_finish":
        acc = mlp_acc_ref(data, job["fc_scale"])
        job["acc_mismatches"] = int((data["out_acc"] != acc).sum())
        real = (acc + data["in_proj_bias"]) * job["acc_scale"]
        expected = np.clip(real / job["out_scale"], -128, 127)
        err = float(np.abs(data["out_y"] - expected).max())
        # an accumulator off by anything is a kernel bug, not rounding
        return (err if job["acc_mismatches"] == 0 else math.inf), 0.5 + 1e-3
    if op == "embedding_lookup":
        # every position whose id is in this slice gets that table row; the
        # count it reports must be how many there were
        rows = data["in_ids"].astype(np.int64) - job["vocab_start"]
        hit = (rows >= 0) & (rows < data["in_table"].shape[0])
        wrong = (data["out_y"][hit] != data["in_table"][rows[hit]]).sum()
        wrong += int(data["out_filled"][0]) != int(hit.sum())
        return float(wrong), 0.0
    if op == "add_position_embeddings":
        off = job["position_offset"]
        pos = data["in_positions"][off : off + data["in_tokens"].shape[0]]
        real = (
            data["in_tokens"] * job["token_scale"] + pos * job["position_scale"]
        )
        expected = np.clip(real / job["out_scale"], -128, 127)
        return float(np.abs(data["out_y"] - expected).max()), 0.5 + 1e-4
    if op == "softmax_lut":
        lo, hi = job["lo"], job["hi"]
        s = SCORE_SCALE * data["in_scores"][lo : hi + 1].astype(np.float64)
        p = np.exp(s - s.max())
        expected = 255 * p / p.sum()
        return float(
            np.abs(data["out_probs"][lo : hi + 1] - expected).max()
        ), 1.0
    if "matmul" in op:
        a = data["in_a"].astype(np.int64)
        expected = a @ data["in_b"].astype(np.int64)
        return float(np.abs(data["out_c"] - expected).max()), 0.0
    if op == "attention_flash_finish":
        expected = attention_ref(
            data["in_q"],
            data["in_kt"],
            data["in_v"],
            job["query_start"],
            job["window"],
        )
        err = float(np.abs(data["out_out"] - expected).max())
        # uint8 probabilities round each p to 1/255, which costs most on
        # long, flat rows (p near 1/255 per key). The kernel must sit within
        # 1 LSB of its own uint8-P algorithm's error vs float.
        u8 = attention_u8_ref(
            data["in_q"],
            data["in_kt"],
            data["in_v"],
            job["query_start"],
            job["window"],
            job["block"],
        )
        u8_err = float(np.abs(to_int8(u8) - expected).max())
        job["uint8_probs_error_lsb"] = u8_err
        return err, u8_err + 1.0
    if op == "softmax_lut_table":
        table = data["out_table"].astype(np.int64)
        gap = np.arange(256)
        exact = 32768 * np.exp(-SCORE_SCALE * np.concatenate([256 * gap, gap]))
        return float(np.abs(table - exact).max()), 1.0
    if op == "gelu":
        return float(np.abs(data["out_y"] - gelu_ref(data["in_x"])).max()), 0.5
    if op == "layernorm":
        expected = layernorm_ref(
            data["in_x"], data["in_gamma"], data["in_beta"]
        )
        return float(np.abs(data["out_out"] - expected).max()), 1.0
    if op == "residual_add":
        real = data["in_a"] * Q["act_scale"] + data["in_b"] * Q["act_scale"]
        expected = np.clip(real / Q["residual_out_scale"], -128, 127)
        return float(np.abs(data["out_out"] - expected).max()), 0.5
    raise KeyError(op)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


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
        self.failures: list[str] = []

    @classmethod
    async def load(cls, dut: SimHandleBase, elf: str) -> "Bench":
        """Start a fresh model and load bin/<elf>.elf."""
        fixture = await VerilatorTestFixture.Create(dut)
        path = RUNFILES.Rlocation(
            f"coralnpu_hw/tests/cocotb/reef/bin/{elf}.elf"
        )
        await fixture.load_elf_and_lookup_symbols(
            path, ["arena", "bench_args", "bench_cycles"]
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
        """Run one kernel call, check its outputs and save the record.

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
        assert cycles != 0xFFFFFFFF, f"{name}: op {op} not in this bench ELF"
        outputs = {}
        for key, (offset, dtype, shape) in reads.items():
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            raw = await self.fx.read("arena", nbytes, offset=offset)
            outputs[key] = np.frombuffer(bytes(raw), dtype=dtype).reshape(shape)
        record = {
            "name": name,
            "op": op,
            "cycles": cycles,
            "halted": halted,
            "host_write_bytes": write_bytes,
            "host_write_cycles": write_cycles,
            **meta,
        }
        data = {
            **{f"in_{k}": v for k, (_, v) in writes.items()},
            **{f"in_{k}": v for k, v in (save or {}).items()},
            **{f"out_{k}": v for k, v in outputs.items()},
        }
        status = ""
        if outputs:
            err, tol = check(record, data)
            record.update(error_lsb=err, tolerance_lsb=tol, ok=err <= tol)
            status = f", error {err:.2f} LSB (limit {tol:.2f})"
            if not record["ok"]:
                self.failures.append(name)
        cocotb.log.info(
            f"{name}: halted at cycle {halted}, kernel {cycles} cycles, "
            f"host wrote {write_bytes} B in {write_cycles} cycles{status}"
        )
        np.savez(
            os.path.join(OUT_DIR, f"{name}.npz"),
            meta=json.dumps(record),
            **data,
        )
        return outputs

    def finish(self) -> None:
        """Fail the test if any checked call was out of tolerance."""
        assert not self.failures, f"out of tolerance: {self.failures}"


# --------------------------------------------------------------------------
# calls shared by several tests
# --------------------------------------------------------------------------


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


def attention_inputs(
    rng: np.random.Generator, nq: int, nk: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random q (int8) and K^T, V (uint8 around their zero points)."""
    q = rng.integers(-64, 64, (nq, d)).astype(np.int8)
    kt = np.clip(rng.normal(Q["k_zero"], 40, (d, nk)), 0, 255).astype(np.uint8)
    v = np.clip(rng.normal(Q["v_zero"], 40, (nk, d)), 0, 255).astype(np.uint8)
    return q, kt, v


async def run_chunked(
    bench: Bench,
    name: str,
    nq: int,
    nk: int,
    d: int,
    block: int,
    query_start: int,
    phase: str,
    window: int | None = None,
) -> None:
    """Stream K and V through DTCM in `block`-key blocks, then finish.

    Each block is one vme_attention_block call; the finish step turns the
    running state into the int8 output. window defaults to nk (global
    attention); blocks wholly before the first query's window are skipped,
    as the host would skip them.
    """
    window = window or nk
    q, kt, v = attention_inputs(np.random.default_rng(nq * 7 + nk), nq, nk, d)
    arena = Arena()
    oq, ok, ov = (
        arena.take(q.nbytes),
        arena.take(d * block),
        arena.take(block * d),
    )
    om, ol, oa = arena.take(nq * 4), arena.take(nq * 4), arena.take(nq * d * 4)
    # scores / P.V int32 (nq x max(block, d)), then P bytes (nq x block),
    # then the exp table (softmax_lut_table, 512 uint16). The finish step's
    # output reuses the scratch, which is free by then.
    os_ = arena.take(nq * max(block, d) * 4 + nq * block)
    ot = arena.take(1024)
    oo = os_
    op = "vme_attention_block"
    # q, the zeroed state and the table go in once; only K/V blocks stream
    setup = {
        "q": (oq, q),
        "l": (ol, np.zeros(nq, np.float32)),
        "acc": (oa, np.zeros(nq * d, np.float32)),
        "exp_table": (ot, exp_table(SCORE_SCALE)),
    }
    first = max(0, query_start + 1 - window) // block * block
    for start in range(first, nk, block):
        kt_block = np.ascontiguousarray(kt[:, start : start + block])
        await bench.call(
            f"{name}_block{start // block}",
            op,
            [
                nq,
                d,
                query_start,
                window,
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
                os_,
                ot,
            ],
            {
                **(setup if start == first else {}),
                "kt": (ok, kt_block),
                "v": (ov, v[start : start + block]),
            },
            {},
            {
                "kernel": op,
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
            "window": window,
            "block": block,
            "ops": 0,
        },
        save={"q": q, "kt": kt, "v": v},
    )


async def run_linear(
    bench: Bench,
    name: str,
    rows: int,
    d_in: int,
    d_out: int,
    phase: str,
    bias: bool = True,
) -> None:
    """One linear call: int8 x, uint8 weight.T around W_ZERO, int32 bias."""
    rng = np.random.default_rng(rows * 7919 + d_in * 31 + d_out)
    x = rng.integers(-128, 128, (rows, d_in)).astype(np.int8)
    wt = rng.integers(0, 256, (d_in, d_out)).astype(np.uint8)
    b = rng.integers(-10000, 10000, d_out).astype(np.int32)
    acc_scale, out_scale = 0.05 * 0.01, 0.5
    arena = Arena()
    ox, ow, ob, os_, oy = (
        arena.take(s)
        for s in (x.nbytes, wt.nbytes, b.nbytes, rows * d_out * 4, rows * d_out)
    )
    await bench.call(
        name,
        "linear",
        [
            rows,
            d_in,
            d_out,
            ox,
            ow,
            W_ZERO,
            ob if bias else NULL,
            acc_scale,
            out_scale,
            os_,
            oy,
        ],
        {"x": (ox, x), "wt": (ow, wt), **({"bias": (ob, b)} if bias else {})},
        {"y": (oy, np.int8, (rows, d_out))},
        {
            "kernel": "linear",
            "shape": [rows, d_in, d_out],
            "phase": phase,
            "acc_scale": acc_scale,
            "out_scale": out_scale,
            "ops": 2 * rows * d_in * d_out,
        },
    )


async def run_mlp(
    bench: Bench,
    name: str,
    rows: int,
    d: int,
    inner: int,
    chunk: int,
    phase: str,
) -> None:
    """Stream the inner width through DTCM in `chunk` pieces, then finish.

    x, the GeLU table and the running sums go in once; each mlp_block call
    gets that chunk's c_fc.T columns, c_fc bias and c_proj.T rows.
    """
    rng = np.random.default_rng(rows * 7 + d * 13 + inner)
    x = rng.integers(-128, 128, (rows, d)).astype(np.int8)
    fc_wt = rng.integers(0, 256, (d, inner)).astype(np.uint8)
    proj_wt = rng.integers(0, 256, (inner, d)).astype(np.uint8)
    fc_bias = rng.integers(-1000, 1000, inner).astype(np.int32)
    proj_bias = rng.integers(-1000, 1000, d).astype(np.int32)
    table = gelu_table_ref(Q["gelu_in_scale"], Q["gelu_out_scale"])
    # accumulator -> GeLU input LSBs; keeps GeLU's input in range for these
    # random weights (about d * 128 * 74 per sum)
    fc_scale = 1.0 / (16 * d)
    acc_scale, out_scale = 0.05 * 0.01, 0.5
    arena = Arena()
    ox, ot, oa = arena.take(x.nbytes), arena.take(256), arena.take(rows * d * 4)
    ofw, ofb, opw = (arena.take(s) for s in (d * chunk, chunk * 4, chunk * d))
    os_ = arena.take(rows * max(chunk, d) * 4 + rows * chunk)
    opb, oy = arena.take(d * 4), arena.take(rows * d)
    setup = {
        "x": (ox, x),
        "gelu_table": (ot, table),
        "acc": (oa, np.zeros(rows * d, np.int32)),
    }
    for c0 in range(0, inner, chunk):
        await bench.call(
            f"{name}_block{c0 // chunk}",
            "mlp_block",
            [
                rows,
                d,
                chunk,
                ox,
                ofw,
                W_ZERO,
                ofb,
                fc_scale,
                ot,
                opw,
                W_ZERO,
                os_,
                oa,
            ],
            {
                **(setup if c0 == 0 else {}),
                "fc_wt": (ofw, np.ascontiguousarray(fc_wt[:, c0 : c0 + chunk])),
                "fc_bias": (ofb, fc_bias[c0 : c0 + chunk]),
                "proj_wt": (opw, proj_wt[c0 : c0 + chunk]),
            },
            {},
            {
                "kernel": "mlp_block",
                "shape": [rows, d, chunk],
                "phase": phase,
                "stream": name,
                "ops": 4 * rows * d * chunk,
            },
        )
    await bench.call(
        f"{name}_finish",
        "mlp_finish",
        [rows, d, oa, opb, acc_scale, out_scale, oy],
        {"proj_bias": (opb, proj_bias)},
        {"y": (oy, np.int8, (rows, d)), "acc": (oa, np.int32, (rows, d))},
        {
            "kernel": "mlp_finish",
            "shape": [rows, d, inner],
            "phase": phase,
            "stream": name,
            "fc_scale": fc_scale,
            "acc_scale": acc_scale,
            "out_scale": out_scale,
            "ops": 0,
        },
        save={
            "x": x,
            "fc_wt": fc_wt,
            "fc_bias": fc_bias,
            "proj_wt": proj_wt,
            "gelu_table": table,
        },
    )


async def run_embedding_lookup(
    bench: Bench,
    name: str,
    length: int,
    width: int,
    vocab: int,
    slice_rows: int,
    phase: str,
) -> None:
    """Stream a [vocab x width] table through DTCM in `slice_rows` slices.

    Each embedding_lookup call fills the positions whose ids fall in its
    slice and is checked on its own; y stays in the arena across calls.
    """
    rng = np.random.default_rng(length * 31 + vocab)
    table = rng.integers(-128, 128, (vocab, width)).astype(np.int8)
    ids = rng.integers(0, vocab, length).astype(np.uint32)
    arena = Arena()
    oi, ot, oy, of = (
        arena.take(s)
        for s in (ids.nbytes, slice_rows * width, length * width, 4)
    )
    for start in range(0, vocab, slice_rows):
        await bench.call(
            f"{name}_slice{start // slice_rows}",
            "embedding_lookup",
            [length, width, oi, start, slice_rows, ot, oy, of],
            {
                **({"ids": (oi, ids)} if start == 0 else {}),
                "table": (ot, table[start : start + slice_rows]),
            },
            {
                "y": (oy, np.int8, (length, width)),
                "filled": (of, np.uint32, (1,)),
            },
            {
                "kernel": "embedding_lookup",
                "shape": [length, slice_rows, width],
                "phase": phase,
                "stream": name,
                "vocab_start": start,
                "ops": 0,
            },
            save={"ids": ids},
        )


async def run_position_embeddings(
    bench: Bench,
    name: str,
    length: int,
    width: int,
    table_rows: int,
    offset: int,
    phase: str,
) -> None:
    """One add_position_embeddings call on a slice of the position table."""
    rng = np.random.default_rng(length * 17 + offset)
    tokens = rng.integers(-128, 128, (length, width)).astype(np.int8)
    positions = rng.integers(-128, 128, (table_rows, width)).astype(np.int8)
    token_scale, position_scale = Q["embedding_scale"], Q["embedding_scale"]
    out_scale = Q["act_scale"]
    arena = Arena()
    ot, op, oy = (
        arena.take(s) for s in (tokens.nbytes, positions.nbytes, tokens.nbytes)
    )
    await bench.call(
        name,
        "add_position_embeddings",
        [
            length,
            width,
            ot,
            token_scale,
            op,
            position_scale,
            offset,
            out_scale,
            oy,
        ],
        {"tokens": (ot, tokens), "positions": (op, positions)},
        {"y": (oy, np.int8, (length, width))},
        {
            "kernel": "add_position_embeddings",
            "shape": [length, width],
            "phase": phase,
            "position_offset": offset,
            "token_scale": token_scale,
            "position_scale": position_scale,
            "out_scale": out_scale,
            "ops": length * width,
        },
    )
