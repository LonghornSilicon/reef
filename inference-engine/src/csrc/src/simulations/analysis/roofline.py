"""Roofline and end-to-end estimate for GPT-2 small on Reef, from results.json.

    uv run --group sim python src/csrc/src/simulations/analysis/roofline.py

Per operator of one GPT-2 block, for prefill (each config length) and decode
(one token at the decode context):
  * ops                 2 x MACs (matmul/attention) or elementwise ops
  * dtcm_bytes          DTCM <-> core traffic, counted from each kernel's
                        loop structure (derived, not measured)
  * host_bytes          host <-> DTCM traffic: every tile writes its inputs
                        and reads its outputs, since there is no DMA
  * kernel_cycles       measured sim cycles per tile x number of tiles
  * host_cycles         host_bytes / measured host->DTCM rate
Writes analysis/roofline.json, analysis/roofline_{prefill,decode}.png and
prints the block and 12-layer totals in ms at the configured clock.

Tiling schedule (the cheapest the current interface allows): matmuls are
A-stationary (one A tile of 16 x 512 stays in DTCM while every B tile
streams past; K is split in 512-wide chunks whose int32 partial sums the
host adds); attention streams 64-key K/V blocks through
attention_flash_chunk for 16-query blocks.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SIMS = HERE.parent
CONFIG = json.loads((SIMS / "config" / "model.json").read_text())
HARDWARE = json.loads((SIMS / "config" / "hardware.json").read_text())
M = CONFIG["model"]
D, DFF, H, DH = M["d_model"], M["d_ff"], M["n_head"], M["d_head"]

# reference palette (dataviz skill): one slot per engine, all-pairs safe
ENGINE_COLORS = {"VME": "#2a78d6", "RVV": "#eb6834", "scalar": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(-a // b)


class Measured:
    """Per-tile sim measurements pulled out of results.json."""

    def __init__(self, results: dict) -> None:
        """Index kernel jobs by name and read the microbenchmarks."""
        self.jobs = {j["name"]: j for j in results["kernels"]}
        self.micro = results["microbench"]
        self.missing: list[str] = []

    def cycles(self, name: str) -> float:
        """Measured cycles of one job; records it as missing if absent."""
        if name not in self.jobs:
            self.missing.append(name)
            return math.nan
        return self.jobs[name]["cycles"]

    def mean_cycles(self, prefix: str) -> float:
        """Mean cycles over jobs whose name starts with prefix."""
        found = [
            j["cycles"] for n, j in self.jobs.items() if n.startswith(prefix)
        ]
        if not found:
            self.missing.append(prefix + "*")
            return math.nan
        return sum(found) / len(found)


def matmul_op(meas: Measured, name: str, n: int, k: int, m: int) -> dict:
    """One N x K x M int8 matmul, tiled for DTCM (see module docstring).

    K is split into chunks of the tile's k (the last one may be shorter).
    A VME tile of chunk width kc costs the measured k-sweep fit, overhead +
    cycles_per_step * kc / 4; an RVV tile scales its measured 512-wide tile
    linearly in kc (its inner loop is linear in k).
    """
    t = CONFIG["tiles"]
    vme = n % 16 == 0
    tn, tk, tm = (t["matmul_vme" if vme else "matmul_rvv"][x] for x in "nkm")
    chunks = [min(tk, k - k0) for k0 in range(0, k, tk)]
    if vme:  # prefill: 16 x 512 x 16 tiles on the matrix engine
        step = meas.micro.get("vme_cycles_per_matmul_step", math.nan)
        overhead = meas.micro.get("vme_tile_overhead_cycles", math.nan)
        tile_cycles = [overhead + step * kc / 4 for kc in chunks]
        # each 16x16x4 step loads 4 A columns + 4 B rows of 16 bytes
        tile_dtcm = [(kc // 4) * 128 + tn * tm * 4 for kc in chunks]
    else:  # decode: matrix-vector on RVV, 1 x 512 x 32 tiles
        per_tile = meas.cycles(f"matmul_rvv_s8u8_{tn}x{tk}x{tm}")
        tile_cycles = [per_tile * kc / tk for kc in chunks]
        # per row, strip and k: one 16-byte B load and one A byte
        tile_dtcm = [tn * cdiv(tm, 16) * kc * 17 + tn * tm * 4 for kc in chunks]
    a_tiles, m_tiles = cdiv(n, tn), cdiv(m, tm)
    host = a_tiles * sum(
        tn * kc + m_tiles * (kc * tm + tn * tm * 4) for kc in chunks
    )
    return {
        "op": name,
        "engine": "VME" if vme else "RVV",
        "ops": 2 * n * k * m,
        "kernel_cycles": a_tiles * m_tiles * sum(tile_cycles),
        "dtcm_bytes": a_tiles * m_tiles * sum(tile_dtcm),
        "host_bytes": host,
        "tiles": a_tiles * m_tiles * len(chunks),
    }


def attention_op(meas: Measured, nq: int, ctx: int, phase: str) -> dict:
    """All heads of causal attention via streamed chunked flash."""
    block = CONFIG["tiles"]["attention_flash_chunk"]["kv_block"]
    qb = CONFIG["tiles"]["attention_flash_chunk"]["nq"] if nq > 1 else 1
    stream = "chunk_prefill" if nq > 1 else "chunk_decode"
    per_block = meas.mean_cycles(f"{stream}_block")
    finish = meas.cycles(f"{stream}_finish")
    pairs = 0  # (query block, kv block) pairs that see at least one key
    for q0 in range(0, nq, qb):
        last_pos = ctx - nq + min(q0 + qb, nq) - 1
        pairs += cdiv(last_pos + 1, block)
    q_blocks = cdiv(nq, qb)
    visible = sum(ctx - nq + i + 1 for i in range(nq))  # causal key count
    # per visible (query, key): a d-byte kt column, a d-byte v row, and a
    # float acc row read and written (8d); per pair the block itself
    dtcm = H * visible * (DH + DH + 8 * DH)
    host = H * (
        pairs * 2 * block * DH  # K and V blocks in
        + q_blocks * (qb * DH + qb * (DH + 2) * 4 + qb * DH)  # q, state, out
    )
    return {
        "op": "attention",
        "engine": "scalar",
        "ops": H * 4 * visible * DH,
        "kernel_cycles": H * (pairs * per_block + q_blocks * finish),
        "dtcm_bytes": dtcm,
        "host_bytes": host,
        "tiles": H * pairs,
    }


def pointwise_op(meas: Measured, name: str, job: str, elems: int) -> dict:
    """Elementwise or row-wise kernel scaled from its measured tile."""
    j = meas.jobs.get(job)
    if j is None:
        meas.missing.append(job)
        return {
            "op": name,
            "engine": "scalar",
            "ops": math.nan,
            "kernel_cycles": math.nan,
            "dtcm_bytes": math.nan,
            "host_bytes": math.nan,
            "tiles": 0,
        }
    tile_elems = math.prod(j["shape"])
    tiles = cdiv(elems, tile_elems)
    host_tile = j["host_write_bytes"] + j["out_bytes"]
    # reads of every input plus the output write; layernorm reads x twice
    dtcm_tile = host_tile + (tile_elems if name.startswith("ln") else 0)
    return {
        "op": name,
        "engine": "scalar",
        "ops": j["ops"] * tiles,
        "kernel_cycles": j["cycles"] * tiles,
        "dtcm_bytes": dtcm_tile * tiles,
        "host_bytes": host_tile * tiles,
        "tiles": tiles,
    }


def block_ops(meas: Measured, nq: int, ctx: int, phase: str) -> list[dict]:
    """Every operator of one GPT-2 block for nq new tokens at context ctx."""
    return [
        pointwise_op(meas, "ln1", "layernorm", nq * D),
        matmul_op(meas, "qkv", nq, D, 3 * D),
        attention_op(meas, nq, ctx, phase),
        matmul_op(meas, "attn_out", nq, D, D),
        pointwise_op(meas, "residual1", "residual_add", nq * D),
        pointwise_op(meas, "ln2", "layernorm", nq * D),
        matmul_op(meas, "fc1", nq, D, DFF),
        pointwise_op(meas, "gelu", "gelu", nq * DFF),
        matmul_op(meas, "fc2", nq, DFF, D),
        pointwise_op(meas, "residual2", "residual_add", nq * D),
    ]


def plot(ops: list[dict], title: str, ceilings: dict, path: Path) -> None:
    """Roofline vs DTCM bytes: attained ops/cycle against ops per byte."""
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [o["ops"] / o["dtcm_bytes"] for o in ops if o["kernel_cycles"] > 0]
    x_lo, x_hi = min([*xs, 0.1]) / 3, max([*xs, 100]) * 3
    grid = [x_lo * (x_hi / x_lo) ** (i / 200) for i in range(201)]
    peak = ceilings["vme_ops_per_cycle"]
    for label, bw, dashed in [
        ("DTCM 16 B/cycle", ceilings["dtcm_bytes_per_cycle"], False),
        (
            ceilings["host_label"],
            ceilings["host_bytes_per_cycle"],
            not ceilings["host_measured"],
        ),
    ]:
        if not bw or math.isnan(bw):
            continue
        ys = [min(peak, bw * x) if peak == peak else bw * x for x in grid]
        ax.plot(grid, ys, color=MUTED, lw=1.5, ls="--" if dashed else "-")
        ax.annotate(
            label,
            (grid[25], bw * grid[25]),
            color=MUTED,
            fontsize=8,
            rotation=33,
            va="bottom",
        )
    if peak == peak:
        ax.axhline(peak, color=INK, lw=1.5)
        label = f"VME peak {peak:.0f} ops/cycle (measured)"
        ax.annotate(
            label, (x_lo * 1.2, peak), color=INK, fontsize=8, va="bottom"
        )
    for o in ops:
        if not o["kernel_cycles"] > 0:
            continue
        x, y = o["ops"] / o["dtcm_bytes"], o["ops"] / o["kernel_cycles"]
        ax.scatter(
            x,
            y,
            s=46,
            color=ENGINE_COLORS[o["engine"]],
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            o["op"],
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    for engine, color in ENGINE_COLORS.items():
        ax.scatter([], [], color=color, label=engine, s=46)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("arithmetic intensity (ops per DTCM byte)", color=MUTED)
    ax.set_ylabel("attained ops per cycle", color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    """Build per-phase operator tables, plots and the end-to-end estimate."""
    results = json.loads((SIMS / "results" / "results.json").read_text())
    meas = Measured(results)
    micro = results["microbench"]
    clock_hz = HARDWARE["clock_mhz"]["value"] * 1e6
    host_measured = "host_to_dtcm_bytes_per_cycle" in micro
    rtl_bound = HARDWARE["axi_slave_data_bits"]["value"] / 8
    host_bw = micro.get("host_to_dtcm_bytes_per_cycle", rtl_bound)
    source = "measured" if host_measured else "RTL bound, unconfirmed"
    ceilings = {
        "vme_ops_per_cycle": 2 * micro.get("vme_peak_macs_per_cycle", math.nan),
        "dtcm_bytes_per_cycle": HARDWARE["dbus_bytes_per_cycle"]["value"],
        "host_bytes_per_cycle": host_bw,
        "host_measured": host_measured,
        "host_label": f"host->DTCM {host_bw:.2f} B/cycle ({source})",
    }

    lens = CONFIG["sequences"]["prefill_lens"]
    phases = {f"prefill_{n}": (n, n) for n in lens}
    ctx = CONFIG["sequences"]["decode_context"]
    phases["decode"] = (1, ctx)
    out = {"ceilings": ceilings, "clock_hz": clock_hz, "phases": {}}
    for phase, (nq, context) in phases.items():
        ops = block_ops(meas, nq, context, phase)
        for o in ops:
            o["host_cycles"] = o["host_bytes"] / host_bw
            o["total_cycles"] = o["kernel_cycles"] + o["host_cycles"]
            o["intensity_dtcm"] = o["ops"] / o["dtcm_bytes"]
            o["intensity_host"] = o["ops"] / o["host_bytes"]
        block = sum(o["total_cycles"] for o in ops)
        out["phases"][phase] = {
            "nq": nq,
            "context": context,
            "operators": ops,
            "block_cycles": block,
            "block_ms": 1e3 * block / clock_hz,
            "model_ms": 1e3 * M["layers"] * block / clock_hz,
            "kernel_share": sum(o["kernel_cycles"] for o in ops) / block,
        }
        p = out["phases"][phase]
        print(
            f"{phase:<13} block {p['block_ms']:10.2f} ms   "
            f"12 layers {p['model_ms']:10.1f} ms   "
            f"kernel share {p['kernel_share']:.0%}"
        )

    longest = f"prefill_{max(CONFIG['sequences']['prefill_lens'])}"
    for phase, label in [(longest, "Prefill"), ("decode", "Decode")]:
        p = out["phases"][phase]
        plot(
            p["operators"],
            f"{label}: one GPT-2 block ({p['nq']} tokens, "
            f"context {p['context']})",
            ceilings,
            HERE / f"roofline_{label.lower()}.png",
        )
    out["missing"] = sorted(set(meas.missing))
    if out["missing"]:
        print("missing measurements (estimates are NaN):", out["missing"])
    (HERE / "roofline.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
