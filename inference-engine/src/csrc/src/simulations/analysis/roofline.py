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
host adds); attention streams 64-key K/V blocks through attention_vme_block
for 16-query blocks (both its matmuls on the matrix engine; decode's single
query on the vector unit). The scalar attention_flash_chunk estimate is
kept alongside for comparison but not counted in the totals.
"""

import csv
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

    def sum_cycles(self, prefix: str) -> float:
        """Total cycles over jobs whose name starts with prefix."""
        return self.mean_cycles(prefix) * sum(
            n.startswith(prefix) for n in self.jobs
        )

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


def attention_blocks(nq: int, ctx: int) -> tuple[int, int, int, int, int]:
    """(query block, query blocks, KV block, visible pairs, block pairs)."""
    block = CONFIG["tiles"]["attention_flash_chunk"]["kv_block"]
    qb = CONFIG["tiles"]["attention_flash_chunk"]["nq"] if nq > 1 else 1
    pairs = 0  # (query block, kv block) pairs that see at least one key
    for q0 in range(0, nq, qb):
        last_pos = ctx - nq + min(q0 + qb, nq) - 1
        pairs += cdiv(last_pos + 1, block)
    visible = sum(ctx - nq + i + 1 for i in range(nq))  # causal key count
    return qb, cdiv(nq, qb), block, visible, pairs


def attention_host_bytes(nq: int, ctx: int) -> int:
    """Host traffic of streamed attention: K/V per block pair, q/state/out."""
    qb, q_blocks, block, _, pairs = attention_blocks(nq, ctx)
    per_q_block = qb * DH + qb * (DH + 2) * 4 + qb * DH  # q, state, out
    return H * (pairs * 2 * block * DH + q_blocks * per_q_block)


def attention_op(meas: Measured, nq: int, ctx: int) -> dict:
    """All heads of causal attention via attention_vme_block (the default).

    Both matmuls run through matmul(): the matrix engine for prefill's
    16-query blocks, the vector unit for decode's single query. The engine
    computes whole blocks (masked keys still cost work), so each (query
    block, KV block) pair costs the measured cycles of one full block.
    """
    qb, q_blocks, block, visible, pairs = attention_blocks(nq, ctx)
    stream = "vme_prefill" if nq > 1 else "vme_decode"
    per_block = meas.mean_cycles(f"{stream}_block")
    finish = meas.cycles(f"{stream}_finish")
    if nq > 1:  # matrix engine: 16x16x4 steps of 128 operand bytes
        steps = (qb // 16) * (block // 16) * (DH // 4)  # q . kt
        steps += (qb // 16) * (DH // 16) * (block // 4)  # P . v
        mm_bytes = steps * 128
    else:  # vector unit: one 16-byte row load + 1 A byte per k per strip
        mm_bytes = DH * cdiv(block, 16) * 17 + block * cdiv(DH, 16) * 17
    # scores and P.V written then read (int32), P written then read, acc
    # read and written (float)
    state = qb * block * (8 + 2) + qb * DH * (8 + 8)
    return {
        "op": "attention",
        "engine": "VME" if nq > 1 else "RVV",
        "ops": H * 4 * visible * DH,
        "kernel_cycles": H * (pairs * per_block + q_blocks * finish),
        "dtcm_bytes": H * pairs * (mm_bytes + state),
        "host_bytes": attention_host_bytes(nq, ctx),
        "tiles": H * pairs,
    }


def attention_scalar_op(meas: Measured, nq: int, ctx: int) -> dict:
    """Scalar attention_flash_chunk, kept for comparison (not in totals).

    Cost per visible (query, key) pair comes from the measured stream: masked
    keys cost nothing, so diagonal blocks are charged only for what they do.
    """
    _, q_blocks, _, visible, pairs = attention_blocks(nq, ctx)
    stream = "chunk_prefill" if nq > 1 else "chunk_decode"
    finish = meas.cycles(f"{stream}_finish")
    fin = meas.jobs.get(f"{stream}_finish", {})
    per_pair = math.nan
    if fin:
        seen = sum(fin["query_start"] + i + 1 for i in range(fin["shape"][0]))
        per_pair = meas.sum_cycles(f"{stream}_block") / seen
    return {
        "op": "attention (scalar)",
        "engine": "scalar",
        "ops": H * 4 * visible * DH,
        "kernel_cycles": H * (visible * per_pair + q_blocks * finish),
        # per visible pair: a d-byte kt column, a d-byte v row and a float
        # acc row read and written (8d)
        "dtcm_bytes": H * visible * (DH + DH + 8 * DH),
        "host_bytes": attention_host_bytes(nq, ctx),
        "tiles": H * pairs,
        "cycles_per_visible_pair": per_pair,
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


def block_ops(meas: Measured, nq: int, ctx: int) -> list[dict]:
    """Every operator of one GPT-2 block for nq new tokens at context ctx."""
    return [
        pointwise_op(meas, "ln1", "layernorm", nq * D),
        matmul_op(meas, "qkv", nq, D, 3 * D),
        attention_op(meas, nq, ctx),
        matmul_op(meas, "attn_out", nq, D, D),
        pointwise_op(meas, "residual1", "residual_add", nq * D),
        pointwise_op(meas, "ln2", "layernorm", nq * D),
        matmul_op(meas, "fc1", nq, D, DFF),
        pointwise_op(meas, "gelu", "gelu", nq * DFF),
        matmul_op(meas, "fc2", nq, DFF, D),
        pointwise_op(meas, "residual2", "residual_add", nq * D),
    ]


def head_ops(meas: Measured, vocab: int) -> list[dict]:
    """Final LayerNorm and LM head (d_model -> vocab) for one token.

    Only the newest position needs logits (decode, and the first token after
    prefill), so this is a 1-row product: matmul() sends it to the vector
    unit, like the other decode matmuls. Its host traffic is dominated by
    streaming the d_model x vocab weight matrix through DTCM once per token.
    """
    return [
        pointwise_op(meas, "ln_f", "layernorm", D),
        matmul_op(meas, "lm_head", 1, D, vocab),
    ]


def add_costs(ops: list[dict], host_bw: float) -> None:
    """Fill in host cycles, totals and intensities for each operator."""
    for o in ops:
        o["host_cycles"] = o["host_bytes"] / host_bw
        o["total_cycles"] = o["kernel_cycles"] + o["host_cycles"]
        o["intensity_dtcm"] = o["ops"] / o["dtcm_bytes"]
        o["intensity_host"] = o["ops"] / o["host_bytes"]


def roofline_panel(
    ax: plt.Axes,
    ops: list[dict],
    key: str,
    bw: float,
    bw_label: str,
    dashed: bool,
    peak: float,
    xlabel: str,
) -> None:
    """One log-log roofline: ops/cycle vs ops per byte of `key` traffic."""
    # ops that land within ~25% of each other share one point and label
    points: dict[tuple, list[tuple]] = {}
    for o in ops:
        if o["kernel_cycles"] > 0:
            xy = (o["ops"] / o[key], o["ops"] / o["kernel_cycles"])
            cell = (round(math.log10(xy[0]), 1), round(math.log10(xy[1]), 1))
            points.setdefault(cell, []).append((xy, o))
    xs = [g[0][0][0] for g in points.values()]
    x_lo, x_hi = min([*xs, 0.1]) / 3, max([*xs, 100]) * 3
    grid = [x_lo * (x_hi / x_lo) ** (i / 200) for i in range(201)]
    ax.plot(
        grid,
        [min(peak, bw * x) for x in grid],
        color=MUTED,
        lw=1.5,
        ls="--" if dashed else "-",
    )
    ax.annotate(
        bw_label,
        (grid[30], bw * grid[30]),
        color=MUTED,
        fontsize=7,
        rotation=38,
        va="bottom",
        ha="left",
    )
    ax.axhline(peak, color=INK, lw=1.0, alpha=0.6)
    ax.annotate(
        f"VME peak {peak:.1f} ops/cycle (measured)",
        (x_lo * 1.3, peak),
        color=INK,
        fontsize=7,
        va="bottom",
    )
    for group in points.values():
        (x, y), first = group[0]
        color = ENGINE_COLORS[first["engine"]]
        ax.scatter(
            x, y, s=46, color=color, edgecolor="white", linewidth=1.5, zorder=3
        )
        label = ", ".join(o["op"] for _, o in group)
        ax.annotate(
            label,
            (x, y),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)


def plot(ops: list[dict], title: str, ceilings: dict, path: Path) -> None:
    """DTCM and host rooflines side by side for one phase."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.6), dpi=150, sharey=True
    )
    peak = ceilings["vme_ops_per_cycle"]
    dtcm = ceilings["dtcm_bytes_per_cycle"]
    roofline_panel(
        left,
        ops,
        "dtcm_bytes",
        dtcm,
        f"DTCM {dtcm} B/cycle",
        False,
        peak,
        "ops per DTCM<->core byte",
    )
    roofline_panel(
        right,
        ops,
        "host_bytes",
        ceilings["host_bytes_per_cycle"],
        ceilings["host_label"],
        not ceilings["host_measured"],
        peak,
        "ops per host<->DTCM byte (no DMA)",
    )
    left.set_ylabel(
        "attained ops per cycle (kernel only)", color=MUTED, fontsize=8
    )
    engines = {o["engine"] for o in ops}
    for engine, color in ENGINE_COLORS.items():
        if engine in engines:
            right.scatter([], [], color=color, label=engine, s=46)
    right.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(title, x=0.06, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def vocab_sweep(
    meas: Measured, out: dict, host_bw: float, clock_hz: float, longest: str
) -> list[dict]:
    """LM-head cost and full-model time per token at each sweep vocab size.

    Writes analysis/vocab_sweep.csv and prints the table. The 12 blocks
    don't depend on vocab; only the LM head (and the host embedding table,
    timed separately in results.json) does.
    """
    rows = []
    for vocab in CONFIG["sweeps"]["vocab_sizes"]:
        head = head_ops(meas, vocab)
        add_costs(head, host_bw)
        lm = head[1]
        head_cycles = sum(o["total_cycles"] for o in head)
        row = {
            "vocab": vocab,
            "lm_head_ops": lm["ops"],
            "lm_head_kernel_cycles": lm["kernel_cycles"],
            "lm_head_host_bytes": lm["host_bytes"],
            "lm_head_host_cycles": lm["host_cycles"],
            "head_ms": 1e3 * head_cycles / clock_hz,
        }
        for phase in ("decode", longest):
            blocks = M["layers"] * out["phases"][phase]["block_cycles"]
            row[f"{phase}_ms"] = 1e3 * (blocks + head_cycles) / clock_hz
            row[f"{phase}_head_share"] = head_cycles / (blocks + head_cycles)
        rows.append(row)
    with open(HERE / "vocab_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"\n{'vocab':>7} {'head ms':>9} {'decode ms/token':>16} "
        f"{'head share':>11} {f'{longest} ms':>17}"
    )
    for r in rows:
        print(
            f"{r['vocab']:>7} {r['head_ms']:>9.1f} {r['decode_ms']:>16.1f} "
            f"{r['decode_head_share']:>11.1%} {r[f'{longest}_ms']:>17.0f}"
        )
    return rows


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
        ops = block_ops(meas, nq, context)
        scalar_attention = attention_scalar_op(meas, nq, context)
        head = head_ops(meas, M["vocab"])
        add_costs([*ops, scalar_attention, *head], host_bw)
        block = sum(o["total_cycles"] for o in ops)
        head_cycles = sum(o["total_cycles"] for o in head)
        full = M["layers"] * block + head_cycles
        out["phases"][phase] = {
            "nq": nq,
            "context": context,
            "operators": ops,
            "attention_scalar_for_comparison": scalar_attention,
            "block_cycles": block,
            "block_ms": 1e3 * block / clock_hz,
            "model_ms": 1e3 * M["layers"] * block / clock_hz,
            "head_operators": head,
            "head_ms": 1e3 * head_cycles / clock_hz,
            "full_model_ms": 1e3 * full / clock_hz,
            "kernel_share": sum(o["kernel_cycles"] for o in ops) / block,
        }
        p = out["phases"][phase]
        print(
            f"{phase:<13} block {p['block_ms']:10.2f} ms   "
            f"12 layers {p['model_ms']:10.1f} ms   "
            f"+ head {p['head_ms']:8.1f} ms = {p['full_model_ms']:10.1f} ms"
        )

    longest = f"prefill_{max(CONFIG['sequences']['prefill_lens'])}"
    for phase, label in [(longest, "Prefill"), ("decode", "Decode")]:
        p = out["phases"][phase]
        plot(
            p["operators"],
            f"{label}: one GPT-2 block ({p['nq']} "
            f"token{'s' if p['nq'] > 1 else ''}, context {p['context']})",
            ceilings,
            HERE / f"roofline_{label.lower()}.png",
        )
    out["vocab_sweep"] = vocab_sweep(meas, out, host_bw, clock_hz, longest)
    out["missing"] = sorted(set(meas.missing))
    if out["missing"]:
        print("missing measurements (estimates are NaN):", out["missing"])
    (HERE / "roofline.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
