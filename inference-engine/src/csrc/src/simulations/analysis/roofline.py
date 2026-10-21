"""Roofline and end-to-end estimate for GPT-2 small on Reef, from results.json.

    uv run --group sim python analysis/roofline.py   (from simulations/)
    uv run --group sim python analysis/roofline.py \
        --model analysis/gpt_neo_28m.json

--model overlays another model's dims, layer pattern (global / local
attention) and sequences on model.json, and reads the sim jobs it names
(e.g. GPT-Neo 28M's d_head = 32 attention streams). Its outputs carry the
model name as a suffix: roofline_<name>.json and so on.

Per operator of one GPT-2 block, for prefill (each config length) and decode
(one token at the decode context):
  * ops                 2 x MACs (matmul/attention) or elementwise ops
  * dtcm_bytes          DTCM <-> core traffic, counted from each kernel's
                        loop structure (derived, not measured)
  * host_bytes          host <-> DTCM traffic: every tile writes its inputs
                        and reads its outputs, since there is no DMA
  * kernel_cycles       measured sim cycles per tile x number of tiles
  * host_cycles         host_bytes / measured host->DTCM rate
Reads out/results.json (collect_results.py). Writes out/roofline.json,
out/roofline_{prefill,decode}.png and out/vocab_sweep.csv and prints the
block and all-layer totals in ms at the configured clock.

Tiling schedule (the cheapest the current interface allows): matmuls are
A-stationary (one A tile of 16 x 512 stays in DTCM while every B tile
streams past; K is split in 512-wide chunks whose int32 partial sums the
host adds); attention streams 64-key K/V blocks through vme_attention_block
for 16-query blocks (both its matmuls on the matrix engine; decode's single
query on the vector unit). Local-attention layers stream only the KV
blocks their window reaches.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"

CONFIG = json.loads((HERE / "model.json").read_text())
HARDWARE = json.loads((HERE / "hardware.json").read_text())
M = CONFIG["model"]
D, DFF, H, DH = M["d_model"], M["d_ff"], M["n_head"], M["d_head"]
# sim job names this model's estimate reads; --model can point elsewhere
SIM_JOBS = {"attention_stream": "vme", "layernorm": "layernorm"}
SUFFIX = ""  # output file suffix, "_<name>" under --model


def use_model(path: Path) -> None:
    """Overlay another model's config (see the module docstring)."""
    global M, D, DFF, H, DH, SUFFIX
    overlay = json.loads(path.read_text())
    M = {**M, **overlay["model"]}
    CONFIG["sequences"] = {**CONFIG["sequences"], **overlay["sequences"]}
    jobs = overlay.get("sim_jobs", {})
    SIM_JOBS.update({k: v for k, v in jobs.items() if not k.startswith("_")})
    D, DFF, H, DH = M["d_model"], M["d_ff"], M["n_head"], M["d_head"]
    SUFFIX = "_" + M["name"]


def layer_windows() -> list[int]:
    """Attention window of each layer: n_ctx for global, window for local."""
    pattern = M.get("attention_layers", ["global"])
    return [
        M["window"] if pattern[i % len(pattern)] == "local" else M["n_ctx"]
        for i in range(M["layers"])
    ]


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
    tn, tk, tm = (t["vme_matmul" if vme else "matmul_rvv"][x] for x in "nkm")
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


def attention_blocks(
    nq: int, ctx: int, window: int
) -> tuple[int, int, int, int, int]:
    """(query block, query blocks, KV block, visible pairs, block pairs)."""
    block = CONFIG["tiles"]["attention"]["kv_block"]
    qb = CONFIG["tiles"]["attention"]["nq"] if nq > 1 else 1
    pairs = 0  # (query block, kv block) pairs that see at least one key
    for q0 in range(0, nq, qb):
        first_pos = ctx - nq + q0
        last_pos = ctx - nq + min(q0 + qb, nq) - 1
        lo = max(0, first_pos + 1 - window)  # oldest key any row sees
        pairs += last_pos // block - lo // block + 1
    # causal key count, capped by the window
    visible = sum(min(ctx - nq + i + 1, window) for i in range(nq))
    return qb, cdiv(nq, qb), block, visible, pairs


def attention_host_bytes(nq: int, ctx: int, window: int) -> int:
    """Host traffic of streamed attention: K/V per block pair, q/state/out."""
    qb, q_blocks, block, _, pairs = attention_blocks(nq, ctx, window)
    per_q_block = qb * DH + qb * (DH + 2) * 4 + qb * DH  # q, state, out
    return H * (pairs * 2 * block * DH + q_blocks * per_q_block)


def attention_op(meas: Measured, nq: int, ctx: int, window: int) -> dict:
    """All heads of causal attention via vme_attention_block (the default).

    Both matmuls run through matmul(): the matrix engine for prefill's
    16-query blocks, the vector unit for decode's single query. The engine
    computes whole blocks (masked keys still cost work), so each (query
    block, KV block) pair costs the measured cycles of one full block.
    """
    qb, q_blocks, block, visible, pairs = attention_blocks(nq, ctx, window)
    stream = SIM_JOBS["attention_stream"]
    stream += "_prefill" if nq > 1 else "_decode"
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
        "host_bytes": attention_host_bytes(nq, ctx, window),
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


def block_ops(meas: Measured, nq: int, ctx: int, window: int) -> list[dict]:
    """Every operator of one block for nq new tokens at context ctx."""
    ln = SIM_JOBS["layernorm"]
    return [
        pointwise_op(meas, "ln1", ln, nq * D),
        matmul_op(meas, "qkv", nq, D, 3 * D),
        attention_op(meas, nq, ctx, window),
        matmul_op(meas, "attn_out", nq, D, D),
        pointwise_op(meas, "residual1", "residual_add", nq * D),
        pointwise_op(meas, "ln2", ln, nq * D),
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
        pointwise_op(meas, "ln_f", SIM_JOBS["layernorm"], D),
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
    rvv_peak: float | None = None,
) -> None:
    """One log-log roofline: ops/cycle vs ops per byte of `key` traffic.

    rvv_peak, when given, adds the measured RVV matmul rate as a second
    horizontal ceiling (decode runs its matmuls on RVV).
    """
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
    if rvv_peak is not None:
        ax.axhline(rvv_peak, color=ENGINE_COLORS["RVV"], lw=1.0, ls="--")
        ax.annotate(
            f"RVV peak {rvv_peak:.1f} (measured)",
            (x_hi / 1.3, rvv_peak),
            color=INK,
            fontsize=7,
            va="bottom",
            ha="right",
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


def plot(
    ops: list[dict],
    title: str,
    ceilings: dict,
    path: Path,
    rvv: bool = False,
) -> None:
    """DTCM and host rooflines side by side for one phase.

    rvv adds the measured RVV matmul ceiling (for decode).
    """
    rvv_peak = ceilings["rvv_ops_per_cycle"] if rvv else None
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
        rvv_peak,
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
        rvv_peak,
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

    Writes out/vocab_sweep.csv and prints the table. The blocks don't
    depend on vocab; only the LM head (and the host embedding table) does.
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
    with open(OUT / f"vocab_sweep{SUFFIX}.csv", "w", newline="") as f:
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, help="model overlay JSON")
    if (model := parser.parse_args().model) is not None:
        use_model(model)
    results = json.loads((OUT / "results.json").read_text())
    meas = Measured(results)
    micro = results["microbench"]
    clock_hz = HARDWARE["clock_mhz"]["value"] * 1e6
    host = HARDWARE["host_to_dtcm_bytes_per_cycle"]
    host_measured = host["confirmed"]
    rtl_bound = HARDWARE["axi_slave_data_bits"]["value"] / 8
    host_bw = host["value"] if host_measured else rtl_bound
    source = "measured" if host_measured else "RTL bound, unconfirmed"
    ceilings = {
        "vme_ops_per_cycle": 2 * micro.get("vme_peak_macs_per_cycle", math.nan),
        # best measured RVV matmul tile (ops = 2 x MACs)
        "rvv_ops_per_cycle": max(
            (
                j["ops"] / j["cycles"]
                for j in meas.jobs.values()
                if j["op"].startswith("matmul_rvv")
            ),
            default=math.nan,
        ),
        "dtcm_bytes_per_cycle": HARDWARE["dbus_bytes_per_cycle"]["value"],
        "host_bytes_per_cycle": host_bw,
        "host_measured": host_measured,
        "host_label": f"host->DTCM {host_bw:.2f} B/cycle ({source})",
    }

    lens = CONFIG["sequences"]["prefill_lens"]
    phases = {f"prefill_{n}": (n, n) for n in lens}
    # decode_context: one context, or a list (the first is "decode")
    ctxs = CONFIG["sequences"]["decode_context"]
    for i, ctx in enumerate(ctxs if isinstance(ctxs, list) else [ctxs]):
        phases["decode" if i == 0 else f"decode_{ctx}"] = (1, ctx)
    windows = layer_windows()
    out = {"model": M, "ceilings": ceilings, "clock_hz": clock_hz}
    out["phases"] = {}
    for phase, (nq, context) in phases.items():
        # one operator table per distinct window (global, local)
        by_window = {w: block_ops(meas, nq, context, w) for w in set(windows)}
        head = head_ops(meas, M["vocab"])
        layer_ops = [o for ops in by_window.values() for o in ops]
        add_costs([*layer_ops, *head], host_bw)
        layer_cycles = {
            w: sum(o["total_cycles"] for o in ops)
            for w, ops in by_window.items()
        }
        block = sum(layer_cycles[w] for w in windows) / M["layers"]  # mean
        ops = by_window[windows[0]]
        head_cycles = sum(o["total_cycles"] for o in head)
        full = M["layers"] * block + head_cycles
        out["phases"][phase] = {
            "nq": nq,
            "context": context,
            "operators": ops,
            **(
                {"local_operators": by_window[M["window"]]}
                if len(by_window) > 1
                else {}
            ),
            "block_cycles": block,
            "block_ms": 1e3 * block / clock_hz,
            "model_ms": 1e3 * M["layers"] * block / clock_hz,
            "head_operators": head,
            "head_ms": 1e3 * head_cycles / clock_hz,
            "full_model_ms": 1e3 * full / clock_hz,
            "kernel_share": sum(
                sum(o["kernel_cycles"] for o in by_window[w]) for w in windows
            )
            / (M["layers"] * block),
        }
        p = out["phases"][phase]
        print(
            f"{phase:<13} block {p['block_ms']:10.2f} ms   "
            f"{M['layers']} layers {p['model_ms']:10.1f} ms   "
            f"+ head {p['head_ms']:8.1f} ms = {p['full_model_ms']:10.1f} ms"
        )

    longest = f"prefill_{max(CONFIG['sequences']['prefill_lens'])}"
    for phase, label in [(longest, "Prefill"), ("decode", "Decode")]:
        p = out["phases"][phase]
        plot(
            p["operators"],
            f"{label}: one {M['name']} block ({p['nq']} "
            f"token{'s' if p['nq'] > 1 else ''}, context {p['context']})",
            ceilings,
            OUT / f"roofline_{label.lower()}{SUFFIX}.png",
            rvv=label == "Decode",
        )
    out["vocab_sweep"] = vocab_sweep(meas, out, host_bw, clock_hz, longest)
    out["missing"] = sorted(set(meas.missing))
    if out["missing"]:
        print("missing measurements (estimates are NaN):", out["missing"])
    (OUT / f"roofline{SUFFIX}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
