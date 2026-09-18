"""Roofline latency per operator over a compute, bandwidth and SRAM grid."""

import csv
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path

from matplotlib.lines import Line2D

from experiments import tracer
from experiments.plots import RAMP, SERIES, SURFACE, panels

RESULTS = Path(__file__).resolve().parents[1] / "results" / "roofline"
LANGUAGE = ("gpt2", "llama")
VISION = ("googlenet", "resnet", "efficientnet")
LENGTH = 512
LANGUAGE_BATCHES = (1, 16)
VISION_BATCHES = (1, 8)
IMAGE = 224
BYTES_PER_ELEMENT = 2
PEAKS = tuple(0.5 * 2**i for i in range(8))
BANDWIDTHS = tuple(4 << i for i in range(8))
SRAMS = {"256 KB": 256 << 10, "1 MB": 1 << 20, "4 MB": 4 << 20}
# (TMAC/s, GB/s, SRAM bytes) for the per-operator detail.
NAMED = {
    "edge": (2.0, 16, 1 << 20),
    "mid": (16.0, 128, 4 << 20),
    "high": (64.0, 512, 4 << 20),
}
TOTAL_HEADER = [
    "model",
    "phase",
    "batch",
    "peak_tmacs",
    "bandwidth_gbs",
    "sram",
    "compute_ms",
    "memory_ms",
    "latency_ms",
]
OPERATOR_HEADER = [
    "model",
    "phase",
    "batch",
    "point",
    "operator",
    "macs",
    "bytes",
    "intensity",
    "compute_ms",
    "memory_ms",
    "latency_ms",
    "bound",
]
Run = tuple[str, str, int]
# operator -> [macs, bytes]
Costs = dict[str, list[float]]


def keys(family: str) -> list[str]:
    configs = importlib.import_module(f"configs.{family}")
    return list(getattr(configs, f"{family.upper()}_CONFIGS"))


def runs(family: str) -> dict[Run, list[tracer.Record]]:
    results = {}
    for key in keys(family):
        if family in LANGUAGE:
            length = tracer.lengths(family, key, (LENGTH,))[0]
            for batch in LANGUAGE_BATCHES:
                results[key, "prefill", batch] = tracer.prefill(
                    family, key, batch, length
                )
                results[key, "decode", batch] = tracer.decode(
                    family, key, batch, length - 1
                )
        else:
            for batch in VISION_BATCHES:
                results[key, "image", batch] = tracer.classify(
                    family, key, batch, IMAGE
                )
    return results


def gemm_traffic(gemm: tracer.Gemm, sram: int) -> int:
    """Elements moved by an output-tiled GEMM whose tiles fill ``sram``."""
    m, n, k, count = gemm
    tile = math.sqrt(sram / BYTES_PER_ELEMENT)
    # Each operand is re-read once per tile row or column of the other; with
    # m, n <= tile this is one pass over both, as a decode GEMV wants.
    return count * (m * k * math.ceil(n / tile) + n * k * math.ceil(m / tile))


def traffic(record: tracer.Record, sram: int) -> int:
    if not record.gemms:
        return record.numel
    outputs = sum(m * n * count for m, n, _, count in record.gemms)
    return outputs + sum(gemm_traffic(gemm, sram) for gemm in record.gemms)


def costs(records: list[tracer.Record], sram: int) -> Costs:
    totals: Costs = defaultdict(lambda: [0.0, 0.0])
    for record in records:
        totals[record.operator][0] += record.macs
        totals[record.operator][1] += traffic(record, sram) * BYTES_PER_ELEMENT
    return totals


def times(macs: float, bytes_: float, peak: float, bandwidth: int) -> tuple:
    """Compute and memory milliseconds under a roofline."""
    return macs / (peak * 1e12) * 1e3, bytes_ / (bandwidth * 1e9) * 1e3


def latency(totals: Costs, peak: float, bandwidth: int) -> tuple:
    compute = memory = total = 0.0
    for macs, bytes_ in totals.values():
        c, m = times(macs, bytes_, peak, bandwidth)
        compute, memory, total = compute + c, memory + m, total + max(c, m)
    return compute, memory, total


def total_rows(run: Run, records: list[tracer.Record]) -> list[list]:
    rows = []
    for sram_name, sram in SRAMS.items():
        totals = costs(records, sram)
        for peak in PEAKS:
            for bandwidth in BANDWIDTHS:
                compute, memory, total = latency(totals, peak, bandwidth)
                rows.append(
                    [
                        *run,
                        peak,
                        bandwidth,
                        sram_name,
                        f"{compute:.4f}",
                        f"{memory:.4f}",
                        f"{total:.4f}",
                    ]
                )
    return rows


def operator_rows(run: Run, records: list[tracer.Record]) -> list[list]:
    rows = []
    for point, (peak, bandwidth, sram) in NAMED.items():
        for operator, (macs, bytes_) in costs(records, sram).items():
            compute, memory = times(macs, bytes_, peak, bandwidth)
            rows.append(
                [
                    *run,
                    point,
                    operator,
                    int(macs),
                    int(bytes_),
                    f"{macs / bytes_:.4f}",
                    f"{compute:.4f}",
                    f"{memory:.4f}",
                    f"{max(compute, memory):.4f}",
                    "compute" if compute >= memory else "memory",
                ]
            )
    return rows


def plot_rooflines(
    family: str, results: dict[Run, list[tracer.Record]], path: Path
) -> None:
    operators = list(
        dict.fromkeys(
            r.operator for records in results.values() for r in records
        )
    )
    colors = dict(zip(operators, SERIES, strict=False))
    figure, axes = panels(
        f"{family} operator rooflines", len(results), len(NAMED), height=4
    )
    for row, (run, records) in enumerate(results.items()):
        for column, (point, (peak, bandwidth, sram)) in enumerate(
            NAMED.items()
        ):
            ax = axes[row * len(NAMED) + column]
            totals = costs(records, sram)
            total = latency(totals, peak, bandwidth)[2]
            knee = peak * 1e12 / (bandwidth * 1e9)
            ax.plot(
                [knee / 1e3, knee, knee * 1e2],
                [peak * 1e9, peak * 1e12, peak * 1e12],
                color=RAMP[4],
                linewidth=1,
            )
            for operator, (macs, bytes_) in totals.items():
                intensity = macs / bytes_
                attained = min(peak * 1e12, bandwidth * 1e9 * intensity)
                share = max(times(macs, bytes_, peak, bandwidth)) / total
                ax.scatter(
                    intensity,
                    attained,
                    s=20 + 300 * share,
                    color=colors[operator],
                    alpha=0.7,
                    edgecolor=SURFACE,
                    linewidth=0.5,
                    zorder=3,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("MACs per byte")
            ax.set_ylabel("MAC/s attained")
            model, phase, batch = run
            ax.set_title(
                f"{model} · {phase} · B{batch} · {point} · {total:.3g} ms",
                loc="left",
            )
    handles = [
        Line2D([], [], marker="o", linestyle="", color=colors[op], label=op)
        for op in operators
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=8)
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    totals, operators = [], []
    for family in sys.argv[1:] or (*LANGUAGE, *VISION):
        results = runs(family)
        for run, records in results.items():
            totals += total_rows(run, records)
            operators += operator_rows(run, records)
            model, phase, batch = run
            line = f"{model:<18}{phase:<8}B{batch:<4}"
            for point, (peak, bandwidth, sram) in NAMED.items():
                ms = latency(costs(records, sram), peak, bandwidth)[2]
                line += f"  {point} {ms:>9.3f} ms"
            print(line)
        plot_rooflines(family, results, RESULTS / f"{family}_roofline.png")
    for name, header, rows in (
        ("totals", TOTAL_HEADER, totals),
        ("operators", OPERATOR_HEADER, operators),
    ):
        with open(RESULTS / f"{name}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
