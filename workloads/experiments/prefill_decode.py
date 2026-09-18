"""Per-operator MACs and element traffic across batch, length and phase."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments import tracer
from experiments.plots import READABLE, SERIES, SURFACE, panels

RESULTS = Path(__file__).resolve().parents[1] / "results" / "prefill_decode"
MODELS = {"gpt2": "GPT-2", "llama": "SmolLM2-135M"}
PHASES = ("prefill", "decode")
BATCHES = (1, 4, 16, 64)
LENGTHS = (128, 512, 2048, 8192)
BYTES_PER_ELEMENT = 2
HEADER = [
    "model",
    "phase",
    "batch",
    "length",
    "operator",
    "calls",
    "macs",
    "elements",
    "fused_elements",
    "bytes",
    "fused_bytes",
    "intensity",
    "fused_intensity",
]
Point = tuple[str, int, int]
# operator -> [calls, macs, elements, fused elements]
Totals = dict[str, list[int]]


def run(family: str, key: str, phase: str, batch: int, length: int) -> Totals:
    # Decode at length L is the L-th token over L - 1 cached ones, so both
    # phases cover the same positions.
    if phase == "prefill":
        records = tracer.prefill(family, key, batch, length)
    else:
        records = tracer.decode(family, key, batch, length - 1)
    totals: Totals = defaultdict(lambda: [0, 0, 0, 0])
    for record in records:
        entry = totals[record.operator]
        entry[0] += 1
        entry[1] += record.macs
        entry[2] += record.numel
        entry[3] += record.fused_numel
    return totals


def sweep(family: str, key: str) -> dict[Point, Totals]:
    results = {}
    for batch in BATCHES:
        for length in tracer.lengths(family, key, LENGTHS):
            for phase in PHASES:
                results[phase, batch, length] = run(
                    family, key, phase, batch, length
                )
    return results


def intensity(macs: int, elements: int) -> str:
    return f"{macs / (elements * BYTES_PER_ELEMENT):.4f}" if elements else ""


def rows(key: str, results: dict[Point, Totals]) -> list[list]:
    out = []
    for (phase, batch, length), totals in results.items():
        summed = [sum(column) for column in zip(*totals.values(), strict=True)]
        for operator, values in [*totals.items(), ("Total", summed)]:
            calls, macs, elements, fused = values
            out.append(
                [
                    key,
                    phase,
                    batch,
                    length,
                    operator,
                    calls,
                    macs,
                    elements,
                    fused,
                    elements * BYTES_PER_ELEMENT,
                    fused * BYTES_PER_ELEMENT,
                    intensity(macs, elements),
                    intensity(macs, fused),
                ]
            )
    return out


def tokens(phase: str, batch: int, length: int) -> int:
    return batch * (length if phase == "prefill" else 1)


def plot_breakdown(key: str, results: dict[Point, Totals], path: Path) -> None:
    operators = list(
        dict.fromkeys(op for totals in results.values() for op in totals)
    )
    colors = dict(zip(operators, SERIES, strict=False))
    metrics = (("MACs", 1), ("bytes", 2), ("fused bytes", 3))
    figure, axes = panels(f"{key} per token", len(PHASES), len(metrics))
    for row, phase in enumerate(PHASES):
        points = [point for point in results if point[0] == phase]
        labels = [f"B{batch}\nL{length}" for _, batch, length in points]
        for column, (metric, index) in enumerate(metrics):
            ax = axes[row * len(metrics) + column]
            bottom = [0.0] * len(points)
            for operator in operators:
                heights = []
                for point in points:
                    value = results[point].get(operator, [0, 0, 0, 0])[index]
                    scale = BYTES_PER_ELEMENT if index > 1 else 1
                    heights.append(value * scale / tokens(*point))
                ax.bar(
                    range(len(points)),
                    heights,
                    bottom=bottom,
                    color=colors[operator],
                    width=0.7,
                    edgecolor=SURFACE,
                    linewidth=0.5,
                )
                bottom = [b + h for b, h in zip(bottom, heights, strict=True)]
            ax.set_xticks(range(len(points)), labels, fontsize=6)
            ax.yaxis.set_major_formatter(READABLE)
            ax.set_title(f"{phase} · {metric} per token", loc="left")
    handles = [Patch(color=colors[op], label=op) for op in operators]
    figure.legend(handles=handles, loc="outside lower center", ncol=8)
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def plot_intensity(key: str, results: dict[Point, Totals], path: Path) -> None:
    figure, axes = panels(f"{key} arithmetic intensity", 1, len(PHASES))
    for ax, phase in zip(axes, PHASES, strict=True):
        for color, batch in zip(SERIES, BATCHES, strict=False):
            points = [
                (b, length)
                for p, b, length in results
                if p == phase and b == batch
            ]
            lengths = [length for _, length in points]
            for index, dashes in ((2, ""), (3, (4, 2))):
                values = []
                for point in points:
                    totals = results[(phase, *point)]
                    macs = sum(v[1] for v in totals.values())
                    elements = sum(v[index] for v in totals.values())
                    values.append(macs / (elements * BYTES_PER_ELEMENT))
                ax.plot(
                    lengths,
                    values,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    linestyle=(0, dashes) if dashes else "-",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("length")
        ax.set_ylabel("MACs per byte")
        ax.set_title(phase, loc="left")
    handles = [
        Line2D([], [], color=color, linewidth=2, label=f"batch {batch}")
        for color, batch in zip(SERIES, BATCHES, strict=False)
    ]
    handles += [
        Line2D([], [], color="black", linewidth=2, label="unfused attention"),
        Line2D(
            [], [], color="black", linewidth=2, dashes=(4, 2), label="fused"
        ),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=6)
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for family in sys.argv[1:] or MODELS:
        key = MODELS[family]
        results = sweep(family, key)
        with open(RESULTS / f"{family}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(HEADER)
            writer.writerows(rows(key, results))
        for (phase, batch, length), totals in results.items():
            macs = sum(v[1] for v in totals.values())
            elements = sum(v[2] for v in totals.values())
            fused = sum(v[3] for v in totals.values())
            print(
                f"{key:<14}{phase:<8}B{batch:<4}L{length:<6}"
                f"{macs / 1e9:>9.3f} G MACs"
                f"{elements * BYTES_PER_ELEMENT / 1e9:>9.3f} GB"
                f"{fused * BYTES_PER_ELEMENT / 1e9:>9.3f} GB fused"
            )
        plot_breakdown(key, results, RESULTS / f"{family}_breakdown.png")
        plot_intensity(key, results, RESULTS / f"{family}_intensity.png")


if __name__ == "__main__":
    main()
