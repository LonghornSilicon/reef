"""Hand-derived FLOPs per operator for GPT-Neo prefill and decode."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS
from workloads.experiments import tracer
from workloads.experiments.plots import (
    BASELINE,
    GRIDLINE,
    INK,
    MUTED_INK,
    SECONDARY_INK,
    SERIES,
    SURFACE,
    readable,
)
from workloads.experiments.quantization import PRECISIONS, is_quantized

RESULTS = Path(__file__).resolve().parents[4] / "results" / "flops"
BATCH = 1
LENGTHS = tuple(2**exponent for exponent in range(12))
PHASES = ("prefill", "decode")
# Both plots cover one generation of up to 256 tokens, as kv_cache_size's does.
PLOT_TOKENS = 256
# int8 rounds each input element (absolute max, divide, round, clamp) and
# rescales each output by the input scale times the weight scale.
QUANTIZE_FLOPS = 4
RESCALE_FLOPS = 2
# Every nonzero int8 decode operator, largest first, in fixed color order.
OPERATOR_LINES = {
    ("Linear (lm_head)", "fp32"): ("lm_head", SERIES[0]),
    ("Linear", "int8"): ("Linear", SERIES[1]),
    ("GroupedQueryAttention", "fp32"): ("Attention", SERIES[2]),
    ("Softmax", "fp32"): ("Softmax", SERIES[3]),
    ("Dequantize", "fp32"): ("Dequantize", SERIES[4]),
    ("Quantize", "fp32"): ("Quantize", SERIES[5]),
    ("GELU", "fp32"): ("GELU", SERIES[6]),
    ("LayerNorm", "fp32"): ("LayerNorm", SERIES[7]),
}
# A line within this many decades of an earlier one at every length would
# hide under it on the log axis (0.05 decades is 12%), so it is drawn dotted
# on top. Quantize always overlaps GELU: 4 FLOPs on each of 9 hidden widths
# of Linear input equal 9 on each of the 4 hidden-wide GELU input.
OVERLAP_DECADES = 0.05
Totals = dict[tuple[str, str], list[int]]


def trace(key: str, phase: str, length: int) -> list[tracer.Record]:
    """Prefill: an L-token prompt. Decode: token L after L - 1 cached ones."""
    if phase == "prefill":
        return tracer.prefill("gpt_neo", key, BATCH, length)
    return tracer.decode("gpt_neo", key, BATCH, context=length - 1)


def count(records: list[tracer.Record], precision: str) -> Totals:
    """(operator, arithmetic) -> [calls, FLOPs], with quantization overhead."""
    totals: Totals = defaultdict(lambda: [0, 0])
    float_type = "bf16" if precision == "bf16" else "fp32"

    def add(operator: str, arithmetic: str, flops: int) -> None:
        totals[operator, arithmetic][0] += 1
        totals[operator, arithmetic][1] += flops

    for record in records:
        quantized = precision in ("int8", "int4") and is_quantized(
            record.path, record.operator
        )
        operator = record.operator
        if record.path == "lm_head":
            operator = "Linear (lm_head)"
        arithmetic = "int8" if quantized and precision == "int8" else float_type
        add(operator, arithmetic, record.flops)
        if not quantized:
            continue
        if precision == "int8":
            add("Quantize", "fp32", QUANTIZE_FLOPS * record.input_numel)
            add("Dequantize", "fp32", RESCALE_FLOPS * record.output_numel)
        else:
            # Every int4 weight is multiplied by its group scale on each call.
            _, n, k, gemms = record.gemms[0]
            add("Dequantize", "fp32", n * k * gemms)
    return totals


def total(totals: Totals) -> int:
    return sum(value for _, value in totals.values())


def generation(key: str, precision: str) -> dict[str, list[int]]:
    """Total FLOPs to reach each length 1..PLOT_TOKENS, per phase.

    Prefill processes an L-token prompt in one pass. Decode generates tokens
    1 to L one at a time: token 1 has no cache, so it is a 1-token prefill.
    """
    first = total(count(trace(key, "prefill", 1), precision))
    prefill, decode = [first], [first]
    for length in range(2, PLOT_TOKENS + 1):
        prefill.append(total(count(trace(key, "prefill", length), precision)))
        step = total(count(trace(key, "decode", length), precision))
        decode.append(decode[-1] + step)
    return {"prefill": prefill, "decode": decode}


def style(ax: object) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)


def grid(keys: list[str], height: float) -> tuple[Figure, list]:
    columns = min(3, len(keys))
    rows = math.ceil(len(keys) / columns)
    figure = Figure(
        figsize=(5.5 * columns, height * rows),
        facecolor=SURFACE,
        layout="constrained",
    )
    figure.get_layout_engine().set(h_pad=0.25)
    axes = list(figure.subplots(rows, columns, squeeze=False).flat)
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    return figure, axes


def title(figure: Figure, text: str) -> None:
    figure.suptitle(text, x=0.01, ha="left", color=INK, fontsize=13)


def label_ends(ax: object, ends: list[tuple[float, str]]) -> None:
    """Label line ends, nudged apart where the lines nearly meet."""
    bottom, top = ax.get_ylim()
    gap = 0.09 * (top - bottom)
    placed = -math.inf
    for y, text in sorted(ends):
        placed = max(y, placed + gap)
        ax.annotate(
            text,
            (PLOT_TOKENS, y),
            xytext=(PLOT_TOKENS + 8, placed),
            va="center",
            fontsize=8,
            color=SECONDARY_INK,
        )


def plot_generation(keys: list[str], path: Path) -> None:
    figure, axes = grid(keys, 3.6)
    title(
        figure,
        f"GPT-Neo int8 FLOPs to reach 1 to {PLOT_TOKENS} tokens, batch 1",
    )
    color = SERIES[2]
    lengths = range(1, PLOT_TOKENS + 1)
    for ax, key in zip(axes, keys, strict=False):
        totals = generation(key, "int8")
        for phase, line in zip(PHASES, ("--", "-"), strict=True):
            ax.plot(lengths, totals[phase], color=color, linestyle=line)
        ax.set_title(key, loc="left", fontsize=10, color=INK)
        # Headroom so the end labels are not clipped.
        ax.set_xlim(0, PLOT_TOKENS * 1.35)
        ax.set_ylim(0, None)
        label_ends(
            ax,
            [
                (totals[phase][-1], f"{phase} {readable(totals[phase][-1])}")
                for phase in PHASES
            ],
        )
        ax.set_xticks([1, 64, 128, 192, 256])
        ax.set_xlabel("sequence length (tokens)", fontsize=8, color=MUTED_INK)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: readable(y)))
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                linestyle="--",
                label="prefill: one L-token prompt",
            ),
            Line2D(
                [],
                [],
                color=color,
                linestyle="-",
                label="decode: tokens 1 to L, one at a time",
            ),
        ],
        loc="outside lower center",
        ncols=2,
        frameon=False,
        fontsize=8,
        labelcolor=SECONDARY_INK,
    )
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def plot_operators(
    results: dict[tuple, Totals], keys: list[str], path: Path
) -> None:
    figure, axes = grid(keys, 3.6)
    title(
        figure,
        f"GPT-Neo int8 FLOPs by operator to decode token L, 2 to "
        f"{PLOT_TOKENS}, batch 1",
    )
    lengths = [length for length in LENGTHS if 2 <= length <= PLOT_TOKENS]
    for ax, key in zip(axes, keys, strict=False):
        plotted: list[int] = []
        drawn: list[list[int]] = []
        for (operator, arithmetic), (_, color) in OPERATOR_LINES.items():
            flops = [
                results[key, "int8", "decode", length].get(
                    (operator, arithmetic), (0, 0)
                )[1]
                for length in lengths
            ]
            plotted += flops
            overlaps = any(
                all(
                    abs(math.log10(a / b)) < OVERLAP_DECADES
                    for a, b in zip(flops, other, strict=True)
                )
                for other in drawn
            )
            drawn.append(flops)
            ax.plot(
                lengths,
                flops,
                color=color,
                linestyle=":" if overlaps else "-",
                linewidth=2.5 if overlaps else 2,
                zorder=3 if overlaps else 2,
            )
        ax.set_title(key, loc="left", fontsize=10, color=INK)
        # Every power of ten labeled, from the one at or below the smallest
        # value to the one at or above the largest.
        ax.set_yscale("log")
        low = math.floor(math.log10(min(plotted)))
        high = math.ceil(math.log10(max(plotted)))
        ax.set_ylim(10**low, 10**high)
        ax.set_yticks([10**power for power in range(low, high + 1)])
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: ""))
        ax.set_xlim(0, PLOT_TOKENS)
        ax.set_xticks([2, 64, 128, 192, 256])
        ax.set_xlabel("sequence length (tokens)", fontsize=8, color=MUTED_INK)
        ax.set_ylabel("FLOPs, log scale", fontsize=8, color=MUTED_INK)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: readable(y)))
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
    figure.legend(
        handles=[
            Line2D([], [], color=color, linewidth=2, label=label)
            for label, color in OPERATOR_LINES.values()
        ]
        + [
            Line2D(
                [],
                [],
                color=SECONDARY_INK,
                linestyle=":",
                linewidth=2.5,
                label="dotted: overlaps a line above",
            )
        ],
        loc="outside lower center",
        ncols=len(OPERATOR_LINES) + 1,
        frameon=False,
        fontsize=8,
        labelcolor=SECONDARY_INK,
    )
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "keys",
        nargs="*",
        choices=list(GPT_NEO_CONFIGS),
        metavar="model",
        help=f"any of {', '.join(GPT_NEO_CONFIGS)} (default: all)",
    )
    keys = parser.parse_args().keys or list(GPT_NEO_CONFIGS)
    results: dict[tuple, Totals] = {}
    rows = []
    for key in keys:
        for phase in PHASES:
            # Decode needs at least one cached token; at L = 1 it is prefill.
            lengths = LENGTHS if phase == "prefill" else LENGTHS[1:]
            for length in lengths:
                records = trace(key, phase, length)
                for precision in PRECISIONS:
                    totals = count(records, precision)
                    results[key, precision, phase, length] = totals
                    ranked = sorted(
                        totals.items(), key=lambda item: -item[1][1]
                    )
                    for (operator, arithmetic), (calls, flops) in ranked:
                        rows.append(
                            [
                                key,
                                precision,
                                phase,
                                length,
                                operator,
                                arithmetic,
                                calls,
                                flops,
                            ]
                        )
        prefill = total(results[key, "fp32", "prefill", PLOT_TOKENS])
        decode = total(results[key, "fp32", "decode", PLOT_TOKENS])
        print(
            f"{key:<26}fp32 prefill of {PLOT_TOKENS} tokens: "
            f"{readable(prefill)}FLOPs; decode of token {PLOT_TOKENS}: "
            f"{readable(decode)}FLOPs"
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "gpt_neo.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "precision",
                "phase",
                "tokens",
                "operator",
                "arithmetic",
                "calls",
                "flops",
            ]
        )
        writer.writerows(rows)
    plot_generation(keys, RESULTS / "prefill_decode.png")
    plot_operators(results, keys, RESULTS / "operators.png")


if __name__ == "__main__":
    main()
