"""KV cache bytes per GPT-Neo size, sequence length and precision."""

import argparse
import csv
import math
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS, GPTNeoConfig
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

RESULTS = Path(__file__).resolve().parents[4] / "results" / "kv_cache_size"
# Bits per element.
PRECISIONS = {"fp32": 32, "bf16": 16, "int8": 8, "int4": 4}
# int8 and int4 add one fp16 scale per token, per head, for K and for V.
SCALE_BYTES = 2
# Stored: every layer keeps every token, as models/gpt_neo.py does. Needed:
# local layers keep only the last window_size tokens.
CACHES = ("stored", "needed")
# Size is linear in length except for one bend at window_size (256), itself a
# power of two, so straight lines through these points are exact.
LENGTHS = tuple(2**exponent for exponent in range(12))
# The plot stops at window_size (256): up to there stored and needed
# are the same size, so it shows one line per precision.
PLOT_TOKENS = 256


def kv_bytes(
    config: GPTNeoConfig, length: int, precision: str, windowed: bool
) -> int:
    assert length <= config.max_position_embeddings
    total = 0
    for layer in range(config.num_layers):
        tokens = length
        if windowed and config.attention_type(layer) == "local":
            tokens = min(length, config.window_size)
        bits = PRECISIONS[precision]
        # K and V, each (heads, tokens, head_dim) = tokens * hidden elements.
        total += 2 * tokens * config.hidden_size * bits // 8
        if bits <= 8:
            total += 2 * tokens * config.num_heads * SCALE_BYTES
    return total


def megabytes(value: float) -> str:
    return "0" if value == 0 else readable(value) + "B"


def plot(keys: list[str], path: Path) -> None:
    lengths = [length for length in LENGTHS if length <= PLOT_TOKENS]
    columns = min(3, len(keys))
    rows = math.ceil(len(keys) / columns)
    figure = Figure(
        figsize=(5.5 * columns, 3.6 * rows),
        facecolor=SURFACE,
        layout="constrained",
    )
    figure.get_layout_engine().set(h_pad=0.25)
    figure.suptitle(
        f"GPT-Neo KV cache, batch 1, 1 to {PLOT_TOKENS} tokens",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    axes = list(figure.subplots(rows, columns, squeeze=False).flat)
    for ax, key in zip(axes, keys, strict=False):
        config = GPT_NEO_CONFIGS[key]
        for color, precision in zip(SERIES, PRECISIONS, strict=True):
            sizes = [
                kv_bytes(config, length, precision, windowed=False)
                for length in lengths
            ]
            ax.plot(lengths, sizes, color=color, linewidth=2)
            ax.annotate(
                f"{precision} {megabytes(sizes[-1])}",
                (lengths[-1], sizes[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=SECONDARY_INK,
            )
        ax.set_title(key, loc="left", fontsize=10, color=INK)
        # Headroom so the end labels are not clipped.
        ax.set_xlim(0, PLOT_TOKENS * 1.32)
        ax.set_ylim(0, None)
        ax.set_xticks([1, 64, 128, 192, 256])
        ax.set_xlabel("sequence length (tokens)", fontsize=8, color=MUTED_INK)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: megabytes(y)))
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    figure.legend(
        handles=[
            Line2D([], [], color=color, linewidth=2, label=precision)
            for color, precision in zip(SERIES, PRECISIONS, strict=True)
        ],
        loc="outside lower center",
        ncols=len(PRECISIONS),
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
    rows = [
        [
            key,
            precision,
            cache,
            length,
            kv_bytes(
                GPT_NEO_CONFIGS[key], length, precision, cache == "needed"
            ),
        ]
        for key in keys
        for precision in PRECISIONS
        for cache in CACHES
        for length in LENGTHS
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "gpt_neo.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "precision", "cache", "tokens", "bytes"])
        writer.writerows(rows)
    plot(keys, RESULTS / "gpt_neo.png")
    limit = max(LENGTHS)
    for key in keys:
        config = GPT_NEO_CONFIGS[key]
        sizes = ", ".join(
            f"{p} {megabytes(kv_bytes(config, limit, p, False))}"
            for p in PRECISIONS
        )
        print(f"{key:<26}at {limit} tokens: {sizes}")


if __name__ == "__main__":
    main()
