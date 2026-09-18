"""KV-cache capacity and decode bandwidth versus context length and batch."""

import csv
import sys
from pathlib import Path

from matplotlib.lines import Line2D

from experiments import quant, tracer
from experiments.plots import RAMP, READABLE, SERIES, SURFACE, panels

RESULTS = Path(__file__).resolve().parents[1] / "results" / "kv_context"
MODELS = {"gpt2": "GPT-2", "llama": "SmolLM2-135M"}
PRECISIONS = ("bf16", "int8", "int4")
BATCHES = (1, 4, 16, 64)
LENGTHS = tuple(128 << i for i in range(7))
BUDGETS = {
    "256 MB": 256 << 20,
    "512 MB": 512 << 20,
    "1 GB": 1 << 30,
    "2 GB": 2 << 30,
    "4 GB": 4 << 30,
    "8 GB": 8 << 30,
}
BANDWIDTHS = tuple(4 << i for i in range(8))
# Ramp steps 300, 400, 500 and 700: the lighter ones vanish on the surface.
SHOWN = dict(
    zip((8, 32, 128, 512), (RAMP[2], RAMP[3], RAMP[4], RAMP[6]), strict=True)
)
CAPACITY_HEADER = [
    "model",
    "weights",
    "kv",
    "budget",
    "batch",
    "weight_bytes",
    "kv_bytes_per_token",
    "max_length",
]
DECODE_HEADER = [
    "model",
    "weights",
    "kv",
    "batch",
    "length",
    "bytes_per_step",
    "bandwidth_gbs",
    "step_ms",
    "stream_tokens_per_s",
    "total_tokens_per_s",
]


class Model:
    """Weight and KV footprints of one model, from its meta-device instance."""

    def __init__(self, family: str, key: str) -> None:
        model = tracer.build(family, key)
        self.key = key
        self.limit = tracer.max_positions(model)
        self.weights = {
            p: quant.model_weight_bytes(model, p) for p in PRECISIONS
        }
        self.kv = {p: quant.kv_bytes_per_token(model, p) for p in PRECISIONS}

    def max_length(self, weights: str, kv: str, budget: int, batch: int) -> int:
        free = budget - self.weights[weights]
        if free <= 0:
            return 0
        return min(self.limit, int(free / (batch * self.kv[kv])))

    def bytes_per_step(
        self, weights: str, kv: str, batch: int, length: int
    ) -> float:
        # Activations are ignored: at decode they are one token wide.
        return self.weights[weights] + batch * length * self.kv[kv]


def capacity_rows(model: Model) -> list[list]:
    rows = []
    for weights in PRECISIONS:
        for kv in PRECISIONS:
            for budget, size in BUDGETS.items():
                for batch in BATCHES:
                    rows.append(
                        [
                            model.key,
                            weights,
                            kv,
                            budget,
                            batch,
                            f"{model.weights[weights]:.0f}",
                            f"{model.kv[kv]:.0f}",
                            model.max_length(weights, kv, size, batch),
                        ]
                    )
    return rows


def decode_rows(model: Model) -> list[list]:
    rows = []
    for weights in PRECISIONS:
        for kv in PRECISIONS:
            for batch in BATCHES:
                for length in LENGTHS:
                    if length > model.limit:
                        continue
                    step = model.bytes_per_step(weights, kv, batch, length)
                    for bandwidth in BANDWIDTHS:
                        seconds = step / (bandwidth * 1e9)
                        rows.append(
                            [
                                model.key,
                                weights,
                                kv,
                                batch,
                                length,
                                f"{step:.0f}",
                                bandwidth,
                                f"{seconds * 1e3:.3f}",
                                f"{1 / seconds:.1f}",
                                f"{batch / seconds:.1f}",
                            ]
                        )
    return rows


def plot_capacity(model: Model, path: Path) -> None:
    figure, axes = panels(f"{model.key} context that fits", 1, len(PRECISIONS))
    sizes = list(BUDGETS.values())
    for ax, kv in zip(axes, PRECISIONS, strict=True):
        for color, batch in zip(SERIES, BATCHES, strict=False):
            for weights, dashes in (("bf16", ""), ("int4", (4, 2))):
                lengths = [
                    model.max_length(weights, kv, size, batch) for size in sizes
                ]
                ax.plot(
                    sizes,
                    # Weights alone overflow the budget: leave a gap, not a 1.
                    [length or float("nan") for length in lengths],
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    linestyle=(0, dashes) if dashes else "-",
                )
        ax.axhline(model.limit, color="black", linewidth=0.8, dashes=(2, 2))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(sizes, list(BUDGETS))
        ax.set_xlabel("memory budget")
        ax.set_ylabel("max context length")
        ax.set_title(f"KV {kv}", loc="left")
    handles = [
        Line2D([], [], color=color, linewidth=2, label=f"batch {batch}")
        for color, batch in zip(SERIES, BATCHES, strict=False)
    ]
    handles += [
        Line2D([], [], color="black", linewidth=2, label="bf16 weights"),
        Line2D(
            [],
            [],
            color="black",
            linewidth=2,
            dashes=(4, 2),
            label="int4 weights",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=0.8,
            dashes=(2, 2),
            label="position limit",
        ),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=7)
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def plot_decode(model: Model, path: Path) -> None:
    figure, axes = panels(
        f"{model.key} decode throughput, bf16 weights", 1, len(PRECISIONS)
    )
    lengths = [length for length in LENGTHS if length <= model.limit]
    for ax, kv in zip(axes, PRECISIONS, strict=True):
        for bandwidth, color in SHOWN.items():
            for batch, dashes in ((1, ""), (16, (4, 2))):
                rates = [
                    batch
                    * bandwidth
                    * 1e9
                    / model.bytes_per_step("bf16", kv, batch, length)
                    for length in lengths
                ]
                ax.plot(
                    lengths,
                    rates,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    linestyle=(0, dashes) if dashes else "-",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(READABLE)
        ax.set_xlabel("context length")
        ax.set_ylabel("tokens per second")
        ax.set_title(f"KV {kv}", loc="left")
    handles = [
        Line2D([], [], color=color, linewidth=2, label=f"{bandwidth} GB/s")
        for bandwidth, color in SHOWN.items()
    ]
    handles += [
        Line2D([], [], color="black", linewidth=2, label="batch 1"),
        Line2D(
            [], [], color="black", linewidth=2, dashes=(4, 2), label="batch 16"
        ),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=6)
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    capacity, decode = [], []
    for family in sys.argv[1:] or MODELS:
        model = Model(family, MODELS[family])
        capacity += capacity_rows(model)
        decode += decode_rows(model)
        for precision in PRECISIONS:
            megabytes = model.weights[precision] / 1e6
            print(
                f"{model.key:<14}{precision:<6}weights {megabytes:>8.1f} MB"
                f"   KV/token {model.kv[precision]:>8.0f} B"
            )
        plot_capacity(model, RESULTS / f"{family}_capacity.png")
        plot_decode(model, RESULTS / f"{family}_decode.png")
    for name, header, rows in (
        ("capacity", CAPACITY_HEADER, capacity),
        ("decode", DECODE_HEADER, decode),
    ):
        with open(RESULTS / f"{name}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
