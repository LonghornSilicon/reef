"""Hand-derived multiply-add (MAC) counts per operator for every reef model."""

import csv
import importlib
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from experiments import tracer
from experiments.plots import (
    BAR,
    BASELINE,
    GRIDLINE,
    INK,
    MUTED_INK,
    SECONDARY_INK,
    SURFACE,
    percent,
    readable,
)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "operator_macs"
BATCH = 1
SEQ_LEN = 128
Inputs = tuple[torch.Tensor, ...]


def image(size: int) -> Callable[[object], Inputs]:
    return lambda config: (torch.zeros(BATCH, 3, size, size, device="meta"),)


def tokens(config: object) -> Inputs:
    return (torch.zeros(BATCH, SEQ_LEN, dtype=torch.long, device="meta"),)


FAMILIES: dict[str, Callable[[object], Inputs]] = {
    "googlenet": image(224),
    "resnet": image(224),
    "efficientnet": image(224),
    "gpt2": tokens,
    "llama": tokens,
    "gpt_neo": tokens,
}


def count(family: str, key: str, config: object) -> dict[str, list[int]]:
    inputs = FAMILIES[family](config)
    model = tracer.build(family, key)
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in tracer.trace(model, *inputs):
        totals[record.operator][0] += 1
        totals[record.operator][1] += record.macs
    return totals


COLUMNS = 3
Ranked = list[tuple[str, list[int]]]


def plot(family: str, results: dict[str, Ranked], path: Path) -> None:
    columns = min(COLUMNS, len(results))
    rows = math.ceil(len(results) / columns)
    bars = max(len(ranked) for ranked in results.values())
    figure = Figure(
        figsize=(5.5 * columns, (0.35 * bars + 1.2) * rows),
        facecolor=SURFACE,
        layout="constrained",
    )
    figure.get_layout_engine().set(h_pad=0.25)
    figure.suptitle(
        family,
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    axes = list(figure.subplots(rows, columns, squeeze=False).flat)
    for ax, (key, ranked) in zip(axes, results.items(), strict=False):
        names = [operator for operator, _ in ranked]
        macs = [value for _, (_, value) in ranked]
        total = sum(macs)
        positions = range(len(ranked))
        # 0.45 of a 0.35 in slot at 150 dpi keeps bars under 24 px thick.
        ax.barh(positions, macs, height=0.45, color=BAR)
        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        # Headroom so the value label on the longest bar is not clipped.
        ax.set_xlim(0, max(macs) * 1.45)
        for y, value in zip(positions, macs, strict=True):
            label = f"  {readable(value)}  ({percent(value / total)})"
            ax.text(
                value, y, label, va="center", fontsize=8, color=SECONDARY_INK
            )
        ax.set_title(
            f"{key}  ·  {readable(total)} MACs",
            loc="left",
            fontsize=10,
            color=INK,
        )
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: readable(x)))
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
        ax.tick_params(axis="y", labelcolor=SECONDARY_INK)
    for ax in axes[len(results) :]:
        ax.set_visible(False)
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def run(family: str) -> None:
    configs = getattr(
        importlib.import_module(f"configs.{family}"),
        f"{family.upper()}_CONFIGS",
    )
    rows = []
    results = {}
    for key, config in configs.items():
        totals = count(family, key, config)
        total = sum(macs for _, macs in totals.values())
        print(f"\n{key}: {total / 1e9:.3f} G MACs")
        ranked = sorted(totals.items(), key=lambda item: -item[1][1])
        results[key] = ranked
        for operator, (calls, macs) in ranked:
            share = macs / total
            millions = macs / 1e6
            print(f"  {operator:<24}{calls:>6}{millions:>14.2f} M{share:>8.1%}")
            rows.append([key, operator, calls, macs, f"{share:.6f}"])
        calls = sum(calls for calls, _ in totals.values())
        rows.append([key, "Total", calls, total, "1"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{family}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "operator", "calls", "macs", "share"])
        writer.writerows(rows)
    plot(family, results, RESULTS / f"{family}.png")


def main() -> None:
    for family in sys.argv[1:] or FAMILIES:
        run(family)


if __name__ == "__main__":
    main()
