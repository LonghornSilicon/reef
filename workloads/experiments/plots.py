"""Number formatting, palette and axis styling shared by experiment plots."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
BAR = "#2a78d6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
# Categorical slots in fixed order; a series keeps its slot across panels.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
# One-hue sequential ramp, light to dark.
RAMP = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)


def readable(value: float) -> str:
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if value >= scale:
            scaled = value / scale
            # Not :.3g, which switches to exponent notation from 1000 up.
            if scaled >= 100:
                return f"{scaled:.0f} {unit}"
            digits = 1 if scaled >= 10 else 2
            return f"{scaled:.{digits}f}".rstrip("0").rstrip(".") + f" {unit}"
    return f"{value:.0f}"


def percent(share: float) -> str:
    if 0 < share < 0.001:
        return "<0.1%"
    return f"{share:.1%}"


READABLE = FuncFormatter(lambda value, _: readable(value))


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
    ax.tick_params(axis="y", labelcolor=SECONDARY_INK)
    ax.title.set_color(INK)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def panels(
    title: str, rows: int, columns: int, width: float = 5.5, height: float = 3.6
) -> tuple[Figure, list[Axes]]:
    figure = Figure(
        figsize=(width * columns, height * rows),
        facecolor=SURFACE,
        layout="constrained",
    )
    figure.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    axes = list(figure.subplots(rows, columns, squeeze=False).flat)
    for ax in axes:
        style(ax)
    return figure, axes
