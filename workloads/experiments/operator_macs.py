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
from torch import nn

from models.gpt2 import Conv1D
from operators.pooling import AdaptiveAvgPool2d

RESULTS = Path(__file__).resolve().parents[1] / "results" / "operator_macs"
BATCH = 1
SEQ_LEN = 128
Inputs = tuple[torch.Tensor, ...]
Formula = Callable[[nn.Module, tuple, dict, object], int]


def argument(args: tuple, kwargs: dict, index: int, name: str) -> object:
    return args[index] if len(args) > index else kwargs.get(name)


# Convention: one MAC per scalar multiply, add, subtract, divide, compare or
# nonlinearity (exp, erf, tanh, rsqrt, ...); a d-wide dot product costs d.
# Reshapes, copies, selects and masked fills cost nothing.


def linear(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    d_in, d_out = args[0].shape[-1], out.shape[-1]
    positions = out.numel() // d_out
    bias = positions * d_out if module.bias is not None else 0
    return positions * d_in * d_out + bias


def conv2d(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    taps = module.in_channels // module.groups * module.kernel_size**2
    bias = out.numel() if module.bias is not None else 0
    return out.numel() * taps + bias


def attention(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    batch, heads, query_len, head_dim = args[0].shape
    key_len = args[1].shape[2]
    scores = batch * heads * query_len * key_len
    bias = scores if argument(args, kwargs, 5, "bias") is not None else 0
    # QKᵀ and softmax(S)·V are each a head_dim dot product per score; the
    # scale is one multiply per score. Softmax is counted under its own row.
    return 2 * scores * head_dim + scores + bias


def elementwise(ops: int) -> Formula:
    def formula(
        module: nn.Module, args: tuple, kwargs: dict, out: object
    ) -> int:
        return ops * args[0].numel()

    return formula


def layer_norm(
    module: nn.Module, args: tuple, kwargs: dict, out: object
) -> int:
    x = args[0]
    vectors = x.numel() // x.shape[-1]
    # mean, subtract, square, mean, multiply, gain (+ shift) per element;
    # two mean divides, eps add and rsqrt per vector.
    per_element = 6 if module.bias is not None else 5
    return per_element * x.numel() + 4 * vectors


def rms_norm(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    x = args[0]
    vectors = x.numel() // x.shape[-1]
    return 4 * x.numel() + 3 * vectors


def batch_norm(
    module: nn.Module, args: tuple, kwargs: dict, out: object
) -> int:
    x = args[0]
    return 4 * x.numel() + 2 * x.shape[1]


def max_pool(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    return out.numel() * module.kernel_size**2


def window_span(size: int, out_size: int) -> int:
    """Total input rows covered by all adaptive-pooling windows on one axis."""
    return sum(
        -(-(i + 1) * size // out_size) - i * size // out_size
        for i in range(out_size)
    )


def adaptive_avg_pool(
    module: AdaptiveAvgPool2d, args: tuple, kwargs: dict, out: object
) -> int:
    batch, channels, height, width = args[0].shape
    out_h, out_w = module.output_size
    area = window_span(height, out_h) * window_span(width, out_w)
    return batch * channels * area + out.numel()


def rotary_tables(
    module: nn.Module, args: tuple, kwargs: dict, out: object
) -> int:
    length, pairs = args[0].numel(), module.rotary_dim // 2
    # position * frequency, then cos and sin over both halves.
    return length * pairs + 2 * length * module.rotary_dim


def rotary_apply(module: nn.Module, x: torch.Tensor) -> int:
    rotated = x.numel() // module.head_dim * module.rotary_dim
    # x·cos, x·sin and their sum per element, plus negating half of them.
    return 3 * rotated + rotated // 2


def free(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    return 0


FORMULAS: dict[str, Formula] = {
    "Linear": linear,
    "Conv2d": conv2d,
    "GroupedQueryAttention": attention,
    "Softmax": elementwise(5),
    "ReLU": elementwise(1),
    "Sigmoid": elementwise(1),
    "SiLU": elementwise(2),
    "GELU": elementwise(9),
    "LayerNorm": layer_norm,
    "RMSNorm": rms_norm,
    "BatchNorm2d": batch_norm,
    "MaxPool2d": max_pool,
    "AdaptiveAvgPool2d": adaptive_avg_pool,
    "RotaryEmbedding": rotary_tables,
    "Embedding": free,
    "Dropout": free,
}


def operator_name(module: nn.Module) -> str | None:
    # GPT-2's Conv1D is a Linear with a transposed weight, defined in the model.
    if isinstance(module, Conv1D):
        return "Linear"
    if type(module).__module__.startswith("operators."):
        return type(module).__name__
    return None


def instrument(model: nn.Module) -> dict[str, list[int]]:
    """Hook every operator; return live ``{operator: [calls, macs]}``."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for module in model.modules():
        name = operator_name(module)
        if name is None:
            continue
        formula = FORMULAS[name]

        def hook(
            module: nn.Module,
            args: tuple,
            kwargs: dict,
            out: object,
            name: str = name,
            formula: Formula = formula,
        ) -> None:
            totals[name][0] += 1
            totals[name][1] += formula(module, args, kwargs, out)

        module.register_forward_hook(hook, with_kwargs=True)
        if name == "RotaryEmbedding":
            # Models call apply_rotary directly, bypassing forward hooks.
            original = module.apply_rotary

            def apply_rotary(
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                module: nn.Module = module,
                original: Callable = original,
            ) -> torch.Tensor:
                totals["RotaryEmbedding"][0] += 1
                totals["RotaryEmbedding"][1] += rotary_apply(module, x)
                return original(x, cos, sin)

            module.apply_rotary = apply_rotary
    return totals


def image(size: int) -> Callable[[object], Inputs]:
    return lambda config: (torch.randn(BATCH, 3, size, size),)


def tokens(config: object) -> Inputs:
    return (torch.randint(0, 100, (BATCH, SEQ_LEN)),)


FAMILIES: dict[str, Callable[[object], Inputs]] = {
    "googlenet": image(224),
    "resnet": image(224),
    "efficientnet": image(224),
    "gpt2": tokens,
    "llama": tokens,
}


def count(family: str, key: str, config: object) -> dict[str, list[int]]:
    inputs = FAMILIES[family]
    module = importlib.import_module(f"models.{family}")
    build = getattr(module, family)
    # Meta tensors carry shapes without storage, so even the billion-parameter
    # models cost nothing to run.
    with torch.device("meta"), torch.no_grad():
        model = build(key).eval()
        totals = instrument(model)
        model(*inputs(config))
    return totals


SURFACE = "#fcfcfb"
BAR = "#2a78d6"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLUMNS = 3
Ranked = list[tuple[str, list[int]]]


def readable(macs: float) -> str:
    for unit, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if macs >= scale:
            value = macs / scale
            # Not :.3g, which switches to exponent notation from 1000 up.
            if value >= 100:
                return f"{value:.0f} {unit}"
            digits = 1 if value >= 10 else 2
            return f"{value:.{digits}f}".rstrip("0").rstrip(".") + f" {unit}"
    return f"{macs:.0f}"


def percent(share: float) -> str:
    if 0 < share < 0.001:
        return "<0.1%"
    return f"{share:.1%}"


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
