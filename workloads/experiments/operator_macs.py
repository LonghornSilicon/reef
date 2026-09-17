"""Hand-derived multiply-add (MAC) counts per operator for every reef model."""

import csv
import importlib
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from models.language.gpt2 import Conv1D
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


def l2_norm(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    x = args[0]
    vectors = x.numel() // x.shape[module.dim]
    return 3 * x.numel() + 2 * vectors


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


INTERPOLATION_TAPS = {"nearest": 1, "bilinear": 2, "bicubic": 4}


def interpolate(
    module: nn.Module, args: tuple, kwargs: dict, out: object
) -> int:
    x = args[0]
    height, width = x.shape[-2:]
    out_h, out_w = out.shape[-2:]
    leading = x.numel() // (height * width)
    taps = INTERPOLATION_TAPS[module.mode]
    # Separable R x Qᵀ counted by its nonzero taps, not the dense matmul the
    # kernel runs: columns first, then rows.
    return leading * taps * (height * out_w + out_h * out_w)


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


def distance_to_box(
    module: nn.Module, args: tuple, kwargs: dict, out: object
) -> int:
    # Two subtracts and two adds per anchor; xywh adds six more.
    return out.numel() * 5 // 2 if module.xywh else out.numel()


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
    "ErfGELU": elementwise(5),
    "GELU": elementwise(9),
    "LayerNorm": layer_norm,
    "RMSNorm": rms_norm,
    "BatchNorm2d": batch_norm,
    "L2Norm": l2_norm,
    "MaxPool2d": max_pool,
    "AdaptiveAvgPool2d": adaptive_avg_pool,
    "Interpolate": interpolate,
    "RotaryEmbedding": rotary_tables,
    "Embedding": free,
    "Dropout": free,
    "DFL": free,
    "DistanceToBox": distance_to_box,
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


def config_image(config: object) -> Inputs:
    size = config.image_size
    return (torch.randn(BATCH, 3, size, size),)


def tokens(config: object) -> Inputs:
    return (torch.randint(0, 100, (BATCH, SEQ_LEN)),)


def text_image(config: object) -> Inputs:
    length = config.text.max_position_embeddings
    size = config.vision.image_size
    return (
        torch.randint(0, 100, (BATCH, length)),
        torch.randn(BATCH, 3, size, size),
    )


FAMILIES: dict[str, tuple[str, Callable[[object], Inputs]]] = {
    "vgg": ("classification", image(224)),
    "googlenet": ("classification", image(224)),
    "resnet": ("classification", image(224)),
    "mobilenet_v3": ("classification", image(224)),
    "efficientnet": ("classification", image(224)),
    "convnext": ("classification", image(224)),
    "yolov8": ("detection", image(640)),
    "yolo11": ("detection", image(640)),
    "vit": ("vision_transformer", config_image),
    "swin": ("vision_transformer", config_image),
    "mobilevit": ("vision_transformer", config_image),
    "dinov2": ("vision_transformer", config_image),
    "segformer": ("vision_transformer", image(512)),
    "clip": ("vision_language", text_image),
    "siglip": ("vision_language", text_image),
    "gpt2": ("language", tokens),
    "bloom": ("language", tokens),
    "llama": ("language", tokens),
    "qwen2": ("language", tokens),
    "qwen3": ("language", tokens),
}


def count(family: str, key: str, config: object) -> dict[str, list[int]]:
    category, inputs = FAMILIES[family]
    module = importlib.import_module(f"models.{category}.{family}")
    build = getattr(module, family)
    # Meta tensors carry shapes without storage, so even the billion-parameter
    # models cost nothing to run.
    with torch.device("meta"), torch.no_grad():
        model = build(key).eval()
        totals = instrument(model)
        model(*inputs(config))
    return totals


def run(family: str) -> None:
    category = FAMILIES[family][0]
    configs = getattr(
        importlib.import_module(f"configs.{category}.{family}"),
        f"{family.upper()}_CONFIGS",
    )
    rows = []
    for key, config in configs.items():
        totals = count(family, key, config)
        total = sum(macs for _, macs in totals.values())
        print(f"\n{key}: {total / 1e9:.3f} G MACs")
        ranked = sorted(totals.items(), key=lambda item: -item[1][1])
        for operator, (calls, macs) in ranked:
            share = macs / total
            millions = macs / 1e6
            print(f"  {operator:<24}{calls:>6}{millions:>14.2f} M{share:>8.1%}")
            rows.append([key, operator, calls, macs, f"{share:.6f}"])
        calls = sum(calls for calls, _ in totals.values())
        rows.append([key, "Total", calls, total, "1"])
    folder = RESULTS / category
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{family}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "operator", "calls", "macs", "share"])
        writer.writerows(rows)


def main() -> None:
    for family in sys.argv[1:] or FAMILIES:
        run(family)


if __name__ == "__main__":
    main()
