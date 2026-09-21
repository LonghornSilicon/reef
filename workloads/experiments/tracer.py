"""Per-call operator records (MACs, element traffic, GEMM shapes) on meta."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from models.gpt2 import Conv1D
from operators.attention import GroupedQueryAttention
from operators.pooling import AdaptiveAvgPool2d

Formula = Callable[[nn.Module, tuple, dict, object], int]
# (M, N, K, count): count independent (M, K) by (K, N) products.
Gemm = tuple[int, int, int, int]
Gemms = Callable[[nn.Module, tuple, object], list[Gemm]]


@dataclass
class Record:
    """One operator call; element counts are dtype-free (meta is fp32)."""

    operator: str
    path: str
    macs: int
    input_numel: int
    weight_numel: int
    output_numel: int
    # Materialized attention scores and probabilities; zero when fused.
    intermediate_numel: int = 0
    gemms: list[Gemm] = field(default_factory=list)
    depthwise: bool = False
    # False for the Softmax inside attention, which a fused kernel absorbs.
    in_fused: bool = True

    @property
    def numel(self) -> int:
        return (
            self.input_numel
            + self.weight_numel
            + self.output_numel
            + self.intermediate_numel
        )

    @property
    def fused_numel(self) -> int:
        if not self.in_fused:
            return 0
        return self.input_numel + self.weight_numel + self.output_numel


def argument(args: tuple, kwargs: dict, index: int, name: str) -> object:
    return args[index] if len(args) > index else kwargs.get(name)


# Convention: one MAC per scalar multiply, add, subtract, divide, compare or
# nonlinearity (exp, erf, tanh, rsqrt, ...); a d-wide dot product costs d.
# Reshapes, copies, selects and masked fills cost nothing.
#
# Traffic is what each operator module reads and writes: input, parameters
# and buffers, output. Module hooks cannot see residual adds, the SwiGLU gate
# multiply, GPT-2's wte + wpe add or EfficientNet's squeeze-excite multiply,
# so none of those are counted.


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


def linear_gemms(module: nn.Module, args: tuple, out: object) -> list[Gemm]:
    # Shapes come from the activations, not the weight: Conv1D stores its
    # weight as (in, out).
    d_in, d_out = args[0].shape[-1], out.shape[-1]
    return [(out.numel() // d_out, d_out, d_in, 1)]


def conv2d_gemms(module: nn.Module, args: tuple, out: object) -> list[Gemm]:
    if is_depthwise(module):
        return []
    batch, out_channels, out_h, out_w = out.shape
    groups = module.groups
    taps = module.in_channels // groups * module.kernel_size**2
    return [(batch * out_h * out_w, out_channels // groups, taps, groups)]


def attention_gemms(module: nn.Module, args: tuple, out: object) -> list[Gemm]:
    batch, heads, query_len, head_dim = args[0].shape
    key_len = args[1].shape[2]
    count = batch * heads
    return [
        (query_len, key_len, head_dim, count),
        (query_len, head_dim, key_len, count),
    ]


GEMMS: dict[str, Gemms] = {
    "Linear": linear_gemms,
    "Conv2d": conv2d_gemms,
    "GroupedQueryAttention": attention_gemms,
}


def is_depthwise(module: nn.Module) -> bool:
    return module.groups == module.in_channels == module.out_channels


def operator_name(module: nn.Module) -> str | None:
    # GPT-2's Conv1D is a Linear with a transposed weight, defined in the model.
    if isinstance(module, Conv1D):
        return "Linear"
    if type(module).__module__.startswith("operators."):
        return type(module).__name__
    return None


def numel(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, tuple | list):
        return sum(numel(item) for item in value)
    return 0


def state_numel(module: nn.Module) -> int:
    tensors = list(module.parameters(recurse=False))
    tensors += list(module.buffers(recurse=False))
    return sum(tensor.numel() for tensor in tensors)


def record(
    name: str,
    path: str,
    module: nn.Module,
    args: tuple,
    kwargs: dict,
    out: object,
) -> Record:
    gemms = GEMMS[name](module, args, out) if name in GEMMS else []
    if name == "GroupedQueryAttention":
        batch, heads, query_len, _ = args[0].shape
        # Scores are written once and read back as probabilities.
        intermediate = 2 * batch * heads * query_len * args[1].shape[2]
        inputs = numel(args[:3])
    else:
        intermediate = 0
        inputs = numel(args[0])
    return Record(
        operator=name,
        path=path,
        macs=FORMULAS[name](module, args, kwargs, out),
        input_numel=inputs,
        weight_numel=state_numel(module),
        output_numel=numel(out),
        intermediate_numel=intermediate,
        gemms=gemms,
        depthwise=name == "Conv2d" and is_depthwise(module),
    )


def instrument(model: nn.Module) -> list[Record]:
    """Hook every operator; return the live list the hooks append to."""
    records: list[Record] = []
    fused_softmax = {
        module.softmax
        for module in model.modules()
        if isinstance(module, GroupedQueryAttention)
    }
    for path, module in model.named_modules():
        name = operator_name(module)
        if name is None:
            continue

        def hook(
            module: nn.Module,
            args: tuple,
            kwargs: dict,
            out: object,
            name: str = name,
            path: str = path,
        ) -> None:
            entry = record(name, path, module, args, kwargs, out)
            entry.in_fused = module not in fused_softmax
            records.append(entry)

        module.register_forward_hook(hook, with_kwargs=True)
        if name == "RotaryEmbedding":
            # Models call apply_rotary directly, bypassing forward hooks.
            original = module.apply_rotary

            def apply_rotary(
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                module: nn.Module = module,
                path: str = path,
                original: Callable = original,
            ) -> torch.Tensor:
                records.append(
                    Record(
                        operator="RotaryEmbedding",
                        path=path,
                        macs=rotary_apply(module, x),
                        input_numel=x.numel() + cos.numel() + sin.numel(),
                        weight_numel=0,
                        output_numel=x.numel(),
                    )
                )
                return original(x, cos, sin)

            module.apply_rotary = apply_rotary
    return records


def build(family: str, key: str) -> nn.Module:
    """Construct a model on the meta device: shapes only, no storage."""
    module = importlib.import_module(f"models.{family}")
    with torch.device("meta"):
        return getattr(module, family)(key).eval()


def trace(model: nn.Module, *inputs: object) -> list[Record]:
    records = instrument(model)
    with torch.no_grad():
        model(*inputs)
    return records


def max_positions(model: nn.Module) -> int:
    config = model.config
    if hasattr(config, "n_positions"):
        return config.n_positions
    return config.max_position_embeddings


def lengths(family: str, key: str, wanted: tuple[int, ...]) -> tuple[int, ...]:
    """``wanted`` clipped to the model's position limit, without repeats."""
    limit = max_positions(build(family, key))
    return tuple(dict.fromkeys(min(length, limit) for length in wanted))


def tokens(batch: int, length: int) -> torch.Tensor:
    return torch.zeros(batch, length, dtype=torch.long, device="meta")


def prefill(family: str, key: str, batch: int, length: int) -> list[Record]:
    model = build(family, key)
    # Meta indexing does no bounds check, so an over-long prompt would
    # silently trace positions the model cannot represent.
    assert length <= max_positions(model)
    return trace(model, tokens(batch, length))


def decode(family: str, key: str, batch: int, context: int) -> list[Record]:
    """One token after ``context`` cached tokens."""
    model = build(family, key)
    assert context + 1 <= max_positions(model)
    with torch.no_grad():
        _, cache = model(tokens(batch, context))
    return trace(model, tokens(batch, 1), cache)


def classify(family: str, key: str, batch: int, size: int) -> list[Record]:
    model = build(family, key)
    return trace(model, torch.zeros(batch, 3, size, size, device="meta"))
