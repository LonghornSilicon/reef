"""Per-call operator records (FLOPs, element traffic, GEMM shapes) on meta."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from workloads.operators.attention import GroupedQueryAttention

Formula = Callable[[nn.Module, tuple, dict, object], int]
# (M, N, K, count): count independent (M, K) by (K, N) products.
Gemm = tuple[int, int, int, int]
Gemms = Callable[[nn.Module, tuple, object], list[Gemm]]


@dataclass
class Record:
    """One operator call; element counts are dtype-free (meta is fp32)."""

    operator: str
    path: str
    flops: int
    input_numel: int
    weight_numel: int
    output_numel: int
    # Materialized attention scores and probabilities; zero when fused.
    intermediate_numel: int = 0
    gemms: list[Gemm] = field(default_factory=list)
    # False for the Softmax inside attention, which a fused kernel absorbs.
    in_fused: bool = True

    @property
    def numel(self) -> int:
        """Every element this call reads or writes, unfused."""
        return (
            self.input_numel
            + self.weight_numel
            + self.output_numel
            + self.intermediate_numel
        )

    @property
    def fused_numel(self) -> int:
        """Traffic with fused attention: no scores, no inner Softmax."""
        if not self.in_fused:
            return 0
        return self.input_numel + self.weight_numel + self.output_numel


def argument(args: tuple, kwargs: dict, index: int, name: str) -> object:
    """Argument ``index`` if passed positionally, else ``kwargs.get(name)``."""
    return args[index] if len(args) > index else kwargs.get(name)


# Convention: one FLOP per scalar multiply, add, subtract, divide, compare or
# nonlinearity (exp, erf, tanh, rsqrt, ...); a d-wide dot product costs 2d, a
# multiply and an add per term. Reshapes, copies, selects and masked fills
# cost nothing.
#
# Traffic is what each operator module reads and writes: input, parameters
# and buffers, output. Module hooks cannot see residual adds or the wte + wpe
# add, so neither is counted.


def linear(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    """FLOPs of one `Linear` call, bias add included."""
    d_in, d_out = args[0].shape[-1], out.shape[-1]
    positions = out.numel() // d_out
    bias = positions * d_out if module.bias is not None else 0
    return 2 * positions * d_in * d_out + bias


def attention(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    """FLOPs of one `GroupedQueryAttention` call, bias add included."""
    batch, heads, query_len, head_dim = args[0].shape
    key_len = args[1].shape[2]
    scores = batch * heads * query_len * key_len
    bias = scores if argument(args, kwargs, 5, "bias") is not None else 0
    # QKᵀ and softmax(S)·V are each a head_dim dot product per score; the
    # scale is one multiply per score. Softmax is counted under its own row.
    return 2 * 2 * scores * head_dim + scores + bias


def elementwise(ops: int) -> Formula:
    """A formula charging ``ops`` FLOPs per input element."""

    def formula(
        module: nn.Module, args: tuple, kwargs: dict, out: object
    ) -> int:
        return ops * args[0].numel()

    return formula


def layer_norm(
    module: nn.Module, args: tuple, kwargs: dict, out: object
) -> int:
    """FLOPs of one `LayerNorm` call, shift included if present."""
    x = args[0]
    vectors = x.numel() // x.shape[-1]
    # mean, subtract, square, mean, multiply, gain (+ shift) per element;
    # two mean divides, eps add and rsqrt per vector.
    per_element = 6 if module.bias is not None else 5
    return per_element * x.numel() + 4 * vectors


def free(module: nn.Module, args: tuple, kwargs: dict, out: object) -> int:
    """Zero FLOPs, for lookups such as `Embedding`."""
    return 0


FORMULAS: dict[str, Formula] = {
    "Linear": linear,
    "GroupedQueryAttention": attention,
    "Softmax": elementwise(5),
    "GELU": elementwise(9),
    "LayerNorm": layer_norm,
    "Embedding": free,
}


def linear_gemms(module: nn.Module, args: tuple, out: object) -> list[Gemm]:
    """One (positions, out_features, in_features, 1) product."""
    d_in, d_out = args[0].shape[-1], out.shape[-1]
    return [(out.numel() // d_out, d_out, d_in, 1)]


def attention_gemms(module: nn.Module, args: tuple, out: object) -> list[Gemm]:
    """QKᵀ then PV, each repeated once per (batch, head)."""
    batch, heads, query_len, head_dim = args[0].shape
    key_len = args[1].shape[2]
    count = batch * heads
    return [
        (query_len, key_len, head_dim, count),
        (query_len, head_dim, key_len, count),
    ]


GEMMS: dict[str, Gemms] = {
    "Linear": linear_gemms,
    "GroupedQueryAttention": attention_gemms,
}


def operator_name(module: nn.Module) -> str | None:
    """Class name of a `workloads.operators` module, else None."""
    if type(module).__module__.startswith("workloads.operators."):
        return type(module).__name__
    return None


def numel(value: object) -> int:
    """Elements of a tensor or a nested tuple/list of them; 0 otherwise."""
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, tuple | list):
        return sum(numel(item) for item in value)
    return 0


def state_numel(module: nn.Module) -> int:
    """Parameters and buffers owned by ``module`` itself, not children."""
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
    """The `Record` for one hooked call of operator ``name`` at ``path``."""
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
        flops=FORMULAS[name](module, args, kwargs, out),
        input_numel=inputs,
        weight_numel=state_numel(module),
        output_numel=numel(out),
        intermediate_numel=intermediate,
        gemms=gemms,
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
    return records


def build(family: str, key: str) -> nn.Module:
    """Construct a model on the meta device: shapes only, no storage."""
    module = importlib.import_module(f"workloads.models.{family}")
    with torch.device("meta"):
        return getattr(module, family)(key).eval()


def trace(model: nn.Module, *inputs: object) -> list[Record]:
    """Records for one forward pass; the hooks stay on ``model``."""
    records = instrument(model)
    with torch.no_grad():
        model(*inputs)
    return records


def max_positions(model: nn.Module) -> int:
    """Longest sequence ``model``'s learned positions can represent."""
    return model.config.max_position_embeddings


def lengths(family: str, key: str, wanted: tuple[int, ...]) -> tuple[int, ...]:
    """``wanted`` clipped to the model's position limit, without repeats."""
    limit = max_positions(build(family, key))
    return tuple(dict.fromkeys(min(length, limit) for length in wanted))


def tokens(batch: int, length: int) -> torch.Tensor:
    """A (batch, length) meta tensor of token ids."""
    return torch.zeros(batch, length, dtype=torch.long, device="meta")


def prefill(family: str, key: str, batch: int, length: int) -> list[Record]:
    """Records for one ``length``-token prompt on a fresh meta model."""
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
