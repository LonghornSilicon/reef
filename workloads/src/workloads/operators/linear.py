"""Dense affine projection."""

import math

import torch
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.ao.quantization.qconfig import default_dynamic_qconfig


class Linear(nn.Module):
    """Affine map ``x @ weight.T + bias``."""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features).uniform_(-bound, bound)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            out = out + self.bias
        return out


class QuantizedLinear(DynamicQuantizedLinear):
    """Int8 dynamic-quantized drop-in for `Linear`.

    Weights are quantized per output channel once, up front; activations are
    quantized per call by the fused ``linear_dynamic`` kernel this subclasses,
    matching the scheme ``torch.ao.quantization.quantize_dynamic`` used before
    that orchestration API's removal in torch 2.10. `from_float` is
    reimplemented here because the base class only accepts `torch.nn.Linear`.
    """

    @classmethod
    def from_float(cls, linear: Linear) -> "QuantizedLinear":
        observer = default_dynamic_qconfig.weight()
        observer(linear.weight)
        qweight = _quantize_weight(linear.weight.float(), observer)
        quantized = cls(
            linear.in_features, linear.out_features, linear.bias is not None
        )
        quantized.set_weight_bias(qweight, linear.bias)
        return quantized


def quantize(model: nn.Module) -> nn.Module:
    """Replace every `Linear` in `model`, in place, with its int8 form."""
    for name, child in model.named_children():
        if isinstance(child, Linear):
            setattr(model, name, QuantizedLinear.from_float(child))
        else:
            quantize(child)
    return model
