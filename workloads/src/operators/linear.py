"""Dense affine projection."""

import math

import torch
from torch import nn


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
